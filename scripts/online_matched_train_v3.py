#!/usr/bin/env python3
"""Matched v13 online-policy pretraining, GRPO, and greedy validation.

This module is deliberately separate from the legacy 6,341-action trainer.  It
uses the 6,373-action raw-root/per-flow-STOP contract and delegates every
feature vector, legal mask, state transition, frame, and timing measurement to
``native_online_training_server``.  Python owns sampling and gradient updates;
it never implements a graph transition or compression fallback.

The reference architecture is kept intentionally unchanged apart from the
output identity: 49 -> tanh(128) -> tanh(128) -> 6,373.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

import weight_conditioned_grpo_v1 as grpo
from export_weighted_grpo_native_policy_v1 import export_checkpoint


SCHEMA = "native-online-matched-training-v3"
DATA_SCHEMA = "v13-training-data-v3"
TEACHER_SCHEMA = "native-online-teacher-records-v3"
RESUME_SCHEMA = "native-online-matched-resume-v3"
SERVER_SCHEMA = "native-online-training-session-v1"
ACTION_COUNT = 6373
DISABLED_GLOBAL_STOP = 0
FLOW_STOP_BEGIN = 6341
FLOW_STOP_COUNT = 32
REGISTRY_FINGERPRINT = (
    "cbd7756de4ebaf544a4eac138dc92c9eac85c701d09e8855a1b624e456170542"
)
FEATURE_SCHEMA = grpo.FEATURE_SCHEMA
TASK_FEATURE_DIM = 18
STATE_FEATURE_DIM = 28
OBSERVATION_DIM = 49
FIXED_WEIGHTS = grpo.FIXED_WEIGHTS
DTYPE_WIDTH = {"int32": 4, "int64": 8, "float32": 4, "float64": 8}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_positive(value: Any, label: str) -> float:
    result = float(value)
    require(math.isfinite(result) and result > 0.0, f"{label} must be finite and positive")
    return result


def _weights(value: Sequence[float]) -> tuple[float, float, float]:
    result = grpo.validate_weight(value)
    return tuple(float(item) for item in result)  # type: ignore[return-value]


def size_bucket(raw_bytes: int) -> str:
    require(type(raw_bytes) is int and raw_bytes > 0, "B_raw must be positive")
    if raw_bytes < 64 * 1024:
        return "lt64k"
    if raw_bytes < 256 * 1024:
        return "64k_256k"
    if raw_bytes < 1024 * 1024:
        return "256k_1m"
    return "ge1m"


@dataclass(frozen=True)
class OnlineContract:
    action_count: int = ACTION_COUNT
    disabled_global_stop: int = DISABLED_GLOBAL_STOP
    flow_stop_begin: int = FLOW_STOP_BEGIN
    flow_stop_count: int = FLOW_STOP_COUNT
    registry_fingerprint: str = REGISTRY_FINGERPRINT
    feature_schema: str = FEATURE_SCHEMA
    task_feature_dim: int = TASK_FEATURE_DIM
    state_feature_dim: int = STATE_FEATURE_DIM
    max_policy_steps: int = 10
    max_nodes: int = 64
    max_active_streams: int = 32
    max_ordinary_depth: int = 10
    raw_root: bool = True
    online_candidate_search: bool = False

    def __post_init__(self) -> None:
        require(self.action_count == ACTION_COUNT, "v3 action count drift")
        require(self.disabled_global_stop == 0, "global STOP identity drift")
        require(self.flow_stop_begin == FLOW_STOP_BEGIN and
                self.flow_stop_count == FLOW_STOP_COUNT, "flow STOP identity drift")
        require(self.registry_fingerprint == REGISTRY_FINGERPRINT,
                "v3 registry fingerprint drift")
        require(self.task_feature_dim + self.state_feature_dim + 3 == OBSERVATION_DIM,
                "observation dimension drift")
        require(1 <= self.max_policy_steps <= 64, "invalid policy horizon")
        require(1 <= self.max_nodes <= 64 and 1 <= self.max_active_streams <= 32 and
                1 <= self.max_ordinary_depth <= 10, "invalid graph limits")
        require(self.raw_root and not self.online_candidate_search,
                "v3 must use raw roots with no online candidate search")

    @property
    def graph_limits(self) -> dict[str, int]:
        return {"max_nodes": self.max_nodes,
                "max_active_streams": self.max_active_streams,
                "max_ordinary_depth": self.max_ordinary_depth}

    @property
    def fingerprint(self) -> str:
        return digest(canonical_json(asdict(self)))


@dataclass(frozen=True)
class TrainingBlock:
    sample_id: str
    dataset_id: str
    domain: str
    split: str
    calibration_key: str
    root_fingerprint: str
    roots: tuple[dict[str, Any], ...]
    original_numeric_bytes: int
    compression_reference_ns_per_byte: float
    decompression_reference_ns_per_byte: float

    def __post_init__(self) -> None:
        require(self.sample_id != "" and self.dataset_id != "" and self.domain != "" and
                self.calibration_key != "",
                "empty block identity")
        require(self.split in {"train", "validation"}, "test data cannot enter v3 training")
        require(len(self.root_fingerprint) == 64, "invalid root fingerprint")
        require(self.original_numeric_bytes > 0, "zero B_raw is N/A, not a training block")
        _finite_positive(self.compression_reference_ns_per_byte, "E0")
        _finite_positive(self.decompression_reference_ns_per_byte, "D0")
        require(0 < len(self.roots) <= 32, "invalid root count")
        numeric = sum(len(bytes.fromhex(str(root["data_hex"])))
                      for root in self.roots if root.get("original_numeric_payload") is True)
        require(numeric == self.original_numeric_bytes, "root B_raw denominator mismatch")

    @property
    def bucket(self) -> str:
        return size_bucket(self.original_numeric_bytes)

    @property
    def request_roots(self) -> list[dict[str, Any]]:
        return [dict(root) for root in self.roots]


def _resolve_payload(path: str, manifest: Path, materialized_root: Path | None) -> Path:
    raw = Path(path)
    resolved = raw.resolve() if raw.is_absolute() else (manifest.parent / raw).resolve()
    if materialized_root is not None:
        base = materialized_root.resolve()
        require(resolved == base or base in resolved.parents,
                f"payload escapes materialized root: {resolved}")
    require(resolved.is_file(), f"payload file absent: {resolved}")
    return resolved


def _span(path: Path, offset: int, length: int) -> bytes:
    require(type(offset) is int and offset >= 0 and type(length) is int and length >= 0,
            "invalid byte span")
    with path.open("rb") as handle:
        handle.seek(offset)
        value = handle.read(length)
    require(len(value) == length, f"truncated payload span: {path}")
    return value


def _bit_span(path: Path, offset_bits: int, bit_count: int) -> bytes:
    require(type(offset_bits) is int and offset_bits >= 0 and
            type(bit_count) is int and bit_count >= 0, "invalid bitmap span")
    if bit_count == 0:
        return b""
    first = offset_bits // 8
    last = (offset_bits + bit_count + 7) // 8
    raw = _span(path, first, last - first)
    out = bytearray((bit_count + 7) // 8)
    local = offset_bits - first * 8
    for index in range(bit_count):
        if (raw[(local + index) // 8] >> ((local + index) % 8)) & 1:
            out[index // 8] |= 1 << (index % 8)
    return bytes(out)


def _reference_map(path: Path | None) -> dict[str, tuple[float, float]]:
    if path is None:
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    require(document.get("schema") == "native-online-time-reference-v3",
            "unsupported v3 time reference")
    result: dict[str, tuple[float, float]] = {}
    for row in document.get("samples", ()):
        key = str(row["sample_id"])
        require(key not in result, "duplicate reference sample")
        result[key] = (_finite_positive(row["compression_reference_ns_per_byte"], "E0"),
                       _finite_positive(row["decompression_reference_ns_per_byte"], "D0"))
    if "global_reference" in document:
        row = document["global_reference"]
        require(row.get("calibration_split") == "train" and
                row.get("validation_used_for_calibration") is False,
                "global E0/D0 must be calibrated from train only")
        result["__train_global__"] = (
            _finite_positive(row["compression_reference_ns_per_byte"], "global E0"),
            _finite_positive(row["decompression_reference_ns_per_byte"], "global D0"))
    return result


def load_training_data(path: Path | Sequence[Path], *, reference_path: Path | None = None,
                       materialized_root: Path | None = None) -> tuple[TrainingBlock, ...]:
    """Load the v3 train/validation manifest and hash every referenced span."""

    paths = (path,) if isinstance(path, Path) else tuple(Path(item) for item in path)
    require(paths, "at least one v3 split manifest is required")
    documents = [(item, json.loads(item.read_text(encoding="utf-8"))) for item in paths]
    require(all(document.get("schema") == DATA_SCHEMA for _, document in documents),
            "unsupported v13 data manifest")
    require({str(document.get("split")) for _, document in documents} ==
            {"train", "validation"}, "both independent train/validation v3 manifests are required")
    require(len({str(document.get("cohort_sha256")) for _, document in documents}) == 1,
            "train/validation cohort identity mismatch")
    for _, document in documents:
        require(document.get("test_used_for_training_or_statistics") is False and
                document.get("web20_used_for_training_or_statistics") is False,
                "held-out/test source contaminated v3 data")
    references = _reference_map(reference_path)
    if reference_path is not None:
        reference_document = json.loads(reference_path.read_text(encoding="utf-8"))
        bound_hashes = {str(row["sha256"])
                        for row in reference_document.get("manifest_bindings", ())}
        actual_hashes = {digest(item.read_bytes()) for item in paths}
        require(bound_hashes == actual_hashes,
                "time reference is not bound to both exact v3 split manifests")
    blocks: list[TrainingBlock] = []
    seen: set[str] = set()
    for manifest_path, document in documents:
        split = str(document["split"])
        sources = tuple(document.get("sources", ()))
        require(sources, "v3 manifest source list is empty")
        source_ids = {str(row["dataset_id"]) for row in sources}
        require(len(source_ids) == len(sources), "duplicate dataset source identity")
        rows = document.get("samples", ())
        require(isinstance(rows, list) and rows, f"v3 {split} split is empty")
        for row in rows:
            sample_id = str(row["sample_id"])
            require(sample_id not in seen, "sample_id must be globally unique")
            seen.add(sample_id)
            require(str(row["split"]) == split, "split record mismatch")
            dataset_id = str(row["dataset_id"])
            require(dataset_id in source_ids, "sample references an unbound source")
            roots: list[dict[str, Any]] = []
            fingerprints: list[dict[str, Any]] = []
            total_raw = 0
            for root in row["roots"]:
                kind = str(root["kind"])
                if kind == "numeric":
                    dtype = str(root["dtype"])
                    width = int(root["width_bytes"])
                    require(dtype in DTYPE_WIDTH and DTYPE_WIDTH[dtype] == width and
                            root.get("byte_order") == "little" and
                            root.get("original_numeric_payload") is True,
                            "unsupported numeric root contract")
                    numeric_path = _resolve_payload(str(root["numeric_path"]), manifest_path,
                                                    materialized_root)
                    payload = _span(numeric_path, int(root["numeric_offset_bytes"]),
                                    int(root["numeric_bytes"]))
                    require(digest(payload) == str(root["numeric_sha256"]),
                            "numeric span hash drift")
                    raw = int(root["B_raw"])
                    require(raw == len(payload), "numeric root denominator mismatch")
                    total_raw += raw
                    roots.append({"kind": "numeric", "width": width,
                                  "data_hex": payload.hex(), "logical_view": dtype,
                                  "original_numeric_payload": True})
                    fingerprints.append({"kind": kind, "dtype": dtype, "width": width,
                                         "payload_sha256": digest(payload),
                                         "payload_bytes": len(payload), "B_raw": raw})
                elif kind == "serial":
                    require(int(root["width_bytes"]) == 1 and root.get("logical_view") == "bytes" and
                            root.get("original_numeric_payload") is False and
                            root.get("bitmap_semantics") == "one_bit_means_missing_lsb_first" and
                            int(root.get("B_raw", -1)) == 0,
                            "unsupported bitmap root contract")
                    bitmap_path = _resolve_payload(str(root["bitmap_path"]), manifest_path,
                                                   materialized_root)
                    payload = _bit_span(bitmap_path, int(root["bitmap_offset_bits"]),
                                        int(root["bitmap_bits"]))
                    require(len(payload) == int(root["bitmap_bytes"]) and
                            digest(payload) == str(root["bitmap_sha256"]),
                            "bitmap span hash drift")
                    roots.append({"kind": "serial", "width": 1,
                                  "data_hex": payload.hex(), "logical_view": "bytes",
                                  "original_numeric_payload": False})
                    fingerprints.append({"kind": kind, "width": 1,
                                         "payload_sha256": digest(payload),
                                         "payload_bytes": len(payload),
                                         "bitmap_bits": int(root["bitmap_bits"]), "B_raw": 0})
                else:
                    raise ValueError("unknown v3 root kind")
            require(total_raw == int(row["B_raw"]), "sample B_raw mismatch")
            calibration_key = str(row["calibration_key"])
            if "__train_global__" in references:
                E0, D0 = references["__train_global__"]
            elif sample_id in references or calibration_key in references:
                E0, D0 = references.get(sample_id, references.get(calibration_key))  # type: ignore[arg-type]
            else:
                E0 = _finite_positive(row["compression_reference_ns_per_byte"], "E0")
                D0 = _finite_positive(row["decompression_reference_ns_per_byte"], "D0")
            root_fp = digest(canonical_json({
                "sample_id": sample_id, "dataset_id": dataset_id, "split": split,
                "row_offset": int(row["row_offset"]), "row_count": int(row["row_count"]),
                "B_raw": total_raw, "roots": fingerprints,
            }))
            blocks.append(TrainingBlock(
                sample_id, dataset_id, str(row["domain"]), split, calibration_key,
                root_fp, tuple(roots),
                total_raw, E0, D0))
    require(not references or "__train_global__" in references or
            all(block.sample_id in references or block.calibration_key in references
                for block in blocks),
            "time reference does not cover every v3 sample/calibration key")
    return tuple(sorted(blocks, key=lambda value: (value.split, value.dataset_id,
                                                   value.sample_id)))


@dataclass(frozen=True)
class Snapshot:
    terminal: bool
    observation: np.ndarray | None
    legal_action_ids: tuple[int, ...]
    response: Mapping[str, Any]


class OnlineBackend(Protocol):
    model_sha256: str | None

    def load_model(self, model: Path) -> str: ...
    def reset(self, block: TrainingBlock, weight: Sequence[float],
              contract: OnlineContract) -> Snapshot: ...
    def step(self, action_id: int) -> Snapshot: ...
    def finish(self) -> Mapping[str, Any]: ...
    def replay(self, block: TrainingBlock, weight: Sequence[float],
               contract: OnlineContract, recipe: Sequence[int], *,
               warmups: int, repetitions: int,
               capture_transitions: bool) -> Mapping[str, Any]: ...


class OnlineRequestTimeout(TimeoutError):
    """One native request exceeded its watchdog; the owned worker was restarted."""


class OnlineNativeFailure(RuntimeError):
    """C++ rejected/failed one request; no compression metric is fabricated."""


class OnlineTrainingServer:
    """Strict JSONL client for one C++ OnlineSession worker."""

    def __init__(self, executable: Path, catalog: Path, profile: Path,
                 *, cpu: int | None = None, timeout_seconds: float = 3600.0):
        command = [str(executable), str(catalog), str(profile)]
        if cpu is not None:
            command = ["taskset", "-c", str(int(cpu)), *command]
        env = dict(os.environ)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        self.command, self.environment = command, env
        self.timeout_seconds = float(timeout_seconds)
        self.request_id = 0
        self.model_sha256: str | None = None
        self.model_path: Path | None = None
        self._spawn()

    def _spawn(self) -> None:
        def limit_native_worker() -> None:
            import resource
            limit = 64 * 1024 ** 3
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        import sys
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1, env=self.environment,
                                        preexec_fn=(limit_native_worker
                                                    if sys.platform.startswith("linux") else None))

    def _kill(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)
        self._close_pipes()

    def _close_pipes(self) -> None:
        for handle in (self.process.stdin, self.process.stdout, self.process.stderr):
            if handle is not None and not handle.closed:
                handle.close()

    def _restart(self) -> None:
        model = self.model_path
        self._kill()
        self._spawn()
        self.model_sha256 = None
        if model is not None:
            self.load_model(model)

    def call(self, value: Mapping[str, Any]) -> dict[str, Any]:
        require(self.process.poll() is None, "native online training server exited")
        self.request_id += 1
        request = {**value, "request_id": self.request_id}
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(canonical_json(request).decode("utf-8") + "\n")
        self.process.stdin.flush()
        import selectors
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        ready = selector.select(self.timeout_seconds)
        selector.close()
        if not ready:
            operation = str(value.get("op", "unknown"))
            self._restart()
            raise OnlineRequestTimeout(
                f"native request {operation} exceeded {self.timeout_seconds:.3f}s; worker restarted")
        line = self.process.stdout.readline()
        require(line != "", "native online training server closed stdout")
        response = json.loads(line)
        require(response.get("request_id") == self.request_id, "native response id mismatch")
        if response.get("ok") is not True:
            error = str(response.get("error", "unknown error"))
            if value.get("op") in {"reset", "observe", "step", "finish", "replay"}:
                self._restart()
            raise OnlineNativeFailure("native online request failed: " + error)
        return response

    @staticmethod
    def _request(block: TrainingBlock, weight: Sequence[float],
                 contract: OnlineContract) -> dict[str, Any]:
        return {"roots": block.request_roots, "weights": list(_weights(weight)),
                "max_policy_steps": contract.max_policy_steps,
                "graph_limits": contract.graph_limits}

    def _snapshot(self, response: Mapping[str, Any]) -> Snapshot:
        require(response.get("schema") == SERVER_SCHEMA, "training server schema drift")
        terminal = bool(response["terminal"])
        if terminal:
            return Snapshot(True, None, (), response)
        require(response.get("registry_fingerprint") == REGISTRY_FINGERPRINT and
                int(response.get("action_count", -1)) == ACTION_COUNT and
                int(response.get("flow_stop_begin", -1)) == FLOW_STOP_BEGIN,
                "training server action identity drift")
        require(int(response.get("candidate_evaluations", 0)) == 0 and
                int(response.get("replayed_prefix_stages", 0)) == 0,
                "online candidate/replay search is forbidden")
        observation = np.asarray(response["observation"], dtype=np.float64)
        legal = tuple(int(value) for value in response["legal_action_ids"])
        require(observation.shape == (OBSERVATION_DIM,) and np.isfinite(observation).all(),
                "invalid C++ observation")
        require(legal == tuple(sorted(set(legal))) and legal and
                all(0 <= value < ACTION_COUNT for value in legal), "invalid C++ legal mask")
        require(DISABLED_GLOBAL_STOP not in legal, "legacy global STOP became legal")
        return Snapshot(False, observation, legal, response)

    def load_model(self, model: Path) -> str:
        response = self.call({"op": "load_model", "model_path": str(model.resolve())})
        require(response.get("schema") == SERVER_SCHEMA and
                response.get("model_loaded") is True,
                "C++ server did not load the WGPN")
        model_sha = str(response["model_sha256"])
        require(len(model_sha) == 64 and model_sha == digest(model.read_bytes()),
                "loaded WGPN hash mismatch")
        self.model_sha256 = model_sha
        self.model_path = model.resolve()
        return model_sha

    def reset(self, block: TrainingBlock, weight: Sequence[float],
              contract: OnlineContract) -> Snapshot:
        return self._snapshot(self.call({"op": "reset", **self._request(block, weight, contract)}))

    def step(self, action_id: int) -> Snapshot:
        require(type(action_id) is int, "action id must be int")
        return self._snapshot(self.call({"op": "step", "action_id": action_id}))

    def finish(self) -> Mapping[str, Any]:
        return self.call({"op": "finish"})

    def replay(self, block: TrainingBlock, weight: Sequence[float],
               contract: OnlineContract, recipe: Sequence[int], *, warmups: int,
               repetitions: int, capture_transitions: bool) -> Mapping[str, Any]:
        require(all(type(value) is int and 0 < value < ACTION_COUNT for value in recipe),
                "forced recipe contains invalid/global STOP action")
        return self.call({"op": "replay", **self._request(block, weight, contract),
                          "forced_action_ids": list(recipe),
                          "timing_warmups": int(warmups),
                          "timing_repetitions": int(repetitions),
                          "capture_transitions": bool(capture_transitions)})

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.call({"op": "shutdown"})
            except Exception:
                self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        self._close_pipes()

    def __enter__(self) -> "OnlineTrainingServer":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass(frozen=True)
class RecipeMeasurement:
    recipe: tuple[int, ...]
    frame_bytes: int
    original_numeric_bytes: int
    compression_time_ns: int
    decompression_time_ns: int
    model_sha256: str
    frame_sha256: str
    canonical_recipe_key: str
    reward_timing_eligible: bool

    @property
    def compression_ratio(self) -> float:
        return self.frame_bytes / self.original_numeric_bytes

    @property
    def compression_ns_per_byte(self) -> float:
        return self.compression_time_ns / self.original_numeric_bytes

    @property
    def decompression_ns_per_byte(self) -> float:
        return self.decompression_time_ns / self.original_numeric_bytes


def _terminal(response: Mapping[str, Any]) -> Mapping[str, Any]:
    value = response.get("terminal_result", response)
    require(isinstance(value, Mapping), "terminal result is absent")
    return value


def _recipe(response: Mapping[str, Any]) -> tuple[int, ...]:
    value = tuple(int(item) for item in _terminal(response)["recipe_action_ids"])
    require(value and all(0 < action < ACTION_COUNT for action in value),
            "invalid terminal recipe")
    return value


def parse_measurement(response: Mapping[str, Any], block: TrainingBlock,
                      model_sha256: str, *, require_reward_timing: bool = True) -> RecipeMeasurement:
    terminal = _terminal(response)
    eligible = terminal.get("reward_timing_eligible") is True
    if require_reward_timing:
        require(eligible, "model-free/fixed-recipe timing cannot define J_total")
    require(str(terminal.get("model_sha256")) == model_sha256,
            "reward timing used a different model")
    raw = int(terminal["original_numeric_bytes"])
    require(raw == block.original_numeric_bytes, "measurement B_raw mismatch")
    enc = tuple(int(value) for value in terminal["encode_samples_ns"])
    dec = tuple(int(value) for value in terminal["decode_samples_ns"])
    require(enc and dec and all(value >= 0 for value in (*enc, *dec)),
            "invalid native timing samples")
    frame_bytes = int(terminal["frame_bytes"])
    require(frame_bytes > 0, "complete frame is empty")
    frame_sha = str(terminal["frame_sha256"])
    require(len(frame_sha) == 64, "invalid frame hash")
    recipe_key = str(terminal["canonical_recipe_key"])
    require(recipe_key != "", "canonical recipe key is absent")
    require(terminal.get("timing_scope") == "matched_policy_forced_action_total",
            "reward timing scope is not the matched online total")
    return RecipeMeasurement(_recipe(terminal), frame_bytes, raw,
                             int(statistics.median(enc)), int(statistics.median(dec)),
                             model_sha256, frame_sha, recipe_key, eligible)


def objective(measurement: RecipeMeasurement, block: TrainingBlock,
              weight: Sequence[float]) -> float:
    a, b, c = _weights(weight)
    E = measurement.compression_ns_per_byte
    D = measurement.decompression_ns_per_byte
    value = (a * measurement.compression_ratio +
             b * E / (E + block.compression_reference_ns_per_byte) +
             c * D / (D + block.decompression_reference_ns_per_byte))
    require(math.isfinite(value), "non-finite J_total")
    return value


def pareto_front(measurements: Sequence[RecipeMeasurement]) -> tuple[RecipeMeasurement, ...]:
    unique = {row.recipe: row for row in measurements}
    require(len(unique) == len(measurements), "duplicate recipe measurement")
    rows = tuple(unique.values())
    kept = []
    for index, row in enumerate(rows):
        point = (row.compression_ratio, row.compression_ns_per_byte,
                 row.decompression_ns_per_byte)
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            rival = (other.compression_ratio, other.compression_ns_per_byte,
                     other.decompression_ns_per_byte)
            if all(a <= b for a, b in zip(rival, point)) and any(
                    a < b for a, b in zip(rival, point)):
                dominated = True
                break
        if not dominated:
            kept.append(row)
    return tuple(sorted(kept, key=lambda row: row.recipe))


def _mask(legal: Sequence[int]) -> np.ndarray:
    result = np.zeros(ACTION_COUNT, dtype=bool)
    result[list(legal)] = True
    return result


def _probabilities(policy: grpo.TwoHiddenMLP, observation: np.ndarray,
                   legal: Sequence[int]) -> np.ndarray:
    logits, _ = policy.forward(observation[None, :])
    ids = np.asarray(legal, dtype=np.int64)
    selected = logits[0, ids]
    selected -= selected.max()
    values = np.exp(selected)
    values /= values.sum()
    require(np.isfinite(values).all(), "non-finite policy distribution")
    return values


def rollout(backend: OnlineBackend, policy: grpo.TwoHiddenMLP,
            block: TrainingBlock, weight: Sequence[float], contract: OnlineContract,
            rng: np.random.Generator, *, greedy: bool,
            policy_adapter: FlatPolicyAdapter | None = None) -> dict[str, Any]:
    adapter = policy_adapter or FlatPolicyAdapter(policy)
    snapshot = backend.reset(block, weight, contract)
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    old_logp: list[float] = []
    response: Mapping[str, Any] = snapshot.response
    while not snapshot.terminal:
        assert snapshot.observation is not None
        probabilities = adapter.probabilities(snapshot.observation,
                                              snapshot.legal_action_ids)
        index = int(np.argmax(probabilities) if greedy else
                    rng.choice(len(snapshot.legal_action_ids), p=probabilities))
        action = snapshot.legal_action_ids[index]
        features.append(snapshot.observation.copy())
        masks.append(_mask(snapshot.legal_action_ids))
        actions.append(action)
        old_logp.append(float(math.log(max(float(probabilities[index]), 1e-300))))
        try:
            snapshot = backend.step(action)
        except (OnlineRequestTimeout, OnlineNativeFailure) as error:
            return {"features": np.asarray(features, dtype=np.float64),
                    "legal": np.asarray(masks, dtype=bool),
                    "actions": np.asarray(actions, dtype=np.int64),
                    "old_logp": np.asarray(old_logp, dtype=np.float64),
                    "recipe": tuple(actions), "terminal": None,
                    "failure": type(error).__name__ + ": " + str(error)}
        response = snapshot.response
    recipe = _recipe(response)
    require(tuple(actions) == recipe[:len(actions)], "C++ terminal recipe changed policy prefix")
    return {"features": np.asarray(features, dtype=np.float64),
            "legal": np.asarray(masks, dtype=bool),
            "actions": np.asarray(actions, dtype=np.int64),
            "old_logp": np.asarray(old_logp, dtype=np.float64),
            "recipe": recipe, "terminal": response, "failure": None}


def constraint_failure_objective(block: TrainingBlock,
                                 weight: Sequence[float]) -> float:
    a, b, c = _weights(weight)
    return float(1.0 + a * ((1 << 30) / block.original_numeric_bytes) + b + c)


def random_recipe(backend: OnlineBackend, block: TrainingBlock,
                  weight: Sequence[float], contract: OnlineContract,
                  rng: np.random.Generator, *, stop_probability: float = 0.25) -> tuple[int, ...]:
    snapshot = backend.reset(block, weight, contract)
    response: Mapping[str, Any] = snapshot.response
    while not snapshot.terminal:
        ordinary = [value for value in snapshot.legal_action_ids
                    if 0 < value < FLOW_STOP_BEGIN]
        stops = [value for value in snapshot.legal_action_ids if value >= FLOW_STOP_BEGIN]
        choose_stop = bool(stops) and (not ordinary or rng.random() < stop_probability)
        pool = stops if choose_stop else ordinary
        snapshot = backend.step(int(pool[int(rng.integers(0, len(pool)))]))
        response = snapshot.response
    return _recipe(response)


def bounded_depth_recipe(backend: OnlineBackend, block: TrainingBlock,
                         contract: OnlineContract, rng: np.random.Generator,
                         ordinary_steps: int) -> tuple[int, ...]:
    """Generate a legal single/multi-operator prefix, then close every leaf."""

    require(ordinary_steps >= 1, "ordinary candidate depth must be positive")
    snapshot = backend.reset(block, FIXED_WEIGHTS[0], contract)
    response: Mapping[str, Any] = snapshot.response
    applied = 0
    while not snapshot.terminal and applied < ordinary_steps:
        ordinary = [value for value in snapshot.legal_action_ids
                    if 0 < value < FLOW_STOP_BEGIN]
        if not ordinary:
            break
        snapshot = backend.step(int(ordinary[int(rng.integers(0, len(ordinary)))]))
        response = snapshot.response
        applied += 1
    if not snapshot.terminal:
        response = backend.finish()
    require(applied >= 1, "no legal ordinary action for teacher candidate")
    return _recipe(response)


def stop_recipe(backend: OnlineBackend, block: TrainingBlock,
                contract: OnlineContract) -> tuple[int, ...]:
    snapshot = backend.reset(block, FIXED_WEIGHTS[0], contract)
    response: Mapping[str, Any] = snapshot.response
    while not snapshot.terminal:
        stops = [value for value in snapshot.legal_action_ids if value >= FLOW_STOP_BEGIN]
        require(stops, "C++ mask exposes no per-flow STOP")
        snapshot = backend.step(max(stops))
        response = snapshot.response
    return _recipe(response)


QUICK_TEACHER_TEMPLATES: tuple[tuple[str, ...], ...] = (
    (), ("lz4",), ("zstd",), ("delta",), ("bitpack",), ("shuffle",),
    ("delta", "lz4"), ("delta", "zstd"),
    ("shuffle", "lz4"), ("shuffle", "zstd"),
    ("delta", "bitpack"), ("delta", "bitpack", "lz4"),
    ("delta", "bitpack", "zstd"),
)
QUICK_VALIDATION_TEMPLATES: tuple[tuple[str, ...], ...] = ((), ("lz4",), ("zstd",))


def load_candidate_profile(catalog_path: Path) -> dict[str, tuple[int, ...]]:
    """Map semantic template families to frozen action IDs; never changes masks."""

    document = json.loads(catalog_path.read_text(encoding="utf-8"))
    require(document.get("registry_fingerprint") ==
            "50d9be2dbc4e77ee8c7fd8a621e68bc869a55f0630daf6b08d6a5def86daf2fd",
            "candidate catalog base registry drift")
    actions = document["registry_payload"]["actions"]
    families: dict[str, list[int]] = {name: [] for name in
                                     ("lz4", "zstd", "delta", "bitpack", "shuffle")}
    for row in actions:
        action = int(row["action_id"])
        if not 0 < action < FLOW_STOP_BEGIN or not row.get("ordinary_operator"):
            continue
        text = " ".join(str(row.get(key, "")).lower() for key in
                        ("key", "runtime_action", "io_variant", "implementation_id"))
        if "lz4" in text:
            families["lz4"].append(action)
        if "zstd" in text:
            families["zstd"].append(action)
        if any(token in text for token in ("delta", "adjacent_residual", "difference")):
            families["delta"].append(action)
        if any(token in text for token in ("bitpack", "minimum_for", "max_value_bitpack")):
            families["bitpack"].append(action)
        if any(token in text for token in ("shuffle", "transpose", "byte_plane")):
            families["shuffle"].append(action)
    require(families["lz4"] and families["zstd"],
            "frozen catalog lacks LZ4/Zstd teacher actions")
    return {name: tuple(sorted(set(values))) for name, values in families.items()}


def template_recipe(backend: OnlineBackend, block: TrainingBlock,
                    contract: OnlineContract, profile: Mapping[str, Sequence[int]],
                    template: Sequence[str]) -> tuple[int, ...]:
    if not template:
        return stop_recipe(backend, block, contract)
    snapshot = backend.reset(block, FIXED_WEIGHTS[0], contract)
    response: Mapping[str, Any] = snapshot.response
    for family in template:
        legal = set(snapshot.legal_action_ids)
        choices = [action for action in profile.get(family, ()) if action in legal]
        require(choices, f"template stage {family} has no legal action")
        # Stable action ID makes candidate coverage reproducible. C++ remains
        # the sole authority on legality and transition semantics.
        snapshot = backend.step(min(choices))
        response = snapshot.response
        require(not snapshot.terminal or family == template[-1],
                "template terminated before all requested stages")
        if snapshot.terminal:
            break
    if not snapshot.terminal:
        response = backend.finish()
    return _recipe(response)


def replay_measure(backend: OnlineBackend, block: TrainingBlock,
                   weight: Sequence[float], contract: OnlineContract,
                   recipe: Sequence[int], *, warmups: int, repetitions: int) -> RecipeMeasurement:
    require(backend.model_sha256 is not None, "load a v3 WGPN before reward measurement")
    response = backend.replay(block, weight, contract, recipe, warmups=warmups,
                              repetitions=repetitions, capture_transitions=False)
    if _terminal(response).get("terminal_status") not in (None, "ok"):
        raise OnlineNativeFailure("forced replay terminal failure: " +
                                  str(_terminal(response).get("error", "unknown")))
    return parse_measurement(response, block, backend.model_sha256,
                             require_reward_timing=True)


def discover_candidates(backend: OnlineBackend, block: TrainingBlock,
                        contract: OnlineContract, rng: np.random.Generator,
                        count: int, *,
                        candidate_profile: Mapping[str, Sequence[int]] | None = None,
                        templates: Sequence[Sequence[str]] | None = None,
                        failure_log: list[dict[str, Any]] | None = None
                        ) -> tuple[tuple[int, ...], ...]:
    require(count >= 2, "teacher search needs at least two candidate rollouts")
    if candidate_profile is not None:
        selected_templates = tuple(templates or QUICK_TEACHER_TEMPLATES)[:count]
        recipes: set[tuple[int, ...]] = set()
        for template in selected_templates:
            try:
                recipes.add(template_recipe(backend, block, contract,
                                            candidate_profile, template))
            except (ValueError, OnlineRequestTimeout, OnlineNativeFailure) as error:
                if failure_log is not None:
                    failure_log.append({"sample_id": block.sample_id,
                                        "template": list(template),
                                        "error": type(error).__name__ + ": " + str(error)})
        require(any(all(action >= FLOW_STOP_BEGIN for action in recipe)
                    for recipe in recipes), "raw STOP teacher reference failed")
        require(any(any(action < FLOW_STOP_BEGIN for action in recipe)
                    for recipe in recipes),
                "no legal non-STOP teacher template")
        return tuple(sorted(recipes))
    recipes = {stop_recipe(backend, block, contract)}
    failures = 0
    # Explicitly cover legal one-, two-, and three-operator prefixes before
    # stochastic exploration. The full online mask stays intact.
    for depth in (1, 2, 3):
        for _ in range(max(1, count // 6)):
            try:
                recipes.add(bounded_depth_recipe(backend, block, contract, rng, depth))
            except (ValueError, OnlineRequestTimeout):
                failures += 1
    attempts = 0
    while len(recipes) < count and attempts < count:
        attempts += 1
        try:
            recipes.add(random_recipe(backend, block, FIXED_WEIGHTS[0], contract, rng))
        except (OnlineRequestTimeout, OnlineNativeFailure):
            failures += 1
    require(len(recipes) >= 2, "teacher search cannot discover a non-STOP recipe")
    require(len(recipes) >= min(count, 4),
            f"teacher search coverage too small after {failures} failed/timeout candidates")
    return tuple(sorted(recipes))


def replay_transitions(backend: OnlineBackend, block: TrainingBlock,
                       weight: Sequence[float], contract: OnlineContract,
                       recipe: Sequence[int]) -> tuple[dict[str, Any], ...]:
    response = backend.replay(block, weight, contract, recipe, warmups=0,
                              repetitions=1, capture_transitions=True)
    rows = tuple(response.get("transitions", ()))
    require(rows, "forced replay omitted transition labels")
    parsed = []
    for step, row in enumerate(rows):
        observation = np.asarray(row["observation"], dtype=np.float64)
        legal = tuple(int(value) for value in row["legal_action_ids"])
        action = int(row["action_id"])
        require(observation.shape == (OBSERVATION_DIM,) and np.isfinite(observation).all(),
                "teacher observation invalid")
        require(action in legal and DISABLED_GLOBAL_STOP not in legal,
                "teacher action is not legal")
        parsed_row = {"step": step, "observation": observation.tolist(),
                      "legal_action_ids": list(legal), "action_id": action}
        if "greedy_action_id" in row:
            parsed_row["greedy_action_id"] = int(row["greedy_action_id"])
        if "structured_observation" in row:
            # Opaque C++-authored trace for later structured-policy ablations.
            parsed_row["structured_observation"] = row["structured_observation"]
        parsed.append(parsed_row)
    require(tuple(row["action_id"] for row in parsed) == tuple(recipe)[:len(parsed)],
            "teacher replay action sequence drift")
    return tuple(parsed)


def build_teacher_records(backend: OnlineBackend, blocks: Sequence[TrainingBlock],
                          contract: OnlineContract, rng: np.random.Generator, *,
                          candidate_count: int = 64,
                          teacher_weights: Sequence[Sequence[float]] = FIXED_WEIGHTS,
                          warmups: int = 1, repetitions: int = 5,
                          candidate_profile: Mapping[str, Sequence[int]] | None = None,
                          failure_log: list[dict[str, Any]] | None = None
                          ) -> tuple[dict[str, Any], ...]:
    require(blocks and all(block.split == "train" for block in blocks),
            "offline teacher may use training sources only")
    require(backend.model_sha256 is not None, "teacher timing requires a loaded v3 WGPN")
    records: list[dict[str, Any]] = []
    for block in blocks:
        candidates = discover_candidates(
            backend, block, contract, rng, candidate_count,
            candidate_profile=candidate_profile, failure_log=failure_log)
        # Approximate teacher timing uses a frozen, recorded model solely to pay
        # the same feature/mask/MLP cost. It is never called an online search.
        measured = []
        for recipe in candidates:
            try:
                measured.append(replay_measure(
                    backend, block, FIXED_WEIGHTS[0], contract, recipe,
                    warmups=warmups, repetitions=repetitions))
            except (OnlineRequestTimeout, OnlineNativeFailure, ValueError) as error:
                if failure_log is not None:
                    failure_log.append({"sample_id": block.sample_id,
                                        "recipe_action_ids": list(recipe),
                                        "error": type(error).__name__ + ": " + str(error)})
        require(len(measured) >= 2, "fewer than raw plus one measured teacher recipe")
        frontier = pareto_front(measured)
        for weight_value in teacher_weights:
            weight = _weights(weight_value)
            best = min(frontier, key=lambda row: (objective(row, block, weight), row.recipe))
            for transition in replay_transitions(backend, block, weight, contract, best.recipe):
                records.append({
                    "schema": TEACHER_SCHEMA, "split": "train",
                    "sample_id": block.sample_id, "dataset_id": block.dataset_id,
                    "domain": block.domain, "root_fingerprint": block.root_fingerprint,
                    "weight": list(weight), "recipe_action_ids": list(best.recipe),
                    "teacher_model_sha256": backend.model_sha256,
                    "teacher_timing_scope": "cpp_total_forced_action_current_model_forward",
                    "teacher_objective": objective(best, block, weight),
                    "teacher_metrics": {
                        "compression_ratio": best.compression_ratio,
                        "compression_ns_per_byte": best.compression_ns_per_byte,
                        "decompression_ns_per_byte": best.decompression_ns_per_byte,
                    }, **transition,
                })
    return tuple(records)


def write_teacher(path: Path, records: Sequence[Mapping[str, Any]], *,
                  contract: OnlineContract, source_manifest_sha256: str) -> dict[str, Any]:
    require(records and all(row.get("split") == "train" for row in records),
            "teacher output is empty or contaminated")
    lines = b"".join(canonical_json(dict(row)) + b"\n" for row in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lines)
    receipt = {"schema": "native-online-teacher-receipt-v3",
               "records": len(records), "datasets": len({row["dataset_id"] for row in records}),
               "samples": len({row["sample_id"] for row in records}),
               "contract": asdict(contract), "contract_sha256": contract.fingerprint,
               "source_manifest_sha256": source_manifest_sha256,
               "teacher_sha256": digest(lines), "training_split_only": True}
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    receipt_path.write_bytes(canonical_json(receipt) + b"\n")
    return receipt


def load_teacher(path: Path, blocks: Sequence[TrainingBlock],
                 contract: OnlineContract) -> tuple[dict[str, Any], ...]:
    block_map = {block.sample_id: block for block in blocks if block.split == "train"}
    require(block_map, "no training blocks")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        require(row.get("schema") == TEACHER_SCHEMA and row.get("split") == "train",
                "invalid/nontrain teacher record")
        block = block_map.get(str(row["sample_id"]))
        require(block is not None and row["root_fingerprint"] == block.root_fingerprint,
                "teacher/data identity mismatch")
        observation = np.asarray(row["observation"], dtype=np.float64)
        legal = tuple(int(value) for value in row["legal_action_ids"])
        action = int(row["action_id"])
        require(observation.shape == (OBSERVATION_DIM,) and np.isfinite(observation).all() and
                action in legal and 0 not in legal, "invalid teacher transition")
        rows.append({**row, "observation": observation, "legal_action_ids": legal,
                     "action_id": action})
    require(rows, "teacher file is empty")
    return tuple(rows)


def policy_identity(train_blocks: Sequence[TrainingBlock],
                    contract: OnlineContract) -> grpo.PolicyIdentity:
    require(train_blocks and all(block.split == "train" for block in train_blocks),
            "policy identity accepts training blocks only")
    data = tuple(sorted((block.sample_id, digest(canonical_json({
        "dataset_id": block.dataset_id, "domain": block.domain,
        "B_raw": block.original_numeric_bytes}))) for block in train_blocks))
    roots = tuple(sorted((block.sample_id, block.root_fingerprint) for block in train_blocks))
    require(len(dict(data)) == len(data) and len(dict(roots)) == len(roots),
            "duplicate policy task identity")
    return grpo.PolicyIdentity(contract.registry_fingerprint, data, roots,
                               contract.action_count, contract.flow_stop_begin,
                               contract.task_feature_dim, contract.state_feature_dim,
                               (128, 128), contract.feature_schema)


def teacher_arrays(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray([row["observation"] for row in records], dtype=np.float64)
    actions = np.asarray([row["action_id"] for row in records], dtype=np.int64)
    legal = np.zeros((len(records), ACTION_COUNT), dtype=bool)
    for index, row in enumerate(records):
        legal[index, list(row["legal_action_ids"])] = True
    require(features.shape == (len(records), OBSERVATION_DIM) and
            np.all(legal[np.arange(len(records)), actions]), "invalid teacher arrays")
    return features, legal, actions


def masked_cross_entropy_update(policy: grpo.TwoHiddenMLP, optimizer: grpo.Adam,
                                features: np.ndarray, legal: np.ndarray,
                                actions: np.ndarray, *, batch_size: int,
                                rng: np.random.Generator,
                                gradient_max_norm: float = 1.0,
                                device: str | None = None) -> dict[str, float | int | str]:
    """One shuffled masked-CE epoch with portable Adam state."""

    from weighted_grpo_torch_backend_v1 import torch_device
    torch, target = torch_device(device or optimizer.execution_device)
    require(features.ndim == 2 and features.shape[1] == OBSERVATION_DIM and
            legal.shape == (features.shape[0], ACTION_COUNT) and
            actions.shape == (features.shape[0],), "masked CE batch shape mismatch")
    require(np.isfinite(features).all() and legal.any(axis=1).all() and
            legal[np.arange(actions.size), actions].all(), "invalid masked CE data")
    order = rng.permutation(features.shape[0])
    params = [torch.tensor(value, dtype=torch.float64, device=target) for value in policy.params]
    first = [torch.tensor(value, dtype=torch.float64, device=target) for value in optimizer.first]
    second = [torch.tensor(value, dtype=torch.float64, device=target) for value in optimizer.second]
    steps = optimizer.steps
    losses = []
    correct = total = 0
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        x = torch.tensor(features[batch], dtype=torch.float64, device=target)
        masks = torch.tensor(legal[batch], dtype=torch.bool, device=target)
        targets = torch.tensor(actions[batch], dtype=torch.int64, device=target)
        work = [value.detach().requires_grad_(True) for value in params]
        w1, b1, w2, b2, w3, b3 = work
        logits = torch.tanh(torch.tanh(x @ w1 + b1) @ w2 + b2) @ w3 + b3
        masked = logits.masked_fill(~masks, -torch.inf)
        loss = torch.nn.functional.cross_entropy(masked, targets, reduction="mean")
        require(bool(torch.isfinite(loss)), "non-finite masked CE loss")
        loss.backward()
        gradients = [value.grad for value in work]
        require(all(value is not None and bool(torch.isfinite(value).all())
                    for value in gradients), "non-finite masked CE gradient")
        norm = torch.sqrt(sum(value.square().sum() for value in gradients if value is not None))
        scale = torch.clamp(float(gradient_max_norm) / norm.clamp_min(1e-30), max=1.0)
        steps += 1
        with torch.no_grad():
            for index, (parameter, gradient) in enumerate(zip(params, gradients)):
                assert gradient is not None
                gradient = gradient * scale
                first[index] = 0.9 * first[index] + 0.1 * gradient
                second[index] = 0.999 * second[index] + 0.001 * gradient.square()
                moment = first[index] / (1.0 - 0.9 ** steps)
                variance = second[index] / (1.0 - 0.999 ** steps)
                parameter -= optimizer.learning_rate * moment / (torch.sqrt(variance) + 1e-8)
        losses.append(float(loss.detach().cpu()))
        correct += int((masked.argmax(dim=1) == targets).sum().detach().cpu())
        total += len(batch)
    require(all(bool(torch.isfinite(value).all()) for value in (*params, *first, *second)),
            "non-finite masked CE optimizer state")
    for destination, source in zip(policy.params, params):
        destination[:] = source.detach().cpu().numpy()
    optimizer.first = [value.detach().cpu().numpy().copy() for value in first]
    optimizer.second = [value.detach().cpu().numpy().copy() for value in second]
    optimizer.steps = steps
    return {"mean_masked_cross_entropy": float(np.mean(losses)),
            "masked_top1_accuracy": correct / total,
            "records": total, "optimizer_steps": len(losses),
            "policy_update_device": str(target), "policy_update_backend": "pytorch_float64"}


def masked_teacher_accuracy(policy: grpo.TwoHiddenMLP,
                            records: Sequence[Mapping[str, Any]], *,
                            batch_size: int = 256) -> dict[str, float | int]:
    """Evaluate all teacher tokens and the raw-root first-action subset."""

    require(records, "teacher accuracy requires records")
    correct = root_correct = root_total = 0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        features = np.asarray([row["observation"] for row in batch], dtype=np.float64)
        logits, _ = policy.forward(features)
        for index, row in enumerate(batch):
            legal = tuple(int(value) for value in row["legal_action_ids"])
            action = int(row["action_id"])
            predicted = legal[int(np.argmax(logits[index, list(legal)]))]
            hit = predicted == action
            correct += hit
            if int(row["step"]) == 0:
                root_total += 1
                root_correct += hit
    require(root_total > 0, "teacher set has no raw-root first actions")
    return {"teacher_token_accuracy": correct / len(records),
            "raw_root_first_action_accuracy": root_correct / root_total,
            "teacher_tokens": len(records), "raw_root_records": root_total}


class FlatPolicyAdapter:
    """Small policy seam shared by CE and GRPO data collection/update.

    A later structured per-flow model can implement the same methods while
    reusing the C++ teacher, rollout reward, sampling, and validation code.
    """

    schema = "flat-49x128x128x6373-policy-adapter-v3"

    def __init__(self, policy: grpo.TwoHiddenMLP, optimizer: grpo.Adam | None = None):
        self.policy = policy
        self.optimizer = optimizer

    def probabilities(self, observation: np.ndarray,
                      legal_action_ids: Sequence[int]) -> np.ndarray:
        return _probabilities(self.policy, observation, legal_action_ids)

    def supervised_update(self, features: np.ndarray, legal: np.ndarray,
                          actions: np.ndarray, *, batch_size: int,
                          rng: np.random.Generator, device: str) -> dict[str, Any]:
        require(self.optimizer is not None, "policy adapter has no optimizer")
        return masked_cross_entropy_update(
            self.policy, self.optimizer, features, legal, actions,
            batch_size=batch_size, rng=rng, device=device)

    def grpo_update(self, batch: Mapping[str, np.ndarray],
                    config: grpo.GRPOConfig) -> dict[str, Any]:
        require(self.optimizer is not None, "policy adapter has no optimizer")
        return grpo._update(self.policy, self.optimizer, batch, config)


class StratifiedSampler:
    """Uniform domain, dataset, occupied size bucket, then block."""

    def __init__(self, blocks: Sequence[TrainingBlock]):
        require(blocks and all(block.split == "train" for block in blocks),
                "sampler accepts training blocks only")
        table: dict[str, dict[str, dict[str, list[TrainingBlock]]]] = {}
        for block in blocks:
            table.setdefault(block.domain, {}).setdefault(block.dataset_id, {}).setdefault(
                block.bucket, []).append(block)
        self.table = {
            domain: {dataset: {bucket: tuple(sorted(values, key=lambda row: row.sample_id))
                               for bucket, values in buckets.items()}
                     for dataset, buckets in datasets.items()}
            for domain, datasets in table.items()}
        self.domains = tuple(sorted(self.table))

    def sample(self, rng: np.random.Generator) -> TrainingBlock:
        domain = self.domains[int(rng.integers(0, len(self.domains)))]
        datasets = tuple(sorted(self.table[domain]))
        dataset = datasets[int(rng.integers(0, len(datasets)))]
        buckets = tuple(sorted(self.table[domain][dataset]))
        bucket = buckets[int(rng.integers(0, len(buckets)))]
        rows = self.table[domain][dataset][bucket]
        return rows[int(rng.integers(0, len(rows)))]

    def receipt(self) -> dict[str, Any]:
        return {domain: {dataset: {bucket: len(rows)
                                   for bucket, rows in sorted(buckets.items())}
                         for dataset, buckets in sorted(datasets.items())}
                for domain, datasets in sorted(self.table.items())}


def quick_subset(blocks: Sequence[TrainingBlock], *, train_per_dataset: int = 2,
                 validation_per_dataset: int = 1) -> tuple[TrainingBlock, ...]:
    """Bound every source while preferring 64/256 KiB-scale available blocks."""

    grouped: dict[tuple[str, str], list[TrainingBlock]] = {}
    for block in blocks:
        grouped.setdefault((block.split, block.dataset_id), []).append(block)
    selected = []
    targets = (256 * 1024, 64 * 1024)
    for (split, _dataset), rows in sorted(grouped.items()):
        limit = train_per_dataset if split == "train" else validation_per_dataset
        remaining = list(rows)
        for target in targets[:limit]:
            if not remaining:
                break
            chosen = min(remaining, key=lambda row: (
                abs(math.log2(max(row.original_numeric_bytes, 1) / target)),
                -row.original_numeric_bytes, row.sample_id))
            selected.append(chosen)
            remaining.remove(chosen)
        while remaining and sum(row.split == split and row.dataset_id == _dataset
                                for row in selected) < limit:
            chosen = max(remaining, key=lambda row: (row.original_numeric_bytes,
                                                     row.sample_id))
            selected.append(chosen); remaining.remove(chosen)
    return tuple(sorted(selected, key=lambda row: (row.split, row.domain,
                                                   row.dataset_id, row.sample_id)))


def export_current_model(policy: grpo.TwoHiddenMLP, config: grpo.GRPOConfig,
                         seed: int, iteration: int, directory: Path,
                         contract: OnlineContract, *, phase: str) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / f"policy_{phase}_{iteration:08d}.json"
    native = checkpoint.with_suffix(".wgpn")
    grpo.save_checkpoint(grpo.TrainingResult(policy, (), config, seed, iteration), checkpoint,
                         extra={"schema": SCHEMA, "matched_online_contract": asdict(contract),
                                "contract_sha256": contract.fingerprint,
                                "raw_root": True, "online_candidate_search": False,
                                "legacy_stop_weights_copied": False, "phase": phase})
    if native.exists():
        native.unlink()
    export_checkpoint(checkpoint, native)
    return native, digest(native.read_bytes())


def grpo_iteration(backend: OnlineBackend, policy: grpo.TwoHiddenMLP,
                   optimizer: grpo.Adam, sampler: StratifiedSampler,
                   contract: OnlineContract, rng: np.random.Generator,
                   config: grpo.GRPOConfig, *, warmups: int = 1,
                   repetitions: int = 5, group_size: int = 8) -> dict[str, Any]:
    require(backend.model_sha256 is not None, "GRPO reward requires current exported WGPN")
    require(2 <= group_size <= 8, "matched GRPO group size must be in [2,8]")
    episodes: list[dict[str, Any]] = []
    adapter = FlatPolicyAdapter(policy, optimizer)
    groups = []
    weights_rows = grpo.sample_training_weights(config.groups_per_iteration, rng,
                                                config.dirichlet_alpha)
    for group_index, (weight_array, weight_source) in enumerate(weights_rows):
        block = sampler.sample(rng)
        weight = _weights(weight_array)
        start = len(episodes)
        for _ in range(group_size):
            episodes.append(rollout(backend, policy, block, weight, contract, rng,
                                    greedy=False, policy_adapter=adapter))
        group_episodes = episodes[start:]
        unique_recipes = tuple(sorted({row["recipe"] for row in group_episodes}))
        # Cache scope is exactly this block/weight/current-model group.  It is
        # discarded before any model update, so J_total cannot use stale MLP time.
        successful_recipes = tuple(sorted({row["recipe"] for row in group_episodes
                                           if row["failure"] is None}))
        cache: dict[tuple[int, ...], RecipeMeasurement] = {}
        replay_failures: dict[tuple[int, ...], str] = {}
        for recipe in successful_recipes:
            try:
                cache[recipe] = replay_measure(
                    backend, block, weight, contract, recipe,
                    warmups=warmups, repetitions=repetitions)
            except (OnlineRequestTimeout, OnlineNativeFailure, ValueError) as error:
                replay_failures[recipe] = type(error).__name__ + ": " + str(error)
        failure_J = constraint_failure_objective(block, weight)
        returns = np.asarray([
            (-objective(cache[row["recipe"]], block, weight)
             if row["failure"] is None and row["recipe"] in cache else -failure_J)
            for row in group_episodes], dtype=np.float64)
        recipe_divergence = len(unique_recipes) > 1
        return_divergence = float(np.ptp(returns)) > 1e-8
        informative = recipe_divergence and return_divergence
        advantages = np.zeros(group_size, dtype=np.float64)
        if informative:
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        for row, reward, advantage in zip(group_episodes, returns, advantages):
            row["return"] = float(reward)
            row["advantage"] = float(advantage)
        groups.append({"group": group_index, "sample_id": block.sample_id,
                       "dataset_id": block.dataset_id, "size_bucket": block.bucket,
                       "weight": list(weight), "weight_source": weight_source,
                       "unique_recipes": len(unique_recipes),
                       "recipe_divergence": recipe_divergence,
                       "return_divergence": return_divergence,
                       "informative": informative, "mean_return": float(returns.mean()),
                       "std_return": float(returns.std()),
                       "measurement_cache_entries": len(cache),
                       "rollout_failures": sum(row["failure"] is not None
                                               for row in group_episodes),
                       "replay_failures": [{"recipe_action_ids": list(recipe),
                                            "error": error}
                                           for recipe, error in replay_failures.items()],
                       "constraint_failure_objective": failure_J,
                       "model_sha256": backend.model_sha256})
    features = np.concatenate([row["features"] for row in episodes])
    legal = np.concatenate([row["legal"] for row in episodes])
    actions = np.concatenate([row["actions"] for row in episodes])
    old_logp = np.concatenate([row["old_logp"] for row in episodes])
    lengths = np.asarray([len(row["actions"]) for row in episodes], dtype=np.int64)
    episode_ids = np.repeat(np.arange(len(episodes)), lengths)
    advantages = np.asarray([row["advantage"] for row in episodes])[episode_ids]
    batch = {"features": features, "legal": legal, "actions": actions,
             "old_logp": old_logp, "advantages": advantages,
             "token_weights": 1.0 / lengths[episode_ids]}
    before = [value.copy() for value in policy.params]
    metrics = adapter.grpo_update(batch, config)
    require(all(np.isfinite(value).all() for value in policy.params),
            "non-finite GRPO parameters")
    change = math.sqrt(sum(float(np.sum((old - new) ** 2))
                           for old, new in zip(before, policy.params)))
    return {"groups": len(groups), "episodes": len(episodes),
            "transitions": int(actions.size),
            "informative_groups": sum(row["informative"] for row in groups),
            "same_recipe_groups": sum(not row["recipe_divergence"] for row in groups),
            "return_tied_groups": sum(not row["return_divergence"] for row in groups),
            "parameter_change_l2": change,
            "reward": "-J_total_cpp_online_model_timing",
            "informative_definition": "recipe_divergence_and_return_divergence",
            "cache_scope": "same_block_weight_model_group_unique_recipe",
            "groups_detail": groups, **metrics}


def greedy_validation(backend: OnlineBackend, policy: grpo.TwoHiddenMLP,
                      blocks: Sequence[TrainingBlock], contract: OnlineContract,
                      rng: np.random.Generator, *, oracle_candidate_count: int = 32,
                      warmups: int = 1, repetitions: int = 5,
                      candidate_recipes: Mapping[str, Sequence[Sequence[int]]] | None = None,
                      candidate_profile: Mapping[str, Sequence[int]] | None = None,
                      quick_limited_oracle: bool = False,
                      failure_log: list[dict[str, Any]] | None = None
                      ) -> dict[str, Any]:
    require(blocks and all(block.split == "validation" for block in blocks),
            "greedy validation accepts validation blocks only")
    require(backend.model_sha256 is not None, "validation requires current exported WGPN")
    rows = []
    adapter = FlatPolicyAdapter(policy)
    for block in blocks:
        # Offline validation oracle is diagnostic only. It does not supply the
        # selected recipe; the measured policy path below is strictly greedy.
        candidates = (tuple(tuple(int(action) for action in recipe)
                            for recipe in candidate_recipes[block.sample_id])
                      if candidate_recipes is not None else
                      discover_candidates(backend, block, contract, rng,
                                          oracle_candidate_count,
                                          candidate_profile=candidate_profile,
                                          templates=(QUICK_VALIDATION_TEMPLATES
                                                     if quick_limited_oracle else None),
                                          failure_log=failure_log))
        for weight in FIXED_WEIGHTS:
            # Validation regret uses actual current-model E_total for this exact
            # weight; these offline oracle measurements do not select a recipe.
            oracle_measurements = []
            for recipe in candidates:
                try:
                    oracle_measurements.append(replay_measure(
                        backend, block, weight, contract, recipe,
                        warmups=warmups, repetitions=repetitions))
                except (OnlineRequestTimeout, OnlineNativeFailure, ValueError) as error:
                    if failure_log is not None:
                        failure_log.append({"sample_id": block.sample_id,
                                            "phase": "validation_oracle",
                                            "weight": list(weight),
                                            "recipe_action_ids": list(recipe),
                                            "error": type(error).__name__ + ": " + str(error)})
            require(oracle_measurements, "all validation reference candidates failed")
            frontier = pareto_front(oracle_measurements)
            selected = rollout(backend, policy, block, weight, contract, rng,
                               greedy=True, policy_adapter=adapter)
            teacher = min(objective(item, block, weight) for item in frontier)
            measured = None
            error = selected["failure"]
            if error is None:
                try:
                    measured = replay_measure(
                        backend, block, weight, contract, selected["recipe"],
                        warmups=warmups, repetitions=repetitions)
                except (OnlineRequestTimeout, OnlineNativeFailure, ValueError) as failure:
                    error = type(failure).__name__ + ": " + str(failure)
            if error is not None and failure_log is not None:
                failure_log.append({"sample_id": block.sample_id,
                                    "phase": "validation_greedy", "weight": list(weight),
                                    "recipe_action_ids": list(selected["recipe"]),
                                    "error": error})
            value = (objective(measured, block, weight) if measured is not None else
                     constraint_failure_objective(block, weight))
            final_recipe = measured.recipe if measured is not None else selected["recipe"]
            operators = sum(0 < action < FLOW_STOP_BEGIN for action in final_recipe)
            rows.append({"sample_id": block.sample_id, "dataset_id": block.dataset_id,
                         "domain": block.domain, "B_raw": block.original_numeric_bytes,
                         "size_bucket": block.bucket, "weight": list(weight),
                         "recipe_action_ids": list(final_recipe),
                         "raw_stop": operators == 0, "operator_count": operators,
                         "terminal_status": "ok" if measured is not None else "failed",
                         "error": error,
                         "compression_ratio": (measured.compression_ratio
                                               if measured is not None else None),
                         "compression_ns_per_byte": (measured.compression_ns_per_byte
                                                     if measured is not None else None),
                         "decompression_ns_per_byte": (measured.decompression_ns_per_byte
                                                       if measured is not None else None),
                         "J_total": value, "teacher_J": teacher,
                         "teacher_regret": value - teacher,
                         "teacher_regret_scope": ("raw_lz4_zstd_limited_reference"
                                                  if quick_limited_oracle else
                                                  "bounded_offline_candidate_frontier"),
                         "model_sha256": backend.model_sha256,
                         "selection": "greedy_single_path_no_candidate_evaluation"})
    costs = np.asarray([row["J_total"] for row in rows], dtype=np.float64)
    regrets = np.asarray([row["teacher_regret"] for row in rows], dtype=np.float64)
    return {"schema": "native-online-greedy-validation-v3",
            "contract_sha256": contract.fingerprint,
            "model_sha256": backend.model_sha256,
            "greedy_only_checkpoint_selection": True,
            "teacher_regret_scope": ("raw_lz4_zstd_limited_reference"
                                     if quick_limited_oracle else
                                     "bounded_offline_candidate_frontier"),
            "rows": rows, "row_count": len(rows),
            "mean_J_total": float(costs.mean()),
            "mean_teacher_regret": float(regrets.mean()),
            "raw_stop_rate": sum(row["raw_stop"] for row in rows) / len(rows),
            "mean_operator_count": float(np.mean([row["operator_count"] for row in rows])),
            "failed_rows": sum(row["terminal_status"] != "ok" for row in rows),
            "mean_compression_ratio": _optional_mean(rows, "compression_ratio"),
            "mean_compression_ns_per_byte": _optional_mean(
                rows, "compression_ns_per_byte"),
            "mean_decompression_ns_per_byte": _optional_mean(
                rows, "decompression_ns_per_byte")}


def _optional_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def _array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "base64": base64.b64encode(array.tobytes()).decode("ascii")}


def _array_from_record(value: Mapping[str, Any]) -> np.ndarray:
    raw = base64.b64decode(str(value["base64"]), validate=True)
    dtype = np.dtype(str(value["dtype"]))
    result = np.frombuffer(raw, dtype=dtype).reshape(tuple(int(x) for x in value["shape"]))
    return result.copy()


def save_resume(path: Path, *, policy: grpo.TwoHiddenMLP, optimizer: grpo.Adam,
                rng: np.random.Generator, contract: OnlineContract, seed: int,
                pretrain_epochs: int, grpo_iterations: int, cumulative_wall_ns: int,
                trace: Sequence[Mapping[str, Any]]) -> str:
    payload = {"contract": asdict(contract), "contract_sha256": contract.fingerprint,
               "identity": policy.identity.to_json(), "seed": int(seed),
               "pretrain_epochs": int(pretrain_epochs),
               "grpo_iterations": int(grpo_iterations),
               "cumulative_wall_ns": int(cumulative_wall_ns),
               "parameters": [_array_record(value) for value in policy.params],
               "optimizer": {"learning_rate": optimizer.learning_rate,
                             "execution_device": optimizer.execution_device,
                             "steps": optimizer.steps,
                             "first": [_array_record(value) for value in optimizer.first],
                             "second": [_array_record(value) for value in optimizer.second]},
               "rng_state": rng.bit_generator.state, "trace": list(trace)}
    payload_sha = digest(canonical_json(payload))
    document = canonical_json({"schema": RESUME_SCHEMA, "payload_sha256": payload_sha,
                               "payload": payload}) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(document); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise
    return payload_sha


def load_resume(path: Path, *, identity: grpo.PolicyIdentity,
                contract: OnlineContract, device: str) -> tuple[
                    grpo.TwoHiddenMLP, grpo.Adam, np.random.Generator, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    require(document.get("schema") == RESUME_SCHEMA, "unsupported resume schema")
    payload = document["payload"]
    require(document.get("payload_sha256") == digest(canonical_json(payload)),
            "resume payload digest mismatch")
    require(payload["contract_sha256"] == contract.fingerprint and
            payload["identity"] == identity.to_json(), "resume training contract drift")
    policy = grpo.TwoHiddenMLP(identity, np.random.default_rng(0))
    params = [_array_from_record(value) for value in payload["parameters"]]
    require([value.shape for value in params] == [value.shape for value in policy.params] and
            all(np.isfinite(value).all() for value in params), "resume parameter mismatch")
    policy.params = params
    state = payload["optimizer"]
    optimizer = grpo.Adam(policy.params, float(state["learning_rate"]), device=device)
    optimizer.steps = int(state["steps"])
    optimizer.first = [_array_from_record(value) for value in state["first"]]
    optimizer.second = [_array_from_record(value) for value in state["second"]]
    require([value.shape for value in optimizer.first] == [value.shape for value in params] and
            [value.shape for value in optimizer.second] == [value.shape for value in params] and
            all(np.isfinite(value).all() for value in (*optimizer.first, *optimizer.second)),
            "resume optimizer mismatch")
    rng = np.random.default_rng()
    rng.bit_generator.state = payload["rng_state"]
    return policy, optimizer, rng, payload


class DeadlineExpired(RuntimeError):
    pass


def remaining_deadline_ns(cumulative_wall_ns: int, deadline_seconds: float,
                          started_ns: int | None = None) -> int:
    require(math.isfinite(deadline_seconds) and deadline_seconds > 0,
            "deadline must be finite and positive")
    elapsed = 0 if started_ns is None else time.monotonic_ns() - started_ns
    return int(deadline_seconds * 1e9) - int(cumulative_wall_ns) - elapsed


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    data = canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True,
                        help="repeat exactly once for train and once for validation v3 manifests")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--materialized-root", type=Path)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--phase", choices=("teacher", "pretrain", "grpo", "validate"),
                        required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--pretrain-steps", type=int, default=200)
    parser.add_argument("--pretrain-max-seconds", type=float, default=600.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grpo-iterations", type=int, default=70)
    parser.add_argument("--groups-per-iteration", type=int, default=8)
    parser.add_argument("--group-size", type=int, choices=range(2, 9), default=8)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--request-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--deadline-seconds", type=float, default=57600.0)
    parser.add_argument("--quick", action="store_true",
                        help="bounded 20-minute flat-MLP pipeline validation profile")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    contract = OnlineContract()
    require(len(args.data) == 2, "pass independent train and validation v3 manifests")
    blocks = load_training_data(args.data, reference_path=args.reference,
                                materialized_root=args.materialized_root)
    if args.quick:
        blocks = quick_subset(blocks)
        args.candidate_count = min(args.candidate_count, 16)
        args.pretrain_epochs = max(args.pretrain_epochs, 100)
        args.pretrain_steps = min(args.pretrain_steps, 100)
        args.pretrain_max_seconds = min(args.pretrain_max_seconds, 120.0)
        args.grpo_iterations = min(args.grpo_iterations, 10)
        args.groups_per_iteration = min(args.groups_per_iteration, 4)
        args.group_size = min(args.group_size, 4)
        args.warmups = 0
        args.repetitions = 1
        args.deadline_seconds = min(args.deadline_seconds, 1200.0)
        args.request_timeout_seconds = min(args.request_timeout_seconds, 15.0)
    train = tuple(block for block in blocks if block.split == "train")
    validation = tuple(block for block in blocks if block.split == "validation")
    identity = policy_identity(train, contract)
    config = grpo.GRPOConfig(groups_per_iteration=args.groups_per_iteration)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    resume_path = output / "resume_v3.json"
    if resume_path.exists():
        policy, optimizer, rng, state = load_resume(
            resume_path, identity=identity, contract=contract, device=args.device)
        pretrain_epochs = int(state["pretrain_epochs"])
        grpo_iterations = int(state["grpo_iterations"])
        cumulative = int(state["cumulative_wall_ns"])
        trace = list(state["trace"])
    else:
        policy = grpo.TwoHiddenMLP(identity, np.random.default_rng(args.seed))
        optimizer = grpo.Adam(policy.params, config.learning_rate, device=args.device)
        rng = np.random.default_rng(args.seed + 1_000_003)
        pretrain_epochs = grpo_iterations = cumulative = 0
        trace = []
    started = time.monotonic_ns()
    manifest_sha = digest(canonical_json([
        {"path": str(path.resolve()), "sha256": digest(path.read_bytes())}
        for path in args.data]))
    candidate_profile = load_candidate_profile(args.catalog) if args.quick else None
    failure_log: list[dict[str, Any]] = []
    with OnlineTrainingServer(args.server, args.catalog, args.profile, cpu=args.cpu,
                              timeout_seconds=args.request_timeout_seconds) as backend:
        if args.phase == "teacher":
            require(args.teacher is not None and not args.teacher.exists(),
                    "teacher output must be a new path")
            model, model_sha = export_current_model(policy, config, args.seed, 0, output,
                                                    contract, phase="teacher_init")
            require(backend.load_model(model) == model_sha, "teacher model load mismatch")
            records = build_teacher_records(backend, train, contract, rng,
                                            candidate_count=args.candidate_count,
                                            warmups=args.warmups,
                                            repetitions=args.repetitions,
                                            candidate_profile=candidate_profile,
                                            failure_log=failure_log)
            receipt = write_teacher(args.teacher, records, contract=contract,
                                    source_manifest_sha256=manifest_sha)
            _atomic_json(output / "teacher_phase_receipt.json", receipt)
            _atomic_json(output / "teacher_candidate_failures.json",
                         {"schema": SCHEMA, "phase": "teacher",
                          "failures": failure_log, "failure_count": len(failure_log)})
            return 0
        if args.phase == "pretrain":
            require(args.teacher is not None and args.teacher.is_file(),
                    "pretrain requires teacher records")
            records = load_teacher(args.teacher, train, contract)
            features, legal, actions = teacher_arrays(records)
            completed_pretrain_steps = sum(int(row.get("optimizer_steps", 0))
                                           for row in trace if row.get("phase") == "pretrain")
            pretrain_started = time.monotonic_ns()
            while (pretrain_epochs < args.pretrain_epochs and
                   completed_pretrain_steps < args.pretrain_steps and
                   time.monotonic_ns() - pretrain_started <
                   int(args.pretrain_max_seconds * 1e9)):
                if remaining_deadline_ns(cumulative, args.deadline_seconds, started) <= 0:
                    raise DeadlineExpired("pretraining hard deadline reached")
                row = FlatPolicyAdapter(policy, optimizer).supervised_update(
                    features, legal, actions, batch_size=args.batch_size, rng=rng,
                    device=args.device)
                pretrain_epochs += 1
                completed_pretrain_steps += int(row["optimizer_steps"])
                accuracy = masked_teacher_accuracy(policy, records)
                trace.append({"phase": "pretrain", "epoch": pretrain_epochs,
                              "completed_pretrain_optimizer_steps": completed_pretrain_steps,
                              **row, **accuracy})
                elapsed = cumulative + time.monotonic_ns() - started
                save_resume(resume_path, policy=policy, optimizer=optimizer, rng=rng,
                            contract=contract, seed=args.seed, pretrain_epochs=pretrain_epochs,
                            grpo_iterations=grpo_iterations, cumulative_wall_ns=elapsed,
                            trace=trace)
                if (completed_pretrain_steps >= 20 and
                        accuracy["teacher_token_accuracy"] >= 0.98 and
                        accuracy["raw_root_first_action_accuracy"] >= 0.95):
                    break
            model, model_sha = export_current_model(policy, config, args.seed,
                                                    pretrain_epochs, output, contract,
                                                    phase="pretrained")
            _atomic_json(output / "pretrain_phase_receipt.json",
                         {"schema": SCHEMA, "phase": "pretrain", "epochs": pretrain_epochs,
                          "optimizer_steps": completed_pretrain_steps,
                          "model": str(model), "model_sha256": model_sha,
                          "trace": trace, "contract": asdict(contract)})
            return 0
        model, model_sha = export_current_model(policy, config, args.seed,
                                                grpo_iterations, output, contract,
                                                phase="current")
        require(backend.load_model(model) == model_sha, "current model load mismatch")
        if args.phase == "grpo":
            sampler = StratifiedSampler(train)
            while grpo_iterations < args.grpo_iterations:
                if remaining_deadline_ns(cumulative, args.deadline_seconds, started) <= 0:
                    raise DeadlineExpired("GRPO hard deadline reached")
                # Rewards are measured against the exact model that generated
                # this batch. Export/reload only after the batch update.
                row = grpo_iteration(backend, policy, optimizer, sampler, contract, rng,
                                     config, warmups=args.warmups,
                                     repetitions=args.repetitions,
                                     group_size=args.group_size)
                grpo_iterations += 1
                trace.append({"phase": "grpo", "iteration": grpo_iterations, **row})
                model, model_sha = export_current_model(policy, config, args.seed,
                                                        grpo_iterations, output, contract,
                                                        phase="grpo")
                require(backend.load_model(model) == model_sha, "updated model load mismatch")
                if (args.validation_interval > 0 and
                        grpo_iterations % args.validation_interval == 0):
                    validation_row = greedy_validation(
                        backend, policy, validation, contract, rng,
                        oracle_candidate_count=max(2, min(args.candidate_count, 12)),
                        warmups=args.warmups, repetitions=args.repetitions,
                        candidate_profile=candidate_profile,
                        quick_limited_oracle=args.quick,
                        failure_log=failure_log)
                    validation_row["iteration"] = grpo_iterations
                    _atomic_json(output / f"validation_{grpo_iterations:08d}.json",
                                 validation_row)
                    trace.append({"phase": "validation", "iteration": grpo_iterations,
                                  "mean_J_total": validation_row["mean_J_total"],
                                  "raw_stop_rate": validation_row["raw_stop_rate"],
                                  "mean_operator_count": validation_row["mean_operator_count"],
                                  "mean_teacher_regret": validation_row["mean_teacher_regret"]})
                elapsed = cumulative + time.monotonic_ns() - started
                save_resume(resume_path, policy=policy, optimizer=optimizer, rng=rng,
                            contract=contract, seed=args.seed, pretrain_epochs=pretrain_epochs,
                            grpo_iterations=grpo_iterations, cumulative_wall_ns=elapsed,
                            trace=trace)
            _atomic_json(output / "grpo_phase_receipt.json",
                         {"schema": SCHEMA, "phase": "grpo",
                          "iterations": grpo_iterations, "model": str(model),
                          "model_sha256": model_sha, "sampler": sampler.receipt(),
                          "contract": asdict(contract), "trace": trace})
            return 0
        result = greedy_validation(backend, policy, validation, contract, rng,
                                   oracle_candidate_count=args.candidate_count,
                                   warmups=args.warmups, repetitions=args.repetitions,
                                   candidate_profile=candidate_profile,
                                   quick_limited_oracle=args.quick,
                                   failure_log=failure_log)
        _atomic_json(output / "greedy_validation_v3.json", result)
        _atomic_json(output / "validation_failures_v3.json",
                     {"schema": SCHEMA, "phase": "validation",
                      "failures": failure_log, "failure_count": len(failure_log)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
