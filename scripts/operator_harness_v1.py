"""Local-only reversible graph harness with independent, durable case receipts.

The native runtime owns transforms, framing, checksums and codec timers. Python
owns catalog validation, process deadlines, cold verification and result commits.
No server, campaign queue, tmux session or historical result is accessed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import selectors
import signal
import statistics
import subprocess
import tempfile
import time
from typing import Any

from operator_harness_registry_v1 import load_registry, validate_case

ROOT = Path(__file__).resolve().parents[1]
REQUEST_SCHEMA = "operator-harness-request-v1"
RESPONSE_SCHEMA = "operator-harness-response-v1"
SUITE_SCHEMA = "operator-harness-suite-v1"
DEFAULT_DRIVER = ROOT / "cpp/operator_harness_v1/build-local/operator_harness_v1_cli"
MAX_PROTOCOL_BYTES = 64 << 20


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".commit-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True,
                                 ensure_ascii=False, allow_nan=False).encode() + b"\n")


@contextmanager
def output_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another local harness owns this output directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class DriverError(RuntimeError):
    pass


class NativeDriver:
    """One resident local driver; a deadline failure terminates only this process."""

    def __init__(self, binary: Path, timeout: float, stderr_path: Path):
        self.timeout = timeout
        self.buffer = bytearray()
        self.stderr = stderr_path.open("ab")
        environment = dict(os.environ)
        environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            self.process = subprocess.Popen(
                [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr, env=environment, start_new_session=True, bufsize=0)
        except BaseException:
            self.stderr.close()
            raise
        assert self.process.stdout is not None
        os.set_blocking(self.process.stdout.fileno(), False)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def request(self, request: dict) -> dict:
        raw = canonical(request) + b"\n"
        if len(raw) > MAX_PROTOCOL_BYTES:
            raise DriverError("request exceeds local harness protocol limit")
        # Requests are bounded local fixtures. Nonblocking writes also keep a
        # stalled/misbehaving driver within the same deadline as its response.
        assert self.process.stdin is not None and self.process.stdout is not None
        deadline = time.monotonic() + self.timeout
        fd = self.process.stdin.fileno()
        os.set_blocking(fd, False)
        with selectors.DefaultSelector() as writable:
            writable.register(fd, selectors.EVENT_WRITE)
            position = 0
            while position < len(raw):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not writable.select(remaining):
                    raise TimeoutError("native request deadline exceeded")
                try:
                    position += os.write(fd, raw[position:])
                except BrokenPipeError as error:
                    raise DriverError("native driver exited before accepting request") from error
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError("native response deadline exceeded")
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                raise DriverError(f"native driver exited: {self.process.poll()}")
            self.buffer.extend(block)
            if len(self.buffer) > MAX_PROTOCOL_BYTES:
                raise DriverError("native response exceeds protocol limit")
        line, _, remainder = self.buffer.partition(b"\n")
        self.buffer = bytearray(remainder)
        try:
            response = json.loads(line)
        except (UnicodeError, ValueError) as error:
            raise DriverError("native driver returned malformed JSON") from error
        if not isinstance(response, dict) or response.get("schema") != RESPONSE_SCHEMA:
            raise DriverError("native response schema mismatch")
        if response.get("case_id") != request.get("case_id"):
            raise DriverError("native response identity mismatch")
        if response.get("command") != request.get("command"):
            raise DriverError("native response command mismatch")
        return response

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=2)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if self.process.poll() is None:
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.selector.close()
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()
        self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _roots(case: dict) -> list[dict]:
    roots = case.get("roots")
    if not isinstance(roots, list) or not roots or len(roots) > 32:
        raise ValueError("case requires 1..32 roots")
    normalized = []
    for item in roots:
        if not isinstance(item, dict):
            raise ValueError("root must be an object")
        width, kind = item.get("width"), item.get("kind")
        if type(width) is not int or width <= 0 or kind not in ("serial", "numeric", "struct"):
            raise ValueError("invalid root kind/width")
        if kind == "serial" and width != 1:
            raise ValueError("serial root width must be one")
        if kind == "numeric" and width not in (1, 2, 4, 8):
            raise ValueError("numeric root width must be 1/2/4/8")
        data_hex = item.get("data_hex")
        if not isinstance(data_hex, str) or len(data_hex) > MAX_PROTOCOL_BYTES // 2:
            raise ValueError("invalid/oversized root data_hex")
        try:
            data = bytes.fromhex(data_hex)
        except ValueError as error:
            raise ValueError("invalid root hex") from error
        if data.hex() != data_hex or len(data) % width:
            raise ValueError("root must use canonical, width-aligned little-endian hex")
        numeric = item.get("original_numeric_payload", True)
        view = item.get("logical_view", "bytes")
        if type(numeric) is not bool or not isinstance(view, str) or not view:
            raise ValueError("invalid root denominator flag/logical view")
        normalized.append(dict(kind=kind, width=width, data_hex=data_hex,
                               original_numeric_payload=numeric, logical_view=view))
    return normalized


def _samples(values: Any, repeats: int, field: str) -> dict:
    if not isinstance(values, list) or len(values) != repeats or any(
            type(value) is not int or value < 0 for value in values):
        raise DriverError(f"invalid native {field} samples")
    return {"samples_ns": values, "median_ns": statistics.median(values),
            "min_ns": min(values), "max_ns": max(values)}


def _verify_run(response: dict, request: dict) -> tuple[bytes, dict]:
    if response.get("status") != "ok":
        raise DriverError(str(response.get("error", "native execution failed")))
    if response.get("bitexact") is not True:
        raise DriverError("native bitexact evidence is absent")
    frame = bytes.fromhex(response["frame_hex"])
    raw = sum(len(bytes.fromhex(root["data_hex"])) for root in request["roots"]
              if root["original_numeric_payload"])
    if response.get("original_numeric_bytes") != raw or response.get("frame_bytes") != len(frame):
        raise DriverError("native byte accounting mismatch")
    if not isinstance(response.get("timing_boundary"), (dict, str)):
        raise DriverError("native timer boundary is absent")
    metrics = {"original_bytes": raw, "complete_frame_bytes": len(frame),
               "frame_to_original_ratio": len(frame) / raw if raw else None,
               "encode": _samples(response.get("encode_ns"), request["repeats"], "encode"),
               "decode": _samples(response.get("decode_ns"), request["repeats"], "decode"),
               "timing_boundary": response["timing_boundary"]}
    for direction in ("encode", "decode"):
        metrics[direction]["median_ns_per_original_byte"] = (
            metrics[direction]["median_ns"] / raw if raw else None)
    return frame, metrics


def _event(path: Path, event: dict) -> None:
    with path.open("ab") as handle:
        handle.write(canonical({"recorded_unix_ns": time.time_ns(), **event}) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_once(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"existing immutable run identity differs: {path}")
    else:
        atomic_json(path, value)


def _dedicated_output(path: Path) -> Path:
    output = path.resolve()
    if ROOT == output or ROOT.is_relative_to(output):
        raise ValueError("output must be a dedicated harness directory, not the workspace or its ancestor")
    if output.exists():
        if not output.is_dir():
            raise ValueError("output must be a dedicated directory")
        existing = {entry.name for entry in output.iterdir()} - {".lock"}
        if existing:
            marker = output / "RUN.json"
            if not marker.is_file() or json.loads(marker.read_text()).get("schema") != "operator-harness-run-v1":
                raise ValueError("refusing a nonempty directory without a harness run identity")
    return output


def _publish_operator_results(output: Path, case_key: str, record: dict) -> None:
    """Publish per-A-id evidence without attributing graph time to a single op."""
    identifiers = record["implementation_ids"]
    if not isinstance(identifiers, list):
        return
    for identifier in identifiers:
        # Receipt ids were filtered against the frozen registry, including for
        # malformed/failed cases; keep the path-component check as defense.
        if not isinstance(identifier, str) or len(identifier) != 4 or identifier[0] != "A" or not identifier[1:].isdigit():
            continue
        evidence = {
            "schema": "operator-harness-implementation-result-v1",
            "implementation_id": identifier,
            "case_id": record["case_id"], "case_key": case_key,
            "run_sha256": record["run_sha256"], "status": record["status"],
            "bitexact": record["bitexact"],
            "independent_single_operator_case": len(identifiers) == 1,
            "evidence_scope": "only this input and exact IO variant; not full-domain qualification",
            "timing_scope": "complete fixed-recipe graph; not attributed per-kernel time",
            "case_result": str((output / "cases" / case_key / "RESULT.json").resolve()),
            "case_receipt_sha256": record["receipt_sha256"],
        }
        evidence["receipt_sha256"] = digest(evidence)
        _write_once(output / "operators" / identifier / (case_key + ".json"), evidence)


def _source_receipt() -> dict:
    paths = [Path(__file__), ROOT / "scripts/operator_harness_registry_v1.py",
             ROOT / "scripts/operator_harness_suite_v1.py"]
    paths += sorted((ROOT / "cpp/operator_harness_v1").glob("*.cpp"))
    paths += sorted((ROOT / "cpp/operator_harness_v1").glob("*.hpp"))
    paths += sorted((ROOT / "cpp/operator_harness_v1").glob("CMakeLists.txt"))
    paths += [ROOT / "cpp/native_graph_v1" / name for name in
              ("graph.cpp", "frame.cpp", "graph.hpp", "stage_dispatch.cpp", "stage_dispatch.cmake")]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths if path.is_file()}


def run_suite(suite: dict, binary: Path, output: Path, *, warmups: int = 1,
              repeats: int = 5, timeout: float = 30.0) -> dict:
    if suite.get("schema") != SUITE_SCHEMA or not isinstance(suite.get("cases"), list):
        raise ValueError("invalid harness suite schema")
    cases = suite["cases"]
    if not cases or any(not isinstance(case, dict) or not isinstance(case.get("case_id"), str)
                        or not case["case_id"] for case in cases):
        raise ValueError("suite requires cases with nonempty case_id")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case_id in suite")
    if type(warmups) is not int or not 0 <= warmups <= 31 or type(repeats) is not int or not 1 <= repeats <= 1001:
        raise ValueError("invalid native warmups/repeats")
    if not 0 < timeout <= 3600:
        raise ValueError("timeout must be in (0,3600] seconds")
    output = _dedicated_output(output)
    binary = binary.resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("driver must be a local executable file")
    registry = load_registry()
    registered_ids = {operator["trace_id"] for operator in registry["operators"]}
    identity = {"schema": "operator-harness-run-v1", "suite_sha256": digest(suite),
                "registry_sha256": registry["registry_sha256"],
                "driver_sha256": file_sha256(binary), "warmups": warmups, "repeats": repeats,
                "timeout_seconds": timeout, "sources": _source_receipt(),
                "host": {"system": platform.system(), "machine": platform.machine(),
                         "release": platform.release(), "cpu_count": os.cpu_count()},
                "execution_scope": "local-only; no remote/campaign access",
                "nncp_policy": "disabled_by_policy; no enable override",
                "measurement_scope": "fixed-recipe native graph; selection and process startup excluded"}
    identity["run_sha256"] = digest(identity)
    with output_lock(output):
        _write_once(output / "RUN.json", identity)
        _write_once(output / "REGISTRY.json", registry)
        _write_once(output / "SUITE.json", suite)
        records = []
        client = None
        try:
            for case in cases:
                key = digest({"run_sha256": identity["run_sha256"], "case": case})
                case_root = output / "cases" / key
                result_path = case_root / "RESULT.json"
                if result_path.exists():
                    record = json.loads(result_path.read_text())
                    receipt_hash = record.pop("receipt_sha256", None)
                    if receipt_hash != digest(record):
                        raise ValueError("committed result receipt changed")
                    record["receipt_sha256"] = receipt_hash
                    if record.get("case_sha256") != digest(case) or record.get("run_sha256") != identity["run_sha256"]:
                        raise ValueError("committed result identity mismatch")
                    if record.get("status") == "ok":
                        frame_path = case_root / "frame.ngf"
                        if not frame_path.is_file() or file_sha256(frame_path) != record["frame_sha256"]:
                            raise ValueError("committed frame missing or changed")
                    _publish_operator_results(output, key, record)
                    records.append(record)
                    continue
                case_root.mkdir(parents=True, exist_ok=True)
                _event(output / "events.jsonl", {"event": "case_started", "case_id": case["case_id"], "case_key": key})
                started = time.monotonic_ns()
                raw_ids = case.get("implementation_ids")
                receipt_ids = list(dict.fromkeys(identifier for identifier in raw_ids
                    if isinstance(identifier, str) and identifier in registered_ids)) if isinstance(raw_ids, list) else []
                record = {"schema": "operator-harness-case-result-v1", "case_id": case["case_id"],
                          "implementation_ids": receipt_ids,
                          "block_id": case.get("block_id", "complete_case"),
                          "case_sha256": digest(case), "run_sha256": identity["run_sha256"],
                          "status": "failed", "bitexact": False}
                try:
                    normalized = validate_case(copy.deepcopy(case), registry=registry)
                    request = {"schema": REQUEST_SCHEMA, "command": "run", "case_id": case["case_id"],
                               "roots": _roots(normalized), "recipe": normalized["recipe"],
                               "warmups": warmups, "repeats": repeats}
                    atomic_json(case_root / "REQUEST.json", request)
                    if client is None:
                        client = NativeDriver(binary, timeout, output / "driver.stderr.log")
                    response = client.request(request)
                    atomic_json(case_root / "NATIVE_RESPONSE.json", response)
                    frame, metrics = _verify_run(response, request)
                    atomic_bytes(case_root / "frame.ngf", frame)
                    _event(output / "events.jsonl", {"event": "native_graph_completed", "case_id": case["case_id"]})
                    # A new OS process sees only the complete frame and decoder.
                    # No encoder roots, session, allocation or dictionary is shared.
                    with NativeDriver(binary, timeout, output / "driver.stderr.log") as cold:
                        decoded = cold.request({"schema": REQUEST_SCHEMA, "command": "decode",
                                                "case_id": case["case_id"], "frame_hex": frame.hex(),
                                                "measure_decode": True, "warmups": 0, "repeats": 1})
                    atomic_json(case_root / "COLD_DECODE.json", decoded)
                    if decoded.get("status") != "ok" or decoded.get("roots") != request["roots"]:
                        raise DriverError("fresh-process decoded roots differ bit-for-bit or in type")
                    if decoded.get("original_numeric_bytes") != metrics["original_bytes"]:
                        raise DriverError("fresh-process denominator mismatch")
                    cold_samples = _samples(decoded.get("decode_ns"), 1, "cold decode")
                    record.update(status="ok", bitexact=True, cold_decode_verified=True,
                                  frame_sha256=hashlib.sha256(frame).hexdigest(),
                                  dispatcher=response.get("dispatcher"), metrics=metrics,
                                  cold_decode=cold_samples,
                                  cold_decode_boundary="fresh OS process; native frame decode only; startup/JSON excluded")
                except Exception as error:
                    record["status"] = "timeout" if isinstance(error, TimeoutError) else (
                        "disabled" if "disabled_by_policy" in str(error) else "failed")
                    record["error"] = {"class": type(error).__name__, "message": str(error)}
                    if client is not None:
                        client.close()
                        client = None
                record["elapsed_wall_ns_not_codec_timing"] = time.monotonic_ns() - started
                record["receipt_sha256"] = digest(record)
                # This is the commit point. A later sibling failure cannot
                # retract this receipt, and a crash before it reruns only this case.
                atomic_json(result_path, record)
                _publish_operator_results(output, key, record)
                _event(output / "events.jsonl", {"event": "case_committed", "case_id": case["case_id"],
                                                 "case_key": key, "status": record["status"]})
                records.append(record)
                print(json.dumps({"case_id": case["case_id"], "status": record["status"]}), flush=True)
        finally:
            if client is not None:
                client.close()
        summary = {"schema": "operator-harness-summary-v1", "run_sha256": identity["run_sha256"],
                   "case_count": len(records), "status_counts": dict(Counter(record["status"] for record in records)),
                   "implementation_ids_in_passing_cases": sorted({identifier for record in records if record["status"] == "ok"
                                                                    for identifier in record["implementation_ids"]}),
                   "independently_verified_implementation_ids": sorted({record["implementation_ids"][0]
                       for record in records if record["status"] == "ok" and len(record["implementation_ids"]) == 1}),
                   "registered_implementation_count": registry["operator_count"],
                   "nncp_policy": identity["nncp_policy"],
                   "complete": len(records) == len(cases),
                   "all_cases_passed": all(record["status"] == "ok" for record in records),
                   "results": [{"case_id": record["case_id"], "status": record["status"],
                                "result_path": str((output / "cases" / digest({"run_sha256": identity["run_sha256"], "case": case}) / "RESULT.json").resolve())}
                               for record, case in zip(records, cases)]}
        atomic_json(output / "SUMMARY.json", summary)
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("registry", help="export all102 typed registrations; never run kernels")
    catalog.add_argument("--out", type=Path)
    demo = commands.add_parser("demo-suite", help="generate deterministic local representative inputs")
    demo.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run", help="run/resume an immutable local suite")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.command == "registry":
        registry = load_registry()
        if args.out:
            atomic_json(args.out, registry)
        else:
            print(json.dumps(registry, indent=2, ensure_ascii=False))
        return 0
    if args.command == "demo-suite":
        from operator_harness_suite_v1 import representative_suite
        atomic_json(args.out, representative_suite())
        print(str(args.out.resolve()))
        return 0
    summary = run_suite(json.loads(args.suite.read_text()), args.driver, args.out,
                        warmups=args.warmups, repeats=args.repeats, timeout=args.timeout)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["all_cases_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
