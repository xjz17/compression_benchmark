#!/usr/bin/env python3
"""Measure lossless operator-pair compression rates for one dataset."""

from __future__ import annotations

import argparse
import csv
import math
import os
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))
from myapp.experiment_data_policy import prepare_numeric_column  # noqa: E402
OUTPUT_DIR = REPO_ROOT / "analysis_outputs"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    label: str


@dataclass(frozen=True)
class LosslessState:
    """Immutable C++ transform result; no Python forward/inverse closures."""

    values: np.ndarray
    metadata: bytes
    action_names: tuple[str, ...]
    compact_payload: bytes

    def serialized(self) -> bytes:
        return self.metadata + self.compact_payload

    def restore(self) -> np.ndarray:
        raw = _run_reference("invert-state", self.serialized(), self.action_names)
        return np.frombuffer(raw, dtype="<i8").copy()


def _run_reference(command: str, payload: bytes, names: tuple[str, ...] = ()) -> bytes:
    """CLI bridge only. Wall-clock users also measure process/IPC overhead."""
    executable = Path(os.environ.get(
        "REFERENCE_OPERATOR_ACTIONS_CLI",
        str(REPO_ROOT / "cpp" / "build" / "reference_operator_actions_cli"),
    ))
    if not executable.is_file():
        raise RuntimeError(f"build reference_operator_actions_cli first: {executable}")
    result = subprocess.run(
        [str(executable), command, *names], input=payload, capture_output=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("eraser", "eraser"),
    OperatorSpec("frame_of_reference", "for"),
    OperatorSpec("value_reorder", "value reorder"),
    OperatorSpec("regression_predictor", "regression"),
    OperatorSpec("outlier_separation", "outlier sep."),
    OperatorSpec("bitpack_int", "bit-packing"),
    OperatorSpec("delta", "delta"),
    OperatorSpec("xor", "xor"),
    OperatorSpec("trailing_zero_pack", "trailing-zero"),
    OperatorSpec("dictionary", "dictionary"),
    OperatorSpec("zigzag", "zigzag"),
    OperatorSpec("block_transform", "block transform"),
    OperatorSpec("symbol_recode", "symbol recoding"),
    OperatorSpec("run_length", "run length"),
    OperatorSpec("lz_match", "lz matching"),
    OperatorSpec("match_parse_policy", "match policy"),
    OperatorSpec("match_window_model", "window model"),
    OperatorSpec("match_token_format", "match tokens"),
    OperatorSpec("integer_code", "integer coding"),
    OperatorSpec("bitstream_pack", "bitstream packing"),
    OperatorSpec("entropy_huffman", "huffman coding"),
    OperatorSpec("entropy_range", "range coding"),
    OperatorSpec("entropy_fse", "fse/ans coding"),
    OperatorSpec("entropy_special", "special coding"),
    OperatorSpec("literal_entropy", "literal coding"),
    OperatorSpec("block_state_control", "block/state"),
    OperatorSpec("container_frame", "container framing"),
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def as_i64(values: np.ndarray | list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def i64_bytes(values: np.ndarray) -> bytes:
    return as_i64(values).astype("<i8", copy=False).tobytes()


def compact_int_payload(values: np.ndarray) -> bytes:
    return _run_reference("compact-encode", i64_bytes(values))


def meta(tag: str, *arrays: np.ndarray, extra: bytes = b"") -> bytes:
    out = bytearray(tag.encode("ascii")[:12].ljust(12, b"\0"))
    out.extend(len(arrays).to_bytes(2, "little"))
    out.extend(len(extra).to_bytes(4, "little"))
    out.extend(extra)
    for array in arrays:
        payload = i64_bytes(array)
        out.extend(len(array).to_bytes(4, "little"))
        out.extend(len(payload).to_bytes(4, "little"))
        out.extend(payload)
    return bytes(out)


def numeric_columns(column_info: Path) -> list[dict[str, str]]:
    with column_info.open("r", encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if str(row.get("logical_type", "")).strip().lower()
            in {"integer", "int", "long", "float", "double"}
        ]


def column_values(series: pd.Series, column: dict[str, str]) -> np.ndarray:
    logical = str(column.get("logical_type", "")).strip().lower()
    prepared = prepare_numeric_column(series.tolist(), logical)
    prepared_metadata = prepared.manifest()
    for key in ("input_policy", "source_rows", "rows", "removed_nan_count"):
        column[key] = str(prepared_metadata[key])
    if prepared.logical_type == "integer":
        return as_i64(np.asarray(prepared.values, dtype=np.int64))
    values = np.asarray(prepared.values, dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("integer-normalized operator runner cannot represent infinity; input policy retains it")
    places = int(column.get("max_point_number", 0) or 0)
    if logical in {"float", "double"} and places > 0:
        values = np.rint(values * (10**places))
    return as_i64(np.rint(values))


def zlib_rate(payload: bytes, original_size: int) -> float:
    if original_size <= 0 or not payload:
        return math.nan
    return len(zlib.compress(payload, level=6)) / float(original_size)


def _cpp_state(
    values: np.ndarray,
    names: tuple[str, ...],
    previous_metadata: bytes = b"",
    previous_names: tuple[str, ...] = (),
) -> LosslessState:
    packet = _run_reference("inspect", i64_bytes(values), names)
    if len(packet) < 16:
        raise RuntimeError("truncated C++ reference state packet")
    metadata_size, value_size = struct.unpack_from("<QQ", packet)
    value_start = 16 + metadata_size
    compact_start = value_start + value_size
    if value_size % 8 or compact_start + 6 > len(packet):
        raise RuntimeError("invalid C++ reference state packet lengths")
    output = np.frombuffer(packet[value_start:compact_start], dtype="<i8").copy()
    output.setflags(write=False)
    return LosslessState(
        values=output,
        metadata=previous_metadata + packet[16:value_start],
        action_names=previous_names + names,
        compact_payload=packet[compact_start:],
    )


def initial_state(values: np.ndarray) -> LosslessState:
    return _cpp_state(as_i64(values), ())


def apply_operator(state: LosslessState, operator_id: str) -> LosslessState:
    """Apply one actual C++ stage to the previous stage's current values."""
    if operator_id not in {operator.operator_id for operator in OPERATORS}:
        raise ValueError(f"unsupported operator: {operator_id}")
    return _cpp_state(
        state.values, (f"op.{operator_id}.reference_v1",),
        state.metadata, state.action_names,
    )


def checked_rate(state: LosslessState, original: np.ndarray, original_size: int) -> tuple[float, bool]:
    compressed = zlib.compress(state.serialized(), level=6)
    zlib.decompress(compressed)
    restored = state.restore()
    lossless = restored.shape == original.shape and bool(np.array_equal(restored, original))
    if not lossless:
        return math.nan, False
    return len(compressed) / float(original_size), True


def compute_pair_metrics(dataset_key: str, max_points: int | None) -> pd.DataFrame:
    dataset_csv = REPO_ROOT / "orignial_datasets" / f"{dataset_key}.csv"
    if not dataset_csv.exists():
        dataset_csv = REPO_ROOT / "backend" / "dataset_csv" / f"{dataset_key}.csv"
    column_info = REPO_ROOT / "backend" / "dataset" / dataset_key / f"{dataset_key}_column_info.csv"
    columns = numeric_columns(column_info)
    df = pd.read_csv(
        dataset_csv,
        usecols=[row["column_name"] for row in columns],
        nrows=max_points if max_points and max_points > 0 else None,
        low_memory=False,
        dtype=str,
    )

    rows: list[dict[str, object]] = []
    for column in columns:
        name = column["column_name"]
        original = column_values(df[name], column)
        if max_points is not None and max_points > 0:
            original = original[:max_points]
        original_size = len(i64_bytes(original))
        if original_size == 0:
            continue
        base_state = initial_state(original)
        first_cache: dict[str, LosslessState] = {}
        first_rates: dict[str, float] = {}
        first_lossless: dict[str, bool] = {}
        for first in OPERATORS:
            try:
                first_state = apply_operator(base_state, first.operator_id)
                first_rate, ok = checked_rate(first_state, original, original_size)
                first_cache[first.operator_id] = first_state
                first_rates[first.operator_id] = first_rate
                first_lossless[first.operator_id] = ok
            except Exception:
                first_rates[first.operator_id] = math.nan
                first_lossless[first.operator_id] = False

        for first in OPERATORS:
            first_state = first_cache.get(first.operator_id)
            base_rate = first_rates.get(first.operator_id, math.nan)
            for second in OPERATORS:
                pair_rate = math.nan
                ratio = math.nan
                lossless = False
                status = "ok"
                if first_state is None or not first_lossless.get(first.operator_id, False):
                    status = "first_not_lossless"
                elif not math.isfinite(base_rate) or base_rate <= 0:
                    status = "first_rate_invalid"
                else:
                    try:
                        pair_state = apply_operator(first_state, second.operator_id)
                        pair_rate, lossless = checked_rate(pair_state, original, original_size)
                        if lossless and math.isfinite(pair_rate):
                            ratio = pair_rate / base_rate
                        else:
                            status = "pair_not_lossless"
                    except Exception as exc:
                        status = f"pair_failed:{type(exc).__name__}"
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "column_index": int(column["column_index"]),
                        "column_name": name,
                        "logical_type": str(column.get("logical_type", "")).strip().lower(),
                        "first_operator": first.operator_id,
                        "second_operator": second.operator_id,
                        "first_rate": base_rate,
                        "pair_rate": pair_rate,
                        "pair_to_first_rate_ratio": ratio,
                        "lossless": int(lossless),
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_matrix(rows: pd.DataFrame, value_column: str) -> pd.DataFrame:
    usable = rows[rows["lossless"] == 1]
    pivot = usable.pivot_table(
        index="first_operator",
        columns="second_operator",
        values=value_column,
        aggfunc="mean",
    )
    operator_ids = [operator.operator_id for operator in OPERATORS]
    return pivot.reindex(index=operator_ids, columns=operator_ids)


def draw_heatmap(matrix: pd.DataFrame, output_stem: Path, dataset_key: str) -> None:
    labels = {operator.operator_id: operator.label for operator in OPERATORS}
    arr = matrix.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise RuntimeError("No finite lossless operator-pair measurements.")

    low = float(np.nanpercentile(finite, 2))
    high = float(np.nanpercentile(finite, 98))
    norm = Normalize(vmin=max(0.0, low), vmax=max(high, low + 1e-9))
    cmap = mpl.colormaps["RdYlGn_r"].copy()
    cmap.set_bad("#eeeeee")

    fig, ax = plt.subplots(figsize=(6.85, 6.15))
    image = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(OPERATORS)))
    ax.set_yticks(np.arange(len(OPERATORS)))
    ax.set_xticklabels([labels[operator.operator_id] for operator in OPERATORS], rotation=45, ha="right")
    ax.set_yticklabels([labels[operator.operator_id] for operator in OPERATORS])
    ax.set_xlabel("Second operator")
    ax.set_ylabel("First operator")
    ax.tick_params(axis="both", labelsize=6.0)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(OPERATORS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(OPERATORS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.018)
    colorbar.set_label("Compression rate", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=6.2)
    fig.suptitle(
        f"{dataset_key} lossless operator-pair compression rate",
        y=0.985,
        fontsize=8.2,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.20, right=0.94, top=0.92, bottom=0.23)

    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_stem.with_suffix(".eps"), format="eps", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Air-Quality", help="Dataset key under backend/dataset.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional per-column cap for quick previews. Omit for full columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_key = args.dataset.lower().replace("-", "_")
    if args.max_points:
        output_key = f"{output_key}_first{args.max_points}"

    rows = compute_pair_metrics(args.dataset, args.max_points)
    row_path = OUTPUT_DIR / f"operator_pair_{output_key}_rates.csv"
    rows.to_csv(row_path, index=False)

    ratio = aggregate_matrix(rows, "pair_to_first_rate_ratio")
    rate = aggregate_matrix(rows, "pair_rate")
    ratio_path = OUTPUT_DIR / f"operator_pair_{output_key}_rate_ratio_matrix.csv"
    rate_path = OUTPUT_DIR / f"operator_pair_{output_key}_pair_rate_matrix.csv"
    ratio.to_csv(ratio_path)
    rate.to_csv(rate_path)

    output_stem = OUTPUT_DIR / f"operator_pair_{output_key}_compression_rate_heatmap"
    draw_heatmap(rate, output_stem, args.dataset)
    measured = int(rows["pair_rate"].notna().sum())
    lossless = int(rows["lossless"].sum())
    total = len(OPERATORS) * len(OPERATORS) * len(numeric_columns(REPO_ROOT / "backend" / "dataset" / args.dataset / f"{args.dataset}_column_info.csv"))
    print(f"Wrote {row_path}")
    print(f"Wrote {ratio_path}")
    print(f"Wrote {rate_path}")
    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {output_stem.with_suffix('.eps')}")
    print(f"Measured finite lossless pairs: {measured}/{total}")
    print(f"Lossless checks passed: {lossless}/{total}")


if __name__ == "__main__":
    main()
