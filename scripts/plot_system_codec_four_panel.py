#!/usr/bin/env python3
"""Plot TsFile and InfluxDB read/write codec time in four panels."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SYSTEMS = ("tsfile", "influxdb")
METRICS = (
    ("median_write_ms", "Write"),
    ("median_read_ms", "Read"),
)
CODEC_ORDER = {
    "tsfile": ("BOS", "SUBCOLUMN", "PLAIN", "GORILLA_V1", "GORILLA", "CHIMP", "RLBE", "CAMEL"),
    "influxdb": ("BOS", "SUBCOLUMN", "GORILLA", "INTEGER_DELTA_ADAPTIVE_BITCAST"),
}
COLORS = {
    "BOS": "#D55E00",
    "SUBCOLUMN": "#0072B2",
    "PLAIN": "#999999",
    "GORILLA_V1": "#56B4E9",
    "GORILLA": "#009E73",
    "CHIMP": "#CC79A7",
    "RLBE": "#E69F00",
    "CAMEL": "#F0E442",
    "INTEGER_DELTA_ADAPTIVE_BITCAST": "#6A3D9A",
}
DISPLAY_NAMES = {
    "SUBCOLUMN": "Sub-column",
    "GORILLA_V1": "Gorilla v1",
    "GORILLA": "Gorilla",
    "INTEGER_DELTA_ADAPTIVE_BITCAST": "Integer delta adaptive\n(bitcast)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path, help="system_codec_summary.csv")
    parser.add_argument("output_prefix", type=Path, help="path without extension")
    parser.add_argument("--expected-datasets", type=int, default=5)
    return parser.parse_args()


def read_summary(path: Path) -> tuple[list[str], dict[tuple[str, str, str], dict[str, float]]]:
    datasets: list[str] = []
    seen_datasets: set[str] = set()
    values: dict[tuple[str, str, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            system = row["system"].strip().lower()
            if system not in SYSTEMS:
                continue
            if row.get("all_hash_ok", "").strip().lower() != "true":
                raise ValueError(f"unvalidated result: {system}/{row['dataset']}/{row['codec']}")
            if int(row.get("repetitions", "0")) != 10:
                raise ValueError(
                    f"expected 10 repetitions: {system}/{row['dataset']}/{row['codec']}"
                )
            dataset = row["dataset"].strip()
            codec = row["codec"].strip().upper()
            if dataset not in seen_datasets:
                datasets.append(dataset)
                seen_datasets.add(dataset)
            metrics = {}
            for field, _ in METRICS:
                value = float(row[field])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"invalid {field}: {system}/{dataset}/{codec}={value}")
                metrics[field] = value
            key = (system, dataset, codec)
            if key in values:
                raise ValueError(f"duplicate summary row: {key}")
            values[key] = metrics
    return datasets, values


def validate_matrix(
    datasets: list[str],
    values: dict[tuple[str, str, str], dict[str, float]],
    expected_datasets: int,
) -> dict[str, list[str]]:
    if len(datasets) != expected_datasets:
        raise ValueError(f"expected {expected_datasets} datasets, found {len(datasets)}: {datasets}")
    codecs_by_system: dict[str, list[str]] = {}
    for system in SYSTEMS:
        present = {codec for candidate, _, codec in values if candidate == system}
        expected = set(CODEC_ORDER[system])
        if present != expected:
            raise ValueError(
                f"unexpected codec set for {system}: missing={sorted(expected - present)}, "
                f"extra={sorted(present - expected)}"
            )
        codecs = list(CODEC_ORDER[system])
        missing = [
            (dataset, codec)
            for dataset in datasets
            for codec in codecs
            if (system, dataset, codec) not in values
        ]
        if missing:
            raise ValueError(f"incomplete {system} matrix, missing {missing}")
        codecs_by_system[system] = codecs
    return codecs_by_system


def plot_panel(
    axis: plt.Axes,
    system: str,
    metric: str,
    operation: str,
    datasets: list[str],
    codecs: list[str],
    values: dict[tuple[str, str, str], dict[str, float]],
) -> None:
    positions = np.arange(len(datasets), dtype=float)
    width = 0.82 / len(codecs)
    all_values: list[float] = []
    for codec_index, codec in enumerate(codecs):
        heights = [values[(system, dataset, codec)][metric] for dataset in datasets]
        all_values.extend(heights)
        offset = (codec_index - (len(codecs) - 1) / 2) * width
        axis.bar(
            positions + offset,
            heights,
            width=width * 0.92,
            color=COLORS.get(codec, "#777777"),
            edgecolor="white",
            linewidth=0.35,
            label=DISPLAY_NAMES.get(codec, codec.title()),
        )
    if max(all_values) / min(all_values) >= 30:
        axis.set_yscale("log")
        axis.set_ylabel("Time (ms, log scale)")
    else:
        axis.set_ylabel("Time (ms)")
    axis.set_title(f"{'TsFile' if system == 'tsfile' else 'InfluxDB TSM'} {operation.lower()}", loc="left", weight="bold")
    axis.set_xticks(positions, datasets, rotation=18, ha="right")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(0.99, 0.97, "Lower is better", transform=axis.transAxes, ha="right", va="top", fontsize=8, color="#555555")


def main() -> None:
    args = parse_args()
    datasets, values = read_summary(args.summary)
    codecs_by_system = validate_matrix(datasets, values, args.expected_datasets)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "legend.fontsize": 7.5})
    figure, axes = plt.subplots(2, 2, figsize=(15.2, 8.8), constrained_layout=True)
    for row, system in enumerate(SYSTEMS):
        for column, (metric, operation) in enumerate(METRICS):
            plot_panel(axes[row, column], system, metric, operation, datasets, codecs_by_system[system], values)
            axes[row, column].legend(ncol=min(4, len(codecs_by_system[system])), frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    figure.suptitle("Encoding performance on the five smallest datasets", fontsize=14, weight="bold")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(f"png={png_path}")
    print(f"pdf={pdf_path}")


if __name__ == "__main__":
    main()
