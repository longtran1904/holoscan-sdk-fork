#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Plot native operator latency summaries from a run folder's comparison.csv."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_folder(folder):
    folder = Path(folder)
    with (folder / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("comparison.csv has no data rows")

    plots = (
        ("median_operator_median_us", 1, "Median per-operator GEMM latency",
         "Latency (microseconds)", "median_per_operator_gemm_latency.png", "µs"),
        ("median_case_wall_ms", 1 / 1000, "Median per-operator latency",
         "Latency (seconds)", "median_per_operator_latency.png", "s"),
    )
    try:
        counts = [int(row["operators"]) for row in rows]
        values = [[float(row[column]) * scale for row in rows] for column, scale, *_ in plots]
        p99s = [json.loads(row["operator_p99_us"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid comparison.csv value: {error}") from error
    if any(count <= 0 for count in counts) or any(
        not math.isfinite(value) or value < 0 for series in values for value in series
    ):
        raise ValueError("Operator counts must be positive and latencies finite and nonnegative")
    if any(
        not isinstance(series, list) or len(series) != count or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 for value in series
        ) for count, series in zip(counts, p99s)
    ):
        raise ValueError("operator_p99_us must contain one finite latency per operator")

    order = sorted(range(len(counts)), key=counts.__getitem__)
    counts = [counts[index] for index in order]
    p99s = [p99s[index] for index in order]
    outputs = []
    for series, (_, _, title, ylabel, filename, unit) in zip(values, plots):
        series = [series[index] for index in order]
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.plot(counts, series, marker="o", linewidth=2)
        axis.set(title=title, xlabel="Number of operators", ylabel=ylabel)
        axis.set_xticks(counts)
        axis.set_ylim(bottom=0, top=max(series) * 1.18 if max(series) else 1)
        axis.grid(axis="y", alpha=0.3)
        for count, value in zip(counts, series):
            axis.annotate(f"{value:.3f} {unit}", (count, value), xytext=(0, 8),
                          textcoords="offset points", ha="center")
        figure.tight_layout()
        output = folder / filename
        figure.savefig(output, dpi=150)
        plt.close(figure)
        outputs.append(output)

    medians = [statistics.median(series) for series in p99s]
    positions = list(range(len(counts)))
    figure, axis = plt.subplots(figsize=(9, 5))
    for position, series in zip(positions, p99s):
        axis.scatter([position] * len(series), series, color="#e89032", s=18, alpha=0.55,
                     label="Operator P99" if position == 0 else None)
    axis.plot(positions, medians, marker="o", linewidth=2, color="#3278b8",
              label="Median operator P99")
    axis.set(title="Per-operator GEMM P99 latency", xlabel="Number of operators",
             ylabel="Latency (microseconds)")
    axis.set_xticks(positions, labels=counts)
    axis.set_ylim(bottom=0, top=max(max(series) for series in p99s) * 1.18 or 1)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    for position, value in zip(positions, medians):
        axis.annotate(f"{value:.3f} µs", (position, value), xytext=(0, 8),
                      textcoords="offset points", ha="center", fontsize=9)
    figure.tight_layout()
    output = folder / "p99_per_operator_gemm_latency.png"
    figure.savefig(output, dpi=150)
    plt.close(figure)
    outputs.append(output)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()
    try:
        outputs = plot_folder(args.results_folder)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Plotting failed: {error}\n")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
