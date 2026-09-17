#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and plot counter deltas against operator completion on a common clock."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from run import COUNTERS, validate_trace, write_json


def read_csv(path):
    with path.open() as stream:
        return [
            {key: value if key == "phase" else int(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def intervals(samples, progress):
    completions = sorted(
        row["monotonic_ns"] for row in progress if row["completed_operations"] == 1000
    )
    origin = samples[0]["monotonic_ns"]
    rows = []
    for previous, current in zip(samples, samples[1:], strict=False):
        elapsed = (current["monotonic_ns"] - previous["monotonic_ns"]) / 1e9
        if elapsed <= 0:
            raise ValueError("Nonpositive sample interval")
        row = {
            "monotonic_ns": current["monotonic_ns"],
            "elapsed_s": (current["monotonic_ns"] - origin) / 1e9,
            "interval_s": elapsed,
        }
        for name in COUNTERS:
            delta = current[name] - previous[name]
            if delta < 0:
                raise ValueError(f"Counter reset: {name}")
            row[name + "_delta"] = delta
            row[name + "_per_s"] = delta / elapsed
        row["unfinished_operators"] = len(completions) - sum(
            t <= current["monotonic_ns"] for t in completions
        )
        # Gauge is recorded as-is; it is NOT an unfinished-operator count.
        row["running_threads"] = current["running_threads"]
        rows.append(row)
    return rows, completions


def tail_bounds(samples, first_completion):
    before = [r for r in samples if r["read_end_ns"] <= first_completion]
    after = [r for r in samples if r["monotonic_ns"] > first_completion]
    if not before or not after:
        raise ValueError("Completion is not bracketed by samples")
    total = samples[-1]["steal_attempts"] - samples[0]["steal_attempts"]
    return {
        "attempts_after_first_completion_lower": samples[-1]["steal_attempts"]
        - after[0]["steal_attempts"],
        "attempts_after_first_completion_upper": samples[-1]["steal_attempts"]
        - before[-1]["steal_attempts"],
        "attempts_total": total,
        "boundary_uncertainty_ms": (after[0]["read_end_ns"] - before[-1]["monotonic_ns"]) / 1e6,
    }


def draw_run(axes, samples, progress, rows, completions, title):
    origin = samples[0]["monotonic_ns"]
    times = [r["elapsed_s"] for r in rows]
    axes[0].plot(times, [r["steal_attempts_per_s"] for r in rows], color="tab:orange")
    axes[0].set_ylabel("Steal attempts/s")
    axes[1].plot(times, [r["steal_successes_per_s"] for r in rows], color="tab:green")
    axes[1].set_ylabel("Successful steals/s")
    axes[2].plot(
        times, [r["worker_waitforjob_total_us_delta"] / 1000 for r in rows], color="tab:purple"
    )
    axes[2].set_ylabel("Worker wait\nms/interval (sum)")
    twin = axes[2].twinx()
    twin.step(
        times, [r["unfinished_operators"] for r in rows], where="post", color="black", alpha=0.55
    )
    twin.set_ylabel("Unfinished operators")
    twin.set_ylim(0, 17)
    for op in range(16):
        points = [r for r in progress if r["operator_id"] == op]
        axes[3].plot(
            [(r["monotonic_ns"] - origin) / 1e9 for r in points],
            [r["completed_operations"] for r in points],
            linewidth=1,
            label=str(op),
        )
    axes[3].set_ylabel("Completed ops\nper operator")
    axes[3].set_xlabel("Seconds since initial counter sample (CLOCK_MONOTONIC)")
    axes[0].set_title(title)
    for axis in axes:
        for completion in completions:
            axis.axvline((completion - origin) / 1e9, color="gray", alpha=0.12, linewidth=0.7)
        axis.grid(alpha=0.2)
        axis.set_xlim(0, (samples[-1]["monotonic_ns"] - origin) / 1e9 * 1.02)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import matplotlib  # noqa: PLC0415 - plotting is optional for validation/tests

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415 - select the backend before pyplot

    out = args.output.resolve()
    experiment = json.loads((out / "experiment.json").read_text())
    if experiment["status"] != "complete":
        raise ValueError("Experiment is incomplete")
    summaries, data = [], {}
    representative_names = set()
    for config in ["queue_stealing", "both"]:
        candidates = sorted(
            [
                r
                for r in experiment["records"]
                if r["instrumented"]
                and r["config"] == config
                and r["interval_ms"] == experiment["selected_interval_ms"]
            ],
            key=lambda r: r["throughput_hz"],
        )
        representative_names.add(candidates[len(candidates) // 2]["run"])
    for record in experiment["records"]:
        if not record["instrumented"]:
            continue
        directory = out / record["run"]
        validate_trace(directory)
        samples = read_csv(next(directory.glob("ebs-*.csv")))
        progress = read_csv(next(directory.glob("progress_log_*.txt")))
        rows, completions = intervals(samples, progress)
        with (directory / "intervals.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        origin = samples[0]["monotonic_ns"]
        summary = {
            **record,
            **tail_bounds(samples, completions[0]),
            "first_completion_s": (completions[0] - origin) / 1e9,
            "last_completion_s": (completions[-1] - origin) / 1e9,
            "completion_spread_s": (completions[-1] - completions[0]) / 1e9,
            "max_sample_read_us": max(
                (r["read_end_ns"] - r["monotonic_ns"]) / 1000 for r in samples
            ),
        }
        summaries.append(summary)
        run_data = (samples, progress, rows, completions)
        if record["run"] in representative_names:
            data[record["run"]] = run_data
        figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
        draw_run(
            axes,
            *run_data,
            f"{record['config']} · {record['throughput_hz']:,.1f} ops/s · {record['run']}",
        )
        figure.tight_layout()
        figure.savefig(directory / "timeline.png", dpi=150)
        plt.close(figure)

    selected = [r for r in summaries if r["interval_ms"] == experiment["selected_interval_ms"]]
    aggregate = {}
    figure, axes = plt.subplots(4, 2, figsize=(20, 12), sharex="col")
    for column, config in enumerate(["queue_stealing", "both"]):
        records = [r for r in selected if r["config"] == config]
        representative = sorted(records, key=lambda r: r["throughput_hz"])[len(records) // 2]
        draw_run(
            axes[:, column],
            *data[representative["run"]],
            f"{config} · representative run {representative['run']}",
        )
        aggregate[config] = {
            "repetitions": len(records),
            "representative_run": representative["run"],
            "median_throughput_hz": statistics.median(r["throughput_hz"] for r in records),
            "median_steal_attempts": statistics.median(r["steal_attempts"] for r in records),
            "median_steal_successes": statistics.median(r["steal_successes"] for r in records),
            "median_completion_spread_s": statistics.median(
                r["completion_spread_s"] for r in records
            ),
            "median_tail_attempt_percent_lower": statistics.median(
                100 * r["attempts_after_first_completion_lower"] / max(1, r["attempts_total"])
                for r in records
            ),
            "median_tail_attempt_percent_upper": statistics.median(
                100 * r["attempts_after_first_completion_upper"] / max(1, r["attempts_total"])
                for r in records
            ),
        }
    figure.tight_layout()
    figure.savefig(out / "comparison.png", dpi=150)
    figure.savefig(out / "comparison.pdf")
    plt.close(figure)
    write_json(out / "analysis.json", {"aggregate": aggregate, "runs": summaries})
    with (out / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# Periodic GXF steal counters",
        "",
        "13 workers, 16 operators, 1,000 operations each, 1 ms busy wait. "
        "Dispatcher: CPU 13, SCHED_FIFO priority 99. Workers retain their original affinity.",
        "",
        f"Selected sampling interval: {experiment['selected_interval_ms']} ms. "
        "Throughput overhead includes both sampling and operator progress instrumentation.",
        "",
        "| Configuration | Median ops/s | Attempts | Successful steals | Completion spread | "
        "Attempts after first completion | Overhead |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config, result in aggregate.items():
        lines.append(
            f"| {config} | {result['median_throughput_hz']:,.1f} | "
            f"{result['median_steal_attempts']:,.0f} | {result['median_steal_successes']:,.0f} | "
            f"{result['median_completion_spread_s']:.3f} s | "
            f"{result['median_tail_attempt_percent_lower']:.1f}–"
            f"{result['median_tail_attempt_percent_upper']:.1f}% | "
            f"{experiment['overhead_percent'][config]:.2f}% |"
        )
    lines += [
        "",
        "Values are medians across repetitions; the two plotted runs are selected "
        "by median throughput. Each trace has its own timeline.png, raw cumulative counters, "
        "interval deltas and operator progress. "
        "Gray lines mark operator completion at its 1,000th compute, not graph shutdown.",
        "",
        "Tail percentages bracket the first completion using neighboring counter samples. "
        "Counters use independent relaxed atomic loads; read_end_ns records the read window. "
        "Worker wait is summed across workers and can exceed elapsed wall time. "
        "running_threads is a GXF gauge, not an operator count.",
        "",
        "Final values for all eight counters match GXF's own final log in every accepted trace. "
        "All 160 progress checkpoints are present in each accepted run.",
        "",
        "These traces show when attempts and completions occur. "
        "They do not directly record queue contents "
        "or establish that a successful steal improved long-term assignment.",
    ]
    if not experiment["overhead_acceptable"]:
        lines += [
            "",
            "Instrumentation changes median throughput by at least 5% even at 50 ms; "
            "treat timing as perturbed and do not make a causal claim from this comparison.",
        ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
