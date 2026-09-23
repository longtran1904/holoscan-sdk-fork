#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Summarize one run_native_operators.sh results folder into comparison.csv."""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)"
LOG_NAME = re.compile(r"run-(?:(?P<run>\d+)-)?operators-(?P<operators>\d+)\.stdout\.log")
OPERATOR = re.compile(r"Operator (?P<index>\d+)/(?P<total>\d+)")
GEMM = re.compile(
    rf"GEMM \(M=\d+, N=\d+, K=\d+, L2 flush=(?:off|on)\): "
    rf"(?P<mean_us>{NUMBER}) ± {NUMBER} us, min={NUMBER} us, "
    rf"median=(?P<median_us>{NUMBER}) us, P99=(?P<p99_us>{NUMBER}) us, "
    rf"CV={NUMBER}%, {NUMBER} TFLOP/s"
)
WALL = re.compile(
    rf"Timed GEMM total: (?P<gpu_total_ms>{NUMBER}) ms, "
    rf"wall time: (?P<case_wall_ms>{NUMBER}) ms, "
    rf"GEMM/wall: (?P<gemm_wall_percent>{NUMBER})%"
)
PREFIX = "LT_SGEMM_NATIVE "
FIELDS = (
    "operators", "completed_cases", "timed_gemms", "compute_calls", "app_wall_ms",
    "process_wall_s", "process_factor_vs_10", "app_gemms_per_s",
    "median_operator_mean_us", "median_operator_median_us",
    "minimum_operator_mean_us", "maximum_operator_mean_us",
    "median_operator_p99_us", "operator_p99_us",
    "sum_gpu_event_ms", "sum_gpu_event_over_app_percent",
    "median_case_wall_ms", "median_gemm_wall_percent", "max_rss_kb", "stderr_bytes",
)


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def parse_run(stdout_path, count):
    stem = stdout_path.name.removesuffix(".stdout.log")
    lines = stdout_path.read_text(encoding="utf-8").splitlines()
    operators, gemms, walls, records = [], [], [], []
    for raw in lines:
        line = raw.strip()
        if line.startswith("Operator "):
            match = OPERATOR.fullmatch(line)
            if match is None or int(match["total"]) != count:
                raise ValueError(f"Malformed operator line in {stdout_path}: {line}")
            operators.append(int(match["index"]))
        elif line.startswith("GEMM ("):
            match = GEMM.fullmatch(line)
            if match is None:
                raise ValueError(f"Malformed GEMM line in {stdout_path}: {line}")
            gemms.append(
                {key: positive(float(value), key) for key, value in match.groupdict().items()}
            )
        elif line.startswith("Timed GEMM total:"):
            match = WALL.fullmatch(line)
            if match is None:
                raise ValueError(f"Malformed timing line in {stdout_path}: {line}")
            walls.append(
                {key: positive(float(value), key) for key, value in match.groupdict().items()}
            )
        elif line.startswith(PREFIX):
            records.append(json.loads(line[len(PREFIX) :]))

    if sorted(operators) != list(range(1, count + 1)) or len(gemms) != count or len(walls) != count:
        raise ValueError(f"Expected {count} complete, unique operator results in {stdout_path}")
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError(f"Expected one native completion record in {stdout_path}")
    record = records[0]
    for key, expected in (
        ("schema_version", 2), ("operator_count", count), ("completed_cases", count),
        ("timed_gemms", count * 1000), ("return_code", 0),
    ):
        if type(record.get(key)) is not int or record[key] != expected:
            raise ValueError(f"Invalid {key} in {stdout_path}: {record.get(key)}")
    gemms_per_tick = record.get("gemms_per_tick")
    if type(gemms_per_tick) is not int or gemms_per_tick < 1:
        raise ValueError(f"Invalid gemms_per_tick in {stdout_path}")
    expected_calls = count * 1000 // gemms_per_tick
    if count * 1000 % gemms_per_tick or record.get("compute_calls") != expected_calls:
        raise ValueError(f"Invalid compute_calls in {stdout_path}")
    if record.get("completed") is not True:
        raise ValueError(f"Incomplete native run in {stdout_path}")
    app_ms = positive(record.get("app_run_wall_ms"), "app_run_wall_ms")

    time_path = stdout_path.with_name(stem + ".time.txt")
    timing = {}
    for line in time_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in timing:
            raise ValueError(f"Malformed time file: {time_path}")
        timing[key] = value
    if timing.get("exit_status") != "0":
        raise ValueError(f"Benchmark exited unsuccessfully: {time_path}")
    process_s = positive(float(timing["process_wall_seconds"]), "process_wall_seconds")
    max_rss_kb = int(timing["max_rss_kb"])
    if max_rss_kb < 0:
        raise ValueError(f"Invalid max_rss_kb in {time_path}")
    stderr_bytes = stdout_path.with_name(stem + ".stderr.log").stat().st_size

    means = [case["mean_us"] for case in gemms]
    operator_p99s = [case["p99_us"] for _, case in sorted(zip(operators, gemms))]
    gpu_ms = sum(case["gpu_total_ms"] for case in walls)
    return dict(
        operators=count,
        completed_cases=record["completed_cases"],
        timed_gemms=record["timed_gemms"],
        compute_calls=record["compute_calls"],
        app_wall_ms=app_ms,
        process_wall_s=process_s,
        process_factor_vs_10="",
        app_gemms_per_s=record["timed_gemms"] * 1000 / app_ms,
        median_operator_mean_us=statistics.median(means),
        median_operator_median_us=statistics.median(case["median_us"] for case in gemms),
        minimum_operator_mean_us=min(means),
        maximum_operator_mean_us=max(means),
        median_operator_p99_us=statistics.median(case["p99_us"] for case in gemms),
        operator_p99_us=json.dumps(operator_p99s, separators=(",", ":")),
        sum_gpu_event_ms=gpu_ms,
        sum_gpu_event_over_app_percent=100 * gpu_ms / app_ms,
        median_case_wall_ms=statistics.median(case["case_wall_ms"] for case in walls),
        median_gemm_wall_percent=statistics.median(case["gemm_wall_percent"] for case in walls),
        max_rss_kb=max_rss_kb,
        stderr_bytes=stderr_bytes,
    )


def summarize_folder(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Not a results directory: {folder}")
    logs = sorted(folder.glob("*.stdout.log"))
    if not logs:
        raise ValueError(f"No native stdout logs in {folder}")
    runs = []
    for log in logs:
        match = LOG_NAME.fullmatch(log.name)
        if match is None:
            raise ValueError(f"Unexpected stdout log name: {log.name}")
        count = int(match["operators"])
        run_number = int(match["run"]) if match["run"] is not None else count
        runs.append((run_number, parse_run(log, count)))
    runs.sort(key=lambda entry: entry[0])
    numbers = [number for number, _ in runs]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"Duplicate run number in {folder}")
    rows = [row for _, row in runs]
    baseline = next((row["process_wall_s"] for row in rows if row["operators"] == 10), None)
    if baseline is not None:
        for row in rows:
            row["process_factor_vs_10"] = row["process_wall_s"] / baseline

    output = folder / "comparison.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_folder", help="app-run folder name or path")
    args = parser.parse_args()
    folder = Path(args.results_folder)
    if not folder.is_absolute() and not folder.is_dir():
        folder = RESULTS_ROOT / folder
    try:
        output = summarize_folder(folder)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, f"Comparison failed: {error}\n")
    print(f"Comparison: {output}")


if __name__ == "__main__":
    sys.exit(main())
