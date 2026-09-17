# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import math
import re

from ..benchmark_configurations import (
    CONFIGURATIONS,
    EVENT_THREADS,
    GREEDY_OPERATORS,
    configuration_key,
)

FIELDS = (
    "scheduler",
    "queue_stealing",
    "postcheck_fastpath",
    "busy_wait",
    "trial",
    "threads",
    "operators",
    "total_operations",
    "throughput_hz",
)


def parse_results(output, config):
    """Read the result table, rejecting incomplete or invalid workload results."""
    event = config.scheduler == "event_based"
    header = ["Trial", *(["Threads"] if event else []), "Operators", "Total Ops", "Ops/s"]
    rows = []
    in_table = False
    for line in output.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [cell.strip() for cell in line[1:-1].split("|")]
        if cells == header:
            if in_table:
                raise ValueError("Duplicate result table")
            in_table = True
            continue
        if not in_table or all(re.fullmatch(r"[-:]+", cell) for cell in cells):
            continue
        if len(cells) != len(header):
            raise ValueError(f"Malformed result row: {line}")
        try:
            numbers = [int(cell) for cell in cells[:-1]]
            throughput = float(cells[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric result: {line}") from exc
        trial, *coordinates, total = numbers
        threads, operators = coordinates if event else (1, coordinates[0])
        rows.append(
            dict(
                zip(
                    FIELDS,
                    (
                        config.scheduler,
                        int(config.queue_stealing) if event else "",
                        int(config.postcheck_fastpath) if event else "",
                        int(config.busy_wait),
                        trial,
                        threads,
                        operators,
                        total,
                        throughput,
                    ),
                    strict=True,
                )
            )
        )

    return validate_rows(rows, config)


def validate_rows(rows, config):
    event = config.scheduler == "event_based"
    for row in rows:
        if (
            row["trial"] < 0
            or min(row["threads"], row["operators"], row["total_operations"]) <= 0
            or not math.isfinite(row["throughput_hz"])
            or row["throughput_hz"] <= 0
        ):
            raise ValueError(f"Nonpositive or nonfinite result: {row}")
        expected_flags = (
            (int(config.queue_stealing), int(config.postcheck_fastpath)) if event else ("", "")
        )
        if (
            row["scheduler"] != config.scheduler
            or row["busy_wait"] != int(config.busy_wait)
            or (row["queue_stealing"], row["postcheck_fastpath"]) != expected_flags
        ):
            raise ValueError(f"Unexpected configuration fields: {row}")
    expected_x = EVENT_THREADS if event else GREEDY_OPERATORS
    expected = {
        (trial, x if event else 1, 16 if event else x) for trial, x in enumerate(expected_x)
    }
    actual = {(row["trial"], row["threads"], row["operators"]) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError(f"Incomplete or unexpected trial coordinates for {config.label}")
    operations_per_operator = config.operations_per_operator
    for row in rows:
        if row["total_operations"] != row["operators"] * operations_per_operator:
            raise ValueError(f"Unexpected operation count for {config.label}: {row}")
    return rows


def match_baseline(rows, baseline):
    """Match by trial coordinates and workload, independently of table row order."""

    def keyed(values):
        return {
            (
                r["scheduler"],
                r["busy_wait"],
                r["trial"],
                r["threads"],
                r["operators"],
                r["total_operations"],
            ): r
            for r in values
        }

    current, default = keyed(rows), keyed(baseline)
    if (
        len(current) != len(rows)
        or len(default) != len(baseline)
        or current.keys() != default.keys()
    ):
        raise ValueError("Event-based trial coordinates do not match the default configuration")
    return [
        (row, row["throughput_hz"] / default[key]["throughput_hz"])
        for key, row in sorted(current.items(), key=lambda item: item[1]["threads"])
    ]


def validate_results(results, configurations=CONFIGURATIONS):
    expected = {configuration_key(config) for config in configurations}
    if set(results) != expected:
        raise ValueError("Comparison requires all ten configurations")
    for config in configurations:
        validate_rows(results[configuration_key(config)], config)
    return results


def load_results(path):
    """Load the same typed, validated rows produced by benchmark table parsing."""
    results = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"{path}: invalid CSV fields")
        for number, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("incorrect field count")
                event = row["scheduler"] == "event_based"
                for field in FIELDS[1:-1]:
                    if field in ("queue_stealing", "postcheck_fastpath") and not event:
                        if row[field] != "":
                            raise ValueError("greedy flags must be empty")
                    else:
                        row[field] = int(row[field])
                row["throughput_hz"] = float(row["throughput_hz"])
                for field in (
                    "busy_wait",
                    *(("queue_stealing", "postcheck_fastpath") if event else ()),
                ):
                    if row[field] not in (0, 1):
                        raise ValueError("flags must be 0 or 1")
                key = (
                    row["scheduler"],
                    bool(row["busy_wait"]),
                    bool(row["queue_stealing"]),
                    bool(row["postcheck_fastpath"]),
                )
                results.setdefault(key, []).append(row)
            except ValueError as exc:  # noqa: PERF203 - report the failing CSV line
                raise ValueError(f"{path}: invalid CSV row {number}: {exc}") from exc
    return validate_results(results)
