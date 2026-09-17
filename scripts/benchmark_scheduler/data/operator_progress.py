# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import re

from ..benchmark_configurations import CHECKPOINT_OPERATIONS


def progress_metadata(text, operators):
    """Read and validate the optional pool/interval header; retain legacy log support."""
    if not text.startswith("# "):
        return {}, text
    header, _, checkpoints = text.partition("\n")
    metadata = json.loads(header[2:])
    if not isinstance(metadata, dict):
        raise ValueError("Progress metadata must be a JSON object")
    threads = metadata.get("threads")
    if type(threads) is not int or not 1 <= threads <= operators:
        raise ValueError("Invalid total worker count in progress metadata")
    assignment = metadata.get("pool_assignment", "gcd")
    if assignment == "shared-twice" and operators == 16:
        shared = threads if threads <= 8 else 16 - threads
        secondary_operators = 0 if threads <= 8 else 16 - 2 * shared
    elif assignment == "shared-eight" and operators == 16:
        shared = min(threads, 8)
        secondary_operators = 8 if threads > 8 else 0
    elif assignment == "gcd":
        shared = math.gcd(threads, operators)
        secondary_operators = threads - shared
    else:
        raise ValueError("Unknown progress pool assignment")
    secondary = threads - shared
    if (metadata.get("shared_threads"), metadata.get("secondary_threads")) != (shared, secondary):
        raise ValueError("Progress pool sizes do not match the declared assignment")
    expected = ["shared"] * (operators - secondary_operators) + ["secondary"] * secondary_operators
    if metadata.get("operator_pools") != expected:
        raise ValueError("Progress operator assignments do not match the pool split")
    if metadata.get("checkpoint_operations") not in (100, 10000):
        raise ValueError("Invalid checkpoint interval")
    return metadata, checkpoints


def parse_progress(text, operators, operations, *, include_metadata=False):
    """Validate integer millisecond checkpoints in each operator's recorded order."""
    metadata, text = progress_metadata(text, operators)
    interval = metadata.get("checkpoint_operations", CHECKPOINT_OPERATIONS)
    if metadata and interval != (100 if operations == 1000 else 10000):
        raise ValueError("Checkpoint interval does not match workload")
    timestamps = {operator: [] for operator in range(operators)}
    for number, line in enumerate(text.splitlines(), 1):
        if not re.fullmatch(r"[0-9]+,[0-9]+", line):
            raise ValueError(f"Malformed progress checkpoint at line {number}: {line!r}")
        operator, elapsed = map(int, line.split(","))
        if operator not in timestamps:
            raise ValueError(f"Unexpected operator ID at line {number}: {operator}")
        previous = timestamps[operator]
        if previous and elapsed < previous[-1]:
            raise ValueError(f"Decreasing timestamp for operator {operator} at line {number}")
        previous.append(elapsed)
    expected = operations // interval
    for operator, values in timestamps.items():
        if len(values) != expected:
            raise ValueError(
                f"Operator {operator}: expected {expected} checkpoints, got {len(values)}"
            )
    timestamps = timestamps if expected else {}
    return (timestamps, metadata) if include_metadata else timestamps


def read_progress(path, config, *, include_metadata=False):
    try:
        return parse_progress(
            path.read_text(), 16, config.operations_per_operator, include_metadata=include_metadata
        )
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
