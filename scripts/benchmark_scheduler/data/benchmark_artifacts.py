# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import shutil
import tempfile
from pathlib import Path

from ..benchmark_configurations import EVENT_THREADS, PANEL_NAMES
from .benchmark_results import FIELDS


def graph_paths(csv_path):
    """Return the overview followed by the six panels in reading order."""
    return [
        csv_path.with_suffix(".png"),
        *(csv_path.with_name(f"{csv_path.stem}-{name}.png") for name in PANEL_NAMES),
    ]


def progress_directories(csv_path):
    return tuple(
        csv_path.with_name(f"{csv_path.stem}-progress-{kind}") for kind in ("logs", "plots")
    )


def progress_files(directory, config):
    operations = config.operations_per_operator
    return [
        directory / f"progress_log_{operations}ops_{threads}threads_16operators.txt"
        for threads in EVENT_THREADS
    ]


def preserve_progress(workdir, log_dir, config):
    """Copy all available expected files before validation, including failed runs' logs."""
    log_dir.mkdir(exist_ok=True)
    for source in progress_files(workdir, config):
        if source.is_file():
            shutil.copyfile(source, log_dir / source.name)


def refuse_existing(paths):
    for path in paths:
        if path.exists() or path.is_symlink():
            raise ValueError(f"Output already exists: {path}; choose a new path")


def validate_output(output):
    if output.suffix.lower() != ".csv":
        raise ValueError("--output must have a .csv extension")
    refuse_existing((output, *graph_paths(output), *progress_directories(output)))


def open_results(build_dir, output=None):
    if output is not None:
        validate_output(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        return output.open("x", newline=""), output
    directory = build_dir / "benchmark-results"
    directory.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(  # noqa: SIM115 - caller closes the stream
        mode="w", newline="", prefix="scheduler-", suffix=".csv", dir=directory, delete=False
    )
    return stream, Path(stream.name)


def result_writer(stream):
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    stream.flush()
    return writer


def append_results(writer, stream, rows):
    writer.writerows(rows)
    stream.flush()
