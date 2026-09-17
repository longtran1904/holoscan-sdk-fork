# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose independent preparation, execution, and analysis commands."""

import argparse
import subprocess
import sys
from pathlib import Path

from .analysis.operator_progress_plots import save_progress_graphs
from .analysis.plotting_backend import load_pyplot
from .analysis.throughput_plots import save_graphs
from .benchmark_configurations import CONFIGURATIONS
from .data.benchmark_artifacts import (
    graph_paths,
    open_results,
    progress_directories,
    refuse_existing,
    validate_output,
)
from .data.benchmark_results import load_results
from .execution.benchmark_runner import check_binaries, run_configurations
from .infrastructure.docker_executor import inspect_container
from .infrastructure.local_executor import LocalExecutor
from .preparation.benchmark_builder import build_benchmarks, read_build_cache

REPO_ROOT = Path(__file__).resolve().parents[2]


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "run", "plot", "all"):
        stage = commands.add_parser(command)
        if command != "plot":
            stage.add_argument(
                "--build-dir",
                required=True,
                type=Path,
                help="Configured SDK build, relative to the repository root",
            )
            stage.add_argument("--container", help="Existing running container name or ID")
        if command in ("build", "all"):
            stage.add_argument("--jobs", type=positive_int)
        if command in ("run", "all"):
            stage.add_argument("--output", type=Path, help="New CSV path")
        if command == "plot":
            stage.add_argument("--input", required=True, type=Path)
            stage.add_argument("--output-dir", type=Path)
            stage.add_argument("--progress-logs", type=Path)
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv.insert(0, "all")
    return parser.parse_args(argv)


def plot_results(plt, results, csv_path, output_dir=None, progress_logs=None):
    destination = output_dir.resolve() / csv_path.name if output_dir else csv_path
    discovered = progress_directories(csv_path)[0]
    logs = progress_logs.resolve() if progress_logs else discovered
    use_progress = progress_logs is not None or logs.exists() or logs.is_symlink()
    _, plots = progress_directories(destination)
    refuse_existing((*graph_paths(destination), plots))
    if use_progress and not logs.is_dir():
        raise ValueError(f"Progress logs directory not found: {logs}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if use_progress:
        save_progress_graphs(plt, logs, plots)
    for path in save_graphs(plt, results, destination):
        print(f"PNG: {path}")
    if use_progress:
        print(f"Progress plots: {plots}")


def main(argv=None):
    args = arguments(argv)
    csv_path = None
    results = {}
    try:
        # Full workflow preflight must happen before expensive build/run work.
        plt = load_pyplot() if args.command in ("all", "plot") else None
        if args.command == "plot":
            source = args.input.resolve()
            results = load_results(source)
            plot_results(plt, results, source, args.output_dir, args.progress_logs)
            return 0
        build_dir = (REPO_ROOT / args.build_dir).resolve()
        executor = (
            inspect_container(args.container, REPO_ROOT)
            if args.container
            else LocalExecutor(REPO_ROOT)
        )
        if args.container:
            print(f"Using existing container: {executor.name} ({executor.id[:12]})", flush=True)
            print(f"Container image: {executor.image_id}", flush=True)
        runtime_build = executor.path(build_dir)
        cache = read_build_cache(build_dir, executor)
        output = args.output.resolve() if getattr(args, "output", None) else None
        if output:
            validate_output(output)
        print(f"Benchmark build: {runtime_build}", flush=True)
        if args.command in ("build", "all"):
            build_benchmarks(build_dir, args.jobs, executor, CONFIGURATIONS)
        if args.command == "build":
            return 0
        check_binaries(build_dir, CONFIGURATIONS)
        stream, csv_path = open_results(build_dir, output)
        with stream:
            run_configurations(
                build_dir, cache, executor, CONFIGURATIONS, REPO_ROOT, stream, csv_path, results
            )
        print(f"CSV: {csv_path}")
        print(f"Progress logs: {progress_directories(csv_path)[0]}")
        if args.command == "all":
            plot_results(plt, results, csv_path)
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
        KeyboardInterrupt,
    ) as exc:
        print(f"Benchmark failed: {exc or 'interrupted'}", file=sys.stderr)
        if csv_path is not None:
            kind = "Complete" if len(results) == len(CONFIGURATIONS) else "Partial"
            print(f"{kind} CSV preserved: {csv_path}", file=sys.stderr)
            for directory in progress_directories(csv_path):
                if directory.exists():
                    print(f"Progress artifacts preserved: {directory}", file=sys.stderr)
        return 1
