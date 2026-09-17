# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shlex
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path

from ..benchmark_configurations import LIBRARY_PATH_VARIABLES, configuration_key, selected_progress
from ..data.benchmark_artifacts import (
    append_results,
    preserve_progress,
    progress_directories,
    progress_files,
    result_writer,
)
from ..data.benchmark_results import match_baseline, parse_results
from ..data.operator_progress import read_progress
from ..infrastructure.command_executor import CommandExecutor


def run_benchmark(build_dir, config, env, executor, cwd):
    build_dir = executor.path(build_dir)
    executable = build_dir / "examples" / config.target / "cpp" / config.target
    command = [str(executable), *config.flags]
    workload = "1 ms busy wait" if config.busy_wait else "normal"
    print(f"\n=== {config.scheduler}: {config.label} ({workload}) ===", flush=True)
    print(f"Working directory: {cwd}", flush=True)
    print(f"Command: {shlex.join(executor.command(command, env, cwd))}", flush=True)
    lines = []
    with executor.process(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            # Logs are streamed but only result-table candidates need retaining.
            if re.sub(r"\x1b\[[0-9;]*m", "", line).lstrip().startswith("|"):
                lines.append(line)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
    return parse_results("".join(lines), config)


def benchmark_environment(build_dir, cache, inherited):
    env = inherited.copy()
    # Normalize lexically: container paths must not be resolved on the host filesystem.
    library_dirs = [build_dir / "lib", build_dir / "lib64"]
    configured_lib = Path(os.path.normpath(build_dir / cache.get("CMAKE_INSTALL_LIBDIR", "lib")))
    if configured_lib.is_relative_to(build_dir) and configured_lib not in library_dirs:
        library_dirs.append(configured_lib)
    libraries = os.pathsep.join(map(str, library_dirs))
    for variable in LIBRARY_PATH_VARIABLES:
        env[variable] = libraries + (os.pathsep + env[variable] if env.get(variable) else "")
    return env


def check_binaries(build_dir, configurations):
    for target in dict.fromkeys(config.target for config in configurations):
        executable = build_dir / "examples" / target / "cpp" / target
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(
                f"Required benchmark executable missing or not executable: {executable}"
            )


def run_configurations(
    build_dir,
    cache,
    executor: CommandExecutor,
    configurations,
    repo_root,
    stream,
    csv_path,
    results,
):
    env = benchmark_environment(executor.path(build_dir), cache, executor.environment)
    log_dir, _ = progress_directories(csv_path)
    writer = result_writer(stream)
    for config in configurations:
        workspace = (
            tempfile.TemporaryDirectory(prefix="scheduler-progress-", dir=build_dir)
            if config.scheduler == "event_based"
            else nullcontext(str(repo_root))
        )
        with workspace as directory:
            workdir = Path(directory)
            try:
                rows = run_benchmark(build_dir, config, env, executor, workdir)
            finally:
                if selected_progress(config):
                    preserve_progress(workdir, log_dir, config)
        baseline_key = ("event_based", config.busy_wait, False, False)
        if config.scheduler == "event_based" and baseline_key in results:
            match_baseline(rows, results[baseline_key])
        append_results(writer, stream, rows)
        results[configuration_key(config)] = rows
        if selected_progress(config):
            for path in progress_files(log_dir, config):
                read_progress(path, config)
