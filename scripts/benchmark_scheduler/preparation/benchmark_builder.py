# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

from ..infrastructure.command_executor import CommandExecutor


def read_build_cache(build_dir, executor):
    cache = {}
    for line in (build_dir / "CMakeCache.txt").read_text().splitlines():
        match = re.match(r"([^/#][^:]*):[^=]+=(.*)", line)
        if match:
            cache[match[1]] = match[2]
    home = cache.get("CMAKE_HOME_DIRECTORY")
    repo_root = executor.repo_root
    cache_root = Path(home) if home else Path(home or ".").resolve()
    if not home or cache_root != repo_root:
        raise ValueError(
            f"Build cache belongs to {home!r}, not {repo_root}. "
            "Run inside the CUDA container or select it with --container NAME."
        )
    for option in ("HOLOSCAN_BUILD_EXAMPLES", "HOLOSCAN_CPP_EXAMPLES"):
        if cache.get(option, "").upper() not in ("ON", "TRUE", "YES", "1"):
            raise ValueError(f"The configured build must have {option}=ON")
    return cache


def build_benchmarks(build_dir, jobs, executor: CommandExecutor, configurations):
    configure = ["cmake", "-S", str(executor.repo_root), "-B", str(executor.path(build_dir))]
    targets = list(dict.fromkeys(config.target for config in configurations))
    command = ["cmake", "--build", str(executor.path(build_dir)), "--target", *targets]
    if jobs is not None:
        command.extend(["--parallel", str(jobs)])
    for argv in (configure, command):
        executor.run(argv)
