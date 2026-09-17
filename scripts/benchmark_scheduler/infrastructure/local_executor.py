# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..benchmark_configurations import LIBRARY_PATH_VARIABLES
from .command_executor import command_process


@dataclass
class LocalExecutor:
    repo_root: Path

    @property
    def environment(self):
        return os.environ

    def path(self, local_path):
        return local_path

    def command(self, argv, env=None, cwd=None):  # noqa: ARG002 - executor interface
        overrides = [f"{key}={env[key]}" for key in LIBRARY_PATH_VARIABLES if env and key in env]
        return ["env", *overrides, *argv] if overrides else argv

    def process(self, command, *, env=None, **kwargs):
        return command_process(command, env=env, **kwargs)

    def run(self, command):
        subprocess.run(command, check=True)
