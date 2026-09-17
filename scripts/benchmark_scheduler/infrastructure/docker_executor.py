# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..benchmark_configurations import LIBRARY_PATH_VARIABLES
from .command_executor import CONTAINER_SUPERVISOR, command_process


@dataclass(frozen=True)
class DockerExecutor:
    """An existing container with this checkout bind-mounted by ./run launch."""

    id: str
    name: str
    image_id: str
    repo_root: Path
    environment: dict
    local_root: Path

    def path(self, local_path):
        if not local_path.is_relative_to(self.local_root):
            raise ValueError("--container requires --build-dir inside this checkout's bind mount")
        return self.repo_root / local_path.relative_to(self.local_root)

    def command(self, argv, env=None, cwd=None):
        command = ["docker", "exec", "--workdir", str(self.path(cwd) if cwd else self.repo_root)]
        for variable, value in (env or {}).items():
            command.extend(["--env", f"{variable}={value}"])
        return [*command, self.id, *argv]

    def process(self, command, *, env=None, **kwargs):
        command = self.command(
            ["python3", "-c", CONTAINER_SUPERVISOR, *command], env, kwargs.get("cwd")
        )
        command.insert(2, "--interactive")
        return command_process(command, stdin=subprocess.PIPE, **kwargs)

    def run(self, command):
        with self.process(command) as process:
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)


def inspect_container(name, repo_root):
    """Resolve a name once and pin all work to that running container's ID."""
    result = subprocess.run(
        ["docker", "container", "inspect", name], check=True, capture_output=True, text=True
    )
    (info,) = json.loads(result.stdout)
    if not info["State"]["Running"]:
        raise ValueError(f"Container {name!r} is not running; use your ./run launch container")
    roots = [
        Path(mount["Destination"])
        for mount in info["Mounts"]
        if mount["Type"] == "bind" and Path(mount["Source"]).resolve() == repo_root
    ]
    if len(roots) != 1:
        raise ValueError(
            f"Container {name!r} must bind-mount this checkout exactly once: {repo_root}"
        )
    environment = dict(entry.split("=", 1) for entry in info["Config"]["Env"] or [])
    # Docker inherits the remaining container environment itself. Never forward host variables.
    environment = {key: environment[key] for key in LIBRARY_PATH_VARIABLES if key in environment}
    return DockerExecutor(
        info["Id"], info["Name"].lstrip("/"), info["Image"], roots[0], environment, repo_root
    )
