# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Protocol

CONTAINER_SUPERVISOR = """
import os
import signal
import subprocess
import sys
import threading

process = subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL, start_new_session=True)

def stop_on_disconnect():
    while os.read(0, 4096):
        pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

threading.Thread(target=stop_on_disconnect, daemon=True).start()
sys.exit(process.wait())
"""


class CommandExecutor(Protocol):
    repo_root: Path
    environment: dict

    def path(self, local_path: Path) -> Path: ...
    def command(self, argv, env=None, cwd=None) -> list: ...
    def process(self, command, *, env=None, **kwargs) -> AbstractContextManager: ...
    def run(self, command) -> None: ...


@contextmanager
def command_process(command, *, env=None, **kwargs):
    with subprocess.Popen(command, env=env, **kwargs) as process:
        try:
            yield process
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
