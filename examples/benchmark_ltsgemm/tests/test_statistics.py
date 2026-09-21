# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PT009

"""Exercise the native statistics implementation with only a C++17 compiler."""

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class StatisticsTests(unittest.TestCase):
    def test_statistics(self):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or not shutil.which(compiler[0]):
            self.skipTest("a C++17 compiler is required for native statistics tests")
        tests = Path(__file__).resolve().parent
        native = tests.parent / "cpp" / "native"
        with tempfile.TemporaryDirectory(prefix="ltsgemm-statistics-") as directory:
            binary = Path(directory) / "test_statistics"
            build = subprocess.run(
                [
                    *compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-pedantic",
                    "-I",
                    str(native),
                    str(tests / "test_statistics.cpp"),
                    str(native / "support.cpp"),
                    "-o",
                    str(binary),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            process = subprocess.run(
                [str(binary)], capture_output=True, text=True, timeout=10, check=False
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
