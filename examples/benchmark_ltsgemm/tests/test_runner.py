# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Use stdlib unittest so these tests do not require pytest.
# ruff: noqa: PT009, PT027

"""Executable contract tests; GPU sweeps require LTSGEMM_GPU_TESTS=1."""

import json
import math
import os
import re
import subprocess
import unittest
from pathlib import Path

PREFIX = "LT_SGEMM_WRAPPER "
CASE = re.compile(r"^GEMM \(M=(\d+), N=(\d+), K=(\d+), L2 flush=(off|on)\):", re.M)
EXPECTED_CASES = [
    (str(128 << power), "128", "128", flush) for flush in ("off", "on") for power in range(10)
]


class RunnerAssertions:
    def run_binary(self, binary, *args, hide_gpu=False):
        environment = os.environ.copy()
        if hide_gpu:
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
        return subprocess.run(
            [str(binary), *args],
            capture_output=True,
            text=True,
            env=environment,
            timeout=1800 if not hide_gpu else 60,
            check=False,
        )

    def assert_summary(self, process, runner, completed, invocations):
        lines = process.stdout.splitlines()
        records = [line[len(PREFIX) :] for line in lines if line.startswith(PREFIX)]
        self.assertEqual(len(records), 1, process.stdout + process.stderr)
        self.assertTrue(lines[-1].startswith(PREFIX), process.stdout)
        record = json.loads(records[0])
        keys = {
            "schema_version",
            "runner",
            "completed",
            "invocations",
            "return_code",
            "entrypoint_wall_ms",
        }
        if runner == "holoscan":
            keys.add("app_run_wall_ms")
        self.assertEqual(set(record), keys)
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["runner"], runner)
        self.assertIs(record["completed"], completed)
        self.assertIs(type(record["invocations"]), int)
        self.assertEqual(record["invocations"], invocations)
        self.assertIs(type(record["return_code"]), int)
        self.assertEqual(record["return_code"], process.returncode)
        self.assertEqual(process.returncode == 0, completed)
        for name in ("entrypoint_wall_ms", "app_run_wall_ms"):
            if name in record:
                self.assertIn(type(record[name]), (int, float))
                self.assertTrue(math.isfinite(record[name]))
                self.assertGreaterEqual(record[name], 0)
        if completed and runner == "holoscan":
            self.assertGreaterEqual(record["app_run_wall_ms"], record["entrypoint_wall_ms"])
        return record


@unittest.skipUnless(os.environ.get("LTSGEMM_WRAPPER_BINARY"), "set LTSGEMM_WRAPPER_BINARY")
class RunnerCliTests(RunnerAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ["LTSGEMM_WRAPPER_BINARY"]).resolve(strict=True)

    def test_help_without_gpu(self):
        process = self.run_binary(self.binary, "--help", hide_gpu=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--runner=direct|holoscan", process.stdout)
        self.assertNotIn("GEMM (", process.stdout)
        self.assertNotIn(PREFIX, process.stdout)
        self.assertNotIn("cuda API failed", process.stderr)

    def test_invalid_arguments_do_not_invoke_entrypoint(self):
        for args in (
            ("--runner=invalid",),
            ("--runner=",),
            ("--unknown",),
            ("--runner=direct", "--runner=holoscan"),
        ):
            with self.subTest(args=args):
                process = self.run_binary(self.binary, *args, hide_gpu=True)
                runner = "direct" if args[0] == "--runner=direct" else "holoscan"
                record = self.assert_summary(process, runner, False, 0)
                self.assertEqual(record["entrypoint_wall_ms"], 0)
                self.assertNotIn("GEMM (", process.stdout)
                self.assertNotIn("cuda API failed", process.stderr)

    def test_cuda_failure_is_unsuccessful(self):
        for runner in ("direct", "holoscan"):
            with self.subTest(runner=runner):
                process = self.run_binary(self.binary, f"--runner={runner}", hide_gpu=True)
                self.assert_summary(process, runner, False, 1)
                self.assertNotIn("GEMM (", process.stdout)
                self.assertIn("cuda API failed", process.stderr)

    def test_default_runner_is_holoscan(self):
        process = self.run_binary(self.binary, hide_gpu=True)
        self.assert_summary(process, "holoscan", False, 1)


@unittest.skipUnless(os.environ.get("LTSGEMM_GPU_TESTS") == "1", "set LTSGEMM_GPU_TESTS=1")
class RunnerGpuTests(RunnerAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for variable in ("LTSGEMM_WRAPPER_BINARY", "LTSGEMM_ORIGINAL_BINARY"):
            if not os.environ.get(variable):
                raise unittest.SkipTest(f"set {variable}")
        cls.wrapper = Path(os.environ["LTSGEMM_WRAPPER_BINARY"]).resolve(strict=True)
        cls.original = Path(os.environ["LTSGEMM_ORIGINAL_BINARY"]).resolve(strict=True)

    def test_all_three_paths_complete_the_original_sweep(self):
        for runner in ("original", "direct", "holoscan"):
            with self.subTest(runner=runner):
                if runner == "original":
                    process = self.run_binary(self.original)
                    self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                    self.assertNotIn(PREFIX, process.stdout)
                else:
                    process = self.run_binary(self.wrapper, f"--runner={runner}")
                    self.assert_summary(process, runner, True, 1)
                self.assertEqual(CASE.findall(process.stdout), EXPECTED_CASES, process.stdout)
                self.assertEqual(process.stdout.count("  Timed GEMM total:"), 20)


@unittest.skipUnless(os.environ.get("LTSGEMM_NATIVE_BINARY"), "set LTSGEMM_NATIVE_BINARY")
class NativeRunnerTests(RunnerAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ["LTSGEMM_NATIVE_BINARY"]).resolve(strict=True)

    def assert_native_summary(self, process, count, completed):
        prefix = "LT_SGEMM_NATIVE "
        records = [
            line[len(prefix) :] for line in process.stdout.splitlines() if line.startswith(prefix)
        ]
        self.assertEqual(len(records), 1, process.stdout + process.stderr)
        self.assertTrue(process.stdout.splitlines()[-1].startswith(prefix))
        record = json.loads(records[0])
        self.assertEqual(
            set(record),
            {
                "schema_version",
                "gemms_per_tick",
                "completed_cases",
                "timed_gemms",
                "compute_calls",
                "app_run_wall_ms",
                "completed",
                "return_code",
            },
        )
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["gemms_per_tick"], count)
        self.assertIs(record["completed"], completed)
        self.assertEqual(record["return_code"], process.returncode)
        self.assertEqual(process.returncode == 0, completed)
        self.assertTrue(math.isfinite(record["app_run_wall_ms"]))
        self.assertGreaterEqual(record["app_run_wall_ms"], 0)
        for field in ("completed_cases", "timed_gemms", "compute_calls"):
            self.assertIs(type(record[field]), int)
        if completed:
            self.assertEqual(record["completed_cases"], 20)
            self.assertEqual(record["timed_gemms"], 20000)
            self.assertEqual(record["compute_calls"], 20000 // count)
        return record

    def test_help_without_gpu(self):
        for option in (
            (),
            ("--gemms-per-tick=1000",),
            ("--help",),
            ("--help", "--gemms-per-tick=1"),
        ):
            process = self.run_binary(self.binary, *option, "--help", hide_gpu=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("--gemms-per-tick=1|1000", process.stdout)
            self.assertNotIn("LT_SGEMM_NATIVE", process.stdout)
            self.assertNotIn("cuda API failed", process.stdout + process.stderr)

    def test_invalid_arguments_do_not_run(self):
        for args, count, invalid in (
            (("--gemms-per-tick=0",), 1, "--gemms-per-tick=0"),
            (("--gemms-per-tick=2",), 1, "--gemms-per-tick=2"),
            (("--gemms-per-tick=",), 1, "--gemms-per-tick="),
            (("--gemms-per-tick=1x",), 1, "--gemms-per-tick=1x"),
            (("--unknown",), 1, "--unknown"),
            (("--help", "--unknown"), 1, "--unknown"),
            (("--gemms-per-tick=1", "--gemms-per-tick=1000"), 1, "--gemms-per-tick=1000"),
            (("--gemms-per-tick=1000", "--gemms-per-tick=1"), 1000, "--gemms-per-tick=1"),
            (("--gemms-per-tick=1000", "--gemms-per-tick=1000"), 1000, "--gemms-per-tick=1000"),
            (("--gemms-per-tick=1", "--gemms-per-tick=1"), 1, "--gemms-per-tick=1"),
            (("--gemms-per-tick=1000", "--help", "--unknown"), 1000, "--unknown"),
            (("--unknown", "--gemms-per-tick=1000"), 1, "--unknown"),
            (("--gemms-per-tick", "1000"), 1, "--gemms-per-tick"),
            (("",), 1, ""),
        ):
            with self.subTest(args=args):
                process = self.run_binary(self.binary, *args, hide_gpu=True)
                record = self.assert_native_summary(process, count, False)
                self.assertEqual(record["compute_calls"], 0)
                self.assertEqual(record["completed_cases"], 0)
                self.assertEqual(record["timed_gemms"], 0)
                self.assertEqual(record["app_run_wall_ms"], 0)
                self.assertEqual(process.returncode, 2)
                self.assertEqual(process.stderr, f"Invalid argument: {invalid}\n")
                self.assertNotIn("cuda API failed", process.stdout + process.stderr)

    def test_cuda_failure_and_default_mode(self):
        for args, count in (
            ((), 1),
            (("--gemms-per-tick=1",), 1),
            (("--gemms-per-tick=1000",), 1000),
        ):
            with self.subTest(args=args):
                process = self.run_binary(self.binary, *args, hide_gpu=True)
                record = self.assert_native_summary(process, count, False)
                self.assertEqual(record["completed_cases"], 0)
                self.assertEqual(record["timed_gemms"], 0)
                self.assertEqual(record["compute_calls"], 1)
                self.assertNotIn("GEMM (", process.stdout)

    @unittest.skipUnless(os.environ.get("LTSGEMM_GPU_TESTS") == "1", "set LTSGEMM_GPU_TESTS=1")
    def test_both_native_modes_complete_the_sweep(self):
        for count in (1000, 1):
            with self.subTest(count=count):
                process = self.run_binary(self.binary, f"--gemms-per-tick={count}")
                self.assert_native_summary(process, count, True)
                self.assertEqual(CASE.findall(process.stdout), EXPECTED_CASES)
                self.assertEqual(process.stdout.count("  Timed GEMM total:"), 20)


if __name__ == "__main__":
    unittest.main()
