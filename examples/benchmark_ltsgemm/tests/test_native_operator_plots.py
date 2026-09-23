# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_native_operator_latencies.py"
RUNNER = SCRIPT.with_name("run_native_operators.sh")


class NativeOperatorPlotTests(unittest.TestCase):
    def test_runner_checks_plot_python_before_gpu_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            runner = scripts / RUNNER.name
            runner.write_text(RUNNER.read_text(encoding="utf-8"), encoding="utf-8")
            python = root / "benchmark-env" / "bin" / "python3"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            python.chmod(0o755)
            result = subprocess.run(
                ["bash", str(runner)], env={"LTSGEMM_NATIVE_BINARY": "/bin/true"},
                capture_output=True, text=True
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Matplotlib", result.stderr)
        self.assertNotIn("Running --operators", result.stdout)

    def test_creates_three_pngs_from_csv_without_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with (folder / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=("operators", "median_operator_median_us",
                                        "median_case_wall_ms", "operator_p99_us")
                )
                writer.writeheader()
                writer.writerows((
                    {"operators": 4, "median_operator_median_us": 350, "median_case_wall_ms": 2200,
                     "operator_p99_us": "[390,410,420,470]"},
                    {"operators": 1, "median_operator_median_us": 120, "median_case_wall_ms": 900,
                     "operator_p99_us": "[150]"},
                ))

            subprocess.run([sys.executable, str(SCRIPT), str(folder)], check=True, capture_output=True)

            for filename in ("median_per_operator_gemm_latency.png",
                             "median_per_operator_latency.png", "p99_per_operator_gemm_latency.png"):
                data = (folder / filename).read_bytes()
                self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertGreater(len(data), 1000)


if __name__ == "__main__":
    unittest.main()
