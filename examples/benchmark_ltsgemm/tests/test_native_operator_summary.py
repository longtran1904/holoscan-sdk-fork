# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_native_operators.py"
SPEC = importlib.util.spec_from_file_location("summarize_native_operators", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def write_run(folder, number, count, process_seconds, completed=True, indexed=True,
              operator_order=None, gemm_medians=None, gemm_p99s=None):
    stem = f"run-{number:02d}-operators-{count}" if indexed else f"run-operators-{count}"
    indices = operator_order if operator_order is not None else range(1, count + 1)
    cases = "".join(
        f"Operator {index}/{count}\n"
        "GEMM (M=65536, N=128, K=128, L2 flush=on): "
        f"500.000 ± 1.000 us, min=490.000 us, median={gemm_medians[index - 1] if gemm_medians else 499.000:.3f} us, "
        f"P99={gemm_p99s[index - 1] if gemm_p99s else 550.000:.3f} us, CV=0.200%, 4.000 TFLOP/s\n"
        "  Timed GEMM total: 500.000 ms, wall time: 2000.000 ms, GEMM/wall: 25.000%\n"
        for index in indices
    )
    record = dict(
        schema_version=2,
        gemms_per_tick=1,
        operator_count=count,
        completed_cases=count if completed else count - 1,
        timed_gemms=count * 1000,
        compute_calls=count * 1000,
        app_run_wall_ms=process_seconds * 900,
        completed=completed,
        return_code=0 if completed else 1,
    )
    (folder / f"{stem}.stdout.log").write_text(
        cases + "LT_SGEMM_NATIVE " + json.dumps(record) + "\n", encoding="utf-8"
    )
    (folder / f"{stem}.stderr.log").write_bytes(b"")
    (folder / f"{stem}.time.txt").write_text(
        f"process_wall_seconds={process_seconds}\nmax_rss_kb=123456\nexit_status=0\n",
        encoding="utf-8",
    )


class NativeOperatorSummaryTests(unittest.TestCase):
    def test_accepts_operator_completion_order(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_run(folder, 1, 4, 4.0, operator_order=(3, 1, 4, 2))

            output = summary.summarize_folder(folder)
            with output.open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(int(row["completed_cases"]), 4)
            self.assertEqual(int(row["timed_gemms"]), 4000)

    def test_median_of_operator_gemm_medians_with_even_count(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_run(folder, 1, 4, 4.0, gemm_medians=(100.0, 200.0, 400.0, 900.0))

            output = summary.summarize_folder(folder)
            with output.open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))

            self.assertEqual(float(row["median_operator_median_us"]), 300.0)

    def test_preserves_printed_operator_p99s_in_operator_order(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_run(folder, 1, 4, 4.0, operator_order=(3, 1, 4, 2),
                      gemm_p99s=(510.0, 520.0, 540.0, 570.0))

            output = summary.summarize_folder(folder)
            with output.open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))

            self.assertEqual(json.loads(row["operator_p99_us"]), [510.0, 520.0, 540.0, 570.0])
            self.assertEqual(float(row["median_operator_p99_us"]), 530.0)

    def test_rejects_duplicate_operator_result(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_run(folder, 1, 4, 4.0, operator_order=(3, 1, 3, 2))

            with self.assertRaisesRegex(ValueError, "operator results"):
                summary.summarize_folder(folder)
            self.assertFalse((folder / "comparison.csv").exists())

    def test_writes_comparison_with_ten_operator_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_run(folder, 1, 10, 10.0)
            write_run(folder, 2, 2, 3.0)

            output = summary.summarize_folder(folder)
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))

            self.assertEqual([row["operators"] for row in rows], ["10", "2"])
            self.assertEqual(float(rows[1]["process_factor_vs_10"]), 0.3)
            self.assertEqual(float(rows[1]["median_operator_mean_us"]), 500.0)
            self.assertEqual(float(rows[1]["sum_gpu_event_ms"]), 1000.0)
            self.assertEqual(int(rows[1]["timed_gemms"]), 2000)

    def test_missing_baseline_leaves_factor_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_run(folder, 1, 2, 3.0)
            output = summary.summarize_folder(folder)
            with output.open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["process_factor_vs_10"], "")

    def test_accepts_current_runner_log_names(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_run(folder, 1, 20, 20.0, indexed=False)
            write_run(folder, 2, 10, 10.0, indexed=False)

            output = summary.summarize_folder(folder)
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))

            self.assertEqual([row["operators"] for row in rows], ["10", "20"])
            self.assertEqual(float(rows[1]["process_factor_vs_10"]), 2.0)

    def test_incomplete_run_does_not_write_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_run(folder, 1, 2, 3.0, completed=False)
            with self.assertRaises(ValueError):
                summary.summarize_folder(folder)
            self.assertFalse((folder / "comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
