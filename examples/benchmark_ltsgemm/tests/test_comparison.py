# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Use stdlib unittest so these tests do not require pytest.
# ruff: noqa: PT009, PT027

import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_comparison.py"
SPEC = importlib.util.spec_from_file_location("run_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)
CASE = (Path(__file__).parent / "fixtures" / "case.txt").read_text(encoding="utf-8")


def sweep():
    return "L2 cache size: 33554432 bytes (32.00 MiB)\n" + "".join(
        CASE.replace("M=128,", f"M={m},").replace("flush=off", f"flush={flush}")
        for flush in ("off", "on")
        for m in (128 << i for i in range(10))
    )


def wrapper(runner="direct", **changes):
    result = dict(
        schema_version=1,
        runner=runner,
        completed=True,
        invocations=1,
        return_code=0,
        entrypoint_wall_ms=5000.0,
    )
    if runner == "holoscan":
        result["app_run_wall_ms"] = 5100.0
    result.update(changes)
    return "LT_SGEMM_WRAPPER " + json.dumps(result) + "\n"


def native(count=1, **changes):
    result = dict(
        schema_version=1,
        gemms_per_tick=count,
        completed_cases=20,
        timed_gemms=20000,
        compute_calls=20000 // count,
        app_run_wall_ms=5200.0,
        completed=True,
        return_code=0,
    )
    result.update(changes)
    return "LT_SGEMM_NATIVE " + json.dumps(result) + "\n"


class ParserTests(unittest.TestCase):
    def test_native_modes(self):
        for count in (1, 1000):
            cases, timing = comparison.parse_output(sweep() + native(count), f"native-{count}")
            self.assertEqual(len(cases), 20)
            self.assertEqual(timing["compute_calls"], 20000 // count)
            self.assertEqual(timing["timed_gemms"], 20000)
            self.assertNotIn("entrypoint_wall_ms", timing)

    def test_native_completion_is_strict(self):
        for count in (1, 1000):
            invalid = ["", native(count) * 2, wrapper("holoscan"), "LT_SGEMM_NATIVE bad\n"]
            for field, values in {
                "schema_version": (True, 2),
                "completed_cases": (19, 21, True, 20.0),
                "timed_gemms": (19999, 20001, True, 20000.0),
                "compute_calls": (0, 20000 // count + 1, True, 20000 / count),
                "gemms_per_tick": (2, True, 1000 if count == 1 else 1),
                "completed": (False, 1),
                "return_code": (1, False),
                "app_run_wall_ms": (0, -1, math.nan, math.inf, True),
                "entrypoint_wall_ms": (100,),
            }.items():
                invalid.extend(native(count, **{field: value}) for value in values)
            for record in invalid:
                with self.subTest(count=count, record=record), self.assertRaises(ValueError):
                    comparison.parse_output(sweep() + record, f"native-{count}")

    def test_native_rejects_missing_and_malformed_cases(self):
        for count in (1, 1000):
            for text in (
                sweep().replace(CASE, "", 1),
                sweep() + CASE,
                sweep().replace("P99=12.000", "P99=bad", 1),
            ):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    comparison.parse_output(text + native(count), f"native-{count}")

    def test_completion_record_types_cannot_be_mixed(self):
        for runner, text in (
            ("original", native()),
            ("direct", wrapper() + native()),
            ("holoscan", wrapper("holoscan") + native()),
            ("native-1", native() + wrapper()),
        ):
            with self.subTest(runner=runner), self.assertRaises(ValueError):
                comparison.parse_output(sweep() + text, runner)

    def test_complete_original_sweep(self):
        cases, timing = comparison.parse_output(sweep(), "original")
        self.assertEqual(len(cases), 20)
        self.assertEqual(cases[0]["mean_us"], 10)
        self.assertEqual(cases[0]["case_wall_ms"], 100)
        self.assertEqual(cases[-1]["m"], 65536)
        self.assertEqual(cases[-1]["flush"], "on")
        self.assertEqual(timing, {})

    def test_wrapper_modes(self):
        for mode in ("direct", "holoscan"):
            cases, timing = comparison.parse_output(sweep() + wrapper(mode), mode)
            self.assertEqual(len(cases), 20)
            self.assertEqual(timing["entrypoint_wall_ms"], 5000)

    def test_missing_duplicate_or_wrong_shape(self):
        for text in (
            sweep().replace(CASE, "", 1),
            sweep() + CASE,
            sweep().replace("M=128,", "M=129,", 1),
        ):
            with self.subTest(text=text[:80]), self.assertRaises(ValueError):
                comparison.parse_output(text, "original")

    def test_incomplete_and_malformed_lines(self):
        for text in (
            sweep().rsplit("  Timed GEMM total:", 1)[0],
            sweep().replace("P99=12.000 us", "P99=bad us", 1),
            sweep().replace("  Timed GEMM total: 10.000 ms", "  Timed GEMM total: bad ms", 1),
        ):
            with self.subTest(text=text[-60:]), self.assertRaises(ValueError):
                comparison.parse_output(text, "original")

    def test_nonfinite_negative_and_zero_timings(self):
        for value in ("nan", "inf", "-1.000", "0.000"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                comparison.parse_output(
                    sweep().replace(": 10.000 ±", f": {value} ±", 1), "original"
                )

    def test_zero_variance_is_valid(self):
        cases, _ = comparison.parse_output(sweep().replace("± 1.000", "± 0.000"), "original")
        self.assertEqual(cases[0]["stddev_us"], 0)

    def test_completion_required_and_strict(self):
        invalid = [
            "",
            wrapper() + wrapper(),
            wrapper(completed=False),
            wrapper(invocations=0),
            wrapper(invocations=2),
            wrapper(return_code=1),
            wrapper(runner="holoscan"),
            wrapper(entrypoint_wall_ms=math.nan),
            wrapper(invocations=True),
            wrapper(schema_version=2),
            "LT_SGEMM_WRAPPER {}\n",
        ]
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                comparison.parse_output(sweep() + record, "direct")

    def test_original_rejects_wrapper_output(self):
        with self.assertRaises(ValueError):
            comparison.parse_output(sweep() + wrapper(), "original")


class ReportTests(unittest.TestCase):
    def runs(self):
        result = []
        for mode, scale in (("original", 1), ("direct", 2), ("holoscan", 0.5)):
            cases, _ = comparison.parse_output(sweep(), "original")
            for case in cases:
                for metric in comparison.CASE_METRICS:
                    case[metric] *= scale
            result.append(
                dict(
                    trial=1,
                    runner=mode,
                    process_wall_ms=100 * scale,
                    entrypoint_wall_ms=90 * scale,
                    cases=cases,
                )
            )
        return result

    def test_ratios_and_speedups_keep_boundaries(self):
        ratios = comparison.paired_ratios(self.runs())
        row = next(
            r
            for r in ratios
            if r["comparison"] == "C/A"
            and r["metric"] == "mean_us"
            and r["m"] == 65536
            and r["flush"] == "off"
        )
        self.assertEqual(row["factor"], 0.5)
        self.assertEqual(row["slowdown_percent"], -50)
        self.assertEqual(row["delta"], -5)
        self.assertEqual({r["comparison"] for r in ratios if r["scope"] == "entrypoint"}, {"C/B"})
        self.assertEqual(len([r for r in ratios if r["scope"] == "process"]), 3)

    def test_no_pairing_partial_trials(self):
        with self.assertRaises(ValueError):
            comparison.paired_ratios(self.runs()[:2])

    def test_zero_denominator_rejected(self):
        runs = self.runs()
        runs[0]["process_wall_ms"] = 0
        with self.assertRaises(ValueError):
            comparison.paired_ratios(runs)

    def test_summary_groups_cache_modes_and_reports_quartiles(self):
        ratios = comparison.paired_ratios(self.runs())
        summaries = comparison.summarize(ratios)
        means = [r for r in summaries if r["comparison"] == "C/A" and r["metric"] == "mean_us"]
        self.assertEqual(len(means), 20)
        self.assertTrue(all(r["median_factor"] == 0.5 and r["trials"] == 1 for r in means))
        self.assertEqual(comparison.percentile([1, 2, 3, 4], 0.25), 1.75)

    def test_orders_rotate(self):
        self.assertEqual(
            [comparison.run_order(i) for i in range(1, 5)],
            [
                ("original", "direct", "holoscan"),
                ("direct", "holoscan", "original"),
                ("holoscan", "original", "direct"),
                ("original", "direct", "holoscan"),
            ],
        )

    def native_runs(self):
        runs = self.runs()
        runs[2]["app_run_wall_ms"] = 50
        for mode, scale in (("native-1000", 2), ("native-1", 4)):
            cases, timing = comparison.parse_output(
                sweep() + native(comparison.NATIVE_MODES[mode]), mode
            )
            for case in cases:
                for metric in comparison.CASE_METRICS:
                    case[metric] *= scale
            runs.append(
                dict(
                    trial=1,
                    runner=mode,
                    process_wall_ms=100 * scale,
                    app_run_wall_ms=100 * scale,
                    cases=cases,
                    native_record=timing,
                )
            )
        return runs

    def test_five_modes_compare_only_matching_boundaries(self):
        ratios = comparison.paired_ratios(self.native_runs())
        expected = {
            "C/A": 0.5,
            "B/A": 2,
            "C/B": 0.25,
            "D/A": 2,
            "E/A": 4,
            "D/C": 4,
            "E/C": 8,
            "E/D": 2,
        }
        for label, factor in expected.items():
            rows = [r for r in ratios if r["comparison"] == label and r["scope"] != "entrypoint"]
            self.assertEqual(len(rows), 122 if label in ("D/C", "E/C", "E/D") else 121)
            self.assertTrue(all(row["factor"] == factor for row in rows))
        self.assertEqual(
            {r["comparison"] for r in ratios if r["scope"] == "application"}, {"D/C", "E/C", "E/D"}
        )
        self.assertEqual({r["comparison"] for r in ratios if r["scope"] == "entrypoint"}, {"C/B"})

    def test_partial_native_trials_and_cases_rejected(self):
        for mode in comparison.run_order(1, native=True):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                comparison.paired_ratios([r for r in self.native_runs() if r["runner"] != mode])
        for mode_index in range(5):
            runs = self.native_runs()
            runs[mode_index]["cases"].pop()
            with self.subTest(mode_index=mode_index), self.assertRaises(ValueError):
                comparison.paired_ratios(runs)
        runs = self.native_runs()
        runs += [dict(r, trial=2) for r in self.runs()]
        with self.assertRaises(ValueError):
            comparison.paired_ratios(runs)

    def test_five_orders_rotate_every_position(self):
        first = comparison.run_order(1, native=True)
        self.assertEqual(first, ("original", "direct", "holoscan", "native-1000", "native-1"))
        for trial in range(1, 11):
            offset = (trial - 1) % 5
            self.assertEqual(
                comparison.run_order(trial, native=True), first[offset:] + first[:offset]
            )


class ProcessTests(unittest.TestCase):
    def test_native_campaign_and_failure_records(self):
        for failure in (
            None,
            "exit",
            "timeout",
            "missing",
            "malformed",
            "ticks",
            "cases",
            "failed",
        ):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "results"
                order = []

                def fake_execute(command, logs, stem, _timeout, failure=failure, order=order):
                    option = command[-1]
                    if option.startswith("--gemms-per-tick="):
                        count = int(option.split("=")[1])
                        mode = f"native-{count}"
                        record = native(count)
                    elif option.startswith("--runner="):
                        mode = option.split("=")[1]
                        record = wrapper(mode)
                    else:
                        mode, record = "original", ""
                    order.append(mode)
                    text = sweep()
                    fail = mode == "native-1" and failure is not None
                    if fail:
                        if failure == "missing":
                            record = ""
                        elif failure == "malformed":
                            record = "LT_SGEMM_NATIVE [bad]\n"
                        elif failure == "ticks":
                            record = native(1, compute_calls=20)
                        elif failure == "cases":
                            text = text.replace(CASE, "", 1)
                        elif failure == "failed":
                            record = native(1, completed=False, return_code=1)
                    stdout, stderr = logs / f"{stem}.stdout.log", logs / f"{stem}.stderr.log"
                    stdout.write_text(text + record)
                    stderr.write_text("diagnostic\n" if fail else "")
                    return dict(
                        command=command,
                        return_code=1 if fail and failure == "exit" else 0,
                        timed_out=fail and failure == "timeout",
                        process_wall_ms=6000,
                        stdout_path=str(stdout),
                        stderr_path=str(stderr),
                    )

                with (
                    patch.object(comparison, "execute", side_effect=fake_execute),
                    patch.object(comparison, "capture_metadata", return_value={}) as metadata,
                    patch.object(comparison, "gpu_snapshot", return_value={}),
                ):
                    kwargs = dict(
                        trials=5 if failure is None else 1,
                        timeout=5,
                        native_binary=Path(sys.executable),
                    )
                    if failure is None:
                        comparison.run_experiment(
                            Path(sys.executable), Path(sys.executable), output, **kwargs
                        )
                    else:
                        with self.assertRaises(ValueError):
                            comparison.run_experiment(
                                Path(sys.executable), Path(sys.executable), output, **kwargs
                            )
                    self.assertEqual(metadata.call_args.args[4], Path(sys.executable).resolve())
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "failed" if failure else "complete")
                if failure:
                    self.assertFalse((output / "summary.csv").exists())
                    self.assertTrue(Path(manifest["runs"][-1]["stdout_path"]).is_file())
                    continue
                self.assertEqual(len(manifest["runs"]), 25)
                self.assertEqual(
                    order,
                    [mode for i in range(1, 6) for mode in comparison.run_order(i, native=True)],
                )
                self.assertEqual(sum(len(r["cases"]) for r in manifest["runs"]), 500)
                with (output / "processes.csv").open() as stream:
                    native_rows = [
                        r for r in csv.DictReader(stream) if r["runner"].startswith("native")
                    ]
                self.assertEqual(len(native_rows), 10)
                for row in native_rows:
                    self.assertEqual(row["entrypoint_wall_ms"], "")
                    self.assertEqual(row["timed_gemms"], "20000")
                    self.assertEqual(int(row["compute_calls"]), 20000 // int(row["gemms_per_tick"]))
                for run in manifest["runs"]:
                    if run["runner"] in comparison.NATIVE_MODES:
                        self.assertIn("native_record", run)
                        self.assertNotIn("wrapper_record", run)

    def test_native_build_directory_requires_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            with self.assertRaisesRegex(ValueError, "requires --native"):
                comparison.run_experiment(
                    Path(sys.executable),
                    Path(sys.executable),
                    output,
                    native_build_dir=Path(directory),
                )
            self.assertFalse(output.exists())

    def test_real_subprocess_logs_survive_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = comparison.execute(
                [sys.executable, "-c", "print('failure'); raise SystemExit(3)"],
                Path(directory),
                "bad",
                5,
            )
            self.assertEqual(result["return_code"], 3)
            self.assertEqual(Path(result["stdout_path"]).read_text(), "failure\n")
            self.assertGreater(result["process_wall_ms"], 0)

    def test_timeout_keeps_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            result = comparison.execute(
                [sys.executable, "-c", "import time; print('partial', flush=True); time.sleep(10)"],
                Path(directory),
                "timeout",
                0.2,
            )
            self.assertTrue(result["timed_out"])
            self.assertIn("partial", Path(result["stdout_path"]).read_text())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileExistsError):
            comparison.run_experiment(
                Path(sys.executable), Path(sys.executable), Path(directory), trials=1, timeout=5
            )

    def test_sequential_fake_campaign(self):
        order = []

        def fake_execute(command, directory, stem, _timeout):
            mode = (
                command[-1].split("=", 1)[1] if command[-1].startswith("--runner=") else "original"
            )
            order.append(mode)
            stdout = directory / f"{stem}.stdout.log"
            stderr = directory / f"{stem}.stderr.log"
            stdout.write_text(
                sweep() + (wrapper(mode) if mode != "original" else ""), encoding="utf-8"
            )
            stderr.write_text("")
            return dict(
                command=command,
                return_code=0,
                timed_out=False,
                process_wall_ms=6000,
                stdout_path=str(stdout),
                stderr_path=str(stderr),
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(comparison, "execute", side_effect=fake_execute),
            patch.object(comparison, "capture_metadata", return_value={}),
            patch.object(comparison, "gpu_snapshot", return_value={}),
        ):
            output = Path(directory) / "results"
            comparison.run_experiment(
                Path(sys.executable), Path(sys.executable), output, trials=3, timeout=5
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["runs"]), 9)
            with (output / "processes.csv").open() as stream:
                self.assertEqual(
                    csv.DictReader(stream).fieldnames,
                    ["trial", "runner", "process_wall_ms", "entrypoint_wall_ms", "app_run_wall_ms"],
                )
            self.assertNotIn("C/D/E", (output / "summary.md").read_text())
            self.assertEqual(
                order, [mode for trial in range(1, 4) for mode in comparison.run_order(trial)]
            )
            self.assertTrue((output / "paired_ratios.csv").is_file())

    def test_failed_campaign_retains_logs_and_never_summarizes(self):
        for failure in ("exit", "timeout", "missing_completion", "malformed"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "results"

                def fake_execute(command, logs, stem, _timeout, failure=failure):
                    stdout, stderr = logs / f"{stem}.stdout.log", logs / f"{stem}.stderr.log"
                    stdout.write_text("bad output" if failure == "malformed" else sweep())
                    stderr.write_text("diagnostic\n")
                    # Let original finish, then fail the direct wrapper as selected.
                    direct = command[-1] == "--runner=direct"
                    return dict(
                        return_code=3 if direct and failure == "exit" else 0,
                        timed_out=direct and failure == "timeout",
                        process_wall_ms=6000,
                        stdout_path=str(stdout),
                        stderr_path=str(stderr),
                    )

                with (
                    patch.object(comparison, "execute", side_effect=fake_execute),
                    patch.object(comparison, "capture_metadata", return_value={}),
                    patch.object(comparison, "gpu_snapshot", return_value={}),
                    self.assertRaises(ValueError),
                ):
                    comparison.run_experiment(
                        Path(sys.executable), Path(sys.executable), output, trials=1, timeout=5
                    )
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "failed")
                self.assertTrue(manifest["error"])
                self.assertTrue(Path(manifest["runs"][-1]["stderr_path"]).is_file())
                self.assertFalse((output / "summary.csv").exists())
                self.assertFalse((output / "summary.md").exists())

    def test_invalid_options_do_not_create_output(self):
        for trials, timeout in ((0, 5), (-1, 5), (1, 0), (1, math.nan), (1, math.inf)):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "results"
                with self.assertRaises(ValueError):
                    comparison.run_experiment(
                        Path(sys.executable), Path(sys.executable), output, trials, timeout
                    )
                self.assertFalse(output.exists())

    def test_protected_output_path_is_rejected(self):
        output = comparison.REPO / "examples/gpu_kernels/src/LtSgemm/results"
        with self.assertRaisesRegex(ValueError, "protected"):
            comparison.run_experiment(Path(sys.executable), Path(sys.executable), output)

    def test_launch_error_keeps_log_files(self):
        with tempfile.TemporaryDirectory() as directory:
            result = comparison.execute(
                [str(Path(directory) / "nonexistent")], Path(directory), "missing", 5
            )
            self.assertIn("launch_error", result)
            self.assertIsNone(result["return_code"])
            self.assertTrue(Path(result["stdout_path"]).exists())


if __name__ == "__main__":
    unittest.main()
