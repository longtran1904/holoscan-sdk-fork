# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Keep these tests runnable with the standard-library unittest runner.
# ruff: noqa: PT009, PT027

import contextlib
import csv
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import benchmark_scheduler as benchmark


def table(config, scale=1):
    event = config.scheduler == "event_based"
    header = (
        "| Trial | Threads | Operators | Total Ops | Ops/s |"
        if event
        else ("| Trial | Operators | Total Ops | Ops/s |")
    )
    lines = [
        "[info] starting",
        header,
        "|-------|---------|-----------|------------|--------------|",
    ]
    for trial, x in enumerate(
        (1, 2, 4, 8, 10, 12, 13, 14, 16) if event else (1, 2, 4, 8, 12, 14, 16)
    ):
        operators = 16 if event else x
        operations = operators * (1000 if config.busy_wait else (100000 if event else 10000000))
        coordinates = f"{x} | {operators}" if event else str(operators)
        lines.append(f"| {trial} | {coordinates} | {operations} | {scale * x * 100.0} |")
    return "\n".join([*lines, "[info] done"])


class ParsingTests(unittest.TestCase):
    def test_all_configurations(self):
        rows = [
            row
            for config in benchmark.CONFIGURATIONS
            for row in benchmark.parse_results(table(config), config)
        ]
        self.assertEqual(len(rows), 86)
        self.assertEqual(rows[0]["operators"], 16)
        self.assertEqual(rows[-1]["total_operations"], 16000)
        self.assertEqual(rows[-1]["threads"], 1)
        self.assertEqual(rows[-1]["queue_stealing"], "")
        self.assertEqual(rows[0]["busy_wait"], 0)

    def test_workload_flags_and_counts(self):
        self.assertEqual(len(benchmark.CONFIGURATIONS), 10)
        self.assertEqual(
            [c.busy_wait for c in benchmark.CONFIGURATIONS],
            [False] * 4 + [True] * 4 + [False, True],
        )
        self.assertEqual(
            [c.scheduler for c in benchmark.CONFIGURATIONS], ["event_based"] * 8 + ["greedy"] * 2
        )
        for config in benchmark.CONFIGURATIONS:
            with self.subTest(config=config):
                self.assertEqual("--busy_wait" in config.flags, config.busy_wait)
                rows = benchmark.parse_results(table(config), config)
                for row in rows:
                    self.assertEqual(row["busy_wait"], int(config.busy_wait))
                    self.assertEqual(
                        row["total_operations"],
                        row["operators"]
                        * (
                            1000
                            if config.busy_wait
                            else (100000 if config.scheduler == "event_based" else 10000000)
                        ),
                    )
                    self.assertEqual(
                        row["queue_stealing"],
                        int(config.queue_stealing) if config.scheduler == "event_based" else "",
                    )
                    self.assertEqual(
                        row["postcheck_fastpath"],
                        int(config.postcheck_fastpath) if config.scheduler == "event_based" else "",
                    )
                wrong = table(config).replace(str(rows[0]["total_operations"]), "123")
                with self.assertRaises(ValueError):
                    benchmark.parse_results(wrong, config)

    def test_invalid_numeric_and_incomplete_tables(self):
        config = benchmark.CONFIGURATIONS[0]
        valid = table(config)
        for invalid in (
            "",
            valid.replace("100.0", "nan"),
            valid.replace("100.0", "0"),
            valid.replace("1600000", "-1"),
            valid.replace("| 7 |", "| 6 |"),
            "\n".join(valid.splitlines()[:-2]),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                benchmark.parse_results(invalid, config)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.build = Path(self.temp.name)
        (self.build / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={benchmark.REPO_ROOT}\n"
            "HOLOSCAN_BUILD_EXAMPLES:BOOL=ON\nHOLOSCAN_CPP_EXAMPLES:BOOL=ON\n"
        )
        self.output = self.build / "results.csv"
        build_patcher = patch.object(benchmark, "build_benchmarks")
        self.build_mock = build_patcher.start()
        self.addCleanup(build_patcher.stop)
        self.write_fake_executables(benchmark.REPO_ROOT)

    def write_fake_executables(self, expected_cwd):
        for target in dict.fromkeys(c.target for c in benchmark.CONFIGURATIONS):
            executable = self.build / "examples" / target / "cpp" / target
            executable.parent.mkdir(parents=True, exist_ok=True)
            # Actual child processes exercise output streaming, flags, and environment.
            modes = {
                tuple(c.flags): table(c) for c in benchmark.CONFIGURATIONS if c.target == target
            }
            executable.write_text(
                f"#!{benchmark.sys.executable}\nimport os, sys\n"
                f"assert os.getcwd() == {str(expected_cwd)!r}\n"
                f"assert os.environ['LD_LIBRARY_PATH'].startswith({str(self.build / 'lib')!r})\n"
                f"assert os.environ['HOLOSCAN_LIB_PATH'].startswith({str(self.build / 'lib')!r})\n"
                f"print({modes!r}[tuple(sys.argv[1:])])\n"
            )
            executable.chmod(0o755)

    def run_main(self):
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            return benchmark.main(["--build-dir", str(self.build), "--output", str(self.output)])

    def csv_rows(self):
        with self.output.open() as stream:
            return list(csv.DictReader(stream))

    def test_success_and_no_overwrite(self):
        self.assertEqual(self.run_main(), 0, self.stderr.getvalue())
        rows = self.csv_rows()
        self.assertEqual(len(rows), 86)
        self.assertEqual(list(rows[0]), list(benchmark.FIELDS))
        self.assertEqual(rows[0]["busy_wait"], "0")
        self.assertEqual(rows[-1]["queue_stealing"], "")
        png_paths = benchmark.graph_paths(self.output)
        self.assertEqual(len(png_paths), 7)
        self.assertEqual(set(self.build.glob("*.png")), set(png_paths))
        for path in png_paths:
            self.assertGreater(path.stat().st_size, 1000)
            self.assertIn(f"PNG: {path}", self.stdout.getvalue())
        self.assertEqual(self.run_main(), 1)
        self.assertEqual(self.build_mock.call_count, 1)

    def test_run_failure_preserves_flushed_rows(self):
        executable = (
            self.build / "examples" / benchmark.GREEDY_TARGET / "cpp" / benchmark.GREEDY_TARGET
        )
        executable.write_text(f"#!{benchmark.sys.executable}\nimport sys\nsys.exit(7)\n")
        self.assertEqual(self.run_main(), 1)
        self.assertEqual(len(self.csv_rows()), 72)
        self.assertIn(f"Partial CSV preserved: {self.output}", self.stderr.getvalue())
        self.assertFalse(self.output.with_suffix(".png").exists())

    def test_busy_wait_failure_preserves_partial_csv(self):
        run = benchmark.run_benchmark
        for failed in benchmark.CONFIGURATIONS:
            if not failed.busy_wait:
                continue
            with self.subTest(failed=failed):
                seen = []

                def fail(build, config, env, container=None, failed=failed, seen=seen):
                    self.assertEqual(
                        len(self.csv_rows()),
                        sum(9 if c.scheduler == "event_based" else 7 for c in seen),
                    )
                    if config == failed:
                        raise subprocess.CalledProcessError(7, [config.target, *config.flags])
                    seen.append(config)
                    return run(build, config, env, container)

                with (
                    patch.object(benchmark, "run_benchmark", side_effect=fail),
                    patch.object(benchmark, "comparison_figure") as plot,
                ):
                    self.assertEqual(self.run_main(), 1)
                plot.assert_not_called()
                self.assertEqual(
                    seen, list(benchmark.CONFIGURATIONS[: benchmark.CONFIGURATIONS.index(failed)])
                )
                self.assertIn("Partial CSV preserved:", self.stderr.getvalue())
                self.assertFalse(self.output.with_suffix(".png").exists())
                self.output.unlink()

    def test_default_output_is_unique_and_relative_build_uses_repo_root(self):
        with (
            patch.object(benchmark, "REPO_ROOT", self.build),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.write_fake_executables(self.build)
            (self.build / "CMakeCache.txt").write_text(
                f"CMAKE_HOME_DIRECTORY:INTERNAL={self.build}\n"
                "HOLOSCAN_BUILD_EXAMPLES:BOOL=ON\nHOLOSCAN_CPP_EXAMPLES:BOOL=ON\n"
            )
            for _ in range(2):
                self.assertEqual(benchmark.main(["--build-dir", "."]), 0)
        outputs = list((self.build / "benchmark-results").glob("*.csv"))
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(output.with_suffix(".png").exists() for output in outputs))
        self.assertEqual(self.build_mock.call_args.args[0], self.build)

    def test_rows_are_visible_before_next_configuration(self):
        run = benchmark.run_benchmark
        completed = 0

        def verify_flush(*args):
            nonlocal completed
            self.assertEqual(len(self.csv_rows()), completed)
            rows = run(*args)
            completed += len(rows)
            return rows

        with patch.object(benchmark, "run_benchmark", side_effect=verify_flush):
            self.assertEqual(self.run_main(), 0)

    def test_build_failure_prevents_runs(self):
        self.build_mock.side_effect = subprocess.CalledProcessError(1, ["cmake"])
        self.assertEqual(self.run_main(), 1)
        self.assertFalse(self.output.exists())
        self.assertNotIn("===", self.stdout.getvalue())

    def test_plot_failure_preserves_complete_csv(self):
        with patch.object(benchmark, "comparison_figure", side_effect=RuntimeError("plot failed")):
            self.assertEqual(self.run_main(), 1)
        self.assertEqual(len(self.csv_rows()), 86)
        self.assertIn("Complete CSV preserved:", self.stderr.getvalue())

    def test_missing_matplotlib_stops_before_build(self):
        with patch.object(benchmark, "load_pyplot", side_effect=RuntimeError("install matplotlib")):
            self.assertEqual(self.run_main(), 1)
        self.build_mock.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_absolute_install_destination_does_not_override_build_libraries(self):
        with (self.build / "CMakeCache.txt").open("a") as stream:
            stream.write("CMAKE_INSTALL_LIBDIR:PATH=/external/install/lib\n")
        self.assertEqual(self.run_main(), 0)

    def test_wrong_checkout_and_disabled_examples(self):
        cache = self.build / "CMakeCache.txt"
        original = cache.read_text()
        for text in (
            original.replace(str(benchmark.REPO_ROOT), "/unrelated"),
            original.replace("=ON", "=OFF"),
        ):
            cache.write_text(text)
            self.assertEqual(self.run_main(), 1)
            self.build_mock.assert_not_called()

    def test_container_cli_uses_container_environment_and_local_output(self):
        remote_root = Path("/workspace/holoscan-sdk")
        container = benchmark.Container(
            "container-id",
            "manual-container",
            "sha256:image-id",
            remote_root,
            {"LD_LIBRARY_PATH": "/container/cuda/lib", "HOLOSCAN_LIB_PATH": "/container/sdk/lib"},
        )
        cache = self.build / "CMakeCache.txt"
        cache.write_text(cache.read_text().replace(str(benchmark.REPO_ROOT), str(remote_root)))
        seen = []

        def run(build, config, env, selected):
            self.assertEqual(build, self.build)
            self.assertIs(selected, container)
            libraries = "/workspace/holoscan-sdk/lib:/workspace/holoscan-sdk/lib64"
            self.assertEqual(
                env,
                {
                    "LD_LIBRARY_PATH": libraries + ":/container/cuda/lib",
                    "HOLOSCAN_LIB_PATH": libraries + ":/container/sdk/lib",
                },
            )
            seen.append(config)
            return benchmark.parse_results(table(config), config)

        with (
            patch.object(benchmark, "REPO_ROOT", self.build),
            patch.object(benchmark, "inspect_container", return_value=container) as inspect,
            patch.object(benchmark, "run_benchmark", side_effect=run),
            patch.object(benchmark, "save_graphs", return_value=[]) as save,
            patch.object(benchmark, "load_pyplot"),
            patch.dict(benchmark.os.environ, {"LD_LIBRARY_PATH": "/host/wrong/lib"}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = benchmark.main(
                [
                    "--container",
                    "manual-container",
                    "--build-dir",
                    ".",
                    "--output",
                    str(self.output),
                ]
            )
        self.assertEqual(status, 0)
        inspect.assert_called_once_with("manual-container")
        self.build_mock.assert_called_once_with(self.build, None, container)
        self.assertEqual(seen, list(benchmark.CONFIGURATIONS))
        self.assertEqual(len(self.csv_rows()), 86)
        self.assertEqual(save.call_args.args[-1], self.output)


class ContainerTests(unittest.TestCase):
    def inspection(self, *, running=True, source=None):
        return [
            {
                "Id": "container-id",
                "Image": "sha256:image-id",
                "Name": "/manual-container",
                "State": {"Running": running},
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": str(source or benchmark.REPO_ROOT),
                        "Destination": "/workspace/holoscan-sdk",
                    }
                ],
                "Config": {
                    "Env": [
                        "LD_LIBRARY_PATH=/container/cuda/lib",
                        "HOLOSCAN_LIB_PATH=/container/old-build/lib",
                    ]
                },
            }
        ]

    def container(self):
        with patch.object(benchmark.subprocess, "run") as run:
            run.return_value.stdout = json.dumps(self.inspection())
            container = benchmark.inspect_container("manual-container")
        run.assert_called_once_with(
            ["docker", "container", "inspect", "manual-container"],
            check=True,
            capture_output=True,
            text=True,
        )
        return container

    def test_existing_container_build_and_environment(self):
        container = self.container()
        build = benchmark.REPO_ROOT / "build-cuda13"
        with patch.object(benchmark.subprocess, "Popen") as run:
            run.return_value.__enter__.return_value.wait.return_value = 0
            benchmark.build_benchmarks(build, 3, container)
        self.assertEqual(run.call_count, 2)
        for invocation in run.call_args_list:
            command = invocation.args[0]
            self.assertEqual(
                command[:6],
                [
                    "docker",
                    "exec",
                    "--interactive",
                    "--workdir",
                    "/workspace/holoscan-sdk",
                    "container-id",
                ],
            )
            self.assertNotIn(str(benchmark.REPO_ROOT), command)
            self.assertIn("/workspace/holoscan-sdk/build-cuda13", command)
        self.assertEqual(run.call_args.args[0][-2:], ["--parallel", "3"])
        self.assertEqual(container.environment["LD_LIBRARY_PATH"], "/container/cuda/lib")

    def test_benchmark_uses_container_paths_and_preserves_output(self):
        container = self.container()
        build = benchmark.REPO_ROOT / "build-cuda13"
        config = benchmark.Configuration(
            "Postcheck fastpath", "event_based", postcheck_fastpath=True
        )
        env = {"HOLOSCAN_LIB_PATH": "/workspace/holoscan-sdk/build-cuda13/lib"}
        process = MagicMock()
        process.__enter__.return_value = process
        process.stdout = io.StringIO(table(config))
        process.wait.return_value = 0
        with (
            patch.object(benchmark.subprocess, "Popen", return_value=process) as popen,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rows = benchmark.run_benchmark(build, config, env, container)
        command = popen.call_args.args[0]
        self.assertEqual(
            command,
            [
                "docker",
                "exec",
                "--interactive",
                "--workdir",
                "/workspace/holoscan-sdk",
                "--env",
                "HOLOSCAN_LIB_PATH=/workspace/holoscan-sdk/build-cuda13/lib",
                "container-id",
                "python3",
                "-c",
                benchmark.CONTAINER_SUPERVISOR,
                "/workspace/holoscan-sdk/build-cuda13/examples/benchmark_scheduler_throughput/cpp/benchmark_scheduler_throughput",
                "--enable_postcheck_fastpath",
            ],
        )
        self.assertIsNone(popen.call_args.kwargs["env"])
        self.assertEqual(popen.call_args.kwargs["stdin"], subprocess.PIPE)
        self.assertEqual(len(rows), 9)

    def test_supervisor_stops_child_on_disconnect_and_preserves_exit_status(self):
        command = [benchmark.sys.executable, "-c", benchmark.CONTAINER_SUPERVISOR]
        child = "import os, time; print(os.getpid(), flush=True); time.sleep(30)"
        with subprocess.Popen(
            [*command, benchmark.sys.executable, "-c", child],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            pid = int(process.stdout.readline())
            process.stdin.close()
            self.assertNotEqual(process.wait(timeout=5), 0)
            with self.assertRaises(ProcessLookupError):
                benchmark.os.kill(pid, 0)
        with subprocess.Popen(
            [*command, benchmark.sys.executable, "-c", "raise SystemExit(7)"],
            stdin=subprocess.PIPE,
        ) as process:
            self.assertEqual(process.wait(timeout=5), 7)

    def test_rejects_stopped_or_unrelated_container(self):
        for data in (self.inspection(running=False), self.inspection(source="/other/checkout")):
            with (
                patch.object(benchmark.subprocess, "run") as run,
                self.assertRaises(ValueError),
            ):
                run.return_value.stdout = json.dumps(data)
                benchmark.inspect_container("manual-container")
        with self.assertRaises(ValueError):
            self.container().path(Path("/unmounted/build"))


class PlotTests(unittest.TestCase):
    def test_baseline_matching_and_plot_coordinates(self):
        results = {
            benchmark.configuration_key(config): benchmark.parse_results(
                table(config, index + 1), config
            )
            for index, config in enumerate(benchmark.CONFIGURATIONS)
        }
        normal = results[("event_based", False, False, False)]
        busy = results[("event_based", True, False, False)]
        stealing = results[("event_based", False, True, False)]
        stealing.reverse()
        self.assertEqual(
            [ratio for _, ratio in benchmark.match_baseline(stealing, normal)], [2] * 9
        )
        with self.assertRaises(ValueError):
            benchmark.match_baseline(busy, normal)
        # Even equal operation budgets must not allow cross-workload matching.
        disguised = [dict(row, total_operations=1600000) for row in busy]
        with self.assertRaises(ValueError):
            benchmark.match_baseline(disguised, normal)
        plt = benchmark.load_pyplot()
        # Configuration insertion order must not affect grouping or styling.
        fig = benchmark.comparison_figure(plt, dict(reversed(list(results.items()))))
        self.addCleanup(plt.close, fig)
        self.assertEqual(len(fig.axes), 6)
        self.assertEqual([len(ax.lines) for ax in fig.axes], [4, 5, 5, 5, 1, 1])
        for ax in fig.axes[:4]:
            self.assertEqual(list(ax.lines[1].get_xdata()), [1, 2, 4, 8, 10, 12, 13, 14, 16])
        self.assertEqual(list(fig.axes[1].lines[1].get_ydata()), [2] * 9)
        self.assertEqual(list(fig.axes[3].lines[1].get_ydata()), [6 / 5] * 9)
        reference = fig.axes[2].lines[-1]
        self.assertIn("Greedy: 1 operator", reference.get_label())
        self.assertEqual(list(reference.get_ydata()), [1000] * 2)
        for index in range(4):
            self.assertEqual(
                fig.axes[0].lines[index].get_color(), fig.axes[2].lines[index].get_color()
            )
            self.assertEqual(fig.axes[0].lines[index].get_marker(), "o")
        for ax in fig.axes[4:]:
            self.assertEqual(list(ax.lines[0].get_xdata()), [1, 2, 4, 8, 12, 14, 16])
        for ax in (fig.axes[0], fig.axes[2], *fig.axes[4:]):
            self.assertEqual(ax.get_ylim()[0], 0)
            self.assertIn("operations/s", ax.get_ylabel())
            self.assertGreater(ax.get_ylim()[1], max(max(line.get_ydata()) for line in ax.lines))
        stealing[0]["operators"] = 8
        with self.assertRaises(ValueError):
            benchmark.match_baseline(stealing, normal)

    def test_cmake_incremental_commands_and_failure_order(self):
        # Bypass RunnerTests' build mock to verify the actual command construction.
        with patch.object(benchmark.subprocess, "run") as run:
            benchmark.build_benchmarks(Path("/build"), None)
            self.assertEqual(
                run.call_args_list,
                [
                    call(["cmake", "-S", str(benchmark.REPO_ROOT), "-B", "/build"], check=True),
                    call(
                        [
                            "cmake",
                            "--build",
                            "/build",
                            "--target",
                            benchmark.EVENT_TARGET,
                            benchmark.GREEDY_TARGET,
                        ],
                        check=True,
                    ),
                ],
            )
            run.reset_mock()
            benchmark.build_benchmarks(Path("/build"), 3)
            self.assertEqual(run.call_args.args[0][-2:], ["--parallel", "3"])
            run.reset_mock()
            run.side_effect = subprocess.CalledProcessError(1, ["cmake"])
            with self.assertRaises(subprocess.CalledProcessError):
                benchmark.build_benchmarks(Path("/build"), None)
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
