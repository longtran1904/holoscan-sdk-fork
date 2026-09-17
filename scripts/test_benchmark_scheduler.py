# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Keep these tests runnable with the standard-library unittest runner.
# ruff: noqa: PT009, PT027

import contextlib
import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from benchmark_scheduler import benchmark_cli as cli
from benchmark_scheduler import benchmark_configurations as configurations
from benchmark_scheduler.analysis import operator_progress_plots as progress_plots
from benchmark_scheduler.analysis import plotting_backend as backend
from benchmark_scheduler.analysis import throughput_plots as plots
from benchmark_scheduler.data import benchmark_artifacts as artifacts
from benchmark_scheduler.data import benchmark_results as results
from benchmark_scheduler.data import operator_progress as progress
from benchmark_scheduler.execution import benchmark_runner as runner
from benchmark_scheduler.infrastructure import command_executor as commands
from benchmark_scheduler.infrastructure import docker_executor as docker
from benchmark_scheduler.infrastructure.local_executor import LocalExecutor
from benchmark_scheduler.preparation import benchmark_builder as builder


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


def checkpoint_fixture():
    return "".join(f"{op},{i * (op + 1)}\n" for op in range(16) for i in range(1, 11))


class ProgressParsingTests(unittest.TestCase):
    def test_checkpoints_and_equal_timestamps(self):
        self.assertEqual(
            progress.parse_progress("1,0\n0,2\n1,0\n0,3\n", 2, 20000),
            {0: [2, 3], 1: [0, 0]},
        )
        self.assertEqual(progress.parse_progress("", 16, 1000), {})

    def test_annotated_progress_and_pool_validation(self):
        metadata = {
            "threads": 10,
            "shared_threads": 2,
            "secondary_threads": 8,
            "checkpoint_operations": 100,
            "operator_pools": ["shared"] * 8 + ["secondary"] * 8,
        }
        data = "# " + json.dumps(metadata) + "\n" + checkpoint_fixture()
        timestamps, parsed = progress.parse_progress(data, 16, 1000, include_metadata=True)
        self.assertEqual(len(timestamps), 16)
        self.assertEqual(parsed, metadata)
        for invalid in (
            [],
            {**metadata, "threads": 0},
            {**metadata, "secondary_threads": 16},
            {**metadata, "checkpoint_operations": 10000},
            {**metadata, "operator_pools": ["secondary"] * 16},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                progress.parse_progress(
                    "# " + json.dumps(invalid) + "\n" + checkpoint_fixture(), 16, 1000
                )

    def test_shared_eight_assignment(self):
        for threads in configurations.EVENT_THREADS:
            shared = min(threads, 8)
            secondary = threads - shared
            pools = ["shared"] * (8 if secondary else 16) + ["secondary"] * (8 if secondary else 0)
            metadata = {
                "pool_assignment": "shared-eight",
                "threads": threads,
                "shared_threads": shared,
                "secondary_threads": secondary,
                "checkpoint_operations": 100,
                "operator_pools": pools,
            }
            with self.subTest(threads=threads):
                data = "# " + json.dumps(metadata) + "\n" + checkpoint_fixture()
                self.assertEqual(len(progress.parse_progress(data, 16, 1000)), 16)
                metadata["operator_pools"] = ["shared"] * 16 if secondary else ["secondary"] * 16
                with self.assertRaises(ValueError):
                    progress.parse_progress(
                        "# " + json.dumps(metadata) + "\n" + checkpoint_fixture(), 16, 1000
                    )

    def test_shared_twice_assignment(self):
        expected = {
            1: (1, 16),
            2: (2, 16),
            4: (4, 16),
            8: (8, 16),
            10: (6, 12),
            12: (4, 8),
            13: (3, 6),
            14: (2, 4),
            16: (0, 0),
        }
        for threads, (shared, shared_ops) in expected.items():
            metadata = {
                "pool_assignment": "shared-twice",
                "threads": threads,
                "shared_threads": shared,
                "secondary_threads": threads - shared,
                "checkpoint_operations": 100,
                "operator_pools": ["shared"] * shared_ops + ["secondary"] * (16 - shared_ops),
            }
            with self.subTest(threads=threads):
                data = "# " + json.dumps(metadata) + "\n" + checkpoint_fixture()
                self.assertEqual(len(progress.parse_progress(data, 16, 1000)), 16)
                metadata["shared_threads"] += 1
                with self.assertRaises(ValueError):
                    progress.parse_progress(
                        "# " + json.dumps(metadata) + "\n" + checkpoint_fixture(), 16, 1000
                    )

    def test_invalid_checkpoints(self):
        for data in (
            "",
            "0,1",
            "0,-1\n0,2",
            "0,2\n0,1",
            "1,1\n1,2",
            "0,1.5\n0,2",
            "0,1,2",
            "0,1\n0,2\n0,3",
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                progress.parse_progress(data, 1, 20000)
        with self.assertRaises(ValueError):
            progress.parse_progress("0,1", 16, 1000)


class ParsingTests(unittest.TestCase):
    def test_all_configurations(self):
        rows = [
            row
            for config in configurations.CONFIGURATIONS
            for row in results.parse_results(table(config), config)
        ]
        self.assertEqual(len(rows), 86)
        self.assertEqual(rows[0]["operators"], 16)
        self.assertEqual(rows[-1]["total_operations"], 16000)
        self.assertEqual(rows[-1]["threads"], 1)
        self.assertEqual(rows[-1]["queue_stealing"], "")
        self.assertEqual(rows[0]["busy_wait"], 0)

    def test_workload_flags_and_counts(self):
        self.assertEqual(len(configurations.CONFIGURATIONS), 10)
        self.assertEqual(
            [c.busy_wait for c in configurations.CONFIGURATIONS],
            [False] * 4 + [True] * 4 + [False, True],
        )
        self.assertEqual(
            [c.scheduler for c in configurations.CONFIGURATIONS],
            ["event_based"] * 8 + ["greedy"] * 2,
        )
        for config in configurations.CONFIGURATIONS:
            with self.subTest(config=config):
                self.assertEqual("--busy_wait" in config.flags, config.busy_wait)
                rows = results.parse_results(table(config), config)
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
                    results.parse_results(wrong, config)

    def test_invalid_numeric_and_incomplete_tables(self):
        config = configurations.CONFIGURATIONS[0]
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
                results.parse_results(invalid, config)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.build = Path(self.temp.name)
        (self.build / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={cli.REPO_ROOT}\n"
            "HOLOSCAN_BUILD_EXAMPLES:BOOL=ON\nHOLOSCAN_CPP_EXAMPLES:BOOL=ON\n"
        )
        self.output = self.build / "results.csv"
        build_patcher = patch.object(cli, "build_benchmarks")
        self.build_mock = build_patcher.start()
        self.addCleanup(build_patcher.stop)
        plot_patcher = patch.object(cli, "save_progress_graphs")
        self.progress_plot_mock = plot_patcher.start()
        self.addCleanup(plot_patcher.stop)
        self.write_fake_executables(cli.REPO_ROOT)

    def write_fake_executables(self, expected_cwd):
        for target in dict.fromkeys(c.target for c in configurations.CONFIGURATIONS):
            executable = self.build / "examples" / target / "cpp" / target
            executable.parent.mkdir(parents=True, exist_ok=True)
            # Actual child processes exercise output streaming, flags, and environment.
            modes = {
                tuple(c.flags): table(c)
                for c in configurations.CONFIGURATIONS
                if c.target == target
            }
            executable.write_text(
                f"#!{sys.executable}\nimport os, sys\nfrom pathlib import Path\n"
                + (
                    f"assert Path.cwd().parent == Path({str(self.build)!r})\n"
                    "assert not list(Path.cwd().glob('progress_log_*'))\n"
                    "busy = '--busy_wait' in sys.argv\n"
                    "ops = 1000 if busy else 100000\n"
                    "data = '' if busy else ''.join(f'{op},{i * 2}\\n' "
                    "for op in range(16) for i in range(1, 11))\n"
                    "if '--enable_queue_stealing' not in sys.argv "
                    "or '--enable_postcheck_fastpath' not in sys.argv:\n"
                    "    data = 'unselected configuration'\n"
                    f"for threads in {configurations.EVENT_THREADS!r}:\n"
                    "    Path(f'progress_log_{ops}ops_{threads}threads_16operators.txt')"
                    ".write_text(data)\n"
                    if target == configurations.EVENT_TARGET
                    else f"assert os.getcwd() == {str(expected_cwd)!r}\n"
                )
                + f"assert os.environ['LD_LIBRARY_PATH'].startswith({str(self.build / 'lib')!r})\n"
                f"assert os.environ['HOLOSCAN_LIB_PATH'].startswith({str(self.build / 'lib')!r})\n"
                f"print({modes!r}[tuple(sys.argv[1:])])\n"
            )
            executable.chmod(0o755)

    def run_main(self):
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            return cli.main(["--build-dir", str(self.build), "--output", str(self.output)])

    def csv_rows(self):
        with self.output.open() as stream:
            return list(csv.DictReader(stream))

    def test_success_and_no_overwrite(self):
        self.assertEqual(self.run_main(), 0, self.stderr.getvalue())
        rows = self.csv_rows()
        self.assertEqual(len(rows), 86)
        self.assertEqual(list(rows[0]), list(results.FIELDS))
        self.assertEqual(rows[0]["busy_wait"], "0")
        self.assertEqual(rows[-1]["queue_stealing"], "")
        png_paths = artifacts.graph_paths(self.output)
        self.assertEqual(len(png_paths), 7)
        self.assertEqual(set(self.build.glob("*.png")), set(png_paths))
        for path in png_paths:
            self.assertGreater(path.stat().st_size, 1000)
            self.assertIn(f"PNG: {path}", self.stdout.getvalue())
        log_dir, _ = artifacts.progress_directories(self.output)
        self.assertEqual(len(list(log_dir.glob("*.txt"))), 18)
        self.assertEqual(len([p for p in log_dir.glob("*.txt") if p.stat().st_size == 0]), 9)
        self.assertFalse(list(self.build.glob("scheduler-progress-*")))
        self.assertEqual(self.run_main(), 1)
        self.assertEqual(self.build_mock.call_count, 1)

    def test_progress_output_collisions(self):
        for path in artifacts.progress_directories(self.output):
            path.mkdir()
            self.assertEqual(self.run_main(), 1)
            self.assertIn("Output already exists", self.stderr.getvalue())
            self.build_mock.assert_not_called()
            path.rmdir()

    def test_missing_or_malformed_progress_preserves_other_logs_and_rows(self):
        original = runner.run_benchmark
        for malformed in (False, True):

            def run(build, config, env, container, cwd, malformed=malformed):
                rows = original(build, config, env, container, cwd)
                if configurations.selected_progress(config):
                    path = artifacts.progress_files(cwd, config)[0]
                    if malformed:
                        path.write_text("0,invalid")
                    else:
                        path.unlink()
                return rows

            with patch.object(runner, "run_benchmark", side_effect=run):
                self.assertEqual(self.run_main(), 1)
            self.assertEqual(len(self.csv_rows()), 36)
            log_dir, _ = artifacts.progress_directories(self.output)
            self.assertEqual(len(list(log_dir.iterdir())), 9 if malformed else 8)
            self.assertIn(str(log_dir), self.stderr.getvalue())
            shutil.rmtree(log_dir)
            self.output.unlink()

    def test_failed_final_configuration_preserves_its_available_logs(self):
        original = runner.run_benchmark

        def run(build, config, env, container, cwd):
            rows = original(build, config, env, container, cwd)
            if configurations.selected_progress(config) and config.busy_wait:
                artifacts.progress_files(cwd, config)[-1].unlink()
                raise subprocess.CalledProcessError(7, [config.target])
            return rows

        with patch.object(runner, "run_benchmark", side_effect=run):
            self.assertEqual(self.run_main(), 1)
        self.assertEqual(len(self.csv_rows()), 63)
        log_dir, _ = artifacts.progress_directories(self.output)
        self.assertEqual(len(list(log_dir.iterdir())), 17)
        self.assertFalse(list(self.build.glob("scheduler-progress-*")))
        self.progress_plot_mock.assert_not_called()

    def test_progress_plot_failure_preserves_logs(self):
        self.progress_plot_mock.side_effect = RuntimeError("progress plot failed")
        self.assertEqual(self.run_main(), 1)
        self.assertEqual(len(self.csv_rows()), 86)
        self.assertEqual(len(list(artifacts.progress_directories(self.output)[0].iterdir())), 18)
        self.assertIn("Complete CSV preserved", self.stderr.getvalue())

    def test_run_failure_preserves_flushed_rows(self):
        executable = (
            self.build
            / "examples"
            / configurations.GREEDY_TARGET
            / "cpp"
            / configurations.GREEDY_TARGET
        )
        executable.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(7)\n")
        self.assertEqual(self.run_main(), 1)
        self.assertEqual(len(self.csv_rows()), 72)
        self.assertIn(f"Partial CSV preserved: {self.output}", self.stderr.getvalue())
        self.assertFalse(self.output.with_suffix(".png").exists())

    def test_busy_wait_failure_preserves_partial_csv(self):
        run = runner.run_benchmark
        for failed in configurations.CONFIGURATIONS:
            if not failed.busy_wait:
                continue
            with self.subTest(failed=failed):
                seen = []

                def fail(build, config, env, container=None, cwd=None, failed=failed, seen=seen):
                    self.assertEqual(
                        len(self.csv_rows()),
                        sum(9 if c.scheduler == "event_based" else 7 for c in seen),
                    )
                    if config == failed:
                        raise subprocess.CalledProcessError(7, [config.target, *config.flags])
                    seen.append(config)
                    return run(build, config, env, container, cwd)

                with (
                    patch.object(runner, "run_benchmark", side_effect=fail),
                    patch.object(plots, "comparison_figure") as plot,
                ):
                    self.assertEqual(self.run_main(), 1)
                plot.assert_not_called()
                self.assertEqual(
                    seen,
                    list(
                        configurations.CONFIGURATIONS[: configurations.CONFIGURATIONS.index(failed)]
                    ),
                )
                self.assertIn("Partial CSV preserved:", self.stderr.getvalue())
                self.assertFalse(self.output.with_suffix(".png").exists())
                self.output.unlink()
                for directory in artifacts.progress_directories(self.output):
                    if directory.exists():
                        shutil.rmtree(directory)

    def test_default_output_is_unique_and_relative_build_uses_repo_root(self):
        with (
            patch.object(cli, "REPO_ROOT", self.build),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.write_fake_executables(self.build)
            (self.build / "CMakeCache.txt").write_text(
                f"CMAKE_HOME_DIRECTORY:INTERNAL={self.build}\n"
                "HOLOSCAN_BUILD_EXAMPLES:BOOL=ON\nHOLOSCAN_CPP_EXAMPLES:BOOL=ON\n"
            )
            for _ in range(2):
                self.assertEqual(cli.main(["--build-dir", "."]), 0)
        outputs = list((self.build / "benchmark-results").glob("*.csv"))
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(output.with_suffix(".png").exists() for output in outputs))
        self.assertEqual(self.build_mock.call_args.args[0], self.build)

    def test_rows_are_visible_before_next_configuration(self):
        run = runner.run_benchmark
        completed = 0

        def verify_flush(*args):
            nonlocal completed
            self.assertEqual(len(self.csv_rows()), completed)
            rows = run(*args)
            completed += len(rows)
            return rows

        with patch.object(runner, "run_benchmark", side_effect=verify_flush):
            self.assertEqual(self.run_main(), 0)

    def test_build_failure_prevents_runs(self):
        self.build_mock.side_effect = subprocess.CalledProcessError(1, ["cmake"])
        self.assertEqual(self.run_main(), 1)
        self.assertFalse(self.output.exists())
        self.assertNotIn("===", self.stdout.getvalue())

    def test_plot_failure_preserves_complete_csv(self):
        with patch.object(plots, "comparison_figure", side_effect=RuntimeError("plot failed")):
            self.assertEqual(self.run_main(), 1)
        self.assertEqual(len(self.csv_rows()), 86)
        self.assertIn("Complete CSV preserved:", self.stderr.getvalue())

    def test_missing_matplotlib_stops_before_build(self):
        with patch.object(cli, "load_pyplot", side_effect=RuntimeError("install matplotlib")):
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
            original.replace(str(cli.REPO_ROOT), "/unrelated"),
            original.replace("=ON", "=OFF"),
        ):
            cache.write_text(text)
            self.assertEqual(self.run_main(), 1)
            self.build_mock.assert_not_called()

    def test_container_cli_uses_container_environment_and_local_output(self):
        remote_root = Path("/workspace/holoscan-sdk")
        container = docker.DockerExecutor(
            "container-id",
            "manual-container",
            "sha256:image-id",
            remote_root,
            {"LD_LIBRARY_PATH": "/container/cuda/lib", "HOLOSCAN_LIB_PATH": "/container/sdk/lib"},
            self.build,
        )
        cache = self.build / "CMakeCache.txt"
        cache.write_text(cache.read_text().replace(str(cli.REPO_ROOT), str(remote_root)))
        seen = []

        def run(build, config, env, selected, cwd):
            if config.scheduler == "event_based":
                self.assertEqual(cwd.parent, self.build)
                for path in artifacts.progress_files(cwd, config):
                    path.write_text("" if config.busy_wait else checkpoint_fixture())
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
            return results.parse_results(table(config), config)

        with (
            patch.object(cli, "REPO_ROOT", self.build),
            patch.object(cli, "inspect_container", return_value=container) as inspect,
            patch.object(runner, "run_benchmark", side_effect=run),
            patch.object(cli, "save_graphs", return_value=[]) as save,
            patch.object(cli, "load_pyplot"),
            patch.dict(os.environ, {"LD_LIBRARY_PATH": "/host/wrong/lib"}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = cli.main(
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
        inspect.assert_called_once_with("manual-container", self.build)
        self.build_mock.assert_called_once_with(
            self.build, None, container, configurations.CONFIGURATIONS
        )
        self.assertEqual(seen, list(configurations.CONFIGURATIONS))
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
                        "Source": str(source or cli.REPO_ROOT),
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
        with patch.object(subprocess, "run") as run:
            run.return_value.stdout = json.dumps(self.inspection())
            container = docker.inspect_container("manual-container", cli.REPO_ROOT)
        run.assert_called_once_with(
            ["docker", "container", "inspect", "manual-container"],
            check=True,
            capture_output=True,
            text=True,
        )
        return container

    def test_existing_container_build_and_environment(self):
        container = self.container()
        build = cli.REPO_ROOT / "build-cuda13"
        with patch.object(subprocess, "Popen") as run:
            run.return_value.__enter__.return_value.wait.return_value = 0
            builder.build_benchmarks(build, 3, container, configurations.CONFIGURATIONS)
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
            self.assertNotIn(str(cli.REPO_ROOT), command)
            self.assertIn("/workspace/holoscan-sdk/build-cuda13", command)
        self.assertEqual(run.call_args.args[0][-2:], ["--parallel", "3"])
        self.assertEqual(container.environment["LD_LIBRARY_PATH"], "/container/cuda/lib")

    def test_benchmark_uses_container_paths_and_preserves_output(self):
        container = self.container()
        build = cli.REPO_ROOT / "build-cuda13"
        config = configurations.Configuration(
            "Postcheck fastpath", "event_based", postcheck_fastpath=True
        )
        env = {"HOLOSCAN_LIB_PATH": "/workspace/holoscan-sdk/build-cuda13/lib"}
        process = MagicMock()
        process.__enter__.return_value = process
        process.stdout = io.StringIO(table(config))
        process.wait.return_value = 0
        with (
            patch.object(subprocess, "Popen", return_value=process) as popen,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rows = runner.run_benchmark(build, config, env, container, build / "fresh-progress")
        command = popen.call_args.args[0]
        self.assertEqual(
            command,
            [
                "docker",
                "exec",
                "--interactive",
                "--workdir",
                "/workspace/holoscan-sdk/build-cuda13/fresh-progress",
                "--env",
                "HOLOSCAN_LIB_PATH=/workspace/holoscan-sdk/build-cuda13/lib",
                "container-id",
                "python3",
                "-c",
                commands.CONTAINER_SUPERVISOR,
                "/workspace/holoscan-sdk/build-cuda13/examples/benchmark_scheduler_throughput/cpp/benchmark_scheduler_throughput",
                "--enable_postcheck_fastpath",
            ],
        )
        self.assertIsNone(popen.call_args.kwargs["env"])
        self.assertEqual(popen.call_args.kwargs["stdin"], subprocess.PIPE)
        self.assertEqual(len(rows), 9)

    def test_supervisor_stops_child_on_disconnect_and_preserves_exit_status(self):
        command = [sys.executable, "-c", commands.CONTAINER_SUPERVISOR]
        child = "import os, time; print(os.getpid(), flush=True); time.sleep(30)"
        with subprocess.Popen(
            [*command, sys.executable, "-c", child],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            pid = int(process.stdout.readline())
            process.stdin.close()
            self.assertNotEqual(process.wait(timeout=5), 0)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        with subprocess.Popen(
            [*command, sys.executable, "-c", "raise SystemExit(7)"],
            stdin=subprocess.PIPE,
        ) as process:
            self.assertEqual(process.wait(timeout=5), 7)

    def test_rejects_stopped_or_unrelated_container(self):
        for data in (self.inspection(running=False), self.inspection(source="/other/checkout")):
            with (
                patch.object(subprocess, "run") as run,
                self.assertRaises(ValueError),
            ):
                run.return_value.stdout = json.dumps(data)
                docker.inspect_container("manual-container", cli.REPO_ROOT)
        with self.assertRaises(ValueError):
            self.container().path(Path("/unmounted/build"))


class ProgressPlotTests(unittest.TestCase):
    def test_artifacts_and_coordinates(self):
        plt = backend.load_pyplot()
        with tempfile.TemporaryDirectory() as directory:
            log_dir, plot_dir = artifacts.progress_directories(Path(directory) / "results.csv")
            log_dir.mkdir()
            for config in configurations.CONFIGURATIONS:
                if configurations.selected_progress(config):
                    for threads, path in zip(
                        configurations.EVENT_THREADS,
                        artifacts.progress_files(log_dir, config),
                        strict=True,
                    ):
                        shared = math.gcd(threads, 16)
                        secondary = threads - shared
                        metadata = {
                            "threads": threads,
                            "shared_threads": shared,
                            "secondary_threads": secondary,
                            "checkpoint_operations": 100 if config.busy_wait else 10000,
                            "operator_pools": ["shared"] * (16 - secondary)
                            + ["secondary"] * secondary,
                        }
                        path.write_text("# " + json.dumps(metadata) + "\n" + checkpoint_fixture())
            draw = progress_plots.draw_progress
            axes = []

            def record_axis(ax, timestamps, title, metadata=None):
                draw(ax, timestamps, title, metadata)
                axes.append(ax)

            with patch.object(progress_plots, "draw_progress", side_effect=record_axis):
                cli.save_progress_graphs(plt, log_dir, plot_dir)
            overview_axes = axes[-9:]
            self.assertEqual(len({ax.get_xlim() for ax in overview_axes}), 1)
            self.assertEqual(len({ax.get_ylim() for ax in overview_axes}), 1)
            self.assertEqual(len(overview_axes[0].lines), 16)
            self.assertEqual(len(list(plot_dir.glob("*.png"))), 20)
            for name in ("overview.svg", "operator-progress.pdf", "README.md"):
                self.assertTrue((plot_dir / name).is_file())
            pdf = (plot_dir / "operator-progress.pdf").read_bytes()
            self.assertTrue(pdf.startswith(b"%PDF-"))
            self.assertEqual(len(re.findall(rb"/Type /Page\b", pdf)), 20)
            readme = (plot_dir / "README.md").read_text()
            self.assertIn("queue stealing enabled; postcheck fastpath enabled", readme)
            self.assertIn("secondary pool", readme)
            self.assertFalse(plt.get_fignums())
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        progress_plots.draw_progress(ax, {0: [5, 10], 10: [1, 1]}, "normal")
        self.assertEqual(list(ax.lines[0].get_xdata()), [0, 5, 10])
        self.assertEqual(list(ax.lines[0].get_ydata()), [0, 10000, 20000])
        self.assertEqual(ax.lines[0].get_color(), ax.lines[1].get_color())
        self.assertNotEqual(ax.lines[0].get_linestyle(), ax.lines[1].get_linestyle())


class PlotTests(unittest.TestCase):
    def test_baseline_matching_and_plot_coordinates(self):
        measurements = {
            configurations.configuration_key(config): results.parse_results(
                table(config, index + 1), config
            )
            for index, config in enumerate(configurations.CONFIGURATIONS)
        }
        normal = measurements[("event_based", False, False, False)]
        busy = measurements[("event_based", True, False, False)]
        stealing = measurements[("event_based", False, True, False)]
        stealing.reverse()
        self.assertEqual([ratio for _, ratio in results.match_baseline(stealing, normal)], [2] * 9)
        with self.assertRaises(ValueError):
            results.match_baseline(busy, normal)
        # Even equal operation budgets must not allow cross-workload matching.
        disguised = [dict(row, total_operations=1600000) for row in busy]
        with self.assertRaises(ValueError):
            results.match_baseline(disguised, normal)
        plt = backend.load_pyplot()
        # Configuration insertion order must not affect grouping or styling.
        fig = plots.comparison_figure(plt, dict(reversed(list(measurements.items()))))
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
            results.match_baseline(stealing, normal)

    def test_comparison_failure_closes_figure(self):
        plt = backend.load_pyplot()
        before = plt.get_fignums()
        with self.assertRaisesRegex(ValueError, "all ten"):
            plots.comparison_figure(plt, {})
        self.assertEqual(plt.get_fignums(), before)

    def test_cmake_incremental_commands_and_failure_order(self):
        # Bypass RunnerTests' build mock to verify the actual command construction.
        with patch.object(subprocess, "run") as run:
            builder.build_benchmarks(
                Path("/build"), None, LocalExecutor(cli.REPO_ROOT), configurations.CONFIGURATIONS
            )
            self.assertEqual(
                run.call_args_list,
                [
                    call(["cmake", "-S", str(cli.REPO_ROOT), "-B", "/build"], check=True),
                    call(
                        [
                            "cmake",
                            "--build",
                            "/build",
                            "--target",
                            configurations.EVENT_TARGET,
                            configurations.GREEDY_TARGET,
                        ],
                        check=True,
                    ),
                ],
            )
            run.reset_mock()
            builder.build_benchmarks(
                Path("/build"), 3, LocalExecutor(cli.REPO_ROOT), configurations.CONFIGURATIONS
            )
            self.assertEqual(run.call_args.args[0][-2:], ["--parallel", "3"])
            run.reset_mock()
            run.side_effect = subprocess.CalledProcessError(1, ["cmake"])
            with self.assertRaises(subprocess.CalledProcessError):
                builder.build_benchmarks(
                    Path("/build"),
                    None,
                    LocalExecutor(cli.REPO_ROOT),
                    configurations.CONFIGURATIONS,
                )
            self.assertEqual(run.call_count, 1)


class CommandTests(unittest.TestCase):
    setUp = RunnerTests.setUp
    write_fake_executables = RunnerTests.write_fake_executables
    run_main = RunnerTests.run_main
    csv_rows = RunnerTests.csv_rows

    def test_independent_build_and_run_without_matplotlib(self):
        with (
            patch.object(cli, "load_pyplot", side_effect=AssertionError("plot dependency")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cli.main(["build", "--build-dir", str(self.build)]), 0)
            self.assertFalse(self.output.exists())
            self.build_mock.reset_mock()
            self.assertEqual(
                cli.main(["run", "--build-dir", str(self.build), "--output", str(self.output)]), 0
            )
            self.build_mock.assert_not_called()
            self.assertEqual(len(self.csv_rows()), 86)
            self.assertFalse(self.output.with_suffix(".png").exists())

    def test_missing_binary_prevents_any_runs_or_csv(self):
        target = configurations.GREEDY_TARGET
        (self.build / "examples" / target / "cpp" / target).unlink()
        with patch.object(cli, "run_configurations") as run:
            self.assertEqual(self.run_main(), 1)
        run.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertIn("Required benchmark executable", self.stderr.getvalue())

    def test_real_run_without_site_packages(self):
        script = cli.REPO_ROOT / "scripts/benchmark_scheduler/benchmark_scheduler.py"
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(script),
                "run",
                "--build-dir",
                str(self.build),
                "--output",
                str(self.output),
            ],
            cwd=self.build,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.csv_rows()), 86)
        self.assertFalse(self.output.with_suffix(".png").exists())

    def test_explicit_all(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                cli.main(["all", "--build-dir", str(self.build), "--output", str(self.output)]), 0
            )
        self.assertEqual(len(self.csv_rows()), 86)


class SavedResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "results.csv"
        self.expected = {
            configurations.configuration_key(config): results.parse_results(table(config), config)
            for config in configurations.CONFIGURATIONS
        }
        self.rows = [row for rows in self.expected.values() for row in rows]
        self.write_rows(self.rows)

    def write_rows(self, rows):
        with self.source.open("w", newline="") as stream:
            writer = artifacts.result_writer(stream)
            artifacts.append_results(writer, stream, rows)

    def test_round_trip_and_invalid_csv(self):
        self.assertEqual(results.load_results(self.source), self.expected)
        for field, value in (
            ("throughput_hz", "nan"),
            ("throughput_hz", "inf"),
            ("throughput_hz", "0"),
            ("threads", "0"),
            ("trial", "-1"),
            ("total_operations", "123"),
            ("scheduler", "unknown"),
            ("busy_wait", "2"),
            ("queue_stealing", "2"),
        ):
            with self.subTest(field=field, value=value):
                self.write_rows([{**self.rows[0], field: value}, *self.rows[1:]])
                with self.assertRaises(ValueError):
                    results.load_results(self.source)
        for rows in ([], self.rows[:-1], [*self.rows, self.rows[0]]):
            self.write_rows(rows)
            with self.assertRaises(ValueError):
                results.load_results(self.source)
        for text in ("wrong,fields\n", self.source.read_text() + "event_based,0\n"):
            self.source.write_text(text)
            with self.assertRaises(ValueError):
                results.load_results(self.source)

    def plot(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return cli.main(["plot", "--input", str(self.source), *map(str, args)])

    def test_plot_without_build_or_docker_and_regeneration(self):
        with (
            patch.object(cli, "inspect_container", side_effect=AssertionError("Docker")),
            patch.object(cli, "read_build_cache", side_effect=AssertionError("CMake")),
        ):
            self.assertEqual(self.plot(), 0)
            self.assertEqual(len(list(self.directory.glob("*.png"))), 7)
            original = {path: path.read_bytes() for path in artifacts.graph_paths(self.source)}
            self.assertEqual(self.plot(), 1)
            self.assertEqual(original, {path: path.read_bytes() for path in original})
            destination = self.directory / "regenerated"
            self.assertEqual(self.plot("--output-dir", destination), 0)
            self.assertEqual(len(list(destination.glob("*.png"))), 7)
            self.assertEqual(results.load_results(self.source), self.expected)

    def test_explicit_missing_and_malformed_discovered_logs(self):
        self.assertEqual(self.plot("--progress-logs", self.directory / "missing"), 1)
        logs, plots_dir = artifacts.progress_directories(self.source)
        logs.mkdir()
        config = next(
            c for c in configurations.CONFIGURATIONS if configurations.selected_progress(c)
        )
        artifacts.progress_files(logs, config)[0].write_text("0,invalid\n")
        self.assertEqual(self.plot(), 1)
        self.assertFalse(plots_dir.exists())
        self.assertFalse(self.source.with_suffix(".png").exists())

    def test_destination_collisions_including_dangling_symlinks(self):
        for path in (
            *artifacts.graph_paths(self.source),
            artifacts.progress_directories(self.source)[1],
        ):
            path.symlink_to(self.directory / "absent")
            with patch.object(cli, "load_pyplot"):
                self.assertEqual(self.plot(), 1)
            self.assertTrue(path.is_symlink())
            path.unlink()

    def test_incomplete_comparison_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "all ten"):
            plots.draw_comparison([], dict(list(self.expected.items())[:-1]))


class EntryPointTests(unittest.TestCase):
    def test_script_and_packages_from_other_directories_without_optional_dependencies(self):
        script = cli.REPO_ROOT / "scripts/benchmark_scheduler/benchmark_scheduler.py"
        with tempfile.TemporaryDirectory() as directory:
            # Block optional imports in a fresh interpreter, even if installed locally.
            Path(directory, "sitecustomize.py").write_text(
                "import sys\n"
                "class Block:\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                "        if fullname.split('.')[0] in ('matplotlib', 'numpy'):\n"
                "            raise ImportError('blocked optional dependency')\n"
                "sys.meta_path.insert(0, Block())\n"
            )
            env = {
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    (directory, str(cli.REPO_ROOT), str(cli.REPO_ROOT / "scripts"))
                ),
            }
            entries = (
                [str(script)],
                ["-m", "scripts.benchmark_scheduler"],
                ["-m", "benchmark_scheduler"],
            )
            for entry in entries:
                for cwd in (cli.REPO_ROOT, directory):
                    for stage in ("build", "run", "plot", "all"):
                        result = subprocess.run(
                            [sys.executable, *entry, stage, "--help"],
                            cwd=cwd,
                            env=env,
                            capture_output=True,
                            text=True,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                    result = subprocess.run(
                        [sys.executable, *entry, "run", "--build-dir", "missing-benchmark-build"],
                        cwd=cwd,
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(str(cli.REPO_ROOT / "missing-benchmark-build"), result.stderr)
                    self.assertNotIn("Matplotlib", result.stderr)

    def test_process_is_reaped_on_interrupt(self):
        process = MagicMock()
        process.__enter__.return_value = process
        process.poll.return_value = None
        with (
            patch.object(subprocess, "Popen", return_value=process),
            self.assertRaises(KeyboardInterrupt),
            commands.command_process(["fixture"]),
        ):
            raise KeyboardInterrupt
        process.kill.assert_called_once()
        process.wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
