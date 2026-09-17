#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile synthetic lifecycle tests and exercise sequential runs with the real GXF binary."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

from run import BUILD_ID, DEFAULT_SOURCE, HERE, ROOT, Experiment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--container", default="scheduler-graphs-rerun")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-cuda13")
    parser.add_argument("--test-output", type=Path, help="New directory for test artifacts")
    args = parser.parse_args()
    args.source = DEFAULT_SOURCE
    out = args.output.resolve()
    experiment = Experiment(args, out)
    tests = args.test_output.resolve() if args.test_output else out / "lifecycle-tests"
    tests.mkdir(exist_ok=False)
    cp = experiment.cp
    common = [
        "/usr/bin/c++",
        "-std=c++17",
        "-pthread",
        "-I/opt/nvidia/gxf/include",
        "-I" + cp(HERE),
    ]
    with (tests / "build.log").open("w") as log:
        (tests / "bad_identity.cpp").write_text(
            'extern "C" int bad(void*) '
            'asm("_ZN6nvidia3gxf19EventBasedScheduler12deinitializeEv");\n'
            'extern "C" int bad(void*) { return 0; }\n'
        )
        commands = [
            common
            + [
                "-fPIC",
                "-shared",
                cp(HERE / "tests/fixture.cpp"),
                "-Wl,--build-id=0x" + BUILD_ID,
                "-o",
                cp(tests / "libfixture.so"),
            ],
            common
            + [
                cp(HERE / "tests/fixture_main.cpp"),
                "-L" + cp(tests),
                "-lfixture",
                "-Wl,-rpath," + cp(tests),
                "-o",
                cp(tests / "fixture"),
            ],
            common
            + [
                "-shared",
                "-fPIC",
                cp(out / "preload.o"),
                "-L" + cp(tests),
                "-lfixture",
                "-Wl,-rpath," + cp(tests),
                "-ldl",
                "-o",
                cp(tests / "libtest_trace.so"),
            ],
            common
            + [
                "-shared",
                "-fPIC",
                cp(tests / "bad_identity.cpp"),
                "-Wl,--build-id=0x0000",
                "-o",
                cp(tests / "libbad_identity.so"),
            ],
        ]
        for command in commands:
            experiment.docker(command, stdout=log, stderr=subprocess.STDOUT)
        # A real Holoscan compute failure exercises GXF's internal error cleanup.
        source = (out / "source-snapshot/benchmark_progress.cpp").read_text()
        if source.count("    count_++;") != 1:
            raise ValueError("Cannot inject compute failure in isolated test source")
        (tests / "real_failure.cpp").write_text(
            source.replace(
                "    count_++;",
                "    count_++;\n    if (count_ == 100) "
                'throw std::runtime_error("trace test failure");',
            )
        )
        build_commands = json.loads((out / "build-metadata.json").read_text())["commands"]
        compile_command, link_command = (list(c) for c in build_commands[2:4])
        compile_command[compile_command.index("-c") + 1] = cp(tests / "real_failure.cpp")
        compile_command[compile_command.index("-o") + 1] = cp(tests / "real_failure.o")
        link_command = [
            cp(tests / "real_failure.o") if a == cp(out / "progress.o") else a for a in link_command
        ]
        link_command[link_command.index("-o") + 1] = cp(tests / "real_failure")
        for command in [compile_command, link_command]:
            experiment.docker(command, stdout=log, stderr=subprocess.STDOUT)
    checks = {}
    cases = {
        "normal": (["complete"], {}),
        "sequential": (["complete", "complete"], {}),
        "run_failure": (["run_failed"], {"FIXTURE_RUN_FAILURE": "1"}),
        "wait_failure": (["deinitialized_without_wait"], {"FIXTURE_WAIT_FAILURE": "1"}),
        "destroy": (["destroyed_without_wait"], {}),
        "deactivate": (["deactivated_without_wait"], {}),
        "multiple": (["rejected"], {"FIXTURE_MULTIPLE": "1"}),
        "concurrent": (["rejected", "unsupported_concurrent_graph"], {}),
        "invalid_interval": (["rejected"], {"GXF_EBS_TRACE_INTERVAL_MS": "0"}),
        "wrong_build_id": (
            ["rejected"],
            {"LD_PRELOAD": cp(tests / "libtest_trace.so") + ":" + cp(tests / "libbad_identity.so")},
        ),
        "truncated": (["complete"], {"GXF_EBS_TRACE_MAX_SAMPLES": "3"}),
        "disabled": ([], {"GXF_EBS_TRACE_DIR": ""}),
        "abort": (["recording"], {}),
    }
    for mode, (expected, extra) in cases.items():
        directory = tests / mode
        directory.mkdir()
        env = {
            "LD_PRELOAD": cp(tests / "libtest_trace.so"),
            "GXF_EBS_TRACE_DIR": cp(directory),
            "GXF_EBS_TRACE_INTERVAL_MS": "2",
            **extra,
        }
        # Process interruption leaves the recording marker, never a fabricated final sample.
        command = [cp(tests / "fixture"), mode]
        if mode == "abort":
            command = ["timeout", "--signal=KILL", "0.2", *command]
        with (directory / "run.log").open("w") as log:
            try:
                experiment.docker(
                    command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError as error:
                if mode != "abort" or error.returncode not in [124, 137]:
                    raise
        traces = [json.loads(p.read_text()) for p in directory.glob("ebs-*.json")]
        actual = sorted(t["status"] for t in traces)
        if actual != sorted(expected):
            raise ValueError(f"{mode}: {actual} != {expected}")
        if mode == "truncated":
            assert traces[0]["truncated"]
            assert traces[0]["sample_count"] == 3
            rows = list(csv.DictReader(next(directory.glob("ebs-*.csv")).open()))
            assert rows[-1]["phase"] == "final"
            assert int(rows[-1]["steal_attempts"]) == 15
        checks[mode] = actual or ["no trace (disabled)"]

    # Public APIs and private counter layout against the actual GXF, two graphs in one process.
    directory = tests / "real-sequential"
    directory.mkdir()
    env = {
        **experiment.env,
        "LD_PRELOAD": cp(out / "libebs_trace.so"),
        "GXF_EBS_TRACE_DIR": cp(directory),
        "GXF_EBS_TRACE_INTERVAL_MS": "10",
        "EBS_TRACE_TEST_SEQUENTIAL": "1",
    }
    with (directory / "run.log").open("w") as log:
        experiment.docker(
            [
                "timeout",
                "--kill-after=5",
                "30",
                cp(out / "benchmark_progress"),
                "--busy_wait",
                "--enable_queue_stealing",
            ],
            cwd=directory,
            env=env,
            root=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    traces = sorted(directory.glob("ebs-*.json"))
    assert len(traces) == 2
    for path in traces:
        metadata = json.loads(path.read_text())
        assert metadata["status"] == "complete"
        assert metadata["final_stable"]
        rows = list(csv.DictReader(path.with_suffix(".csv").open()))
        assert rows[0]["steal_attempts"] == "0"
        assert int(rows[-1]["steal_attempts"]) > 0
    checks["real-sequential"] = ["two complete, independent counter traces"]
    directory = tests / "real-failure"
    directory.mkdir()
    env.pop("EBS_TRACE_TEST_SEQUENTIAL")
    env["GXF_EBS_TRACE_DIR"] = cp(directory)
    with (directory / "run.log").open("w") as log:
        try:
            experiment.docker(
                [
                    "timeout",
                    "--kill-after=5",
                    "30",
                    cp(tests / "real_failure"),
                    "--busy_wait",
                    "--enable_queue_stealing",
                ],
                cwd=directory,
                env=env,
                root=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as error:
            if error.returncode in [124, 137, 139]:
                raise
        else:
            raise ValueError("Injected compute failure unexpectedly succeeded")
    traces = [json.loads(p.read_text()) for p in directory.glob("ebs-*.json")]
    assert len(traces) == 1
    assert not traces[0]["final_stable"]
    assert traces[0]["status"] in ["deinitialized_without_wait", "wait_failed"]
    checks["real-failure"] = [traces[0]["status"]]
    write_json(tests / "verification.json", {"status": "passed", "checks": checks})
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
