#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the diagnostic shim and run the isolated results-6 comparison in Docker."""

import argparse
import csv
import hashlib
import json
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / (
    "build-cuda13/benchmark-results-6-dispatcher-pinned-20260914T030711Z/"
    "source-snapshot/benchmark_scheduler_throughput.cpp"
)
SOURCE_SHA256 = "58776d062004a83b779c475cf96c63dcc0d7be4e637562170478e26831c6461c"
LIBRARY_SHA256 = "9da4ff855450ec6030460109e6b5774dfffc85e9a008a0f2b3d8d2ebcfc7016d"
BUILD_ID = "7fb77053fb8b71654d71098aba0a8be015c83344"
HEADER_SHA256 = "9c5f7f5925095d92ff34ee9ce11bd74382b7982be585858fc9964e82e5a54171"
PIN_ENV = {
    "GXF_EBS_DISPATCHER_CPU_CORE": "13",
    "GXF_EBS_DISPATCHER_SCHED_POLICY": "SCHED_FIFO",
    "GXF_EBS_DISPATCHER_SCHED_PRIORITY": "99",
}
COUNTERS = [
    "steal_attempts",
    "steal_successes",
    "worker_waitforjob_calls",
    "worker_waitforjob_total_us",
    "worker_execute_calls",
    "worker_execute_total_us",
    "worker_postcheck_fastpath_ready",
    "dispatcher_events_dispatched",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Snapshot transform expected one occurrence of {old!r}")
    return source.replace(old, new, 1)


def benchmark_source(source, progress=False):
    """Modify only an isolated snapshot; fail if the source shape has changed."""
    source, count = re.subn(
        r"  std::vector<BenchmarkSchedulerThroughputApp::Options> trial_options = \{\n"
        r"(?:      make_trial_options\(\d+, 16\),\n)+  \};",
        "  std::vector<BenchmarkSchedulerThroughputApp::Options> trial_options = {\n"
        "      make_trial_options(13, 16),\n  };\n"
        '  if (std::getenv("EBS_TRACE_TEST_SEQUENTIAL"))\n'
        "    trial_options.push_back(make_trial_options(13, 16));",
        source,
    )
    if count != 1:
        raise ValueError("Cannot isolate the 13-worker trial")
    source = replace_once(
        source, "#include <algorithm>", "#include <algorithm>\n#include <cstdlib>"
    )
    if not progress:
        return source
    source = replace_once(source, "#include <chrono>", "#include <chrono>\n#include <time.h>")
    source = replace_once(
        source,
        "  void start() override {",
        """  void set_progress_interval(int64_t interval, int64_t total) {
    progress_interval_ = interval;
    checkpoints_.reserve(total / interval);
  }

  void write_progress() {
    for (const auto& point : checkpoints_)
      (*progress_log) << index.get() << ',' << point.first << ',' << point.second << '\\n';
    progress_log->flush();
  }

  void start() override {""",
    )
    start = source.index("    // Write timestamp checkpoints to file for plotting.")
    end = source.index("\n  }", start)
    source = source[:start] + source[end:]
    source = replace_once(
        source,
        """    if (count_ % 10000 == 0) {
      timestamps_.push_back(make_pair(index.get(), std::chrono::steady_clock::now()));
    }""",
        """    if (count_ % progress_interval_ == 0) {
      timespec now{};
      clock_gettime(CLOCK_MONOTONIC, &now);
      checkpoints_.emplace_back(count_, uint64_t(now.tv_sec) * 1000000000 + now.tv_nsec);
    }""",
    )
    source = replace_once(
        source,
        "  int64_t count_ = 0;",
        """  int64_t count_ = 0;
  int64_t progress_interval_ = 10000;
  std::vector<std::pair<int64_t, uint64_t>> checkpoints_;""",
    )
    source = replace_once(
        source,
        "      count_ops_.back()->set_progress_log(progress_log);",
        """      count_ops_.back()->set_progress_log(progress_log);
      count_ops_.back()->set_progress_interval(options_.busy_wait ? 100 : 10000,
                                               options_.num_operations);""",
    )
    source = replace_once(
        source,
        "    for (int i = 0; i < options_.num_operators; i++) {",
        '    (*progress_log) << "operator_id,completed_operations,monotonic_ns\\n";\n'
        "    for (int i = 0; i < options_.num_operators; i++) {",
    )
    source = replace_once(
        source,
        "      total_operations += count_op->count();",
        "      count_op->write_progress();\n      total_operations += count_op->count();",
    )
    return source


class Experiment:
    def __init__(self, args, out):
        self.args, self.out = args, out
        self.build = args.build_dir.resolve()
        self.info = json.loads(subprocess.check_output(["docker", "inspect", args.container]))[0]
        mounts = [m for m in self.info["Mounts"] if Path(m["Source"]).resolve() == ROOT]
        if len(mounts) != 1:
            raise ValueError("Container must bind mount the repository root")
        self.mount = Path(mounts[0]["Destination"])
        container_env = dict(v.split("=", 1) for v in self.info["Config"]["Env"])
        self.env = dict(PIN_ENV)
        for name in ["LD_LIBRARY_PATH", "HOLOSCAN_LIB_PATH"]:
            self.env[name] = ":".join(
                [
                    self.cp(self.build / "lib"),
                    self.cp(self.build / "lib64"),
                    container_env.get(name, ""),
                ]
            )

    def cp(self, path):
        return str(self.mount / path.resolve().relative_to(ROOT))

    def docker(self, argv, cwd=None, env=None, root=False, **kwargs):
        command = ["docker", "exec"]
        if root:
            command += ["--user", "0"]
        command += ["--workdir", self.cp(cwd or self.build)]
        for key, value in (env or {}).items():
            command += ["--env", f"{key}={value}"]
        command += [self.args.container, *argv]
        return subprocess.run(command, check=True, **kwargs)

    def output(self, argv):
        return self.docker(argv, stdout=subprocess.PIPE, text=True).stdout.strip()

    def build_artifacts(self):
        # Enforce the known binary/header pair, and record the actual container inputs.
        library = "/opt/nvidia/gxf/lib/gxf/std/libgxf_std.so"
        header = "/opt/nvidia/gxf/include/gxf/std/event_based_scheduler.hpp"
        hashes = self.output(["sha256sum", library, header]).splitlines()
        library_hash, header_hash = (line.split()[0] for line in hashes)
        if (
            library_hash != LIBRARY_SHA256
            or digest(self.build / "lib/libgxf_std.so") != library_hash
        ):
            raise ValueError(
                "GXF binary is different from the validated build; revalidate the ABI first"
            )
        if header_hash != HEADER_SHA256:
            raise ValueError("GXF header differs from the validated header/binary pair")
        source = self.args.source.resolve()
        if digest(source) != SOURCE_SHA256:
            raise ValueError("Source does not match the original results-6 snapshot")
        snap = self.out / "source-snapshot"
        snap.mkdir()
        shutil.copy2(source, snap / "original.cpp")
        for name in [
            "counter_reader.hpp",
            "counter_reader.cpp",
            "preload.cpp",
            "run.py",
            "analyze.py",
        ]:
            shutil.copy2(HERE / name, snap / name)
        (snap / "trace_build.hpp").write_text(
            f'#define EBS_TRACE_BUILD_ID "{BUILD_ID}"\n'
            f'#define EBS_TRACE_LIBRARY_SHA256 "{library_hash}"\n'
            f'#define EBS_TRACE_HEADER_SHA256 "{header_hash}"\n'
        )
        entry = next(
            e
            for e in json.loads((self.build / "compile_commands.json").read_text())
            if e["file"].endswith("/benchmark_scheduler_throughput.cpp")
        )
        original_compile = shlex.split(entry["command"])
        target = "examples/benchmark_scheduler_throughput/cpp/benchmark_scheduler_throughput"
        link = self.output(
            ["ninja", "-C", self.cp(self.build), "-t", "commands", target]
        ).splitlines()[-1]
        original_link = shlex.split(link.removeprefix(": && ").removesuffix(" && :"))
        commands = []
        with (self.out / "build.log").open("w") as log:

            def compile_run(args):
                commands.append(args)
                self.docker(args, stdout=log, stderr=subprocess.STDOUT)

            for variant, progress in [("baseline", False), ("progress", True)]:
                source_path = snap / f"benchmark_{variant}.cpp"
                source_path.write_text(benchmark_source(source.read_text(), progress))
                obj = self.cp(self.out / f"{variant}.o")
                command = list(original_compile)
                command[command.index("-c") + 1] = self.cp(source_path)
                command[command.index("-o") + 1] = obj
                compile_run(command)
                command = [
                    obj if a == entry["output"] else a
                    for a in original_link
                    if not a.startswith("-Wl,--dependency-file=")
                ]
                command[command.index("-o") + 1] = self.cp(self.out / f"benchmark_{variant}")
                compile_run(command)
            # Retain SDK include/define/ABI flags; only the reader bypasses access control.
            base = list(original_compile)
            for flag in ["-c", "-o"]:
                index = base.index(flag)
                del base[index : index + 2]
            base = [a for a in base if a != "-fPIE"] + ["-fPIC", "-pthread", "-Wall", "-Wextra"]
            for name in ["counter_reader", "preload"]:
                compile_run(
                    base
                    + (["-fno-access-control"] if name == "counter_reader" else [])
                    + [
                        "-I" + self.cp(snap),
                        "-c",
                        self.cp(snap / f"{name}.cpp"),
                        "-o",
                        self.cp(self.out / f"{name}.o"),
                    ]
                )
            compile_run(
                [
                    "/usr/bin/c++",
                    "-shared",
                    "-pthread",
                    "-o",
                    self.cp(self.out / "libebs_trace.so"),
                    self.cp(self.out / "counter_reader.o"),
                    self.cp(self.out / "preload.o"),
                    "-ldl",
                ]
            )
        metadata = {
            "container_id": self.info["Id"],
            "image_id": self.info["Image"],
            "dispatcher_environment": self.env,
            "source": str(source),
            "library_sha256": library_hash,
            "header_sha256": header_hash,
            "library_build_id": BUILD_ID,
            "compiler": self.output(["/usr/bin/c++", "--version"]),
            "commands": commands,
            "source_sha256": {p.name: digest(p) for p in snap.iterdir()},
            "artifacts_sha256": {
                p.name: digest(p)
                for p in self.out.iterdir()
                if p.name in ["benchmark_baseline", "benchmark_progress", "libebs_trace.so"]
            },
        }
        write_json(self.out / "build-metadata.json", metadata)

    def run_one(self, name, config, instrumented=True, interval=10, extra_env=None):
        directory = self.out / name
        directory.mkdir()
        env = {
            **self.env,
            "LD_PRELOAD": "",
            "GXF_EBS_TRACE_DIR": "",
            "GXF_EBS_TRACE_INTERVAL_MS": str(interval),
            "GXF_EBS_TRACE_MAX_SAMPLES": str(60000 // interval + 2),
        }
        if instrumented:
            env.update(
                LD_PRELOAD=self.cp(self.out / "libebs_trace.so"),
                GXF_EBS_TRACE_DIR=self.cp(directory),
            )
        env.update(extra_env or {})
        binary = self.out / ("benchmark_progress" if instrumented else "benchmark_baseline")
        command = [self.cp(binary), "--busy_wait", "--enable_queue_stealing"]
        if config == "both":
            command.append("--enable_postcheck_fastpath")
        # timeout runs INSIDE Docker, so a timed-out docker exec cannot orphan a benchmark.
        with (directory / "run.log").open("w") as log:
            self.docker(
                ["timeout", "--signal=TERM", "--kill-after=5", "30", *command],
                cwd=directory,
                env=env,
                root=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        log = (directory / "run.log").read_text()
        if (
            "Dispatcher thread pinned to CPU core 13" not in log
            or "Dispatcher thread configured with SCHED_FIFO, priority 99" not in log
        ):
            raise ValueError(f"Dispatcher configuration not confirmed: {name}")
        throughputs = re.findall(
            r"\|\s*\d+\s*\|\s*13\s*\|\s*16\s*\|\s*16000\s*\|\s*([\d.]+)\s*\|", log
        )
        if len(throughputs) != 1:
            raise ValueError(f"Expected one complete throughput measurement: {name}")
        record = {
            "run": name,
            "config": config,
            "instrumented": instrumented,
            "interval_ms": interval if instrumented else None,
            "throughput_hz": float(throughputs[0]),
        }
        if instrumented:
            record.update(validate_trace(directory))
        write_json(directory / "run.json", {**record, "environment": env, "command": command})
        print(f"{name}: {record['throughput_hz']:.1f} ops/s", flush=True)
        return record


def validate_trace(directory):
    files = list(directory.glob("ebs-*.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one trace: {directory}")
    metadata = json.loads(files[0].read_text())
    if metadata["status"] != "complete" or not metadata["final_stable"] or metadata["truncated"]:
        raise ValueError(f"Incomplete trace: {metadata}")
    if metadata["sampler_policy_error"]:
        raise ValueError("Sampler could not use SCHED_OTHER")
    samples = list(csv.DictReader(files[0].with_suffix(".csv").open()))
    if len(samples) < 3 or len(samples) != metadata["sample_count"]:
        raise ValueError("Missing periodic samples")
    fields = ["monotonic_ns", "read_end_ns", *COUNTERS]
    for before, after in zip(samples, samples[1:], strict=False):
        if any(int(after[k]) < int(before[k]) for k in fields):
            raise ValueError("Counter or timestamp decreased")
    for row in samples:
        if int(row["read_end_ns"]) < int(row["monotonic_ns"]) or int(row["running_threads"]) > 13:
            raise ValueError("Invalid timestamp or running-thread gauge")
    if samples[0]["phase"] != "initial" or samples[-1]["phase"] != "final":
        raise ValueError("Missing lifecycle samples")
    log = (directory / "run.log").read_text()
    for counter in COUNTERS:
        name = (
            counter.removeprefix("worker_") if counter.startswith("worker_postcheck") else counter
        )
        values = re.findall(rf"\b{re.escape(name)}=(\d+)", log)
        if len(values) != 1 or int(values[0]) != int(samples[-1][counter]):
            raise ValueError(
                f"Final GXF log mismatch for {counter}: {values} vs {samples[-1][counter]}"
            )
    progress = list(csv.DictReader(next(directory.glob("progress_log_*.txt")).open()))
    if len(progress) != 160:
        raise ValueError(f"Expected 160 checkpoints, got {len(progress)}")
    for op in range(16):
        rows = [r for r in progress if int(r["operator_id"]) == op]
        if [int(r["completed_operations"]) for r in rows] != list(range(100, 1001, 100)):
            raise ValueError(f"Missing progress for operator {op}")
        times = [int(r["monotonic_ns"]) for r in rows]
        if times != sorted(times) or times[-1] > int(samples[-1]["read_end_ns"]):
            raise ValueError("Progress timestamps outside trace")
    return {k: int(samples[-1][k]) for k in COUNTERS} | {"samples": len(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="scheduler-graphs-rerun")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-cuda13")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--reuse-build", type=Path, help="Reuse artifacts in an existing output folder"
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    out = (
        args.reuse_build
        or args.output
        or args.build_dir
        / ("benchmark-steal-trace-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    ).resolve()
    if not args.reuse_build:
        out.mkdir(parents=True, exist_ok=False)
    experiment = Experiment(args, out)
    print(f"Output: {out}", flush=True)
    if not args.reuse_build:
        experiment.build_artifacts()
    else:
        metadata = json.loads((out / "build-metadata.json").read_text())
        for name, expected in metadata["artifacts_sha256"].items():
            if digest(out / name) != expected:
                raise ValueError(f"Built artifact changed: {name}")
    if args.build_only:
        return
    if (out / "experiment.json").exists():
        raise ValueError(
            "This directory already contains an experiment; choose a new output directory"
        )
    records = []
    status = {"status": "running", "repetitions": args.repetitions, "records": records}
    write_json(out / "experiment.json", status)
    try:
        for interval in [10, 20, 50]:
            batch = []
            for repetition in range(args.repetitions):
                configs = ["queue_stealing", "both"]
                if repetition % 2:
                    configs.reverse()
                for config in configs:
                    variants = [False, True] if repetition % 2 == 0 else [True, False]
                    for instrumented in variants:
                        variant = "trace" if instrumented else "baseline"
                        name = f"{interval}ms-r{repetition + 1}-{config}-{variant}"
                        record = experiment.run_one(name, config, instrumented, interval)
                        batch.append(record)
                        records.append(record)
                        write_json(out / "experiment.json", status)
            overhead = {}
            for config in ["queue_stealing", "both"]:
                medians = {
                    traced: statistics.median(
                        r["throughput_hz"]
                        for r in batch
                        if r["config"] == config and r["instrumented"] == traced
                    )
                    for traced in [False, True]
                }

                overhead[config] = 100 * (1 - medians[True] / medians[False])
            status.update(selected_interval_ms=interval, overhead_percent=overhead)
            if max(abs(value) for value in overhead.values()) < 5:
                break
        status["status"] = "complete"
        status["overhead_acceptable"] = max(abs(v) for v in overhead.values()) < 5
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(out / "experiment.json", status)
    subprocess.run([sys.executable, "-s", str(HERE / "analyze.py"), str(out)], check=True)


if __name__ == "__main__":
    main()
