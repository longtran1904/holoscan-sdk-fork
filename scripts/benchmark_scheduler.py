#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and run 10 scheduler configurations; write a CSV and seven comparison PNGs."""

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENT_TARGET = "benchmark_scheduler_throughput"
GREEDY_TARGET = "benchmark_scheduler_throughput_greedy"
LIBRARY_PATH_VARIABLES = ("LD_LIBRARY_PATH", "HOLOSCAN_LIB_PATH")
# Keep container work tied to the Docker client's stdin connection. Killing a
# docker-exec client alone does not terminate its process inside the container.
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
FIELDS = (
    "scheduler",
    "queue_stealing",
    "postcheck_fastpath",
    "busy_wait",
    "trial",
    "threads",
    "operators",
    "total_operations",
    "throughput_hz",
)


@dataclass(frozen=True)
class Configuration:
    label: str
    scheduler: str
    queue_stealing: bool = False
    postcheck_fastpath: bool = False
    busy_wait: bool = False

    @property
    def target(self):
        return EVENT_TARGET if self.scheduler == "event_based" else GREEDY_TARGET

    @property
    def flags(self):
        return [
            f"--{name}"
            for name, enabled in (
                ("enable_queue_stealing", self.queue_stealing),
                ("enable_postcheck_fastpath", self.postcheck_fastpath),
                ("busy_wait", self.busy_wait),
            )
            if enabled
        ]


EVENT_VARIANTS = (
    ("Default", False, False),
    ("Queue stealing", True, False),
    ("Postcheck fastpath", False, True),
    ("Both flags", True, True),
)
CONFIGURATIONS = (
    *(
        Configuration(label, "event_based", stealing, fastpath, busy_wait)
        for busy_wait in (False, True)
        for label, stealing, fastpath in EVENT_VARIANTS
    ),
    Configuration("Greedy: normal", "greedy"),
    Configuration("Greedy: 1 ms busy wait", "greedy", busy_wait=True),
)


PANEL_NAMES = (
    "event-throughput-normal",
    "event-improvement-normal",
    "event-throughput-busy-wait",
    "event-improvement-busy-wait",
    "greedy-normal",
    "greedy-busy-wait",
)


def graph_paths(csv_path):
    """Return the overview followed by the six panels in reading order."""
    return [
        csv_path.with_suffix(".png"),
        *(csv_path.with_name(f"{csv_path.stem}-{name}.png") for name in PANEL_NAMES),
    ]


def configuration_key(config):
    return (config.scheduler, config.busy_wait, config.queue_stealing, config.postcheck_fastpath)


def parse_results(output, config):
    """Read the result table, rejecting incomplete or invalid workload results."""
    event = config.scheduler == "event_based"
    header = ["Trial", *(["Threads"] if event else []), "Operators", "Total Ops", "Ops/s"]
    rows = []
    in_table = False
    for line in output.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [cell.strip() for cell in line[1:-1].split("|")]
        if cells == header:
            if in_table:
                raise ValueError("Duplicate result table")
            in_table = True
            continue
        if not in_table or all(re.fullmatch(r"[-:]+", cell) for cell in cells):
            continue
        if len(cells) != len(header):
            raise ValueError(f"Malformed result row: {line}")
        try:
            numbers = [int(cell) for cell in cells[:-1]]
            throughput = float(cells[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric result: {line}") from exc
        trial, *coordinates, total = numbers
        threads, operators = coordinates if event else (1, coordinates[0])
        if (
            trial < 0
            or min(threads, operators, total) <= 0
            or not math.isfinite(throughput)
            or throughput <= 0
        ):
            raise ValueError(f"Nonpositive or nonfinite result: {line}")
        rows.append(
            dict(
                zip(
                    FIELDS,
                    (
                        config.scheduler,
                        int(config.queue_stealing) if event else "",
                        int(config.postcheck_fastpath) if event else "",
                        int(config.busy_wait),
                        trial,
                        threads,
                        operators,
                        total,
                        throughput,
                    ),
                    strict=True,
                )
            )
        )

    expected_x = (1, 2, 4, 8, 10, 12, 13, 14, 16) if event else (1, 2, 4, 8, 12, 14, 16)
    expected = {
        (trial, x if event else 1, 16 if event else x) for trial, x in enumerate(expected_x)
    }
    actual = {(row["trial"], row["threads"], row["operators"]) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError(f"Incomplete or unexpected trial coordinates for {config.label}")
    operations_per_operator = 1000 if config.busy_wait else (100000 if event else 10000000)
    for row in rows:
        if row["total_operations"] != row["operators"] * operations_per_operator:
            raise ValueError(f"Unexpected operation count for {config.label}: {row}")
    return rows


def match_baseline(rows, baseline):
    """Match by trial coordinates and workload, independently of table row order."""

    def keyed(values):
        return {
            (
                r["scheduler"],
                r["busy_wait"],
                r["trial"],
                r["threads"],
                r["operators"],
                r["total_operations"],
            ): r
            for r in values
        }

    current, default = keyed(rows), keyed(baseline)
    if (
        len(current) != len(rows)
        or len(default) != len(baseline)
        or current.keys() != default.keys()
    ):
        raise ValueError("Event-based trial coordinates do not match the default configuration")
    return [
        (row, row["throughput_hz"] / default[key]["throughput_hz"])
        for key, row in sorted(current.items(), key=lambda item: item[1]["threads"])
    ]


def load_pyplot():
    try:
        import matplotlib  # noqa: PLC0415 - optional dependency preflight

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415 - select backend before pyplot
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required. Install it in the benchmark environment with "
            "`python3 -m pip install matplotlib`, then rerun."
        ) from exc
    return plt


def comparison_figure(plt, results):
    fig, _ = plt.subplots(3, 2, figsize=(13, 13), layout="constrained")
    draw_comparison(fig.axes, results)
    return fig


def draw_comparison(axes, results):
    """Draw the same six panels on overview axes or independent figure axes."""
    for busy_wait in (False, True):
        offset = 2 * int(busy_wait)
        throughput_ax, ratio_ax = axes[offset : offset + 2]
        workload = "1 ms busy wait" if busy_wait else "normal"
        baseline = results[("event_based", busy_wait, False, False)]
        for index, (label, stealing, fastpath) in enumerate(EVENT_VARIANTS):
            rows = results[("event_based", busy_wait, stealing, fastpath)]
            matched = match_baseline(rows, baseline)
            x = [row["threads"] for row, _ in matched]
            style = dict(label=label, color=f"C{index}", marker="o")
            throughput_ax.plot(x, [row["throughput_hz"] for row, _ in matched], **style)
            ratio_ax.plot(x, [ratio for _, ratio in matched], **style)
        throughput_ax.set(
            title=f"Event-based throughput: {workload}, 16 operators",
            xlabel="Worker threads",
            ylabel="Throughput (operations/s)",
        )
        ratio_ax.set(
            title=f"Event-based improvement: {workload}",
            xlabel="Worker threads",
            ylabel="Throughput / workload default (×)",
        )
        ratio_ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        for ax in (throughput_ax, ratio_ax):
            ax.set_xticks(sorted(row["threads"] for row in baseline))

    greedy_busy = results[("greedy", True, False, False)]
    (reference,) = [
        row for row in greedy_busy if row["operators"] == 1 and row["total_operations"] == 1000
    ]
    axes[2].axhline(
        reference["throughput_hz"],
        color="black",
        linestyle="--",
        label="Greedy: 1 operator, 1 ms busy wait",
    )
    for busy_wait, ax in zip((False, True), axes[4:], strict=True):
        rows = results[("greedy", busy_wait, False, False)]
        ordered = sorted(rows, key=lambda row: row["operators"])
        x = [row["operators"] for row in ordered]
        label = "Greedy: 1 ms busy wait" if busy_wait else "Greedy: normal"
        ax.plot(x, [row["throughput_hz"] for row in ordered], marker="o", label=label)
        ax.set(
            title=label,
            xlabel="Operators (one scheduler thread)",
            ylabel="Throughput (operations/s)",
            xticks=x,
        )
    for ax in (axes[0], axes[2], *axes[4:]):
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.05)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()


def save_graphs(plt, results, csv_path):
    """Save the overview and independently laid out panels, closing every figure."""
    paths = graph_paths(csv_path)
    figures = []
    try:
        figures.append(comparison_figure(plt, results))
        panel_axes = []
        for _ in PANEL_NAMES:
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
            figures.append(fig)
            panel_axes.append(ax)
        draw_comparison(panel_axes, results)
        for fig, path in zip(figures, paths, strict=True):
            fig.savefig(path, dpi=160)
    finally:
        for fig in figures:
            plt.close(fig)
    return paths


@dataclass(frozen=True)
class Container:
    """An existing container with this checkout bind-mounted by ./run launch."""

    id: str
    name: str
    image_id: str
    repo_root: Path
    environment: dict

    def path(self, local_path):
        if not local_path.is_relative_to(REPO_ROOT):
            raise ValueError("--container requires --build-dir inside this checkout's bind mount")
        return self.repo_root / local_path.relative_to(REPO_ROOT)

    def command(self, argv, env=None):
        command = ["docker", "exec", "--workdir", str(self.repo_root)]
        for variable, value in (env or {}).items():
            command.extend(["--env", f"{variable}={value}"])
        return [*command, self.id, *argv]


def inspect_container(name):
    """Resolve a name once and pin all work to that running container's ID."""
    result = subprocess.run(
        ["docker", "container", "inspect", name], check=True, capture_output=True, text=True
    )
    (info,) = json.loads(result.stdout)
    if not info["State"]["Running"]:
        raise ValueError(f"Container {name!r} is not running; use your ./run launch container")
    roots = [
        Path(mount["Destination"])
        for mount in info["Mounts"]
        if mount["Type"] == "bind" and Path(mount["Source"]).resolve() == REPO_ROOT
    ]
    if len(roots) != 1:
        raise ValueError(
            f"Container {name!r} must bind-mount this checkout exactly once: {REPO_ROOT}"
        )
    environment = dict(entry.split("=", 1) for entry in info["Config"]["Env"] or [])
    # Docker inherits the remaining container environment itself. Never forward host variables.
    environment = {key: environment[key] for key in LIBRARY_PATH_VARIABLES if key in environment}
    return Container(info["Id"], info["Name"].lstrip("/"), info["Image"], roots[0], environment)


def read_build_cache(build_dir, container=None):
    cache = {}
    for line in (build_dir / "CMakeCache.txt").read_text().splitlines():
        match = re.match(r"([^/#][^:]*):[^=]+=(.*)", line)
        if match:
            cache[match[1]] = match[2]
    home = cache.get("CMAKE_HOME_DIRECTORY")
    repo_root = container.repo_root if container else REPO_ROOT
    cache_root = Path(home) if container and home else Path(home or ".").resolve()
    if not home or cache_root != repo_root:
        raise ValueError(
            f"Build cache belongs to {home!r}, not {repo_root}. "
            "Run inside the CUDA container or select it with --container NAME."
        )
    for option in ("HOLOSCAN_BUILD_EXAMPLES", "HOLOSCAN_CPP_EXAMPLES"):
        if cache.get(option, "").upper() not in ("ON", "TRUE", "YES", "1"):
            raise ValueError(f"The configured build must have {option}=ON")
    return cache


@contextmanager
def command_process(command, *, container=None, env=None, **kwargs):
    if container:
        command = container.command(["python3", "-c", CONTAINER_SUPERVISOR, *command], env)
        command.insert(2, "--interactive")
        kwargs["stdin"] = subprocess.PIPE
        env = None
    with subprocess.Popen(command, env=env, **kwargs) as process:
        try:
            yield process
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def build_benchmarks(build_dir, jobs, container=None):
    repo_root = container.repo_root if container else REPO_ROOT
    build_dir = container.path(build_dir) if container else build_dir
    configure = ["cmake", "-S", str(repo_root), "-B", str(build_dir)]
    command = ["cmake", "--build", str(build_dir), "--target", EVENT_TARGET, GREEDY_TARGET]
    if jobs is not None:
        command.extend(["--parallel", str(jobs)])
    for argv in (configure, command):
        if container:
            with command_process(argv, container=container) as process:
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, argv)
        else:
            subprocess.run(argv, check=True)


def run_benchmark(build_dir, config, env, container=None):
    build_dir = container.path(build_dir) if container else build_dir
    executable = build_dir / "examples" / config.target / "cpp" / config.target
    command = [str(executable), *config.flags]
    workload = "1 ms busy wait" if config.busy_wait else "normal"
    print(f"\n=== {config.scheduler}: {config.label} ({workload}) ===", flush=True)
    if container:
        print(f"Benchmark command: {shlex.join(container.command(command, env))}", flush=True)
    else:
        overrides = [f"{key}={env[key]}" for key in LIBRARY_PATH_VARIABLES]
        print(f"Working directory: {REPO_ROOT}", flush=True)
        print(f"Command: {shlex.join(['env', *overrides, *command])}", flush=True)
    lines = []
    with command_process(
        command,
        container=container,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            # Logs are streamed but only result-table candidates need retaining.
            if re.sub(r"\x1b\[[0-9;]*m", "", line).lstrip().startswith("|"):
                lines.append(line)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
    return parse_results("".join(lines), config)


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def benchmark_environment(build_dir, cache, inherited):
    env = inherited.copy()
    # Normalize lexically: container paths must not be resolved on the host filesystem.
    library_dirs = [build_dir / "lib", build_dir / "lib64"]
    configured_lib = Path(os.path.normpath(build_dir / cache.get("CMAKE_INSTALL_LIBDIR", "lib")))
    if configured_lib.is_relative_to(build_dir) and configured_lib not in library_dirs:
        library_dirs.append(configured_lib)
    libraries = os.pathsep.join(map(str, library_dirs))
    for variable in LIBRARY_PATH_VARIABLES:
        env[variable] = libraries + (os.pathsep + env[variable] if env.get(variable) else "")
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        required=True,
        type=Path,
        help="Configured SDK build (relative paths start at the repository root)",
    )
    parser.add_argument(
        "--jobs", type=positive_int, help="Parallel build jobs (build tool default)"
    )
    parser.add_argument(
        "--output", type=Path, help="New CSV path (default: BUILD/benchmark-results/)"
    )
    parser.add_argument(
        "--container",
        help="Existing ./run launch container name or ID; build/run via docker exec, plot locally",
    )
    args = parser.parse_args(argv)
    csv_path = None
    results = {}
    try:
        plt = load_pyplot()  # Fail before any compilation or benchmark work.
        build_dir = (REPO_ROOT / args.build_dir).resolve()
        container = inspect_container(args.container) if args.container else None
        runtime_build = container.path(build_dir) if container else build_dir
        cache = read_build_cache(build_dir, container)
        if args.output:
            output = args.output.resolve()
            if output.suffix.lower() != ".csv":
                raise ValueError("--output must have a .csv extension")
            for path in (output, *graph_paths(output)):
                if path.exists():
                    raise ValueError(f"Output already exists: {path}; choose a new path")
        if container:
            print(f"Using existing container: {container.name} ({container.id[:12]})", flush=True)
            print(f"Container image: {container.image_id}", flush=True)
        print(f"Benchmark build: {runtime_build}", flush=True)
        build_benchmarks(build_dir, args.jobs, container)
        if args.output:
            output.parent.mkdir(parents=True, exist_ok=True)
            stream = output.open("x", newline="")
            csv_path = output
        else:
            directory = build_dir / "benchmark-results"
            directory.mkdir(parents=True, exist_ok=True)
            stream = tempfile.NamedTemporaryFile(  # noqa: SIM115 - managed by `with stream` below
                mode="w",
                newline="",
                prefix="scheduler-",
                suffix=".csv",
                dir=directory,
                delete=False,
            )
            csv_path = Path(stream.name)
        env = benchmark_environment(
            runtime_build, cache, container.environment if container else os.environ
        )
        with stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            stream.flush()
            for config in CONFIGURATIONS:
                rows = run_benchmark(build_dir, config, env, container)
                baseline_key = ("event_based", config.busy_wait, False, False)
                if config.scheduler == "event_based" and baseline_key in results:
                    match_baseline(rows, results[baseline_key])
                writer.writerows(rows)
                stream.flush()
                results[configuration_key(config)] = rows
        png_paths = save_graphs(plt, results, csv_path)
        print(f"\nCSV: {csv_path}")
        for png_path in png_paths:
            print(f"PNG: {png_path}")
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
        KeyboardInterrupt,
    ) as exc:
        print(f"Benchmark failed: {exc or 'interrupted'}", file=sys.stderr)
        if csv_path is not None:
            kind = "Complete" if len(results) == len(CONFIGURATIONS) else "Partial"
            print(f"{kind} CSV preserved: {csv_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
