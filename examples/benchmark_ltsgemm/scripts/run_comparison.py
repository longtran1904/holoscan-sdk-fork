#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Run LtSgemm in sequential A/B/C processes, optionally adding native D/E modes."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXPECTED = {(128 << i, 128, 128, flush) for flush in ("off", "on") for i in range(10)}
CASE_METRICS = ("mean_us", "min_us", "median_us", "p99_us", "gpu_total_ms", "case_wall_ms")
PREFIX = "LT_SGEMM_WRAPPER "
NATIVE_PREFIX = "LT_SGEMM_NATIVE "
BASE_MODES = ("original", "direct", "holoscan")
NATIVE_MODES = {"native-1000": 1000, "native-1": 1}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)"
GEMM = re.compile(
    r"GEMM \(M=(?P<m>\d+), N=(?P<n>\d+), K=(?P<k>\d+), L2 flush=(?P<flush>off|on)\): "
    + rf"(?P<mean_us>{NUMBER}) ± (?P<stddev_us>{NUMBER}) us, "
    + rf"min=(?P<min_us>{NUMBER}) us, median=(?P<median_us>{NUMBER}) us, "
    + rf"P99=(?P<p99_us>{NUMBER}) us, CV=(?P<cv_percent>{NUMBER})%, "
    + rf"(?P<tflops>{NUMBER}) TFLOP/s",
    re.IGNORECASE,
)
WALL = re.compile(
    rf"Timed GEMM total: (?P<gpu_total_ms>{NUMBER}) ms, "
    rf"wall time: (?P<case_wall_ms>{NUMBER}) ms, "
    rf"GEMM/wall: (?P<gemm_wall_percent>{NUMBER})%",
    re.IGNORECASE,
)


def positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return value


def parse_output(text, runner):
    """Require a complete sweep and the matching wrapper/native success record."""
    if runner not in (*BASE_MODES, *NATIVE_MODES):
        raise ValueError(f"Unknown runner: {runner}")
    cases, records, native_records, keys = [], [], [], set()
    pending = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("GEMM ("):
            match = GEMM.fullmatch(line)
            if pending is not None or match is None:
                raise ValueError(f"Incomplete or malformed GEMM record: {line}")
            pending = match.groupdict()
            for key in ("m", "n", "k"):
                pending[key] = int(pending[key])
            for key in set(pending) - {"m", "n", "k", "flush"}:
                pending[key] = float(pending[key])
        elif line.startswith("Timed GEMM total:"):
            match = WALL.fullmatch(line)
            if pending is None or match is None:
                raise ValueError(f"Unpaired or malformed wall-time record: {line}")
            pending.update({key: float(value) for key, value in match.groupdict().items()})
            key = tuple(pending[name] for name in ("m", "n", "k", "flush"))
            if key not in EXPECTED or key in keys:
                raise ValueError(f"Unexpected or duplicate case: {key}")
            for name in CASE_METRICS + ("tflops",):
                positive(pending[name], name)
            for name in ("stddev_us", "cv_percent", "gemm_wall_percent"):
                positive(pending[name], name, allow_zero=True)
            keys.add(key)
            cases.append(pending)
            pending = None
        elif line.startswith(PREFIX):
            records.append(json.loads(line[len(PREFIX) :]))
        elif line.startswith(NATIVE_PREFIX):
            native_records.append(json.loads(line[len(NATIVE_PREFIX) :]))
    if pending is not None or keys != EXPECTED:
        raise ValueError(
            f"Expected 20 complete cases; got {len(cases)}; missing {sorted(EXPECTED - keys)}"
        )
    if runner in NATIVE_MODES:
        if records or len(native_records) != 1 or not isinstance(native_records[0], dict):
            raise ValueError("Expected exactly one native completion record and no wrapper record")
        timing = native_records[0]
        count = NATIVE_MODES[runner]
        for key, expected in (
            ("schema_version", 1),
            ("gemms_per_tick", count),
            ("completed_cases", 20),
            ("timed_gemms", 20000),
            ("compute_calls", 20000 // count),
            ("return_code", 0),
        ):
            if type(timing.get(key)) is not int or timing[key] != expected:
                raise ValueError(f"Invalid native {key}: {timing.get(key)}")
        if timing.get("completed") is not True:
            raise ValueError("Incomplete native benchmark")
        if "entrypoint_wall_ms" in timing:
            raise ValueError("Native runner must not report an original-entrypoint timing")
        positive(timing.get("app_run_wall_ms"), "app_run_wall_ms")
        return cases, timing
    if native_records:
        raise ValueError("Non-native executable unexpectedly emitted native records")
    if runner == "original":
        if records:
            raise ValueError("Original executable unexpectedly emitted wrapper records")
        return cases, {}
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError("Expected exactly one wrapper completion record")
    timing = records[0]
    for key, expected in (("schema_version", 1), ("invocations", 1), ("return_code", 0)):
        if type(timing.get(key)) is not int or timing[key] != expected:
            raise ValueError(f"Invalid wrapper {key}: {timing.get(key)}")
    if timing.get("runner") != runner or timing.get("completed") is not True:
        raise ValueError("Wrapper runner mismatch or incomplete benchmark")
    positive(timing.get("entrypoint_wall_ms"), "entrypoint_wall_ms")
    if runner == "holoscan":
        positive(timing.get("app_run_wall_ms"), "app_run_wall_ms")
    elif "app_run_wall_ms" in timing:
        raise ValueError("Direct runner must not report an application-run timing")
    return cases, timing


def run_order(trial, native=False):
    modes = BASE_MODES + (tuple(NATIVE_MODES) if native else ())
    offset = (trial - 1) % len(modes)
    return modes[offset:] + modes[:offset]


def paired_ratios(runs):
    native = any(run["runner"] in NATIVE_MODES for run in runs)
    expected_modes = set(run_order(1, native))
    groups = {}
    for run in runs:
        modes = groups.setdefault(run["trial"], {})
        if run["runner"] in modes:
            raise ValueError("Duplicate process run in trial")
        modes[run["runner"]] = run
    result = []

    def append(trial, scope, case, label, metric, candidate, reference):
        positive(candidate, metric)
        positive(reference, metric)
        factor = candidate / reference
        result.append(
            dict(
                trial=trial,
                scope=scope,
                m=case.get("m", ""),
                n=case.get("n", ""),
                k=case.get("k", ""),
                flush=case.get("flush", ""),
                comparison=label,
                metric=metric,
                reference=reference,
                candidate=candidate,
                factor=factor,
                slowdown_percent=100 * (factor - 1),
                delta=candidate - reference,
            )
        )

    for trial, modes in sorted(groups.items()):
        if set(modes) != expected_modes:
            raise ValueError(f"Trial {trial} must contain {sorted(expected_modes)}")
        for run in modes.values():
            keys = [tuple(case[key] for key in ("m", "n", "k", "flush")) for case in run["cases"]]
            if len(keys) != len(EXPECTED) or set(keys) != EXPECTED:
                raise ValueError(f"Trial {trial} {run['runner']} has incomplete or duplicate cases")
        pairs = [
            ("holoscan", "original", "C/A"),
            ("direct", "original", "B/A"),
            ("holoscan", "direct", "C/B"),
        ]
        native_pairs = [
            ("native-1000", "holoscan", "D/C"),
            ("native-1", "holoscan", "E/C"),
            ("native-1", "native-1000", "E/D"),
        ]
        if native:
            pairs += [("native-1000", "original", "D/A"), ("native-1", "original", "E/A")]
            pairs += native_pairs
        for candidate, reference, label in pairs:
            cand, ref = modes[candidate], modes[reference]
            append(
                trial,
                "process",
                {},
                label,
                "process_wall_ms",
                cand["process_wall_ms"],
                ref["process_wall_ms"],
            )
            indexed = {tuple(c[key] for key in ("m", "n", "k", "flush")): c for c in ref["cases"]}
            for case in cand["cases"]:
                key = tuple(case[name] for name in ("m", "n", "k", "flush"))
                for metric in CASE_METRICS:
                    append(trial, "case", case, label, metric, case[metric], indexed[key][metric])
        append(
            trial,
            "entrypoint",
            {},
            "C/B",
            "entrypoint_wall_ms",
            modes["holoscan"]["entrypoint_wall_ms"],
            modes["direct"]["entrypoint_wall_ms"],
        )
        if native:
            for candidate, reference, label in native_pairs:
                append(
                    trial,
                    "application",
                    {},
                    label,
                    "app_run_wall_ms",
                    modes[candidate]["app_run_wall_ms"],
                    modes[reference]["app_run_wall_ms"],
                )
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(ratios):
    fields = ("scope", "m", "n", "k", "flush", "comparison", "metric")
    groups = {}
    for row in ratios:
        groups.setdefault(tuple(row[key] for key in fields), []).append(row)
    result = []
    for key, rows in groups.items():
        factors = [row["factor"] for row in rows]
        result.append(
            dict(
                zip(fields, key, strict=False),
                trials=len(rows),
                median_factor=statistics.median(factors),
                q25_factor=percentile(factors, 0.25),
                q75_factor=percentile(factors, 0.75),
                median_slowdown_percent=statistics.median([r["slowdown_percent"] for r in rows]),
                median_delta=statistics.median([r["delta"] for r in rows]),
            )
        )
    return result


def execute(command, directory, stem, timeout):
    """Time launch through completion, capturing both streams identically in all modes."""
    stdout_path, stderr_path = directory / f"{stem}.stdout.log", directory / f"{stem}.stderr.log"
    result = dict(
        command=[str(arg) for arg in command],
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        timed_out=False,
        return_code=None,
    )
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        start = time.perf_counter_ns()
        try:
            process = subprocess.run(
                command, stdout=stdout, stderr=stderr, timeout=timeout, check=False
            )
            result["return_code"] = process.returncode
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
        except OSError as error:
            result["launch_error"] = str(error)
        finally:
            result["process_wall_ms"] = (time.perf_counter_ns() - start) / 1e6
    return result


def probe(command, cwd=None):
    try:
        process = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=10, check=False
        )
        return dict(
            command=[str(arg) for arg in command],
            return_code=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=[str(arg) for arg in command], unavailable=str(error))


def gpu_snapshot():
    return dict(
        device=probe(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem",
                "--format=csv",
            ]
        ),
        activity=probe(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]
        ),
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes():
    files = [REPO / "examples/gpu_kernels/src/Common/helpers.h"]
    for relative in ("examples/gpu_kernels/src/LtSgemm", "examples/resources/cuda_green_context"):
        files.extend(path for path in (REPO / relative).rglob("*") if path.is_file())
    return {str(path.relative_to(REPO)): sha256(path) for path in sorted(files)}


def build_metadata(binary, requested_dir):
    directory = (
        Path(requested_dir).resolve()
        if requested_dir
        else next((p for p in binary.parents if (p / "CMakeCache.txt").is_file()), None)
    )
    data = dict(
        binary=str(binary),
        binary_sha256=sha256(binary),
        linked_libraries=probe(["ldd", str(binary)]),
    )
    data["resolved_libraries"] = {
        name: str(Path(path).resolve())
        for name, path in re.findall(
            r"^\s*(\S+)\s+=>\s+(/\S+)",
            data["linked_libraries"].get("stdout", ""),
            re.MULTILINE,
        )
    }
    if directory is None:
        data["warning"] = (
            "No build directory found; supply the corresponding --original-build-dir, "
            "--wrapper-build-dir, or --native-build-dir"
        )
        return data
    data["build_dir"] = str(directory)
    cache = directory / "CMakeCache.txt"
    if cache.is_file():
        data["cmake_cache_sha256"] = sha256(cache)
        data["build_settings"] = {}
        for line in cache.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.split(":", 1)[0]
            if name.startswith(
                (
                    "CMAKE_CXX",
                    "CMAKE_CUDA",
                    "CMAKE_EXE_LINKER",
                    "CMAKE_BUILD_TYPE",
                    "CUBLAS",
                    "CUDART",
                    "CUDAToolkit",
                    "holoscan_DIR",
                )
            ):
                data["build_settings"][name] = value
    commands = directory / "compile_commands.json"
    if commands.is_file():
        data["compile_commands"] = json.loads(commands.read_text(encoding="utf-8"))
    else:
        data["warning"] = (
            "compile_commands.json unavailable; configure CMAKE_EXPORT_COMPILE_COMMANDS=ON"
        )
    compiler = data.get("build_settings", {}).get("CMAKE_CUDA_COMPILER") or shutil.which("nvcc")
    if compiler:
        data["nvcc_version"] = probe([compiler, "--version"])
        cublas_header = Path(compiler).resolve().parents[1] / "include/cublas_api.h"
        if cublas_header.is_file():
            data["cublas_header_version"] = dict(
                re.findall(
                    r"#define CUBLAS_VER_(MAJOR|MINOR|PATCH|BUILD)\s+(\d+)",
                    cublas_header.read_text(),
                )
            )
    return data


def capture_metadata(
    original,
    wrapper_binary,
    original_build_dir=None,
    wrapper_build_dir=None,
    native_binary=None,
    native_build_dir=None,
):
    env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "CUDA_CACHE_DISABLE",
        "CUDA_CACHE_PATH",
        "CUDA_FORCE_PTX_JIT",
        "CUDA_MODULE_LOADING",
        "CUBLAS_WORKSPACE_CONFIG",
        "NVIDIA_TF32_OVERRIDE",
        "HOLOSCAN_LOG_LEVEL",
        "HOLOSCAN_EXECUTOR_LOG_LEVEL",
        "OMP_NUM_THREADS",
        "LD_LIBRARY_PATH",
    )
    metadata = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),
        python=sys.version,
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        environment={key: os.environ.get(key) for key in env_keys},
        git_head=probe(["git", "rev-parse", "HEAD"], REPO),
        git_status=probe(["git", "status", "--short", "--untracked-files=all"], REPO),
        original=build_metadata(original, original_build_dir),
        wrapper=build_metadata(wrapper_binary, wrapper_build_dir),
    )
    if native_binary is not None:
        metadata["native"] = build_metadata(native_binary, native_build_dir)
    return metadata


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output, runs):
    measurements, processes = [], []
    native = any(run["runner"] in NATIVE_MODES for run in runs)
    for run in runs:
        measurements.extend(
            dict(trial=run["trial"], runner=run["runner"], **case) for case in run["cases"]
        )
        process = dict(
            trial=run["trial"],
            runner=run["runner"],
            process_wall_ms=run["process_wall_ms"],
            entrypoint_wall_ms=run.get("entrypoint_wall_ms", ""),
            app_run_wall_ms=run.get("app_run_wall_ms", ""),
        )
        if native:
            process.update(
                {
                    key: run.get("native_record", {}).get(key, "")
                    for key in ("gemms_per_tick", "completed_cases", "timed_gemms", "compute_calls")
                }
            )
        processes.append(process)
    ratios = paired_ratios(runs)
    summaries = summarize(ratios)
    for name, rows in (
        ("measurements.csv", measurements),
        ("processes.csv", processes),
        ("paired_ratios.csv", ratios),
        ("summary.csv", summaries),
    ):
        write_csv(output / name, rows)
    lines = [
        "# LtSgemm benchmark-hosting comparison",
        "",
        "A = original; B = direct control; C = single-tick Holoscan wrapper. "
        "Factors are paired within each trial.",
        "",
        *(
            ["D = native 1,000 GEMMs/tick (20 calls); E = native one GEMM/tick (20,000 calls)."]
            if native
            else []
        ),
        "Differences are observed execution costs, not pure scheduler overhead.",
        "A negative slowdown is a speedup. "
        "Case wall time includes gaps between repetitions, setup, copies, warmup, and cleanup.",
        "Event timings use the original CUDA-event boundaries; "
        "they exclude gaps between repetitions and are not pure instruction times.",
        *(
            [
                "Application-run timings surround app->run() in C/D/E. "
                "Native runs have no original-entrypoint timing."
            ]
            if native
            else []
        ),
        "",
        "## Primary case: M=65536, N=K=128, L2 flush off",
        "",
        "| Comparison | Metric | Median factor | Factor IQR | Median slowdown |",
        "| --- | --- | --- | --- | --- |",
    ]

    def table_row(row, label):
        return (
            f"| {row['comparison']} | {row[label]} | {row['median_factor']:.4f}x | "
            f"{row['q25_factor']:.4f}-{row['q75_factor']:.4f} | "
            f"{row['median_slowdown_percent']:+.2f}% |"
        )

    lines.extend(
        table_row(row, "metric")
        for row in summaries
        if row["m"] == 65536
        and row["flush"] == "off"
        and row["metric"] in ("mean_us", "median_us", "p99_us", "case_wall_ms")
    )
    lines.extend(
        [
            "",
            "## Whole-process, entrypoint, and application-run timings",
            "",
            "| Comparison | Boundary | Median factor | Factor IQR | Median slowdown |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    lines.extend(table_row(row, "scope") for row in summaries if row["scope"] != "case")
    lines.extend(
        [
            "",
            "All shapes and both cache modes are in summary.csv; "
            "individual paired values are in paired_ratios.csv.",
            "P99 rows summarize ratios of per-run P99s, not a global P99 pooled across runs.",
            "Inspect B/A, compile commands, linked libraries, and GPU activity "
            "in manifest.json before attributing C/A to Holoscan.",
            "The benchmark does not pin clocks or cuBLASLt algorithms "
            "and inherits the original numerical-validation limitations.",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    original,
    wrapper_binary,
    output,
    trials=10,
    timeout=600,
    original_build_dir=None,
    wrapper_build_dir=None,
    native_binary=None,
    native_build_dir=None,
):
    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be a positive integer")
    positive(timeout, "timeout")
    if native_build_dir is not None and native_binary is None:
        raise ValueError("--native-build-dir requires --native")
    original, wrapper_binary, output = (
        original.resolve(),
        wrapper_binary.resolve(),
        output.resolve(),
    )
    native_binary = native_binary.resolve() if native_binary is not None else None
    for binary in (original, wrapper_binary, *([native_binary] if native_binary else [])):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"Not an executable file: {binary}")
    # Never allow generated outputs under a protected input directory.
    for protected in (
        REPO / "examples/gpu_kernels/src",
        REPO / "examples/resources/cuda_green_context",
    ):
        if output == protected or protected in output.parents:
            raise ValueError(f"Output directory is inside protected inputs: {output}")
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    manifest = dict(
        schema_version=1,
        status="running",
        trials_requested=trials,
        timeout_seconds=timeout,
        timed_repetitions_per_case=1000,
        expected_cases_per_process=20,
        profiling=False,
        modes=list(run_order(1, native_binary is not None)),
        runs=[],
    )
    initial_hashes = protected_hashes()
    manifest["protected_before"] = initial_hashes
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    try:
        manifest["metadata"] = capture_metadata(
            original,
            wrapper_binary,
            original_build_dir,
            wrapper_build_dir,
            native_binary,
            native_build_dir,
        )
        for trial in range(1, trials + 1):
            for runner in run_order(trial, native_binary is not None):
                print(f"Trial {trial}/{trials}: {runner}", flush=True)
                if runner in NATIVE_MODES:
                    command = [str(native_binary), f"--gemms-per-tick={NATIVE_MODES[runner]}"]
                elif runner == "original":
                    command = [str(original)]
                else:
                    command = [str(wrapper_binary), f"--runner={runner}"]
                before = gpu_snapshot()
                result = execute(command, logs, f"trial-{trial:02d}-{runner}", timeout)
                result.update(
                    trial=trial, runner=runner, gpu_before=before, gpu_after=gpu_snapshot()
                )
                manifest["runs"].append(result)
                write_json(manifest_path, manifest)
                if result["timed_out"] or result["return_code"] != 0:
                    raise ValueError(f"Trial {trial} {runner} failed; see {result['stderr_path']}")
                cases, timing = parse_output(
                    Path(result["stdout_path"]).read_text(encoding="utf-8"), runner
                )
                result["cases"] = cases
                for name in ("entrypoint_wall_ms", "app_run_wall_ms"):
                    if name in timing:
                        result[name] = timing[name]
                result["native_record" if runner in NATIVE_MODES else "wrapper_record"] = timing
                write_json(manifest_path, manifest)
        manifest["protected_after"] = protected_hashes()
        if manifest["protected_after"] != initial_hashes:
            raise ValueError("Protected inputs changed during experiment")
        write_reports(output, manifest["runs"])
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original", type=Path, required=True, help="Unchanged original benchmark executable"
    )
    parser.add_argument("--wrapper", type=Path, required=True, help="benchmark_ltsgemm executable")
    parser.add_argument(
        "--native", type=Path, help="benchmark_ltsgemm_native executable (adds D/E)"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New result directory (must not exist)"
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600, help="Seconds per benchmark process")
    parser.add_argument("--original-build-dir", type=Path)
    parser.add_argument("--wrapper-build-dir", type=Path)
    parser.add_argument("--native-build-dir", type=Path)
    args = parser.parse_args()
    try:
        run_experiment(
            args.original,
            args.wrapper,
            args.output,
            args.trials,
            args.timeout,
            args.original_build_dir,
            args.wrapper_build_dir,
            args.native,
            args.native_build_dir,
        )
    except (OSError, ValueError) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        return 1
    print(f"Results: {args.output.resolve() / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
