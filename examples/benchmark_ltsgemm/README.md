# LtSgemm hosting and native scheduling benchmarks

Compare the existing LtSgemm benchmark outside and inside Holoscan, plus two native scheduling modes, without editing the reference sources.
This is an isolated, full-GPU experiment; it does not use CUDA Green Context partitions or concurrent kernels.

| Run | Path | Purpose |
| --- | --- | --- |
| A | Original `sample_cublasLt_LtSgemm` executable | Reference |
| B | `benchmark_ltsgemm --runner=direct` | Wrapper/build/linking control |
| C | `benchmark_ltsgemm --runner=holoscan` (default) | One portless operator, one tick, GreedyScheduler |
| D | `benchmark_ltsgemm_native --gemms-per-tick=1000` | One case per tick, 20 operator calls |
| E | `benchmark_ltsgemm_native --gemms-per-tick=1` (default) | One GEMM per tick, 20,000 operator calls |

Use `--operators=N` to place N identical native operators in the application for contention
experiments. The default is one. This option does not change the configured scheduler or CUDA
synchronization behavior.

The wrapper compiles the original `main.cpp` with a target-local entrypoint rename and links the original CUDA sources.
Both wrapper modes call that same entrypoint once.
The separate native executable owns matrices, workspace, descriptors, selected algorithm, and events in persistent case state.
D/E use the same operator implementation and differ only in how many GEMMs each `compute()` launches.
Each application instance owns one case configuration and runs a one-time no-op before its
portless GPU operator. Every GPU operator uses GreedyScheduler and a CountCondition determined by
the selected GEMMs-per-tick mode.
The native target never calls the original entrypoint or `LtSgemmBench`; it reuses the helper status checks and the unchanged `LtSgemm` warmup implementation.
Its float setup mirrors the reference with resource owners that clean up partial initialization. A small adapter also tracks warmup descriptors on failure without editing the reference source.
No source under `examples/gpu_kernels/src/LtSgemm/`, its shared `Common/helpers.h`, or the CUDA Green Context example needs modification.

## Workload and interpretation

Every process runs the original full sweep: 20 cases, `M=128,256,...,65536`, `N=K=128`, L2 flush off/on, and 1,000 timed repetitions per case.
The default report highlights `M=65536, N=K=128`, flush off, but does not change the executed workload.
The native modes preserve alpha=2, beta=0, the original data initialization and 4 MiB workspace, a 4096³ warmup per case, and stream-0 matmul/event launches.
Allocation and algorithm selection occur once per case on its first tick. Statistics, output copies, and cleanup occur on its last tick.
There is no added tick-boundary synchronization. Flush-on retains the reference device synchronization after cache eviction and before every GEMM; both modes retain the initial and final case synchronization.
Both print the original statistics and wall-time format.

The original comparisons are `C/A`, `B/A`, and `C/B`. With native runs enabled, the reports also include `D/A`, `E/A`, `D/C`, `E/C`, and `E/D` for matching GPU-event, case-wall, and process-wall metrics, plus application-run ratios `D/C`, `E/C`, and `E/D`:

- Factor: candidate time / reference time.
- Slowdown: `100 * (factor - 1)` percent. Negative values are speedups.
- Absolute delta: candidate time - reference time, in the metric's named units.

GPU-event intervals, original case wall times, wrapper entrypoint time, Holoscan application-run time, and external process time have different boundaries.
GPU-event intervals exclude gaps between repetitions, including the time between Holoscan ticks.
Case wall time includes those gaps, setup, transfers, warmup, cache handling, statistics, and cleanup.
The entrypoint timer includes the full sweep and its printing; the native application-run timer
sums the 20 `app->run()` calls.
External process time includes launch, dynamic loading, startup, and shutdown.
Native runs have no original-entrypoint timer. Differences are observed execution costs, not a pure scheduler-overhead measurement.

**C hosts the whole sweep in one tick; D/E schedule the native case state over multiple ticks.**
All modes inherit the original benchmark's numerical-validation limitations.

## Build

Run from the SDK repository root. Requirements are a CUDA-capable GPU, a toolkit compatible with the unchanged baseline, a Holoscan C++ development installation, CMake, and Python 3.10+.
The baseline includes FP4/FP8 and NVTX headers; an older toolkit may not support the current sources.
Do not edit the reference to accommodate an incompatible environment.
The installed SDK used during development required CMake 3.30.4 or newer through a dependency; CMake 3.31.10 was verified.

Use the same CMake, host compiler, toolkit, build type, architecture, and CUDA runtime settings for A, B/C, and D/E:

```bash
cmake -S examples/gpu_kernels/src/LtSgemm -B build-ltsgemm-original \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-ltsgemm-original --target sample_cublasLt_LtSgemm -j 3

cmake -S examples/benchmark_ltsgemm -B build-ltsgemm-holoscan \
  -DCMAKE_PREFIX_PATH=/opt/nvidia/holoscan \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-ltsgemm-holoscan \
  --target benchmark_ltsgemm benchmark_ltsgemm_native -j 3
```

The example is also registered in the SDK's C++ examples build; enable it with `-DHOLOSCAN_BUILD_LTSGEMM_BENCHMARK=ON`.
It is opt-in there so the immutable baseline's toolkit requirements do not affect ordinary SDK example builds.
Standalone builds require the source tree; installation/packaging of this example is intentionally not provided.
To use another Holoscan build, point `holoscan_DIR` at its exported CMake package instead of selecting the installed SDK above.
Record the actual linked SDK version: the checkout's `VERSION` is not proof of the version loaded at runtime.

Audit `compile_commands.json` in both build directories before interpreting measurements.
The reference `main.cpp` remains C++11, while the wrapper and native host/operator code use Holoscan's C++17 standard.
The CUDA objects also retain the original target's default C++11 setting unless `CMAKE_CUDA_STANDARD` is explicitly supplied to both builds.
The local `main` rename and object/output paths are expected differences; investigate optimization, architecture, default-stream, include, and library differences.
The direct control does not remove the need for this audit.

## Run

Keep other GPU workloads stopped. Choose one visible GPU and use the same environment and CPU-affinity policy for every process.
The script inherits these settings, records selected relevant variables, and does not change clocks or power limits.
Set framework logging explicitly because environment settings can override the wrapper's default:

```bash
export CUDA_VISIBLE_DEVICES=0
export HOLOSCAN_LOG_LEVEL=OFF

python3 examples/benchmark_ltsgemm/scripts/run_comparison.py \
  --original build-ltsgemm-original/sample_cublasLt_LtSgemm \
  --wrapper build-ltsgemm-holoscan/cpp/benchmark_ltsgemm \
  --native build-ltsgemm-holoscan/cpp/benchmark_ltsgemm_native \
  --output /tmp/ltsgemm-smoke --trials 1

python3 examples/benchmark_ltsgemm/scripts/run_comparison.py \
  --original build-ltsgemm-original/sample_cublasLt_LtSgemm \
  --wrapper build-ltsgemm-holoscan/cpp/benchmark_ltsgemm \
  --native build-ltsgemm-holoscan/cpp/benchmark_ltsgemm_native \
  --output /tmp/ltsgemm-measured --trials 10
```

Output directories must not already exist and must be outside protected inputs.
Use a new path when repeating an experiment; existing data is never overwritten.
Copy any results needed long-term out of `/tmp`.
The default per-process timeout is 600 seconds; override it with `--timeout` if necessary.
`--original-build-dir` and `--wrapper-build-dir` can supply build locations when they cannot be found above each executable.

Without `--native`, each trial retains the three sequential A/B/C processes, rotating `ABC`, `BCA`, `CAB`.
With `--native`, each trial runs five sequential processes, rotating `ABCDE`, `BCDEA`, `CDEAB`, `DEABC`, `EABCD`.
Use `--native-build-dir` if the native build directory cannot be discovered above the executable; it requires `--native`.
Both native modes can also be run directly with the commands in the table above; `--help` and invalid-argument checks do not require GPU access.
Primary measurements must be unprofiled. For diagnosis, invoke each executable separately under a profiler and keep those results outside the reported campaign.
An apparent `C/A` slowdown with a similar `B/A` slowdown may reflect build/linking effects rather than active framework execution.

### Output files

| File | Contents |
| --- | --- |
| `manifest.json` | Commands, exit status, metadata, protected-source hashes, binary hashes, compile commands, selected CMake settings, linked libraries, GPU snapshots, and raw per-run data |
| `logs/*.stdout.log`, `logs/*.stderr.log` | Separate raw logs for every process |
| `measurements.csv` | One row per case: all original statistics, with explicit microsecond/millisecond field names |
| `processes.csv` | One row per process: external, entrypoint, and application-run timings where available; native mode and completion counts |
| `paired_ratios.csv` | Per-trial matching-boundary comparisons listed above; `C/B` only for entrypoint time |
| `summary.csv` | Median and 25th/75th percentiles of paired factors, grouped by case, cache mode, boundary, and metric |
| `summary.md` | Highlighted case and whole-process comparisons with interpretation limits |

A one-trial A/B/C smoke run has 3 successful processes and 60 case records.
A ten-trial A/B/C campaign has 30 successful processes and 600 case records.
With native modes, smoke produces 5 successful processes and 100 case records; ten trials produce 50 processes and 1,000 case records.
Each native process must report exactly 20 completed cases and 20,000 timed GEMMs.
Its separate `LT_SGEMM_NATIVE` JSON completion record includes `gemms_per_tick`, `operator_count`, `completed_cases`, `timed_gemms`, actual `compute_calls`, `app_run_wall_ms`, `completed`, and `return_code` (schema version 2).
D must report 20 compute calls; E must report 20,000. The manifest stores this as `native_record`, separately from wrapper records.
Failures report unsuccessful completion and return nonzero; invalid arguments return 2. Warmup GEMMs are excluded from the timed count.
P99 summaries describe ratios of each process's P99, not a P99 calculated by pooling unavailable samples.
Original printed precision is retained; individual event samples are not available from this interface.

Missing/duplicate cases, malformed/non-finite measurements, failed processes, timeouts, missing completion records, or changed protected inputs invalidate the experiment.
The script exits unsuccessfully, marks the manifest failed, and retains logs rather than silently dropping trials or filling missing values with zero.
Unavailable metadata probes are recorded as unavailable; inspect them before making attribution claims.
The script records competing GPU activity but does not reserve the GPU or guarantee that it remained idle between snapshots.

### Native operator count sweep

`scripts/run_native_operators.sh` runs the operator counts configured in the script and saves
each run under `results/app-run-*`. Its `benchmark-env/bin/python3` needs Matplotlib (and a compatible
NumPy installation); the script checks that Matplotlib imports before starting GPU work.
Each run folder contains `comparison.csv`, `median_per_operator_gemm_latency.png` (the median
of operator GEMM medians, in microseconds), and `median_per_operator_latency.png` (the median
operator wall interval, in seconds). `p99_per_operator_gemm_latency.png` shows every operator's
printed GEMM P99 latency for each run and the median at each operator count. All plots are made
from the CSV after summarization and
before report generation. To plot an older run folder, rerun
`python3 scripts/summarize_native_operators.py <run-folder>` to add the new CSV column, then run
`benchmark-env/bin/python3 scripts/plot_native_operator_latencies.py <run-folder>`.
The CSV's `operator_p99_us` cell is a JSON array of the printed GEMM P99 latencies, ordered by
operator number; `median_operator_p99_us` remains their median.

## Tests

Pure parser, aggregation, output-safety, failure, and subprocess tests do not need a GPU.
Native statistics tests compile with a C++17 compiler (`CXX`, default `c++`) and require
neither CUDA nor Holoscan. They cover equal samples, population standard deviation,
interpolated percentiles, and single/empty sample sets:

```bash
python3 -m unittest discover -s examples/benchmark_ltsgemm/tests -v
```

Set the wrapper and native paths to additionally test help, invalid arguments, default-mode selection, and CUDA failure propagation with the GPU hidden.
Enable full-sweep integration tests explicitly:

```bash
LTSGEMM_WRAPPER_BINARY="$PWD/build-ltsgemm-holoscan/cpp/benchmark_ltsgemm" \
LTSGEMM_ORIGINAL_BINARY="$PWD/build-ltsgemm-original/sample_cublasLt_LtSgemm" \
LTSGEMM_NATIVE_BINARY="$PWD/build-ltsgemm-holoscan/cpp/benchmark_ltsgemm_native" \
LTSGEMM_GPU_TESTS=1 HOLOSCAN_LOG_LEVEL=OFF \
python3 -m unittest discover -s examples/benchmark_ltsgemm/tests -v
```

The tests never shorten or alter the original sweep.
Concurrency, stream-aware derivatives, and Green Context partitioning remain separate experiments.

## Native source layout

`cpp/benchmark_ltsgemm_native.cpp` handles CLI outcomes, invokes the application, and reports
the final status. Its private modules in `cpp/native/` are compiled only into the native executable:

| Module | Responsibility |
| --- | --- |
| `types.hpp`, `timing.hpp` | Small configuration/result types, workload constants, and steady-clock timing |
| `support.*` | Argument parsing, timing aggregation, and output formatting |
| `execution.*` | `BenchmarkCase` reuses `TestBench<float>` for matrices, transfers, and their CUDA resources, and owns matmul setup, warmup, events, and cleanup; CUDA types stay in the implementation |
| `application.*` | Holoscan application composition, tick scheduling, shared run results, and application timing |

The application and operator receive immutable options separately from mutable run results.
Each tick delegates to `BenchmarkCase::launch()`; the final tick calls `finish()`, destroys the
case's host vectors, then ends the case wall timer and reports the result. Application timing is
the sum of the 20 `app->run()` calls. The CUDA warmup adapter remains in `cpp/native_warmup.cu`.
These modules add no SDK API and do not change the reference executable or wrapper.

## Recorded experiment

See [RESULTS.md](RESULTS.md) for the initial ten-trial Thor/Holoscan 4.4.0 comparison, including timing variability and environment limitations.
See [NATIVE_RESULTS.md](NATIVE_RESULTS.md) for the subsequent five-mode comparison, native tick counts, and failure-path validation.
