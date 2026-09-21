# Initial isolated LtSgemm comparison

Measured on 2026-09-19 with ten paired A/B/C trials: 30 successful processes and 600 complete case records.
All processes executed the unchanged 20-case sweep with 1,000 timed repetitions per case.
Orders rotated ABC, BCA, CAB; runs were sequential and unprofiled.

## Primary case

For `M=65536, N=K=128`, L2 flush off, the median paired Holoscan/original event-mean ratio was **0.9995x (-0.05%)**.
The interquartile range straddles 1.0x; this run does not show a clear GPU-event slowdown for that case.

| Comparison | Boundary/metric | Median factor | Factor 25th-75th percentile | Median slowdown |
| --- | --- | --- | --- | --- |
| C/A | GPU event mean, primary case | 0.9995x | 0.9989-1.0001 | -0.05% |
| C/A | GPU event median, primary case | 0.9990x | 0.9983-1.0000 | -0.10% |
| C/A | GPU event P99, primary case | 0.9996x | 0.9989-1.0004 | -0.04% |
| C/A | Case wall time, primary case | 1.0002x | 0.9979-1.0020 | +0.02% |
| B/A | GPU event mean, primary case | 1.0005x | 1.0002-1.0022 | +0.05% |
| C/B | GPU event mean, primary case | 0.9987x | 0.9977-0.9993 | -0.13% |
| C/A | Whole-process wall time, full sweep | 1.0200x | 1.0094-1.0733 | +2.00% |
| B/A | Whole-process wall time, full sweep | 1.0043x | 0.9949-1.0149 | +0.43% |
| C/B | Whole-process wall time, full sweep | 1.0158x | 1.0064-1.1005 | +1.58% |
| C/B | Entrypoint wall time, full sweep | 0.9861x | 0.9745-1.0769 | -1.39% |

A is the original executable; B is the direct-call wrapper; C is the single-tick Holoscan wrapper.
Factors are paired within each trial before summarizing; they are not ratios of independently summarized times.
The median process durations were 5,940.479 ms (A), 5,971.648 ms (B), and 6,103.526 ms (C).

The process-time increase includes loading, startup/shutdown, and the complete sweep, including flush-on cases.
Its variability is much larger than that of the primary GPU-event metric; do not interpret the +2.00% as pure scheduler cost or as a per-GEMM penalty.
The benchmark makes only one Holoscan operator invocation per process.
P99 comparisons summarize per-run P99 ratios, not pooled event samples.

## Environment and controls

- GPU: NVIDIA Thor, compute capability 11.0; driver 595.78.
- Linked framework: installed Holoscan **4.4.0**, not the checkout's 4.6.0 development build.
- Compiler/toolkit: GCC 13.3.0, CUDA 13.2.78, cuBLAS 13.4.0.1, CMake 3.31.10.
- Both reference builds: Release, `-O3 -DNDEBUG`, native CUDA architecture, C++11 reference main and CUDA translation units, shared CUDA runtime. The wrapper itself uses C++17.
- The renamed main symbol and object paths are intentional differences. Reference includes resolve to the same directories; both executables resolve the same CUDA runtime and cuBLASLt libraries.
- `CUDA_VISIBLE_DEVICES=0`, `HOLOSCAN_LOG_LEVEL=OFF`; inherited CPU affinity covered CPUs 0-13, without explicit pinning.
- GPU snapshots showed no competing compute processes. Temperature rose from 33 to 44 degrees Celsius; SM/memory clocks were unavailable from `nvidia-smi` and were not locked or changed.
- All 18 protected-file hashes matched the pre-implementation snapshot and the pre/post-experiment manifests, including the user's dirty and untracked baseline files.
- Checkout revision: `c6a976a7da96beb680b8901a868c492db06b7fb8`; the dirty-state and file hashes in the manifest, not this revision alone, identify the workload.

These results describe this GPU, workload, environment, and benchmark-hosting design.
They do not establish overhead for other shapes, per-GEMM scheduling, other SDK versions, concurrent execution, or Green Context partitioning.
The full per-shape/cache-mode results remain available in the archived CSV files.

## Local artifacts

The generated measurement files and raw logs are archived at `.cache/benchmark_ltsgemm/2026-09-19-initial/` relative to the repository root.
That directory is intentionally ignored by Git; preserve it separately when sharing the experiment.
It contains `manifest.json`, `measurements.csv`, `processes.csv`, `paired_ratios.csv`, `summary.csv`, `summary.md`, and `logs/`.
The manifest retains the original execution-time paths under `/tmp/ltsgemm-overhead-YYkk3d/`; archived logs have the same filenames under `logs/`.
Build artifacts remain in that temporary directory's `original/` and `wrapper/` subdirectories.

Reproduce using the commands in [README.md](README.md), with a new output directory and the intended Holoscan installation selected explicitly.
