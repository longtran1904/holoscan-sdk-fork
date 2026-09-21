# Native LtSgemm comparison

Measured on 2026-09-20 on NVIDIA Thor with ten sequential, unprofiled A/B/C/D/E trials: **50 successful processes and 1,000 complete case records**. Each process ran the full 20-case sweep with 1,000 timed GEMMs per case. Orders rotated ABCDE, BCDEA, CDEAB, DEABC, EABCD twice.

A is the original executable, B the direct wrapper, C the single-tick Holoscan wrapper, D the native 1,000-GEMMs-per-tick mode, and E the native one-GEMM-per-tick mode. All ten D runs reported exactly 20 compute calls; all ten E runs reported exactly 20,000. Every native run completed 20 cases and 20,000 timed GEMMs.

## Paired comparisons

Values are medians of ratios paired within each trial, not ratios of separately summarized times. GPU-event mean and case-wall columns use `M=65536, N=K=128`, L2 flush off. Process and application columns cover the entire sweep.

| Comparison | GPU-event mean | Case wall | Process wall | Application run |
| --- | --- | --- | --- | --- |
| D/A | 0.9990x | 0.9994x | 1.0193x | — |
| E/A | 0.9990x | 0.9982x | 1.0078x | — |
| D/C | 1.0002x | 1.0004x | 1.0012x | 1.0026x |
| E/C | 0.9996x | 1.0014x | 0.9969x | 0.9980x |
| E/D | 0.9995x | 0.9988x | 0.9944x | 0.9958x |

For E/D, the interquartile ranges were:

| Metric | Factor 25th–75th percentile |
| --- | --- |
| mean_us | 0.9985–1.0019x |
| case_wall_ms | 0.9940–1.0037x |
| process_wall_ms | 0.9904–0.9995x |
| app_run_wall_ms | 0.9892–0.9995x |

GPU-event intervals exclude gaps between repetitions. Case wall time includes those gaps, setup, transfers, warmup, cache handling, statistics, and cleanup. Native and wrapper host implementations also differ. These are observed execution costs, not a pure scheduler-overhead measurement or an isolated per-tick penalty. Native runs have no original-entrypoint timing. No performance threshold was imposed.

## Validation and environment

- All 39 parser, campaign, CLI, and GPU integration tests passed, including the unchanged A/B/C tests. The five-process smoke campaign produced 100 case records; the ten-trial campaign produced complete finite statistics and reports.
- Fourteen separate failure probes (seven injected API failures in each native mode) returned unsuccessful completion with nonzero exit status and no remaining tracked allocations, streams, handles, descriptors, or events. They covered partial matrix allocation, case and warmup descriptor creation, event creation, and cleanup failures. These probes ran separately from measured processes.
- GCC 13.3.0, CUDA 13.2.78, cuBLAS 13.4.0.1, CMake 3.31.10; Release `-O3 -DNDEBUG`, explicit CUDA architecture `110`, shared CUDA runtime. The reference main/CUDA objects and native warmup adapter use C++11; the Holoscan wrapper and native operator use C++17.
- All three executables resolve the same CUDA runtime and cuBLASLt libraries. C/D/E link installed Holoscan **4.4.0**. The native binary contains neither the original benchmark entrypoint nor `LtSgemmBench`.
- `CUDA_VISIBLE_DEVICES=0`, `HOLOSCAN_LOG_LEVEL=OFF`; inherited CPU affinity covered CPUs 0–13. Clocks and power settings were not changed. GPU clock readings were unavailable; inspect the retained GPU snapshots before making attribution claims.
- Reference sources, the pre-existing example registration, and the original `RESULTS.md` matched all 20 pre-task protected hashes. Formatting and Python lint checks passed.

## Artifacts

The raw logs, manifests, binary hashes, build metadata, and CSV/Markdown reports are archived in `.cache/benchmark_ltsgemm/2026-09-20-native/`, with `smoke/`, `measured/`, and `validation/` subdirectories. This directory is ignored by Git; preserve it separately when sharing results. Manifests retain their original execution paths under `/tmp/ltsgemm-native-81nsh5ed/`. The smoke run preceded the final cleanup-error fix; the full GPU tests and ten-trial campaign used the final native binary.

Build artifacts remain in `/tmp/ltsgemm-native-81nsh5ed/original/` and `/tmp/ltsgemm-native-81nsh5ed/wrapper/`. Validation artifacts include the failure-probe source, logs, source hashes, resolved library paths, and test results. The final report writer reproduces every native-enabled measured report byte for byte.

The [initial A/B/C results](RESULTS.md) are preserved. See [README.md](README.md) for build, run, and test commands.
