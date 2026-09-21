# Million-GEMM LtSgemm comparison

Measured on NVIDIA Thor on 2026-09-20 with **1,000,000 timed GEMMs per process**, split across the existing 20 cases: **50,000 per case**. Ten paired trials completed successfully: 50 sequential processes, 1,000 case records, and 50,000,000 timed GEMMs in total. Warmup GEMMs are additional and excluded from these counts.

Orders rotated ABCDE, BCDEA, CDEAB, DEABC, EABCD twice. The unprofiled campaign ran from 16:24:25 to 19:06:08 UTC (about 2 hours 42 minutes).

A is the original standalone executable; B calls the original entrypoint through the direct wrapper; C calls the whole sweep in one Holoscan compute call. D is the native operator launching 1,000 GEMMs per tick; E launches one GEMM per tick. Each native mode retains one case’s resources across its ticks.

## Measured times

Each value below is the median of ten runs. Process and application times cover the full sweep; GPU-event mean and case wall use M=65536, N=K=128, L2 flush off.

| Version | Compute calls/process | Process wall (s) | Application run (s) | GPU-event mean (µs) | Case wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | — | 194.254 | — | 510.037 | 25.715 |
| B | — | 194.054 | — | 510.268 | 25.724 |
| C | 1 | 193.695 | 193.449 | 509.629 | 25.693 |
| D | 1,000 | 193.981 | 193.749 | 509.525 | 25.687 |
| E | 1,000,000 | 194.418 | 194.177 | 510.258 | 25.722 |

## Paired differences

These are medians of percentage differences paired within each trial, not ratios of the separately summarized times above. Negative values mean the candidate took less time. Pairwise medians need not be transitive.

| Comparison | GPU-event mean | Case wall | Process wall | Application run |
| --- | ---: | ---: | ---: | ---: |
| B/A | +0.036% | +0.036% | -0.229% | — |
| C/A | -0.075% | -0.059% | -0.084% | — |
| C/B | -0.099% | -0.091% | -0.008% | — |
| D/A | -0.086% | -0.082% | -0.046% | — |
| E/A | +0.068% | +0.058% | -0.245% | — |
| D/C | +0.012% | -0.003% | -0.099% | -0.100% |
| E/C | +0.098% | +0.091% | -0.088% | -0.091% |
| E/D | +0.121% | +0.120% | +0.476% | +0.459% |

C and D remain very close in this workload. The paired D/C process difference was −0.099% (median delta −0.194 seconds); its GPU-event mean differed by +0.012%. E/D process time was +0.476% (median delta +0.922 seconds).

| Process comparison | 25th–75th percentile paired difference |
| --- | ---: |
| D/C | -0.862% to +0.459% |
| E/C | -0.371% to +0.574% |
| E/D | -0.173% to +0.724% |

GPU-event intervals exclude gaps between repetitions. Case wall time includes those gaps, setup, copies, warmup, cache handling, statistics, and cleanup. The native and wrapper host implementations also differ, so these are observed execution costs, not a pure scheduler-overhead measurement. Native runs have no original-entrypoint timing. No performance threshold was imposed.

## Workload and verification

Only repetition constants, expected completion counts, test fixtures, and report labels were adjusted in an isolated source copy. M still doubles from 128 through 65,536, N=K=128, L2 flushing is off/on, alpha=2, beta=0, workspace is 4 MiB, and each case retains its 4096³ warmup. Matrix initialization, GEMM implementations, event boundaries, and synchronization behavior are unchanged. The repository benchmark sources and earlier results retain their original hashes.

- All 50 processes returned zero and produced every expected case with finite statistics. All 20 native records reported 20 cases and 1,000,000 timed GEMMs; every D run had 1,000 compute calls and every E run had 1,000,000.
- Every case’s printed mean agrees with its GPU-event total divided by 50,000 within output rounding. All comparison groups contain ten paired trials. Regenerating the reports produced byte-identical files.
- The adjusted parser, campaign, and CLI suite passed: 39 tests run, 2 full GPU integration tests skipped. The measured campaign supplied the full real-GPU validation. Help, invalid arguments, and CUDA failure handling were tested before measurement.
- GCC 13.3.0, CUDA 13.2.78, cuBLAS 13.4.0.1, installed Holoscan 4.4.0, driver 595.78. All builds use Release `-O3 -DNDEBUG`, CUDA architecture `110`, and shared CUDA runtime. Reference main/CUDA objects and the native warmup adapter use C++11; Holoscan host code uses C++17.
- `CUDA_VISIBLE_DEVICES=0`, `HOLOSCAN_LOG_LEVEL=OFF`, inherited CPU affinity 0–13. Clock and power settings were unchanged. Clock readings were unavailable; no other compute applications appeared in the before/after GPU snapshots.

## Saved artifacts

The complete experiment is archived in [`.cache/benchmark_ltsgemm/2026-09-20-million/`](../../.cache/benchmark_ltsgemm/2026-09-20-million/). It contains binaries and builds, the exact source snapshot, the workload patch and preparation script, test logs, validation results, and build provenance. This directory is ignored by Git; preserve it separately when sharing the report.

- [Full comparison report](../../.cache/benchmark_ltsgemm/2026-09-20-million/measured/summary.md)
- [All shapes and cache modes](../../.cache/benchmark_ltsgemm/2026-09-20-million/measured/summary.csv)
- [Individual process measurements](../../.cache/benchmark_ltsgemm/2026-09-20-million/measured/processes.csv)
- [Raw manifest and log paths](../../.cache/benchmark_ltsgemm/2026-09-20-million/measured/manifest.json)
- [Workload patch](../../.cache/benchmark_ltsgemm/2026-09-20-million/workload.patch)
- [Build and experiment metadata](../../.cache/benchmark_ltsgemm/2026-09-20-million/experiment.json)

Execution paths in metadata point to `/tmp/ltsgemm-million-rnxp3_vp/`, which remains available. The archived `source/` includes the comparison parser for the larger workload; the repository’s default parser still expects 20,000 GEMMs per process. The archived CMake caches retain their original absolute build paths; use a fresh build directory when rebuilding elsewhere.

The [earlier native results](NATIVE_RESULTS.md) and [initial A/B/C results](RESULTS.md) are preserved.
