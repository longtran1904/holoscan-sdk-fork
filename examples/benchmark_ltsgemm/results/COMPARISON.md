# Native LtSgemm operator-count comparison

## Result

All ten runs completed successfully for `--operators={10,20,...,100}`. Each operator completed one `M=65536, N=K=128, L2 flush=on` case with 1,000 timed GEMMs. The default `--gemms-per-tick=1` was retained, so every timed GEMM was a separate compute call.

Process time scaled nearly linearly with operator count: a least-squares fit is approximately `1.311 + 0.7418 * operators` seconds (`R²=0.99980`). The 100-operator process took 75.55 s, or 8.212x the 10-operator process's 9.20 s while doing 10x the timed GEMMs.

Per-operator CUDA-event time was stable. The median operator mean ranged from 522.752 to 530.182 us across the ten runs, a 1.42% span relative to the across-run average. Application-level throughput rose from 1,117.6 timed GEMMs/s at 10 operators to 1,327.5 timed GEMMs/s at 100 operators, with most counts from 30 onward near 1,306-1,329 GEMMs/s.

Peak RSS also scaled linearly: approximately `626,344 + 67,350 * operators` KiB (`R²=0.9999999`), reaching 7.020 GiB at 100 operators.

## Comparison

| Operators | App wall (s) | Process wall (s) | vs 10-op process | GEMMs/s (app) | Median mean (us) | Median P99 (us) | Sum GPU / app | Max RSS (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 8.947 | 9.20 | 1.000x | 1117.6 | 530.182 | 550.026 | 59.46% | 1.240 |
| 20 | 16.063 | 16.31 | 1.773x | 1245.1 | 525.241 | 543.431 | 65.67% | 1.882 |
| 30 | 22.732 | 22.94 | 2.493x | 1319.7 | 522.752 | 539.128 | 69.11% | 2.524 |
| 40 | 30.630 | 30.84 | 3.352x | 1305.9 | 524.822 | 540.307 | 68.63% | 3.166 |
| 50 | 38.305 | 38.51 | 4.186x | 1305.3 | 524.578 | 539.892 | 68.57% | 3.808 |
| 60 | 45.478 | 45.69 | 4.966x | 1319.3 | 523.511 | 539.649 | 69.15% | 4.452 |
| 70 | 52.689 | 52.90 | 5.750x | 1328.6 | 522.916 | 538.625 | 69.55% | 5.094 |
| 80 | 60.629 | 60.85 | 6.614x | 1319.5 | 524.435 | 539.058 | 69.29% | 5.736 |
| 90 | 68.077 | 68.29 | 7.423x | 1322.0 | 524.771 | 540.370 | 69.44% | 6.378 |
| 100 | 75.328 | 75.55 | 8.212x | 1327.5 | 524.538 | 539.155 | 69.71% | 7.020 |

`Sum GPU / app` is the sum of all operators' reported CUDA-event totals divided by application wall time. It is a campaign-level utilization indicator, not a concurrency measurement. The per-operator `GEMM/wall` percentage in the raw output falls as operators increase because each operator's case wall interval includes time spent executing the other operators.

## Run configuration

- Date: 2026-09-20, America/Chicago
- Command pattern: `CUDA_VISIBLE_DEVICES=0 HOLOSCAN_LOG_LEVEL=OFF benchmark_ltsgemm_native --operators=N`
- Operator counts: 10 through 100 in increments of 10, run sequentially once each
- GEMMs per tick: 1 (program default)
- Case: `M=65536, N=128, K=128, L2 flush=on`
- Repetitions: 1,000 per operator
- Scheduler: Holoscan `GreedyScheduler`
- GPU: NVIDIA Thor; driver 595.78; CUDA 13.2 reported by `nvidia-smi`
- Build: Release, `-O3 -DNDEBUG -march=native -fopt-info-vec-optimized`
- Linked runtime: Holoscan 4 libraries from `/opt/nvidia/holoscan`, CUDA 13.2 `libcudart.so.13` and `libcublasLt.so.13`
- Git HEAD: `640cb4b4a9305e695fa5413e17d52583837adbcd`
- Binary SHA-256: `e3bf6b71af7b58c2f79456e0010c727325d5f4a12aa72cf1912d979116944fd2`

The worktree contained pre-existing uncommitted benchmark changes, and this report characterizes the exact binary hash above. The binary was already newer than the native source files, so it was not rebuilt.

## Artifacts

- `comparison.csv`: machine-readable comparison and validation fields
- `operators-NNN.stdout.log`: complete native benchmark output for each count
- `operators-NNN.stderr.log`: stderr for each count (all are empty)
- `operators-NNN.time.txt`: external process time, CPU time, peak RSS, and exit status

This is a single-trial scaling sweep. Treat small differences between operator counts as descriptive rather than statistically conclusive.
