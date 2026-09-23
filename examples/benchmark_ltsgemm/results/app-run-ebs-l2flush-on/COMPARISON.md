# Native LtSgemm operator-count comparison

## Result

All seven runs completed successfully for `--operators={1,2,4,8,16,32,64}`. Each operator completed one `M=65536, N=128, K=128, L2 flush=on` case containing 1,000 timed GEMMs. With `--gemms-per-tick=1`, each timed GEMM used a separate compute call. The sweep therefore completed 127 cases and 127,000 timed GEMMs.

Application time scaled nearly linearly from 2 through 64 operators: a least-squares fit over those counts is approximately `0.258 + 0.6761 * operators` seconds (`R²=0.999996`). The one-operator run was startup dominated: its 2.066 s application time exceeded the two-operator run's 1.621 s. Consequently, the 64-operator process took 19.118x as long as the one-operator process while performing 64x as many timed GEMMs.

Application throughput increased from 484.1 timed GEMMs/s at one operator to 1,469.2 GEMMs/s at 64 operators, a 3.035x increase. From two operators onward, throughput rose more gradually from 1,234.0 to 1,469.2 GEMMs/s.

Per-operator CUDA-event means were broadly stable. The median operator mean ranged from 524.466 to 540.719 us across the sweep, a 3.06% span relative to the across-run average. Some operators in the 16-, 32-, and 64-operator logs had rare long events that raised their means and coefficients of variation even though their medians and P99 values remained much closer to the other operators.

Peak RSS rose from 0.664 GiB at one operator to 5.643 GiB at 64 operators. Peak RSS measures system RAM used by the process, not GPU memory.

## Comparison

| Operators | Cases | Timed GEMMs | App wall (s) | Process wall (s) | vs 1-op process | GEMMs/s (app) | Median mean (us) | Median P99 (us) | Sum GPU event / app | Peak RSS (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1,000 | 2.066 | 2.29 | 1.000x | 484.1 | 530.715 | 585.734 | 25.69% | 0.664 |
| 2 | 2 | 2,000 | 1.621 | 1.79 | 0.782x | 1,234.0 | 524.466 | 536.981 | 64.72% | 0.914 |
| 4 | 4 | 4,000 | 2.977 | 3.17 | 1.384x | 1,343.5 | 529.648 | 549.577 | 70.91% | 1.416 |
| 8 | 8 | 8,000 | 5.686 | 5.88 | 2.568x | 1,407.1 | 526.311 | 537.270 | 74.04% | 2.417 |
| 16 | 16 | 16,000 | 11.064 | 11.26 | 4.917x | 1,446.1 | 540.719 | 552.212 | 81.16% | 3.670 |
| 32 | 32 | 32,000 | 21.833 | 22.04 | 9.624x | 1,465.7 | 537.418 | 534.465 | 80.71% | 3.672 |
| 64 | 64 | 64,000 | 43.560 | 43.78 | 19.118x | 1,469.2 | 527.026 | 532.717 | 78.47% | 5.643 |

`Median mean` is the median of the operators' mean CUDA-event times; `Median P99` is the median of their P99 times. Each operator statistic covers 1,000 timed GEMMs.

`Sum GPU event / app` is the sum of all operators' reported CUDA-event totals divided by application wall time. Summing event durations can count overlapping intervals more than once and omits work outside those events, so this ratio is neither a GPU utilization measurement nor a concurrency measurement.

The two-operator result finishing faster than the one-operator result shows that fixed startup effects materially affect the smallest run. Comparisons against the one-operator baseline should therefore be interpreted cautiously.

## Run configuration

- Run-directory timestamp: `2026-09-22_10-11-45_499358053`
- Operator counts: 1, 2, 4, 8, 16, 32, and 64
- Scheduler: Event-Based Scheduler with 13 threads
- GEMMs per tick: 1
- Case: `M=65536, N=128, K=128, L2 flush=on`
- Timed GEMMs: 1,000 per operator
- Compute calls: 1,000 per operator
- Reported L2 cache size: 33,554,432 bytes (32.00 MiB)
- Result schema: `LT_SGEMM_NATIVE` version 2
- Completion: every run reported `completed=true`, `return_code=0`, and external exit status 0
- Standard error: empty for every run

The run artifacts do not record the GPU model, driver version, binary hash, build options, command environment, or source revision, so those details cannot be established from this run.

This was a single-trial sweep with one observation at each operator count. The results describe this execution, but small differences between counts and the apparent throughput plateau are not statistically conclusive. Repeated randomized trials would be required to estimate run-to-run variance, confidence intervals, and stable scaling trends.

## Artifacts

- `comparison.csv`: machine-readable aggregate measurements and validation fields
- `run-operators-N.stdout.log`: complete benchmark output and per-operator measurements for each operator count
- `run-operators-N.stderr.log`: standard error logs; all are empty
- `run-operators-N.time.txt`: external process wall time, peak system RSS, and exit status for each run