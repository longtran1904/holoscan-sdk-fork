# Native LtSgemm operator-count comparison

## Result

All seven runs completed successfully for `--operators={1,2,4,8,16,32,64}`. Every operator completed one `M=65536, N=128, K=128, L2 flush=off` case with 1,000 timed GEMMs. One GEMM was issued per compute call. All processes returned status 0, and every stderr log was empty.

Application throughput increased from 501.2 timed GEMMs/s with one operator to 1,789.4 GEMMs/s with 64 operators, a 3.571x increase while the timed workload grew by 64x. Most of the gain occurred at lower counts: throughput increased only 1.34% from 16 to 64 operators.

The 64-operator application wall time was 35.766 s, 17.925x the one-operator time. External process wall time was 35.96 s, 15.982x the one-operator time. The two-operator run was shorter than the one-operator run despite performing twice as many timed GEMMs; a single trial cannot separate repeatable scaling behavior from startup, scheduling, or run-to-run effects.

Reported per-operator CUDA-event durations increased and developed broad tails as the count rose. The median operator mean increased from 509.945 us at one operator to 3,603.020 us at 64 operators, a 7.066x increase. Median operator P99 rose from 521.314 us to 18,898.815 us, although it was not monotonic across the sweep.

Peak RSS increased from 0.664 GiB to 5.812 GiB. Peak RSS measures system RAM consumed by the process; it is not GPU memory usage.

## Comparison

| Operators | Completed cases | Timed GEMMs | App wall (s) | Process wall (s) | Process vs 1 op | GEMMs/s | Median operator mean (us) | Operator mean range (us) | Median P99 (us) | Sum GPU events / app | Median GEMM/wall | Peak RSS (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1,000 | 1.995 | 2.25 | 1.000x | 501.2 | 509.945 | 509.945–509.945 | 521.314 | 25.56% | 73.286% | 0.664 |
| 2 | 2 | 2,000 | 1.377 | 1.56 | 0.693x | 1,451.9 | 666.686 | 650.061–683.311 | 1,050.649 | 96.80% | 55.987% | 0.914 |
| 4 | 4 | 4,000 | 2.495 | 2.69 | 1.196x | 1,602.9 | 1,174.807 | 582.327–1,242.665 | 3,124.844 | 167.29% | 50.816% | 1.415 |
| 8 | 8 | 8,000 | 4.726 | 4.92 | 2.187x | 1,692.8 | 1,930.742 | 1,474.517–1,985.551 | 6,996.192 | 316.45% | 42.512% | 2.417 |
| 16 | 16 | 16,000 | 9.061 | 9.25 | 4.111x | 1,765.8 | 2,921.872 | 1,466.394–3,576.418 | 13,117.956 | 469.16% | 36.751% | 3.671 |
| 32 | 32 | 32,000 | 17.994 | 18.20 | 8.089x | 1,778.4 | 3,037.412 | 1,797.756–5,956.646 | 19,264.899 | 596.62% | 17.728% | 4.506 |
| 64 | 64 | 64,000 | 35.766 | 35.96 | 15.982x | 1,789.4 | 3,603.020 | 2,673.052–4,205.433 | 18,898.815 | 640.09% | 10.691% | 5.812 |

`Sum GPU events / app` is the sum of every operator’s reported CUDA-event total divided by application wall time. Because overlapping operator totals are added, the value can exceed 100%. It is an arithmetic aggregate and does not measure GPU utilization or concurrency.

The per-operator `GEMM/wall` percentage falls as the operator count increases. Each operator’s case wall interval spans time during which work from other operators may also execute, so these percentages cannot be added or interpreted as shares of total application time.

## Run configuration

- Run directory timestamp: `2026-09-21_21-01-34_106264293` (timezone is not recorded in the artifacts)
- Operator counts: 1, 2, 4, 8, 16, 32, and 64, run once each in ascending order
- Scheduler: Event-Based Scheduler with 13 threads
- GEMMs per tick: 1
- Case: `M=65536, N=128, K=128`
- L2 flushing: disabled
- Reported L2 cache size: 33,554,432 bytes (32.00 MiB)
- Timed repetitions: 1,000 per operator
- Compute calls: one per timed GEMM
- Total timed workload: 127,000 GEMMs across the sweep
- Process results: all exit status 0; all stderr logs empty

The artifacts do not record the exact shell command, GPU model, driver version, CUDA version, build flags, git revision, or binary hash. Those details cannot be established from this run alone.

This is a single-trial sweep with no repeated observations at any operator count. The results describe this execution, but small differences, non-monotonic values, and apparent scaling trends are not statistically conclusive. Repeated trials would be required to estimate variability or confidence intervals.

## Artifacts

- `comparison.csv`: aggregated measurements and validation fields for all seven operator counts
- `run-operators-N.stdout.log`: complete benchmark output for each operator count
- `run-operators-N.stderr.log`: stderr for each count; all seven files are empty
- `run-operators-N.time.txt`: external process wall time, peak RSS, and exit status for each count