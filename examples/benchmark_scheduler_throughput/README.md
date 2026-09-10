# Scheduler Throughput Benchmark

This application benchmarks scheduler throughput by counting simple increment operations over a
period of time using EventBasedScheduler. Each trial has 16 operators and uses
1, 2, 4, 8, 10, 12, 14, or 16 worker threads.

## Run the Benchmark

From the build directory:

```bash
./examples/benchmark_scheduler_throughput/cpp/benchmark_scheduler_throughput
```

Optional workload and scheduler flags (independently combinable):

```bash
./examples/benchmark_scheduler_throughput/cpp/benchmark_scheduler_throughput \
  --busy_wait \
  --enable_queue_stealing \
  --enable_postcheck_fastpath
```

- `--enable_queue_stealing`: Enables the `enable_queue_stealing` argument on the
  `EventBasedScheduler`.
- `--enable_postcheck_fastpath`: Enables the
  `enable_worker_postcheck_fastpath` argument on the `EventBasedScheduler`.
- `--busy_wait`: Actively polls `std::chrono::steady_clock` for at least 1 ms before
  each increment, without sleeping or yielding. Runs 1,000 operations per operator
  for exactly 16,000 operations per trial, matching the greedy busy-wait workload.
  Without this flag, `CountOp` runs 100,000 operations per operator (1,600,000 total).
- `--help` / `-h`: Prints usage without running trials.
- If either scheduler flag is omitted, the corresponding scheduler argument remains `false`.

## Metrics

For each trial, the benchmark results will display:

- **Threads**: The number of worker threads used by the scheduler.
- **Operators**: The number of (unconnected) operators in the graph.
- **Total Operations**: The total number of operations performed across all operators.
- **Throughput (ops / s)**: The number of operations performed per second across all operators.

For the combined 10-configuration CSV and six-panel comparison figure, see
[`benchmark_scheduler.py`](../../scripts/README.md#benchmark_schedulerpy).
