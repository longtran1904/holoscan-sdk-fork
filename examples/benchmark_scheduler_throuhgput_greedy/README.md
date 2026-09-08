# Greedy Scheduler Throughput Benchmark

This application benchmarks `GreedyScheduler` throughput by counting simple increment operations
over a period of time. Each trial changes the number of independent operators while keeping the
scheduler single-threaded. The trials use 1, 2, 4, 8, and 16 operators, each executing
100,000 operations.

## Run the Benchmark

From the build directory:

```bash
./examples/benchmark_scheduler_throuhgput_greedy/cpp/benchmark_scheduler_throuhgput_greedy
```

Use `--busy_wait` to select `BusyWaitCountOp`, which actively waits for at least 1 ms
before each increment:

```bash
./examples/benchmark_scheduler_throuhgput_greedy/cpp/benchmark_scheduler_throuhgput_greedy --busy_wait
```

This mode uses 1,000 operations per operator, so the full sweep takes at least 31 seconds.
The operator continuously polls `std::chrono::steady_clock`; it does not sleep or yield,
keeping the CPU busy while it waits. The operating system can still preempt the thread.
Without this flag, the benchmark uses the original `CountOp` and 100,000 operations per operator.

`--help` (or `-h`) prints usage.

## Metrics

For each trial, the benchmark reports:

- **Operators**: The number of independent operators in the graph.
- **Total Operations**: The total number of operations performed across all operators.
- **Throughput (ops / s)**: The number of operations performed per second across all operators.

Timing uses `std::chrono::steady_clock`, from the earliest operator start to the latest
operator stop, matching the reference throughput benchmark.
