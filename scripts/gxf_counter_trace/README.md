# Periodic GXF counter tracing

This diagnostic `LD_PRELOAD` library samples the existing atomic counters in
`nvidia::gxf::EventBasedScheduler`. It does not add counters or intercept individual
steal attempts. Only `counter_reader.cpp` uses GCC `-fno-access-control`, allowing
the compiler to calculate private member addresses from the installed header.
GXF itself and the working benchmark source are unchanged.

The runner currently supports the validated CUDA13/aarch64 GXF build and the
original results-6 source snapshot. It refuses other header/library hashes. The
shim also checks the loaded library's ELF build ID before reading private members.
These checks identify the validated build; this is not a portable GXF ABI.

Run the complete comparison from the repository root:

```bash
python3 -s scripts/gxf_counter_trace/run.py
```

The default container is `scheduler-graphs-rerun`, with the repository mounted in
it. The container must already have permission to set dispatcher FIFO priority.
Benchmark children run as container root. The runner builds into a new
`build-cuda13/benchmark-steal-trace-<UTC timestamp>/` directory and runs:

- 13 workers, 16 operators, 1,000 executions per operator, 1 ms busy wait;
- queue stealing only versus both queue stealing and postcheck fastpath;
- dispatcher CPU 13, `SCHED_FIFO`, priority 99;
- five repetitions of each configuration, each with an uninstrumented baseline;
- alternating configuration order and alternating baseline/trace order;
- 10 ms sampling, retrying at 20 then 50 ms if the absolute median throughput
  change reaches 5%. Other benchmark parameters remain identical.

CPU 13 is a Linux logical CPU ID. Pinning the dispatcher does not exclude workers
from that CPU. The sampler uses `SCHED_OTHER` and inherits the caller's affinity;
it does not add affinity restrictions. No other CPU-heavy tests run concurrently
with the measurements.

For build-only use:

```bash
python3 -s scripts/gxf_counter_trace/run.py --build-only --output build-cuda13/my-trace-build
python3 -s scripts/gxf_counter_trace/run.py --reuse-build build-cuda13/my-trace-build
```

`--reuse-build` uses an existing build-only directory; it does not resume or
overwrite a partially completed experiment. A directory collision stops execution.

To sample another compatible application after building the shim, set these
variables **inside the same container**, using its paths:

```bash
GXF_EBS_TRACE_DIR=/workspace/holoscan-sdk/build-cuda13/my-app-trace \
GXF_EBS_TRACE_INTERVAL_MS=10 \
GXF_EBS_DISPATCHER_CPU_CORE=13 \
GXF_EBS_DISPATCHER_SCHED_POLICY=SCHED_FIFO \
GXF_EBS_DISPATCHER_SCHED_PRIORITY=99 \
LD_PRELOAD=/workspace/holoscan-sdk/build-cuda13/my-trace-build/libebs_trace.so \
./your_holoscan_app
```

The trace directory enables sampling; omitting or emptying it disables sampling
even with the library preloaded. Existing application library paths still need
to resolve Holoscan and GXF. Enable GXF `log_perf_stats` when validating a new app.

## What is recorded

Each graph gets a unique `ebs-<pid>-<sequence>-<monotonic ns>` prefix:

- `.csv`: initial, periodic, and final snapshots with `CLOCK_MONOTONIC` nanoseconds
  before/after the reads; cumulative attempts, successful steals, worker wait
  calls/time, execute calls/time, postcheck-ready and dispatched events, plus GXF's
  running-thread gauge.
- `.json`: completion status, sample capacity/truncation, sampling interval,
  loaded library path/build ID, build-time header/library hashes, compiler, and
  calculated member offsets.

Each field is an independent relaxed atomic load. A row is not an instantaneous,
transactional snapshot of all fields. Rates use actual elapsed sample time, not
the requested interval. An attempt counts a victim-queue probe; one unsuccessful
scan can make multiple attempts. A successful steal counts a queued execution,
not a distinct operator or a permanent assignment change.

The buffer normally holds about 60 seconds and reserves a final-snapshot slot.
`GXF_EBS_TRACE_MAX_SAMPLES` can override its capacity (2–600002). On exhaustion,
the trace is marked `truncated`; the last stable counters are still retained.
There is no periodic disk I/O. The benchmark separately buffers every 100th
busy-wait execution, then writes progress after `app->run()`. Its 1,000th
execution timestamp marks operator completion, regardless of later graph shutdown.

The shim discovers the scheduler through GXF's public C APIs before
`GxfGraphRunAsync`, and samples until `GxfGraphWait` returns. Before graph
deactivation, context destruction, or EBS deinitialization, it stops and joins the
sampler before forwarding the original call. Successful wait yields a stable
final counter snapshot. Earlier teardown/failure is marked incomplete. Original
GXF return codes are preserved. Abrupt process termination leaves a `recording`
metadata file, which consumers must treat as incomplete; no signal handler is
installed. Single-scheduler graphs and sequential runs are supported; multiple
schedulers and concurrent graphs are rejected for tracing.

## Results and validation

The result directory contains isolated source/binaries, build commands and hashes,
`experiment.json`, per-run logs and raw traces, `summary.csv`, `analysis.json`,
`comparison.png`/`.pdf`, and individual `timeline.png`/`intervals.csv` files.
The report brackets attempts after first operator completion using neighboring
samples, retaining the sampling uncertainty.

The runner checks dispatcher settings, all 16,000 operations, all 160 progress
checkpoints, monotonic counters/timestamps, complete nontruncated traces, and exact
agreement of **all eight final counters** with GXF's final statistics log.
It includes both sampler and operator-progress cost in the overhead comparison.

```bash
python3 -s -m pytest scripts/gxf_counter_trace/test_trace.py -q
python3 -s scripts/gxf_counter_trace/check_lifecycle.py build-cuda13/my-trace-build
python3 -s scripts/gxf_counter_trace/analyze.py build-cuda13/my-trace-build
```

Lifecycle tests use a synthetic GXF library and reader to detect any read after
teardown, exercise failures, unsupported graphs, truncation, disabled mode, and
abrupt termination. They also run two sequential graphs against the **real** GXF.
The synthetic library's deliberately supplied build ID is for tests only; the
real counter layout is validated by the benchmark/log comparison.

Plotting uses system Python's NumPy and Matplotlib (`python3 -s` avoids the local
user-site NumPy/Matplotlib ABI mismatch on this machine). Removing `LD_PRELOAD`
and the trace variables disables the diagnostic entirely.
