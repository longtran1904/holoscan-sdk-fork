# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for source isolation, validation, and time-series calculations."""

import json
import tempfile
from pathlib import Path

import pytest
from analyze import intervals, tail_bounds
from run import COUNTERS, DEFAULT_SOURCE, benchmark_source, validate_trace


class TestTrace:
    def test_snapshot_changes_only_selected_trial_and_progress(self):
        original = DEFAULT_SOURCE.read_text()
        baseline = benchmark_source(original)
        traced = benchmark_source(original, True)
        assert "count_ % 10000 == 0" in baseline
        assert "checkpoints_.reserve(total / interval)" in traced
        assert "options_.busy_wait ? 100 : 10000" in traced
        assert "clock_gettime(CLOCK_MONOTONIC" in traced
        # CSV writes happen in results(), after app->run(), not compute()/stop().
        assert "count_op->write_progress();" in traced
        assert "timestamps_.push_back" not in traced
        trial = baseline.split(
            "  std::vector<BenchmarkSchedulerThroughputApp::Options> trial_options = {"
        )[-1]
        assert trial.split("};")[0].count("make_trial_options") == 1
        assert DEFAULT_SOURCE.read_text() == original

    def test_changed_snapshot_rejected(self):
        with pytest.raises(ValueError, match="Cannot isolate"):
            benchmark_source("unrecognized source")

    def sample(self, ns, attempts):
        return {
            "monotonic_ns": ns,
            "read_end_ns": ns + 1,
            "running_threads": 2,
            **{name: attempts if name == "steal_attempts" else 0 for name in COUNTERS},
        }

    def test_deltas_use_actual_elapsed_time_and_completion_count(self):
        samples = [self.sample(0, 0), self.sample(10_000_000, 12), self.sample(30_000_000, 36)]
        progress = [
            {"monotonic_ns": 15_000_000, "completed_operations": 1000},
            {"monotonic_ns": 20_000_000, "completed_operations": 1000},
        ]
        rows, completions = intervals(samples, progress)
        assert [r["steal_attempts_per_s"] for r in rows] == [1200, 1200]
        assert [r["unfinished_operators"] for r in rows] == [2, 0]
        bounds = tail_bounds(samples, completions[0])
        assert bounds["attempts_after_first_completion_lower"] == 0
        assert bounds["attempts_after_first_completion_upper"] == 24

    def test_counter_reset_rejected(self):
        with pytest.raises(ValueError, match="Counter reset"):
            intervals([self.sample(0, 10), self.sample(1000, 5)], [])

    def test_zero_interval_rejected(self):
        with pytest.raises(ValueError, match="Nonpositive sample interval"):
            intervals([self.sample(10, 0), self.sample(10, 1)], [])

    def test_incomplete_truncated_and_missing_trace_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            with pytest.raises(ValueError, match="Expected one trace"):
                validate_trace(directory)
            for status, truncated in [
                ("recording", False),
                ("complete", True),
                ("wait_failed", False),
            ]:
                (directory / "ebs-test.json").write_text(
                    json.dumps(
                        {
                            "status": status,
                            "truncated": truncated,
                            "final_stable": status == "complete",
                        }
                    )
                )
                with pytest.raises(ValueError, match="Incomplete trace"):
                    validate_trace(directory)
