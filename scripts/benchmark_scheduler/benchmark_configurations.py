# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

EVENT_TARGET = "benchmark_scheduler_throughput"
GREEDY_TARGET = "benchmark_scheduler_throughput_greedy"
LIBRARY_PATH_VARIABLES = ("LD_LIBRARY_PATH", "HOLOSCAN_LIB_PATH")


@dataclass(frozen=True)
class Configuration:
    label: str
    scheduler: str
    queue_stealing: bool = False
    postcheck_fastpath: bool = False
    busy_wait: bool = False

    @property
    def target(self):
        return EVENT_TARGET if self.scheduler == "event_based" else GREEDY_TARGET

    @property
    def operations_per_operator(self):
        return 1000 if self.busy_wait else (100000 if self.scheduler == "event_based" else 10000000)

    @property
    def flags(self):
        return [
            f"--{name}"
            for name, enabled in (
                ("enable_queue_stealing", self.queue_stealing),
                ("enable_postcheck_fastpath", self.postcheck_fastpath),
                ("busy_wait", self.busy_wait),
            )
            if enabled
        ]


EVENT_VARIANTS = (
    ("Default", False, False),
    ("Queue stealing", True, False),
    ("Postcheck fastpath", False, True),
    ("Both flags", True, True),
)
CONFIGURATIONS = (
    *(
        Configuration(label, "event_based", stealing, fastpath, busy_wait)
        for busy_wait in (False, True)
        for label, stealing, fastpath in EVENT_VARIANTS
    ),
    Configuration("Greedy: normal", "greedy"),
    Configuration("Greedy: 1 ms busy wait", "greedy", busy_wait=True),
)


PANEL_NAMES = (
    "event-throughput-normal",
    "event-improvement-normal",
    "event-throughput-busy-wait",
    "event-improvement-busy-wait",
    "greedy-normal",
    "greedy-busy-wait",
)


CHECKPOINT_OPERATIONS = 10000
EVENT_THREADS = (1, 2, 4, 8, 10, 12, 13, 14, 16)
GREEDY_OPERATORS = (1, 2, 4, 8, 12, 14, 16)


def configuration_key(config):
    return (config.scheduler, config.busy_wait, config.queue_stealing, config.postcheck_fastpath)


def selected_progress(config):
    return config.scheduler == "event_based" and config.queue_stealing and config.postcheck_fastpath
