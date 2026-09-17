#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable entry point, including invocation from outside the checkout."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmark_scheduler.benchmark_cli import main
else:
    from .benchmark_cli import main

if __name__ == "__main__":
    sys.exit(main())
