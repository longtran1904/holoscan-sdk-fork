# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def load_pyplot():
    try:
        import matplotlib  # noqa: PLC0415 - optional dependency preflight

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415 - select backend before pyplot
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required. Install it in the benchmark environment with "
            "`python3 -m pip install matplotlib`, then rerun."
        ) from exc
    return plt
