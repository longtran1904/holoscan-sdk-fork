# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..benchmark_configurations import EVENT_VARIANTS, PANEL_NAMES
from ..data.benchmark_artifacts import graph_paths
from ..data.benchmark_results import match_baseline, validate_results


def comparison_figure(plt, results):
    fig, _ = plt.subplots(3, 2, figsize=(13, 14), layout="constrained")
    try:
        draw_comparison(fig.axes, results)
    except BaseException:
        plt.close(fig)
        raise
    return fig


def draw_comparison(axes, results):
    """Draw the same event-based panels on overview axes or independent figure axes."""
    validate_results(results)
    for busy_wait in (False, True):
        offset = 2 * int(busy_wait)
        throughput_ax, ratio_ax = axes[offset : offset + 2]
        workload = "1 ms busy wait" if busy_wait else "normal"
        baseline = results[("event_based", busy_wait, False, False)]
        for index, (label, stealing, fastpath) in enumerate(EVENT_VARIANTS):
            rows = results[("event_based", busy_wait, stealing, fastpath)]
            matched = match_baseline(rows, baseline)
            x = [row["threads"] for row, _ in matched]
            style = dict(label=label, color=f"C{index}", marker="o")
            throughput_ax.plot(x, [row["throughput_hz"] for row, _ in matched], **style)
            ratio_ax.plot(x, [ratio for _, ratio in matched], **style)
        throughput_ax.set(
            title=f"Event-based throughput: {workload}, 16 operators",
            xlabel="Worker threads",
            ylabel="Throughput (operations/s)",
        )
        ratio_ax.set(
            title=f"Event-based improvement: {workload}",
            xlabel="Worker threads",
            ylabel="Throughput / workload default (×)",
        )
        ratio_ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        for ax in (throughput_ax, ratio_ax):
            ax.set_xticks(sorted(row["threads"] for row in baseline))

    greedy_busy = results[("greedy", True, False, False)]
    (reference,) = [
        row for row in greedy_busy if row["operators"] == 1 and row["total_operations"] == 1000
    ]
    axes[2].axhline(
        reference["throughput_hz"],
        color="black",
        linestyle="--",
        label="Greedy: 1 operator, 1 ms busy wait",
    )
    for busy_wait, ax in zip((False, True), axes[4:], strict=True):
        rows = results[("greedy", busy_wait, False, False)]
        ordered = sorted(rows, key=lambda row: row["operators"])
        x = [row["operators"] for row in ordered]
        label = "Greedy: 1 ms busy wait" if busy_wait else "Greedy: normal"
        ax.plot(x, [row["throughput_hz"] for row in ordered], marker="o", label=label)
        ax.set(
            title=label,
            xlabel="Operators (one scheduler thread)",
            ylabel="Throughput (operations/s)",
            xticks=x,
        )
    for ax in (axes[0], axes[2], *axes[4:]):
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.05)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()


def save_graphs(plt, results, csv_path):
    """Save the overview and independently laid out panels, closing every figure."""
    paths = graph_paths(csv_path)
    figures = []
    try:
        figures.append(comparison_figure(plt, results))
        panel_axes = []
        for _ in PANEL_NAMES:
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
            figures.append(fig)
            panel_axes.append(ax)
        draw_comparison(panel_axes, results)
        for fig, path in zip(figures, paths, strict=True):
            fig.savefig(path, dpi=160)
    finally:
        for fig in figures:
            plt.close(fig)
    return paths
