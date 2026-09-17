# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

from ..benchmark_configurations import (
    CHECKPOINT_OPERATIONS,
    CONFIGURATIONS,
    EVENT_THREADS,
    selected_progress,
)
from ..data.benchmark_artifacts import progress_files
from ..data.operator_progress import read_progress


def draw_progress(ax, timestamps, title, metadata=None):
    metadata = metadata or {}
    interval = metadata.get("checkpoint_operations", CHECKPOINT_OPERATIONS)
    pools = metadata.get("operator_pools", [])
    for operator, elapsed in sorted(timestamps.items()):
        ax.plot(
            [0, *elapsed],
            [i * interval for i in range(len(elapsed) + 1)],
            color=f"C{operator % 10}",
            linestyle=("-", "--", "-.", ":")[operator // 10 % 4],
            linewidth=1.3,
            label=f"Operator {operator}" + (f" ({pools[operator]})" if pools else ""),
        )
    ax.set_title(title)
    ax.set_xlabel("Elapsed milliseconds from each operator's start")
    ax.set_ylabel("Completed operations")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)


def save_progress_graphs(plt, log_dir, plot_dir):
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: PLC0415

    charts = []
    for config in CONFIGURATIONS:
        if not selected_progress(config):
            continue
        for threads, path in zip(EVENT_THREADS, progress_files(log_dir, config), strict=True):
            timestamps, metadata = read_progress(path, config, include_metadata=True)
            if metadata and metadata["threads"] != threads:
                raise ValueError(f"{path}: metadata worker count does not match filename")
            if timestamps:
                workload = "1 ms busy wait" if config.busy_wait else "Normal workload"
                title = f"{workload} • {threads} threads • 16 operators"
                if metadata:
                    title += (
                        f"\nShared: {metadata['shared_threads']} threads; "
                        f"secondary: {metadata['secondary_threads']} threads"
                    )
                charts.append((path, timestamps, title, metadata, config.busy_wait))
    plot_dir.mkdir()
    description = "Event-based scheduler: queue stealing enabled; postcheck fastpath enabled"
    (plot_dir / "README.md").write_text(
        f"# Operator progress\n\n{description}. Only the final configuration for each workload "
        "is included. Normal: 100,000 operations per operator, checkpoints every 10,000. "
        "Busy wait: 1,000 operations per operator with 1 ms busy wait, checkpoints every 100.\n\n"
        "For the shared-twice assignment, Y<=8 puts all 16 operators in the shared pool "
        "with Y workers. For Y>8, X=16-Y workers handle the first 2X operators in the "
        "shared pool; the remaining operators and workers use the secondary pool. "
        "For 8<Y<16, shared operators use the default pool and secondary operators "
        "are pinned one per named secondary worker. At Y=16 the shared pool is absent "
        "and all 16 operators use the default pool, labeled secondary. "
        "Historical assignments retain their original meanings, recorded in their headers. "
        "Each log starts with a JSON comment recording pool sizes, every operator's pool, "
        "and the checkpoint interval. Subsequent rows contain operator ID,elapsed milliseconds. "
        "Times are measured from each operator's own steady-clock start. Equal timestamps "
        "are valid. Plots add an origin at (0,0). Individual legends identify pool membership; "
        "overview colors identify operator IDs, whose pool may change between panels. "
        "Overviews use separate axes for each workload.\n"
    )
    with PdfPages(plot_dir / "operator-progress.pdf") as pdf:
        for path, timestamps, title, metadata, _ in charts:
            fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
            try:
                draw_progress(ax, timestamps, title, metadata)
                fig.suptitle(description, fontsize=11)
                ax.legend(ncol=2, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
                fig.savefig(plot_dir / f"{path.stem}.png", dpi=160)
                pdf.savefig(fig)
            finally:
                plt.close(fig)
        for busy_wait in (False, True):
            group = [chart for chart in charts if chart[4] == busy_wait]
            if not group:
                continue
            fig, axes = plt.subplots(
                math.ceil(len(group) / 3),
                3,
                figsize=(18, 12),
                sharex=True,
                sharey=True,
                squeeze=False,
                layout="constrained",
            )
            try:
                for ax, (_, timestamps, title, metadata, _) in zip(axes.flat, group, strict=False):
                    draw_progress(ax, timestamps, title, metadata)
                    ax.set_xlabel("Elapsed milliseconds")
                for ax in list(axes.flat)[len(group) :]:
                    ax.set_visible(False)
                maximum = max(max(values) for _, data, _, _, _ in group for values in data.values())
                axes.flat[0].set_xlim(0, max(1, maximum * 1.03))
                axes.flat[0].set_ylim(
                    0,
                    max(
                        len(v) * meta.get("checkpoint_operations", CHECKPOINT_OPERATIONS)
                        for _, data, _, meta, _ in group
                        for v in data.values()
                    )
                    * 1.03,
                )
                handles, _ = axes.flat[0].get_legend_handles_labels()
                fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.945))
                fig.legend(
                    handles,
                    [f"Operator {i}" for i in range(16)],
                    loc="lower center",
                    ncol=8,
                    fontsize=9,
                )
                fig.suptitle(
                    description + "\nElapsed time is relative to each operator's own start"
                )
                stem = "overview-busy-wait" if busy_wait else "overview"
                for suffix in ("png", "svg"):
                    fig.savefig(plot_dir / f"{stem}.{suffix}", dpi=160)
                pdf.savefig(fig)
            finally:
                plt.close(fig)
