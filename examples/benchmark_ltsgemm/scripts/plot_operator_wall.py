#!/usr/bin/env python3
"""Plot median per-operator wall time against application wall time from comparison.csv."""

import argparse
import csv
from html import escape
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("output_svg", type=Path)
    args = parser.parse_args()

    with args.comparison_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        parser.error("comparison CSV has no data rows")

    values = [
        (
            int(row["operators"]),
            float(row["median_case_wall_ms"]) / 1000,
            float(row["app_wall_ms"]) / 1000,
        )
        for row in rows
    ]
    if any(count <= 0 or case < 0 or app < 0 for count, case, app in values):
        parser.error("operator counts must be positive and wall times nonnegative")

    width, height = 1120, 630
    left, right, top, bottom = 90, 1070, 100, 515
    plot_height = bottom - top
    y_max = max(max(case, app) for _, case, app in values)
    y_max = max(10, ((int(y_max) + 9) // 10) * 10)
    tick_step = 10 if y_max <= 50 else 20

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Grouped bar chart comparing median per-operator wall time and application wall time by operator count">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="DejaVu Sans, Arial, sans-serif" fill="#17212f">',
        '<text x="90" y="43" font-size="24" font-weight="bold">Wall time by operator count</text>',
        '<text x="90" y="70" font-size="14" fill="#536171">Median per-operator interval vs. full application run</text>',
    ]

    for tick in range(0, y_max + 1, tick_step):
        y = bottom - tick / y_max * plot_height
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
                'stroke="#e2e8f0"/>',
                f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" '
                f'font-size="13" fill="#536171">{tick}</text>',
            ]
        )

    group_width = (right - left) / len(values)
    bar_width = min(34, group_width * 0.34)
    for index, (count, case, app) in enumerate(values):
        center = left + group_width * (index + 0.5)
        for x, value, color in (
            (center - bar_width - 2, case, "#3278b8"),
            (center + 2, app, "#e89032"),
        ):
            bar_height = value / y_max * plot_height
            parts.append(
                f'<rect x="{x:.2f}" y="{bottom - bar_height:.2f}" '
                f'width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}">'
                f'<title>{escape(str(count))} operators: {value:.3f} s</title></rect>'
            )
        parts.append(
            f'<text x="{center:.2f}" y="{bottom + 22}" text-anchor="middle" '
            f'font-size="13">{count}</text>'
        )

    parts.extend(
        [
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#536171"/>',
            '<text x="580" y="572" text-anchor="middle" font-size="15">Number of operators</text>',
            '<text x="28" y="310" text-anchor="middle" font-size="15" '
            'transform="rotate(-90 28 310)">Wall time (seconds)</text>',
            '<rect x="660" y="48" width="15" height="15" fill="#3278b8"/>',
            '<text x="682" y="61" font-size="13">Median per-operator</text>',
            '<rect x="875" y="48" width="15" height="15" fill="#e89032"/>',
            '<text x="897" y="61" font-size="13">Application</text>',
            '<text x="90" y="611" font-size="12" fill="#536171">'
            'Per-operator intervals overlap; do not sum them to obtain application wall time.</text>',
            '</g></svg>',
        ]
    )
    args.output_svg.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(args.output_svg)


if __name__ == "__main__":
    main()
