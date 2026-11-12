#!/usr/bin/env python3
"""Render the paper process-scaling latency figure as a publication SVG."""
from __future__ import annotations

import argparse
from pathlib import Path


# Matched process-scaling cohort values: the same pooled latency and speedup
# summary shown in the accompanying process-scaling table.
PROCESSES = (1, 2, 4, 8, 16)
LATENCY_MS = (3546, 2986, 2911, 2941, 3237)
SPEEDUP = (1.00, 1.19, 1.22, 1.21, 1.10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the QoE figure's single-column width but use a compact panel. The
    # latency axis is intentionally truncated to the observed process range.
    width, height = 860, 500
    left, right, top, bottom = 132, 44, 60, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max, y_min, y_max = 0.85, 19.0, 2700.0, 3800.0
    # Process values are spaced logarithmically, exactly as the reference
    # Matplotlib plot's base-two x-axis.
    import math
    x = lambda value: left + (math.log2(value) - math.log2(x_min)) * plot_w / (math.log2(x_max) - math.log2(x_min))
    y = lambda value: top + (y_max - value) * plot_h / (y_max - y_min)
    baseline_y = y(y_min)
    blue, orange = "#1F69A7", "#C77719"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>.annotation{font:25px Arial,Helvetica,sans-serif;fill:#C77719}.axislabel{font:35px Arial,Helvetica,sans-serif;fill:#15191f}.tick{font:27px Arial,Helvetica,sans-serif;fill:#29323d}.grid{stroke:#D8DEE6;stroke-width:1.3}.axis{stroke:#1F2937;stroke-width:1.8}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for tick in (2700, 2900, 3100, 3300, 3500, 3700):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 14}" y="{yy + 7:.1f}" class="tick" text-anchor="end">{tick:,}</text>')
    reference_x = x(4)
    parts.append(f'<line x1="{reference_x:.1f}" y1="{top}" x2="{reference_x:.1f}" y2="{baseline_y:.1f}" stroke="#6B7280" stroke-width="2" stroke-dasharray="8 6"/>')
    points = " ".join(f'{x(process):.1f},{y(latency):.1f}' for process, latency in zip(PROCESSES, LATENCY_MS))
    area = f'{x(PROCESSES[0]):.1f},{baseline_y:.1f} {points} {x(PROCESSES[-1]):.1f},{baseline_y:.1f}'
    parts.append(f'<polygon points="{area}" fill="{blue}" fill-opacity="0.10"/>')
    parts.append(f'<polyline points="{points}" fill="none" stroke="{blue}" stroke-width="4"/>')
    for process, latency, speedup in zip(PROCESSES, LATENCY_MS, SPEEDUP):
        xx, yy = x(process), y(latency)
        parts.append(f'<rect x="{xx - 7:.1f}" y="{yy - 7:.1f}" width="14" height="14" fill="{blue}"/>')
        parts.append(f'<text x="{xx:.1f}" y="{yy - 17:.1f}" class="annotation" text-anchor="middle">{speedup:.2f}×</text>')
        parts.append(f'<line x1="{xx:.1f}" y1="{baseline_y:.1f}" x2="{xx:.1f}" y2="{baseline_y + 9:.1f}" class="axis"/>')
        parts.append(f'<text x="{xx:.1f}" y="{baseline_y + 42:.1f}" class="tick" text-anchor="middle">{process}</text>')
    parts.extend((
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{baseline_y}" class="axis"/>',
        f'<line x1="{left}" y1="{baseline_y}" x2="{width - right}" y2="{baseline_y}" class="axis"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 25}" class="axislabel" text-anchor="middle">Worker processes</text>',
        f'<text x="36" y="{top + plot_h / 2:.1f}" class="axislabel" text-anchor="middle" transform="rotate(-90 36 {top + plot_h / 2:.1f})">Latency (ms)</text>',
        '</svg>',
    ))
    (args.output_dir / "raycast_process_scaling.svg").write_text("\n".join(parts), encoding="utf-8")
    print("raycast_process_scaling.svg")


if __name__ == "__main__":
    main()
