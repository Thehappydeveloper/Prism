#!/usr/bin/env python3
"""Render the paper utility-versus-bandwidth figure for the baseline sweep."""
from __future__ import annotations

import argparse
import html
from pathlib import Path

from report_baseline_bandwidth_qoe import (
    NETWORKS,
    PREDICTORS,
    by_dataset,
    condition_runs,
    final_qoe,
    gather_diagnostics_for_run_ids,
    load_runs,
    FovAwarePlaybackUtility,
)
from report_baselines_rank_budget import estimate


SERIES = {
    "Ground-truth FoV oracle": ("GT", "#138A72", "diamond", "5 4"),
    "Argus (8 raycasting processes)": ("Argus", "#1F69A7", "circle", ""),
    "Last observed": ("LO", "#4D4D4D", "square", ""),
    "Linear extrapolation": ("LE", "#969696", "triangle", "4 3"),
    "Linear pose extrapolation": ("LP", "#6B7280", "triangle_down", "2 2"),
    "Trained LSTM": ("LSTM", "#C77719", "down", ""),
    "Trained MLP": ("MLP", "#8B5AA6", "cross", "5 3"),
    "Distance-based": ("DB", "#C95443", "star", ""),
}


def point_marker(parts: list[str], kind: str, x: float, y: float, color: str) -> None:
    if kind == "circle":
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.6" fill="{color}"/>')
    elif kind == "square":
        parts.append(f'<rect x="{x - 5.6:.1f}" y="{y - 5.6:.1f}" width="11.2" height="11.2" fill="{color}"/>')
    elif kind == "diamond":
        parts.append(f'<path d="M {x:.1f} {y - 7:.1f} L {x + 7:.1f} {y:.1f} L {x:.1f} {y + 7:.1f} L {x - 7:.1f} {y:.1f} Z" fill="{color}"/>')
    elif kind == "triangle":
        parts.append(f'<path d="M {x:.1f} {y - 7:.1f} L {x + 7:.1f} {y + 6:.1f} L {x - 7:.1f} {y + 6:.1f} Z" fill="{color}"/>')
    elif kind == "triangle_down":
        parts.append(f'<path d="M {x - 7:.1f} {y - 6:.1f} L {x + 7:.1f} {y - 6:.1f} L {x:.1f} {y + 7:.1f} Z" fill="{color}"/>')
    elif kind == "down":
        parts.append(f'<path d="M {x - 7:.1f} {y - 6:.1f} L {x + 7:.1f} {y - 6:.1f} L {x:.1f} {y + 7:.1f} Z" fill="{color}"/>')
    elif kind == "cross":
        parts.append(f'<path d="M {x - 6:.1f} {y - 6:.1f} L {x + 6:.1f} {y + 6:.1f} M {x + 6:.1f} {y - 6:.1f} L {x - 6:.1f} {y + 6:.1f}" stroke="{color}" stroke-width="3.4"/>')
    else:  # star
        parts.append(f'<path d="M {x:.1f} {y - 7:.1f} L {x + 2.5:.1f} {y - 2.5:.1f} L {x + 7:.1f} {y:.1f} L {x + 2.5:.1f} {y + 2.5:.1f} L {x:.1f} {y + 7:.1f} L {x - 2.5:.1f} {y + 2.5:.1f} L {x - 7:.1f} {y:.1f} L {x - 2.5:.1f} {y - 2.5:.1f} Z" fill="{color}"/>')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    metrics = gather_diagnostics_for_run_ids(
        sorted(runs), datasets={"qoe_series"}, qoe_model=FovAwarePlaybackUtility()
    )
    estimates: dict[str, list[tuple[float, float, float]]] = {}
    for predictor, preset, label in PREDICTORS:
        estimates[label] = [
            estimate(by_dataset(condition_runs(runs, predictor, preset, profile, label), lambda run_id: final_qoe(metrics, run_id)))
            for profile, _rate in NETWORKS
        ]

    # Keep the paper's single-column width while making the plot slightly
    # more compact vertically.
    width, height = 860, 740
    # Reserve a larger bottom band for the x-axis title without reducing the
    # plot area or moving any data/tick coordinates.
    left, right, top, bottom = 132, 44, 80, 106
    plot_w, plot_h = width - left - right, height - top - bottom
    rates = [rate for _profile, rate in NETWORKS]
    x_min, x_max = min(rates), max(rates)
    y_min, y_max = 0.35, 0.76
    x = lambda rate: left + (rate - x_min) * plot_w / (x_max - x_min)
    y = lambda value: top + (y_max - value) * plot_h / (y_max - y_min)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>.legend{font:31px Arial,Helvetica,sans-serif;fill:#202124}.axislabel{font:41px Arial,Helvetica,sans-serif;fill:#15191f}.tick{font:31px Arial,Helvetica,sans-serif;fill:#29323d}.grid{stroke:#D8DEE6;stroke-width:1.3}.axis{stroke:#1F2937;stroke-width:1.8}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    legend_order = (
        "Ground-truth FoV oracle", "Argus (8 raycasting processes)", "Last observed", "Linear extrapolation",
        "Linear pose extrapolation", "Trained LSTM", "Trained MLP", "Distance-based",
    )
    # One evenly spaced legend row, centred over the complete figure.
    legend_positions = [(30, 38), (125, 38), (260, 38), (350, 38), (430, 38), (515, 38), (635, 38), (750, 38)]
    for label, (lx, ly) in zip(legend_order, legend_positions):
        short, color, marker, dash = SERIES[label]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{lx}" y1="{ly - 5}" x2="{lx + 29}" y2="{ly - 5}" stroke="{color}" stroke-width="3.2"{dash_attr}/>')
        point_marker(parts, marker, lx + 14.5, ly - 5, color)
        parts.append(f'<text x="{lx + 38}" y="{ly + 2}" class="legend">{html.escape(short)}</text>')
    for tick in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 14}" y="{yy + 5:.1f}" class="tick" text-anchor="end">{tick:.2f}</text>')
    for rate in rates:
        xx = x(rate)
        parts.append(f'<line x1="{xx:.1f}" y1="{top + plot_h}" x2="{xx:.1f}" y2="{top + plot_h + 9}" class="axis"/>')
        parts.append(f'<text x="{xx:.1f}" y="{top + plot_h + 34}" class="tick" text-anchor="middle">{rate}</text>')
    parts.extend((
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" class="axis"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" class="axislabel" text-anchor="middle">Bandwidth (Mb/s)</text>',
        f'<text x="34" y="{top + plot_h / 2:.1f}" class="axislabel" text-anchor="middle" transform="rotate(-90 34 {top + plot_h / 2:.1f})">FoV-aware playback utility</text>',
    ))
    for label, (short, color, marker, dash) in SERIES.items():
        values = estimates[label]
        upper = " ".join(f'{x(rate):.1f},{y(high):.1f}' for rate, (_point, _low, high) in zip(rates, values))
        lower = " ".join(f'{x(rate):.1f},{y(low):.1f}' for rate, (_point, low, _high) in reversed(list(zip(rates, values))))
        parts.append(f'<polygon points="{upper} {lower}" fill="{color}" fill-opacity="0.10"/>')
        points = " ".join(f'{x(rate):.1f},{y(point):.1f}' for rate, (point, _low, _high) in zip(rates, values))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3.4"{dash_attr}/>')
        for rate, (point, _low, _high) in zip(rates, values):
            point_marker(parts, marker, x(rate), y(point), color)
    parts.append('</svg>')
    (args.output_dir / "baseline_bandwidth_qoe.svg").write_text("\n".join(parts), encoding="utf-8")
    print("baseline_bandwidth_qoe.svg")


if __name__ == "__main__":
    main()
