"""Generate the compact SVG figures embedded in README.md."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np

from circular_interpolation import interpolate_circular_auto

DOCS = Path(__file__).resolve().parent
W, H = 960, 470
PLOT = (74.0, 58.0, 846.0, 345.0)


def _map(values: np.ndarray, lo: float, hi: float, start: float, size: float) -> np.ndarray:
    return start + size * (values - lo) / (hi - lo)


def _path(x: np.ndarray, y: np.ndarray) -> str:
    return " ".join(
        ("M" if i == 0 else "L") + f"{a:.1f},{b:.1f}"
        for i, (a, b) in enumerate(zip(x, y))
    )


def _svg(
    *,
    title: str,
    x: np.ndarray,
    y_limits: tuple[float, float],
    lines: Iterable[tuple[np.ndarray, str, str]],
    points: Iterable[tuple[np.ndarray, np.ndarray, str, str]],
    bands: Iterable[tuple[float, float, str, str]],
    legend: Iterable[tuple[str, str, str]],
    note: str,
) -> str:
    left, top, width, height = PLOT
    x0, x1 = float(np.min(x)), float(np.max(x))
    y0, y1 = y_limits
    mx = lambda q: _map(np.asarray(q), x0, x1, left, width)
    my = lambda q: top + height - _map(np.asarray(q), y0, y1, 0.0, height)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" fill="#344054">',
        '<rect width="100%" height="100%" rx="18" fill="#fbfcfe"/>',
        f'<text x="{W/2:.0f}" y="30" text-anchor="middle" font-size="20" font-weight="700" fill="#172033">{escape(title)}</text>',
    ]
    for i in range(5):
        yy = top + i * height / 4
        value = y1 - i * (y1 - y0) / 4
        out += [
            f'<line x1="{left}" y1="{yy:.1f}" x2="{left+width}" y2="{yy:.1f}" stroke="#dce2ec"/>',
            f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#596579">{value:.0f}</text>',
        ]
    for i in range(6):
        xx = left + i * width / 5
        value = x0 + i * (x1 - x0) / 5
        out += [
            f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+height}" stroke="#edf0f5"/>',
            f'<text x="{xx:.1f}" y="{top+height+21}" text-anchor="middle" font-size="11" fill="#596579">{value:.3f}</text>',
        ]
    for start, end, fill, label in bands:
        a, b = float(mx([start])[0]), float(mx([end])[0])
        out.append(f'<rect x="{a:.1f}" y="{top}" width="{b-a:.1f}" height="{height}" fill="{fill}" opacity=".16"><title>{escape(label)}</title></rect>')
    for values, stroke, dash in lines:
        out.append(f'<path d="{_path(mx(x), my(values))}" fill="none" stroke="{stroke}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"{dash}/>')
    for px, py, fill, shape in points:
        sx, sy = mx(px), my(py)
        if shape == "x":
            for a, b in zip(sx, sy):
                out.append(f'<path d="M{a-3:.1f},{b-3:.1f}L{a+3:.1f},{b+3:.1f}M{a-3:.1f},{b+3:.1f}L{a+3:.1f},{b-3:.1f}" stroke="{fill}" stroke-width="1.6"/>')
        else:
            for a, b in zip(sx, sy):
                out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3.1" fill="{fill}" stroke="#fff" stroke-width=".8"/>')
    out += [
        f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="#687386"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" stroke="#687386"/>',
        f'<text x="{left+width/2:.1f}" y="{H-17}" text-anchor="middle" font-size="13" fill="#344054">Time [s]</text>',
        f'<text x="18" y="{top+height/2:.1f}" transform="rotate(-90 18 {top+height/2:.1f})" text-anchor="middle" font-size="13" fill="#344054">Unwrapped angle [deg]</text>',
    ]
    lx, ly = left + 14, top + 18
    for index, (label, color, kind) in enumerate(legend):
        xx = lx + (index % 2) * 300
        yy = ly + (index // 2) * 22
        if kind == "point":
            out.append(f'<circle cx="{xx}" cy="{yy-4}" r="3.2" fill="{color}"/>')
        else:
            out.append(f'<line x1="{xx-8}" y1="{yy-4}" x2="{xx+8}" y2="{yy-4}" stroke="{color}" stroke-width="2.7"/>')
        out.append(f'<text x="{xx+14}" y="{yy}" font-size="11" fill="#344054">{escape(label)}</text>')
    out.append(f'<text x="{left+width-8}" y="{top+height-12}" text-anchor="end" font-size="11" font-weight="600" fill="#344054">{escape(note)}</text>')
    out.append('</svg>')
    return "".join(out)


def gap_repair() -> None:
    t = np.arange(0.0, 0.181, 0.001)
    omega = 2 * np.pi * 4
    truth = 330 + 6000 * t + 4000 / omega * (1 - np.cos(omega * t))
    measured = np.mod(truth, 360.0)
    gap = (t >= 0.070) & (t <= 0.115)
    held = measured.copy(); held[gap] = held[np.flatnonzero(gap)[0] - 1]
    dense_t = np.linspace(t[0], t[-1], 81)
    dense = interpolate_circular_auto(t, held, gap, target_time_s=dense_t)
    source = interpolate_circular_auto(t, held, gap)
    valid_i = np.flatnonzero(~gap)[::10]
    gap_i = np.flatnonzero(gap)[::10]
    svg = _svg(
        title="Held-value dropout repaired on the plausible revolution",
        x=dense_t,
        y_limits=(300, 1550),
        lines=[(np.interp(dense_t, t, truth), "#8090a7", ' stroke-dasharray="7 5"'), (dense.unwrapped_deg, "#1769aa", "")],
        points=[(t[valid_i], source.unwrapped_deg[valid_i], "#26364d", "circle"), (t[gap_i], np.full(gap_i.size, source.unwrapped_deg[gap_i[0]-1]), "#d1495b", "x")],
        bands=[(0.070, 0.115, "#efb366", "detected gap")],
        legend=[("hidden physical motion", "#8090a7", "line"), ("direct reconstruction", "#1769aa", "line"), ("valid samples", "#26364d", "point"), ("held sensor value", "#d1495b", "point")],
        note="The winding number is retained.",
    )
    (DOCS / "gap_repair.svg").write_text(svg)


def target_grid() -> None:
    t = np.array([0, .004, .009, .015, .022, .030, .039, .049, .060, .072, .085, .099, .114, .130])
    truth = 42 + 4200 * t + 18000 * t**2
    measured = np.mod(truth, 360.0)
    gap = (t >= .049) & (t <= .085); measured[gap] = np.nan
    target = np.unique(np.r_[np.linspace(-.025, .160, 51), [.054, .0665, .079, .145]])
    result = interpolate_circular_auto(t, measured, gap, target_time_s=target)
    valid = ~gap
    svg = _svg(
        title="One call repairs, resamples, and extrapolates",
        x=target,
        y_limits=(-80, 1200),
        lines=[(result.unwrapped_deg, "#1769aa", "")],
        points=[(target[::6], result.unwrapped_deg[::6], "#3a86c6", "circle"), (t[valid], truth[valid], "#26364d", "circle")],
        bands=[(-.025, 0, "#7dc9b1", "left extrapolation"), (.049, .085, "#efb366", "gap model"), (.130, .160, "#7dc9b1", "right extrapolation")],
        legend=[("continuous target-grid result", "#1769aa", "line"), ("requested samples", "#3a86c6", "point"), ("valid source samples", "#26364d", "point")],
        note="No repair-then-resample chain.",
    )
    (DOCS / "target_time_grid.svg").write_text(svg)


def main() -> None:
    gap_repair(); target_grid()


if __name__ == "__main__":
    main()
