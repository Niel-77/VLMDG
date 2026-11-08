"""Matplotlib rendering of a :class:`TrussDesign`.

The renderer is used both for run diagnostics and to attach a picture of the
current design to the VLM prompt (visual prompting). A non-interactive backend is
forced so the module is import-safe on headless machines.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlmdg.design.truss import TrussDesign

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_truss(
    design: TrussDesign,
    ax: "Axes | None" = None,
    *,
    title: str | None = None,
    show_ids: bool = True,
    annotate_area: bool = False,
    member_scale: float = 4.0,
) -> "Axes":
    """Draw ``design`` onto ``ax`` (created if omitted) and return the axes.

    Members are drawn with a line width proportional to ``sqrt(area)``. Supports
    are marked with triangles, loaded joints with arrows.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    areas = [design.area(i, j) for i, j in design.members()] or [1.0]
    a_max = max(areas)
    for i, j in design.members():
        (xi, yi), (xj, yj) = design.coords(i), design.coords(j)
        width = member_scale * (design.area(i, j) / a_max) ** 0.5
        ax.plot([xi, xj], [yi, yj], "-", color="#33507a", linewidth=max(0.6, width), zorder=1)
        if annotate_area:
            ax.annotate(
                f"{design.area(i, j):.2g}",
                ((xi + xj) / 2, (yi + yj) / 2),
                fontsize=7,
                color="#7a3333",
            )

    xs = [design.coords(n)[0] for n in design.node_order]
    ys = [design.coords(n)[1] for n in design.node_order]
    ax.scatter(xs, ys, s=28, color="#222222", zorder=3)

    for n in design.node_order:
        x, y = design.coords(n)
        if any(design.support(n)):
            ax.scatter([x], [y], marker="^", s=150, color="#2a7a2a", zorder=2)
        fx, fy = design.load(n)
        if fx or fy:
            norm = (fx ** 2 + fy ** 2) ** 0.5
            span = max(xs) - min(xs) or 1.0
            scale = 0.25 * span / norm
            ax.annotate(
                "",
                xy=(x + fx * scale, y + fy * scale),
                xytext=(x, y),
                arrowprops={"arrowstyle": "-|>", "color": "#b03030", "lw": 2},
                zorder=4,
            )
        if show_ids:
            ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.15)
    ax.grid(True, linestyle=":", alpha=0.4)
    if title:
        ax.set_title(title)
    return ax


def plot_truss_figure(
    design: TrussDesign,
    ax: "Axes | None" = None,
    *,
    member_scale: float = 3.0,
    node_size: float = 14.0,
    font_size: float = 6.0,
    color: str = "#1c5cab",
) -> "Axes":
    """Draw ``design`` for reports: beams, nodes, and node numbers only.

    No title, area labels, grid, axes, support or load symbols. Line width is
    proportional to ``sqrt(area)``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    members = design.members()
    a_max = max((design.area(i, j) for i, j in members), default=1.0) or 1.0
    for i, j in members:
        (xi, yi), (xj, yj) = design.coords(i), design.coords(j)
        width = max(0.4, member_scale * (design.area(i, j) / a_max) ** 0.5)
        ax.plot([xi, xj], [yi, yj], "-", color=color, linewidth=width,
                solid_capstyle="round", zorder=1)
    for n in design.node_order:
        x, y = design.coords(n)
        ax.scatter([x], [y], s=node_size, color="#222222", zorder=3,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(3, 3),
                    fontsize=font_size, color="#222222", zorder=4,
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none",
                          "alpha": 0.85})
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.06)
    ax.axis("off")
    return ax


def render_truss_figure_png(design: TrussDesign, path: str | Path, *, dpi: int = 200) -> Path:
    """Render ``design`` with :func:`plot_truss_figure` to a PNG file."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    plot_truss_figure(design, ax=ax)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def render_truss_png(
    design: TrussDesign,
    path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 130,
    **plot_kwargs,
) -> Path:
    """Render ``design`` to a PNG file and return its path."""
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_truss(design, ax=ax, title=title, **plot_kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
