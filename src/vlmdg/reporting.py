"""Plotting helpers for a run: convergence history and design snapshots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:  # pragma: no cover
    from vlmdg.optimize.shape_annealing import AnnealResult


def plot_convergence(
    result: "AnnealResult",
    path: str | Path,
    *,
    title: str | None = None,
    unit: str | None = "kN",
    scale: float = 1e3,
    dpi: int = 130,
) -> Path:
    """Plot the current and best-so-far energy against iteration and save to ``path``.

    The energy is the weight times the penalty factor, so it equals the weight
    whenever no stress or buckling limit is exceeded. Values are divided by
    ``scale`` and the axis is labelled with ``unit`` (``None`` for problems in
    normalized units). Stalls are drawn as vertical dotted lines; a marker
    annotates any stall at which a rule was added to the grammar. The figure
    has one y-axis and no grid, and a title is drawn only when ``title`` is
    given.
    """
    history = result.history
    if not history:
        raise ValueError("AnnealResult has no history to plot")

    def scaled(values):
        return [v / scale if math.isfinite(v) else None for v in values]

    iters = [h["iteration"] for h in history]
    best = scaled(h["best_energy"] for h in history)
    current = scaled(h["current_energy"] for h in history)

    finite = [v for v in best if v is not None]
    floor = min(finite) * 0.5 if finite else 1.0

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(iters, current, color="#b8c4d8", lw=1, label="current design")
    ax.plot(iters, best, color="#1f3b6f", lw=2, label="best design so far")
    ax.set_xlabel("Iteration")
    label = "Weight \u00d7 penalty factor"
    ax.set_ylabel(f"{label} ({unit})" if unit else label)
    if finite and max(finite) / max(floor, 1e-12) > 50:
        ax.set_yscale("log")

    for k, stall in enumerate(result.stalls):
        it = stall["diagnostics"]["iteration"]
        added = stall["outcome"].get("rule_added")
        ax.axvline(it, color="#999999", ls=":", lw=1, label="stall" if k == 0 else None)
        if added:
            ax.annotate(
                f"+{added}",
                xy=(it, ax.get_ylim()[1]),
                xytext=(0, -10),
                textcoords="offset points",
                rotation=90,
                fontsize=7,
                ha="right",
                va="top",
                color="#7a3333",
            )

    ax.legend(loc="upper right", fontsize=8, frameon=False)
    if title:
        ax.set_title(title)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
