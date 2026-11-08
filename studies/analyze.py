"""Summarize a study: statistics, convergence plot, best designs, and GIFs.

Reads runs/study_<problem>/<condition>/seed<k>/ and writes to
runs/study_<problem>/analysis/:

  summary.json, summary.md      best weight per condition, comparison with the
                                fixed grammar on the same seeds, rule counts,
                                VLM calls and tokens
  convergence.pdf/.png          mean best weight against iteration
  best_design_<condition>.png   best design of each condition
  evolution_<condition>_seed<k>.gif
                                the run of the median seed; stars mark the
                                stalls where a rule was added

Run from vlmdg/:

    python studies/analyze.py --problem ten_bar
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from vlmdg.design.io import design_from_dict  # noqa: E402
from vlmdg.design.plotting import plot_truss_figure  # noqa: E402

# Fixed order and identity of conditions: colour plus line style.
CONDITIONS = {
    "baseline": ("Fixed grammar", "#2a78d6", "-"),
    "random": ("Random rules", "#eb6834", "--"),
    "uninformed": ("VLM, uninformed", "#1baf7a", "-."),
    "informed": ("VLM, informed", "#eda100", ":"),
    "expanded": ("Expanded grammar", "#e87ba4", (0, (5, 1, 1, 1))),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#888888",
        "axes.labelcolor": "#222222",
        "axes.grid": False,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
    }
)


def load_condition(root: Path, condition: str) -> list[dict]:
    """Return the finished runs of one condition, sorted by seed."""
    runs = []
    for run_dir in sorted((root / condition).glob("seed*")):
        if not run_dir.is_dir() or not (run_dir / "result.json").exists():
            continue
        seed = int(run_dir.name.rsplit("seed", 1)[1])
        proposals_path = run_dir / "rulegen_proposals.json"
        runs.append(
            {
                "seed": seed,
                "dir": run_dir,
                "result": json.loads((run_dir / "result.json").read_text()),
                "proposals": json.loads(proposals_path.read_text())
                if proposals_path.exists()
                else [],
            }
        )
    return sorted(runs, key=lambda r: r["seed"])


def best_curve(result: dict, grid: np.ndarray) -> np.ndarray:
    """Best weight at every grid iteration (step interpolation of the history)."""
    hist = result["history"]
    its = np.array([h["iteration"] for h in hist])
    best = np.array([h["best_mass"] for h in hist])
    idx = np.searchsorted(its, grid, side="right") - 1
    return best[np.clip(idx, 0, None)]


def iterations_to_reach(result: dict, target: float) -> float:
    """First iteration whose best weight is at or below ``target`` (nan if never)."""
    for h in result["history"]:
        if h["best_mass"] <= target:
            return h["iteration"]
    return float("nan")


def _tokens(proposal: dict) -> int:
    total = 0
    for item in [proposal, *proposal.get("rejected_attempts", [])]:
        usage = item.get("usage") or {}
        total += int(usage.get("total_tokens", 0) or 0) or (
            int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0)
        )
    return total


def summarize(runs: list[dict], baseline: list[dict] | None) -> dict:
    """Statistics for one condition."""
    from scipy.stats import wilcoxon

    masses = [r["result"]["summary"]["best_mass"] for r in runs]
    out = {
        "n_runs": len(runs),
        "seeds": [r["seed"] for r in runs],
        "best_mass_n": [round(m, 1) for m in masses],
        "mean_n": statistics.mean(masses),
        "std_n": statistics.stdev(masses) if len(masses) > 1 else 0.0,
        "median_n": statistics.median(masses),
        "min_n": min(masses),
        "max_n": max(masses),
        "all_feasible": all(r["result"]["summary"]["best_feasible"] for r in runs),
        "n_stalls": [r["result"]["summary"]["n_stalls"] for r in runs],
        "rules_added": [r["result"]["summary"]["rules_added"] for r in runs],
        "grammar_size_end": [r["result"]["summary"]["grammar_size"] for r in runs],
    }

    proposals = [p for r in runs for p in r["proposals"]]
    if proposals:
        out["proposals"] = len(proposals)
        out["admitted"] = sum(1 for p in proposals if p.get("accepted"))
        out["llm_calls"] = sum(1 + len(p.get("rejected_attempts", [])) for p in proposals)
        out["tokens"] = sum(_tokens(p) for p in proposals)

    added = [
        u
        for r in runs
        for u in r["result"]["summary"]["rule_utility"]
        if u["origin"] != "base"
    ]
    if added:
        out["added_rules"] = len(added)
        out["added_rule_applications"] = sum(u["applied"] for u in added)
        out["added_rule_accepted"] = sum(u["accepted"] for u in added)
        out["added_rule_improved_best"] = sum(u["improved"] for u in added)
        out["added_rules_never_applied"] = sum(1 for u in added if u["applied"] == 0)
        all_rows = [u for r in runs for u in r["result"]["summary"]["rule_utility"]]
        out["share_of_applications_by_added_rules"] = out["added_rule_applications"] / max(
            1, sum(u["applied"] for u in all_rows)
        )
        out["share_of_improvements_by_added_rules"] = out["added_rule_improved_best"] / max(
            1, sum(u["improved"] for u in all_rows)
        )

    if baseline:
        base = {r["seed"]: r["result"]["summary"]["best_mass"] for r in baseline}
        pairs = [(r["result"]["summary"]["best_mass"], base[r["seed"]])
                 for r in runs if r["seed"] in base]
        if pairs:
            diffs = [a - b for a, b in pairs]
            out["paired_vs_baseline"] = {
                "n_pairs": len(pairs),
                "mean_diff_n": statistics.mean(diffs),
                "wins": sum(d < -1e-6 for d in diffs),
                "ties": sum(abs(d) <= 1e-6 for d in diffs),
                "losses": sum(d > 1e-6 for d in diffs),
            }
            if any(abs(d) > 1e-6 for d in diffs) and len(pairs) >= 5:
                out["paired_vs_baseline"]["wilcoxon_p"] = float(wilcoxon(diffs).pvalue)
            # Convergence speed: iterations to reach the baseline's final weight
            # of the same seed (within 1%).
            base_runs = {r["seed"]: r for r in baseline}
            its = [
                iterations_to_reach(r["result"], 1.01 * base[r["seed"]])
                for r in runs
                if r["seed"] in base_runs
            ]
            out["iterations_to_baseline_final"] = its
    return out


def plot_convergence(data: dict[str, list[dict]], n_iter: int, out: Path) -> None:
    """Mean best weight against iteration, one line per condition; no grid."""
    grid = np.arange(0, n_iter + 1)
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    finals = []
    for cond, runs in data.items():
        if not runs:
            continue
        label, color, style = CONDITIONS[cond]
        curves = np.array([best_curve(r["result"], grid) for r in runs]) / 1e3
        mean = curves.mean(axis=0)
        ax.plot(grid, mean, color=color, linestyle=style, linewidth=1.6,
                label=f"{label} (n={len(runs)})")
        finals.append(mean[-1])
    # End labels, nudged apart so that close values do not overlap.
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    placed: list[float] = []
    for final in sorted(finals):
        y = final
        while any(abs(y - p) < 0.045 * span for p in placed):
            y += 0.01 * span
        placed.append(y)
        ax.annotate(f"{final:.1f}", (grid[-1], final), xytext=(n_iter * 1.01, y),
                    textcoords="data", va="center", fontsize=7, color="#222222")
    ax.set_xlabel("Iteration (one sized candidate each)")
    ax.set_ylabel("Mean best-so-far weight (kN)")
    ax.set_xlim(0, n_iter * 1.06)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"convergence.{ext}", dpi=200)
    plt.close(fig)


def _limits(designs) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [design.coords(n)[0] for design in designs for n in design.node_order]
    ys = [design.coords(n)[1] for design in designs for n in design.node_order]
    px, py = 0.06 * (max(xs) - min(xs) or 1), 0.08 * (max(ys) - min(ys) or 1)
    return (min(xs) - px, max(xs) + px), (min(ys) - py, max(ys) + py)


def plot_best_designs(data: dict[str, list[dict]], out: Path) -> dict:
    """Save the best design of each condition as ``best_design_{condition}.png``.

    Drawings show only bars, joints, and joint numbers; the seed and weight are
    returned (and written to ``summary.json``) instead of being drawn.
    """
    info = {}
    for cond, runs in data.items():
        run = min(runs, key=lambda r: r["result"]["summary"]["best_mass"])
        design = design_from_dict(json.loads((run["dir"] / "best_design.json").read_text()))
        fig, ax = plt.subplots(figsize=(5.0, 2.0))
        plot_truss_figure(design, ax, font_size=5.5, node_size=9.0)
        fig.tight_layout(pad=0.1)
        fig.savefig(out / f"best_design_{cond}.png", dpi=200)
        plt.close(fig)
        info[cond] = {"seed": run["seed"], "best_mass_n": run["result"]["summary"]["best_mass"]}
    return info


def make_gif(run: dict, cond: str, out: Path, fps: int = 8) -> Path | None:
    """Current and best-so-far design per snapshot with a moving convergence marker."""
    snap_path = run["dir"] / "snapshots.json"
    if not snap_path.exists():
        return None
    snaps = json.loads(snap_path.read_text())
    if not snaps:
        return None
    current = [design_from_dict(s["current"]) for s in snaps]
    best = [design_from_dict(s["best"]) for s in snaps]
    xlim, ylim = _limits(current + best)
    its = [s["iteration"] for s in snaps]
    best_kn = [s["best_mass"] / 1e3 for s in snaps]
    cur_kn = [s["current_mass"] / 1e3 for s in snaps]
    _, color, _ = CONDITIONS[cond]

    fig = plt.figure(figsize=(7.2, 4.8))
    ax_cur = fig.add_axes([0.01, 0.37, 0.48, 0.47])
    ax_best = fig.add_axes([0.51, 0.37, 0.48, 0.47])
    ax_conv = fig.add_axes([0.09, 0.09, 0.86, 0.24])
    ax_conv.plot(its, cur_kn, color="#b0b0b0", linewidth=0.7, label="current")
    ax_conv.plot(its, best_kn, color=color, linewidth=1.6, label="best so far")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Weight (kN)")
    hi = np.percentile(cur_kn, 95)
    ax_conv.set_ylim(min(best_kn) * 0.95, max(hi, best_kn[0]) * 1.05)
    # Stall events: a star on the best-so-far line where a rule joined the
    # grammar (numbered in order), a small grey tick where the stall only reheated.
    descriptions = {p["rule_name"]: p.get("description", "")
                    for p in run["proposals"] if p.get("accepted")}
    added, reheats = [], []
    for stall in run["result"]["stalls"]:
        it, name = stall["outcome"]["iteration"], stall["outcome"]["rule_added"]
        (added if name else reheats).append((it, name))
    best_at = lambda it: best_curve(run["result"], np.array([it]))[0] / 1e3  # noqa: E731
    y_floor = min(best_kn) * 0.965
    if reheats:
        ax_conv.plot([it for it, _ in reheats], [y_floor] * len(reheats), linestyle="none",
                     marker="|", markersize=6, color="#888888", label="stall, reheat only")
    if added:
        xs = [it for it, _ in added]
        ys = [best_at(it) for it in xs]
        ax_conv.plot(xs, ys, linestyle="none", marker="*", markersize=10, color="#222222",
                     markeredgecolor="white", markeredgewidth=0.5, zorder=5,
                     label="rule added to grammar")
        for n, (x, y) in enumerate(zip(xs, ys), start=1):
            ax_conv.annotate(str(n), (x, y), xytext=(0, 6), textcoords="offset points",
                             ha="center", fontsize=6.5, color="#222222")
    ax_conv.legend(frameon=False, fontsize=7, loc="upper right", ncol=4)
    (marker,) = ax_conv.plot([], [], "o", color=color, markersize=5)
    vline = ax_conv.axvline(0, color="#555555", linewidth=0.6)

    def draw(k: int):
        for ax, design in ((ax_cur, current[k]), (ax_best, best[k])):
            ax.clear()
            plot_truss_figure(design, ax, font_size=5.5, node_size=10.0)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        marker.set_data([its[k]], [best_kn[k]])
        vline.set_xdata([its[k], its[k]])
        fig.suptitle(f"{CONDITIONS[cond][0]}, seed {run['seed']}: iteration {its[k]}, "
                     f"T = {snaps[k]['temperature']:.2g}, "
                     f"{snaps[k]['grammar_size']} rules\n"
                     f"left: current design {snaps[k]['current_mass'] / 1e3:.1f} kN    "
                     f"right: best so far {snaps[k]['best_mass'] / 1e3:.1f} kN\n"
                     + _rule_line(added, descriptions, its[k]),
                     fontsize=9, y=0.99)
        return []

    frames = list(range(len(snaps))) + [len(snaps) - 1] * fps * 2  # hold last frame
    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    path = out / f"evolution_{cond}_seed{run['seed']}.gif"
    anim.save(path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return path


def _rule_line(added: list, descriptions: dict, iteration: int) -> str:
    """Header line naming the most recent rule added up to ``iteration``."""
    done = [(n, it, name) for n, (it, name) in enumerate(added, start=1) if it <= iteration]
    if not added:
        return "no rules added during this run"
    if not done:
        return "no rule added yet"
    n, it, name = done[-1]
    desc = descriptions.get(name, "").strip().split("\n")[0]
    if len(desc) > 70:
        desc = desc[:67] + "..."
    return f"rule {n} added at iteration {it}: {desc or name}"


def median_run(runs: list[dict]) -> dict:
    """The run whose best weight is the median (lower median for even n)."""
    ordered = sorted(runs, key=lambda r: r["result"]["summary"]["best_mass"])
    return ordered[(len(ordered) - 1) // 2]


def write_markdown(summary: dict, out: Path) -> None:
    lines = [
        "| Condition | n | Mean (N) | Std (N) | Median (N) | Min (N) | Max (N) | "
        "Rules added | Wins/ties/losses vs fixed grammar | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for cond, s in summary.items():
        pv = s.get("paired_vs_baseline", {})
        wtl = f"{pv['wins']}/{pv['ties']}/{pv['losses']}" if pv and cond != "baseline" else ""
        p = f"{pv['wilcoxon_p']:.3f}" if "wilcoxon_p" in pv and cond != "baseline" else ""
        lines.append(
            f"| {CONDITIONS[cond][0]} | {s['n_runs']} | {s['mean_n']:,.0f} | {s['std_n']:,.0f} | "
            f"{s['median_n']:,.0f} | {s['min_n']:,.0f} | {s['max_n']:,.0f} | "
            f"{statistics.mean(s['rules_added']):.1f} | {wtl} | {p} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--problem", default="ten_bar")
    parser.add_argument("--root", default=None)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--gif-conditions", default="baseline,random,uninformed,informed,expanded")
    args = parser.parse_args()

    root = Path(args.root or f"runs/study_{args.problem}")
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    data = {c: load_condition(root, c) for c in CONDITIONS}
    data = {c: r for c, r in data.items() if r}
    n_iter = max(r["result"]["summary"]["iterations"] for runs in data.values() for r in runs)

    summary = {c: summarize(runs, data.get("baseline")) for c, runs in data.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary, out)
    plot_convergence(data, n_iter, out)
    best = plot_best_designs(data, out)
    for cond, item in best.items():
        summary[cond]["best_run"] = item
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if not args.no_gif:
        for cond in args.gif_conditions.split(","):
            if cond in data:
                print("gif", make_gif(median_run(data[cond]), cond, out))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
