"""Draw what each accepted VLM rule does to the cantilever design.

Each rule is tried on the cantilever start design and, if it does not apply
there, after one step of the fixed grammar. The first valid change is drawn;
rules that never apply are listed instead.

Run from vlmdg/:

    python studies/plot_accepted_rules.py runs/previews/accepted_rules_cantilever.png
"""

from __future__ import annotations

import argparse
import json
import math
import random
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from vlmdg.grammar.base_truss import base_truss_grammar  # noqa: E402
from vlmdg.grammar.validate import admissibility_gate  # noqa: E402
from vlmdg.problems import get_material, get_problem  # noqa: E402
from vlmdg.rulegen.sandbox import compile_rule  # noqa: E402

INK, MUTED, ADD, REM, SUB = "#2b2b2b", "#b8b8b8", "#1f6fd1", "#d4542a", "#555555"
SEEDS = range(20)


def accepted_rules(problems, conditions):
    """Yield ``(label, description, source)`` for every accepted proposal."""
    for problem in problems:
        for condition in conditions:
            root = Path(f"runs/study_{problem}/{condition}")
            for path in sorted(root.glob("seed*/rulegen_proposals.json"),
                               key=lambda p: int(p.parent.name[4:])):
                for prop in json.loads(path.read_text()):
                    if prop["accepted"]:
                        label = f"{problem}, {condition}, {path.parent.name}, stall {prop['stall_index']}"
                        yield label, prop.get("description", ""), prop["source"]


def clearance(design):
    """Smallest distance from a joint to a member it is not an end of."""
    best = float("inf")
    for i, j in design.members():
        (ax_, ay), (bx, by) = design.coords(i), design.coords(j)
        dx, dy = bx - ax_, by - ay
        length2 = dx * dx + dy * dy
        for n in design.joints():
            if n in (i, j):
                continue
            px, py = design.coords(n)
            t = max(0.0, min(1.0, ((px - ax_) * dx + (py - ay) * dy) / length2))
            best = min(best, math.hypot(px - ax_ - t * dx, py - ay - t * dy))
    return best


def contexts(design0, spec, grammar):
    """Return the start design, then gate-passing results of one G0 step.

    For each rule and match, the random draw that leaves the most room between
    joints and members is kept, and within each rule the clearest results come
    first, so the drawings stay readable.
    """
    out = [("start design", design0)]
    for rule in grammar.rules:
        if rule.name == "move_joint":
            continue
        found = []
        for match in rule.find_matches(design0):
            best = None
            for seed in range(10):
                after = rule.apply_match(design0.copy(), match, random.Random(seed))
                if after is None or after == design0 or not admissibility_gate(after, spec).passed:
                    continue
                score = clearance(after)
                if best is None or score > best[0]:
                    best = (score, after)
            if best is not None:
                found.append(best)
        found.sort(key=lambda item: -item[0])
        out += [(f"design after {rule.name}", after) for _, after in found]
    return out


def same_shape(a, b):
    """True when two designs have the same joints, positions, and members."""
    if set(a.joints()) != set(b.joints()):
        return False
    if any(a.coords(n) != b.coords(n) for n in a.joints()):
        return False
    key = lambda d: {tuple(sorted(m)) for m in d.members()}  # noqa: E731
    return key(a) == key(b)


def find_example(rule, ctxs, spec):
    """Return ``(context_label, before, after)`` for the first valid change, else None."""
    for label, before in ctxs:
        for seed in SEEDS:
            try:
                matches = rule.find_matches(before)
                if not matches:
                    break
                after = rule.apply_match(before.copy(), matches[0], random.Random(seed))
            except Exception:  # pylint: disable=broad-except
                break
            if after is None or same_shape(before, after):
                continue
            if admissibility_gate(after, spec).passed:
                return label, before, after
    return None


def seg(ax, p, q, **kw):
    ax.plot([p[0], q[0]], [p[1], q[1]], solid_capstyle="round", **kw)


def draw(ax, spec, before, after):
    key = lambda m: tuple(sorted(m))  # noqa: E731
    b_mem = {key(m) for m in before.members()}
    a_mem = {key(m) for m in after.members()}
    moved = {n for n in after.joints()
             if n in before.joints() and before.coords(n) != after.coords(n)}
    for i, j in sorted(b_mem - a_mem):
        seg(ax, before.coords(i), before.coords(j), color=REM, lw=2, ls=(0, (4, 3)), zorder=2)
    for i, j in sorted(a_mem):
        changed = (i, j) not in b_mem or i in moved or j in moved
        if i in moved or j in moved:
            if (i, j) in b_mem:
                seg(ax, before.coords(i), before.coords(j), color=MUTED, lw=1.4,
                    ls=(0, (4, 3)), zorder=1)
        if changed:
            seg(ax, after.coords(i), after.coords(j), color="white", lw=5.4, zorder=3)
            seg(ax, after.coords(i), after.coords(j), color=ADD, lw=2.4, zorder=3)
        else:
            seg(ax, after.coords(i), after.coords(j), color=INK, lw=1.4, zorder=2)
    for n in before.joints():
        if n not in after.joints():
            x, y = before.coords(n)
            ax.scatter([x], [y], s=46, facecolor="white", edgecolor=REM, linewidth=1.5, zorder=6)
    for n in after.joints():
        x, y = after.coords(n)
        if n in moved:
            ox, oy = before.coords(n)
            ax.scatter([ox], [oy], s=40, facecolor="white", edgecolor=MUTED, zorder=4)
            ax.annotate("", xy=(x, y), xytext=(ox, oy), zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color=ADD, lw=1.4, shrinkA=4, shrinkB=5))
        new = n not in before.joints()
        ax.scatter([x], [y], s=60 if new else 30, color=ADD if new else INK,
                   edgecolor="white", linewidth=1.2, zorder=6)
        ax.annotate(str(n), (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=INK, zorder=7)
    for n in spec.supports:
        x, y = after.coords(n)
        ax.plot([x - 0.07], [y], marker=">", ms=8, color=INK)
        ax.plot([x - 0.11, x - 0.11], [y - 0.09, y + 0.09], color=INK, lw=1.5)
    for n in spec.loads:
        x, y = after.coords(n)
        ax.annotate("", xy=(x, y - 0.38), xytext=(x, y - 0.03),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    xs = [after.coords(n)[0] for n in after.joints()] + [before.coords(n)[0] for n in before.joints()]
    ys = [after.coords(n)[1] for n in after.joints()] + [before.coords(n)[1] for n in before.joints()]
    ax.set_xlim(min(xs) - 0.3, max(xs) + 0.35)
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.25)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--problems", default="ten_bar,bridge")
    parser.add_argument("--conditions", default="uninformed,informed")
    args = parser.parse_args()

    design0, spec = get_problem("cantilever")
    ctxs = contexts(design0, spec, base_truss_grammar(spec, get_material("cantilever")))
    rules = list(accepted_rules(args.problems.split(","), args.conditions.split(",")))
    if not rules:
        raise SystemExit("no accepted rules found")

    ncols = 3
    nrows = -(-len(rules) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.3 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (label, description, source) in zip(axes.flat, rules):
        rule = compile_rule(source, label, description)
        example = find_example(rule, ctxs, spec)
        caption = "\n".join(textwrap.wrap(description, 46))
        if example is None:
            ax.text(0.5, 0.55, "Does not change the cantilever\n(start design or one G0 step)",
                    ha="center", va="center", fontsize=9, color=SUB, transform=ax.transAxes)
        else:
            ctx_label, before, after = example
            draw(ax, spec, before, after)
            caption += f"\n(applied to the {ctx_label})"
        ax.set_title(label, fontsize=9.5, fontweight="bold", color=INK, loc="left")
        ax.text(0, -0.02, caption, transform=ax.transAxes, fontsize=8.3, color=SUB, va="top")

    handles = [
        Line2D([], [], color=INK, lw=1.4, label="unchanged member"),
        Line2D([], [], color=ADD, lw=2.4, label="new or changed member"),
        Line2D([], [], color=REM, lw=2, ls=(0, (4, 3)), label="removed member"),
        Line2D([], [], color=ADD, marker="o", lw=0, ms=8, label="new joint"),
        Line2D([], [], color=REM, marker="o", mfc="white", lw=0, ms=7, label="removed joint"),
        Line2D([], [], color=MUTED, lw=1.4, ls=(0, (4, 3)), label="position before a move"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print("saved", args.out, f"({len(rules)} rules)")


if __name__ == "__main__":
    main()
