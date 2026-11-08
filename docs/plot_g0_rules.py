"""Illustrate the five base rules of G0 on the cantilever test problem.

Each panel applies one rule with the package's own code and draws the
before/after difference. Examples are chosen to pass the admissibility gate
and to add no crossing members, so the change is easy to see.
"""
import random
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from vlmdg.grammar.base_truss import base_truss_grammar
from vlmdg.grammar.validate import admissibility_gate, check_planarity
from vlmdg.problems import get_material, get_problem

OUT = sys.argv[1]
INK, MUTED, ADD, REM, SUB = "#2b2b2b", "#b8b8b8", "#1f6fd1", "#d4542a", "#555555"

design0, spec = get_problem("cantilever")
grammar = base_truss_grammar(spec, get_material("cantilever"))
rules = {r.name: r for r in grammar.rules}


def ok(before, after):
    return (
        after != before
        and admissibility_gate(after, spec).passed
        and len(check_planarity(after)) <= len(check_planarity(before))
    )


def example(name, before, matches=None, exclude=(), prefer=lambda b, a: 0.0):
    """Return the clearest gate-passing result of ``name`` on ``before``."""
    rule = rules[name]
    cands = [m for m in (matches or rule.find_matches(before)) if m not in exclude]
    best = None
    for m in cands:
        assert m in rule.find_matches(before), (name, m)
        for seed in range(30):
            after = rule.apply_match(before.copy(), m, random.Random(seed))
            if ok(before, after):
                score = prefer(before, after)
                if best is None or score > best[0]:
                    best = (score, after)
    assert best is not None, name
    return best[1]


def displacement(before, after):
    return sum(
        abs(a - b) for n in before.joints() if n in after.joints()
        for a, b in zip(after.coords(n), before.coords(n))
    )


# Build the examples. connect_joints and remove_member use the design after
# add_triangle, which has the free joint and the redundant member they need.
tri = example("add_triangle", design0, matches=[(3, 5)], prefer=lambda b, a: a.coords(6)[1])
conn = example("connect_joints", tri, exclude=[(0, 4), (1, 5)])
CASES = [
    ("add_triangle", design0, tri,
     "Erect a new joint beside a member and\nconnect it to both ends of that member"),
    ("split_member", design0, example("split_member", design0, matches=[(3, 5)]),
     "Insert a joint at a member's midpoint, replacing\nthe member by two, and brace the new joint\nto its nearest other joint"),
    ("connect_joints", tri, conn,
     "Add a member between two unconnected joints\nif it crosses no existing member\n(shown after add_triangle)"),
    ("remove_member", conn, example("remove_member", conn, exclude=[(0, 1)]),
     "Delete a member if the truss stays connected\n(shown after add_triangle and connect_joints)"),
    ("move_joint", design0, example("move_joint", design0, matches=[5], prefer=displacement),
     "Move a joint that is neither a support nor\nloaded by a random step inside the domain"),
]


def mkey(i, j):
    return (min(i, j), max(i, j))


def seg(ax, p, q, **kw):
    ax.plot([p[0], q[0]], [p[1], q[1]], solid_capstyle="round", **kw)


def draw(ax, before, after, name, caption):

    b_mem = {mkey(*m) for m in before.members()}
    a_mem = {mkey(*m) for m in after.members()}
    moved = {n for n in after.joints() if n in before.joints() and before.coords(n) != after.coords(n)}

    for i, j in sorted(b_mem - a_mem):
        seg(ax, before.coords(i), before.coords(j), color=REM, lw=2, ls=(0, (4, 3)), zorder=2)
    for i, j in sorted(a_mem):
        p, q = after.coords(i), after.coords(j)
        new = (i, j) not in b_mem
        ratio = after.area(i, j) / before.area(i, j) if not new else 1.0
        changed = new or abs(ratio - 1.0) > 1e-12 or i in moved or j in moved
        if i in moved or j in moved:
            seg(ax, before.coords(i), before.coords(j), color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)
        if changed:
            lw = 2.4 * ratio ** 0.5 * (1.8 if abs(ratio - 1) > 1e-12 else 1.0)
            seg(ax, p, q, color="white", lw=lw + 3, zorder=3)
            seg(ax, p, q, color=ADD, lw=lw, zorder=3)
        else:
            seg(ax, p, q, color=INK, lw=1.4, zorder=2)

    for n in after.joints():
        x, y = after.coords(n)
        new = n not in before.joints()
        if n in moved:
            ox, oy = before.coords(n)
            ax.scatter([ox], [oy], s=40, facecolor="white", edgecolor=MUTED, zorder=4)
            ax.annotate("", xy=(x, y), xytext=(ox, oy), zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color=ADD, lw=1.4, shrinkA=4, shrinkB=5))
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

    ax.set_title(name, fontsize=11, fontweight="bold", color=INK, loc="left", family="monospace")
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 2.6)
    ax.set_ylim(-0.45, 1.65)
    ax.axis("off")


fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
for ax, (name, before, after, caption) in zip(axes.flat, CASES):
    draw(ax, before, after, name, caption)

handles = [
    Line2D([], [], color=INK, lw=1.4, label="unchanged member"),
    Line2D([], [], color=ADD, lw=2.4, label="new or changed member"),
    Line2D([], [], color=REM, lw=2, ls=(0, (4, 3)), label="removed member"),
    Line2D([], [], color=MUTED, lw=1.4, ls=(0, (4, 3)), label="position before the move"),
    Line2D([], [], color=ADD, marker="o", lw=0, ms=8, label="new joint"),
]
# The sixth grid cell holds the legend.
axes.flat[-1].axis("off")
axes.flat[-1].legend(handles=handles, loc="center left", frameon=False, fontsize=10)
fig.tight_layout(h_pad=2)
fig.savefig(OUT, dpi=150)
print("saved", OUT)
