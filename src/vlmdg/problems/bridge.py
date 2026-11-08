r"""Planar deck bridge: 24 m span, six 4 m deck panels.

                       7 *
                      /|||\         start: a fan from apex joint 7 at (12, 4)
                    /  |||  \
      0 *----*----*----*----*----*----* 6
       pin   1    2    3    4    5   roller
             100 kN down at joints 1-5

The truss may be built only above the deck and must stay planar: bars may
not cross, and no joint may lie on a bar it does not end. Deck joints stay
fixed. The material is that of the ten-bar truss (Material.ten_bar()).
"""

from __future__ import annotations

from typing import Tuple

from vlmdg.config import DesignSpec
from vlmdg.design.truss import TrussDesign

#: Deck panel length (m) and number of panels.
PANEL = 4.0
N_PANELS = 6
#: Downward load at each interior deck joint (N).
LOAD = 100_000.0
#: Height of the design domain above the deck (m).
HEIGHT = 6.0


def build_bridge(default_area: float = 1.0e-3) -> Tuple[TrussDesign, DesignSpec]:
    """Return ``(design0, spec)`` for the planar deck bridge."""
    span = PANEL * N_PANELS
    joints = {k: (PANEL * k, 0.0) for k in range(N_PANELS + 1)}
    apex = N_PANELS + 1
    joints[apex] = (span / 2.0, 4.0)
    members = [(k, k + 1) for k in range(N_PANELS)]
    members += [(k, apex) for k in range(N_PANELS + 1)]
    supports = {0: (True, True), N_PANELS: (False, True)}
    loads = {k: (0.0, -LOAD) for k in range(1, N_PANELS)}

    design = TrussDesign.from_spec(
        joints=joints,
        members=members,
        supports=supports,
        loads=loads,
        default_area=default_area,
    )
    spec = DesignSpec(
        name="bridge",
        x_bounds=(0.0, span),
        y_bounds=(0.0, HEIGHT),
        fixed_joints={k: joints[k] for k in range(N_PANELS + 1)},
        supports=supports,
        loads=loads,
        min_joint_spacing=1.0,
        min_member_length=1.0,
        max_member_length=span + 1.0,
        max_joints=20,
        max_members=45,
        planar=True,
    )
    return design, spec


def reference_designs() -> dict:
    """Hand-made planar bridge layouts used as reference points (not optimized).

    Returns ``name -> (joints, members)`` on the same deck as :func:`build_bridge`.
    """
    deck = {k: (PANEL * k, 0.0) for k in range(N_PANELS + 1)}
    chord = [(k, k + 1) for k in range(N_PANELS)]
    out = {}
    for h in (3.0, 4.0, 5.0, 6.0):
        # Pratt: top joints above deck joints 1..5, verticals, diagonals toward
        # mid-span.
        joints = dict(deck)
        top = {k: 10 + k for k in range(1, N_PANELS)}
        for k, t in top.items():
            joints[t] = (PANEL * k, h)
        members = list(chord)
        members += [(top[k], top[k + 1]) for k in range(1, N_PANELS - 1)]
        members += [(0, top[1]), (N_PANELS, top[N_PANELS - 1])]
        members += [(k, top[k]) for k in top]
        members += [(top[k], k + 1) for k in range(1, N_PANELS // 2)]
        members += [(top[k], k - 1) for k in range(N_PANELS // 2 + 1, N_PANELS)]
        out[f"pratt_h{h:g}"] = (joints, members)

        # Warren: top joints above panel mid-points.
        joints = dict(deck)
        members = list(chord)
        for k in range(N_PANELS):
            t = 10 + k
            joints[t] = (PANEL * (k + 0.5), h)
            members += [(k, t), (t, k + 1)]
            if k:
                members.append((t - 1, t))
        out[f"warren_h{h:g}"] = (joints, members)
    return out
