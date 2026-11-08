"""Two-panel planar cantilever: a small test problem in normalized units.

    1 *------* 3 ------* 5
      |    / |       / |
      |   /  |      /  |
    0 *------* 2 ------* 4
                       |
                       v  P = 1

Joints 0 and 1 are pinned to the wall; a unit load acts down at joint 4.
Modulus, allowable stress and density are 1; bar areas are in [0.1, 10].
Design domain: x in [0, 3], y in [0, 2].
"""

from __future__ import annotations

from typing import Tuple

from vlmdg.config import DesignSpec
from vlmdg.design.truss import TrussDesign

_TIP_LOAD = 1.0


def build_cantilever(default_area: float = 1.0) -> Tuple[TrussDesign, DesignSpec]:
    """Return ``(design0, spec)`` for the two-panel cantilever."""
    joints = {
        0: (0.0, 0.0),
        1: (0.0, 1.0),
        2: (1.0, 0.0),
        3: (1.0, 1.0),
        4: (2.0, 0.0),
        5: (2.0, 1.0),
    }
    members = [
        (0, 1),  # wall vertical
        (0, 2), (2, 4),  # bottom chord
        (1, 3), (3, 5),  # top chord
        (2, 3), (4, 5),  # panel verticals
        (0, 3), (2, 5),  # panel diagonals
    ]
    supports = {0: (True, True), 1: (True, True)}
    loads = {4: (0.0, -_TIP_LOAD)}

    design = TrussDesign.from_spec(
        joints=joints,
        members=members,
        supports=supports,
        loads=loads,
        default_area=default_area,
    )

    spec = DesignSpec(
        name="cantilever",
        x_bounds=(0.0, 3.0),
        y_bounds=(0.0, 2.0),
        fixed_joints={0: (0.0, 0.0), 1: (0.0, 1.0)},
        supports=supports,
        loads=loads,
        min_joint_spacing=0.15,
        min_member_length=0.15,
        max_member_length=4.0,
        max_joints=24,
        max_members=60,
    )
    return design, spec
