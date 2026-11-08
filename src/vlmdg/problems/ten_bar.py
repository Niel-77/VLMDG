r"""Ten-bar cantilever truss (Reddy and Cagan, 1995, Fig. 3a).

Two 9.1 m bays, 9.1 m tall:

      0 *----------* 2 ----------* 4
        |\        /|\           /|
        |  \    /  |  \       /  |
        |   \  /   |   \     /   |
      1 *----------* 3 ----------* 5
                   |             |
                   v 445 kN      v 445 kN

Joints 0 and 1 are pinned; 445 kN acts down at joints 3 and 5. The ten bars
are four chord bars, two verticals, and both diagonals of each bay. Minimize
weight (27 kN/m^3) with a 172 MPa stress limit and Euler buckling.
Material: Material.ten_bar().
"""

from __future__ import annotations

from typing import Tuple

from vlmdg.config import DesignSpec
from vlmdg.design.truss import TrussDesign

#: Bay width / truss height (m), as printed in the paper.
BAY = 9.1
#: Point load at each bottom free joint (N), as printed in the paper.
LOAD = 445_000.0


def build_ten_bar(default_area: float = 1.0e-3) -> Tuple[TrussDesign, DesignSpec]:
    """Return ``(design0, spec)`` for the Reddy & Cagan (1995) 10-bar truss.

    Parameters
    ----------
    default_area:
        Cross-sectional area (m^2) assigned to every bar of the starting design.
        The starting areas are deliberately *not* optimal - the per-topology
        sizing step (the paper's "shape optimization") drives them to the
        minimum-weight feasible values.
    """
    joints = {
        0: (0.0, BAY),          # top-left  - pinned support
        1: (0.0, 0.0),          # bottom-left - pinned support
        2: (BAY, BAY),          # top-middle
        3: (BAY, 0.0),          # bottom-middle - loaded
        4: (2.0 * BAY, BAY),    # top-right
        5: (2.0 * BAY, 0.0),    # bottom-right - loaded
    }
    members = [
        (0, 2), (2, 4),         # top chord
        (1, 3), (3, 5),         # bottom chord
        (2, 3), (4, 5),         # verticals at x = 9.1, x = 18.2
        (0, 3), (1, 2),         # left-bay diagonals (the "X")
        (2, 5), (3, 4),         # right-bay diagonals (the "X")
    ]
    supports = {0: (True, True), 1: (True, True)}
    loads = {3: (0.0, -LOAD), 5: (0.0, -LOAD)}

    design = TrussDesign.from_spec(
        joints=joints,
        members=members,
        supports=supports,
        loads=loads,
        default_area=default_area,
    )

    spec = DesignSpec(
        name="ten_bar",
        # The design envelope is the reference 10-bar bounding box: the search
        # rearranges material between the supports and the load points, matching
        # the paper's shape-annealing results (Fig. 3c, 3d).
        x_bounds=(0.0, 2.0 * BAY),
        y_bounds=(0.0, BAY),
        fixed_joints={0: (0.0, BAY), 1: (0.0, 0.0)},
        supports=supports,
        loads=loads,
        min_joint_spacing=0.5,
        min_member_length=0.5,
        max_member_length=25.0,
        max_joints=24,
        max_members=60,
    )
    return design, spec
