"""Small planar-geometry helpers shared by the base grammar rules."""

from __future__ import annotations

import math
from typing import Tuple

Point = Tuple[float, float]


def orientation(a: Point, b: Point, c: Point) -> float:
    """Signed area of triangle ``abc`` (``> 0`` counter-clockwise)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point, eps: float = 1e-9) -> bool:
    """Return ``True`` when collinear point ``p`` lies within segment ``ab``."""
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def segments_properly_intersect(
    p1: Point, p2: Point, p3: Point, p4: Point, eps: float = 1e-9
) -> bool:
    """Return ``True`` when open segments ``p1p2`` and ``p3p4`` cross transversally.

    Shared endpoints and pure touching (T-junctions) do **not** count as a
    crossing - those are legal in a truss.
    """
    d1 = orientation(p3, p4, p1)
    d2 = orientation(p3, p4, p2)
    d3 = orientation(p1, p2, p3)
    d4 = orientation(p1, p2, p4)
    if ((d1 > eps) != (d2 > eps)) and ((d3 > eps) != (d4 > eps)):
        # Strictly opposite sides on both tests -> genuine crossing.
        if abs(d1) > eps and abs(d2) > eps and abs(d3) > eps and abs(d4) > eps:
            return True
    return False


def perpendicular_apex(
    xi: float, yi: float, xj: float, yj: float, offset: float, side: int
) -> Point:
    """Return the point ``offset`` away from the midpoint of ``ij``, normal to it.

    ``side`` is ``+1`` or ``-1`` to choose the half-plane.
    """
    mx, my = (xi + xj) / 2.0, (yi + yj) / 2.0
    dx, dy = xj - xi, yj - yi
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    return (mx + side * offset * nx, my + side * offset * ny)
