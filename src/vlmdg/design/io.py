"""Serialization and human/VLM-readable descriptions of a :class:`TrussDesign`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from vlmdg.design.truss import TrussDesign


def design_to_dict(design: TrussDesign) -> Dict[str, Any]:
    """Convert a design to a plain, JSON-serializable dictionary."""
    return {
        "joints": [
            {
                "id": int(n),
                "x": float(design.coords(n)[0]),
                "y": float(design.coords(n)[1]),
                "support": [bool(design.support(n)[0]), bool(design.support(n)[1])],
                "load": [float(design.load(n)[0]), float(design.load(n)[1])],
            }
            for n in design.node_order
        ],
        "members": [
            {
                "i": int(i),
                "j": int(j),
                "area": float(design.area(i, j)),
                "label": design.label(i, j),
            }
            for i, j in design.members()
        ],
    }


def design_from_dict(data: Dict[str, Any]) -> TrussDesign:
    """Rebuild a design from :func:`design_to_dict` output."""
    design = TrussDesign()
    for joint in data["joints"]:
        design.add_joint(
            joint["x"],
            joint["y"],
            support=tuple(joint.get("support", (False, False))),
            load=tuple(joint.get("load", (0.0, 0.0))),
            joint_id=int(joint["id"]),
        )
    for member in data["members"]:
        design.add_member(
            int(member["i"]),
            int(member["j"]),
            area=float(member.get("area", 1.0)),
            label=member.get("label"),
        )
    return design


def save_design(design: TrussDesign, path: str | Path) -> None:
    """Write a design to ``path`` as pretty-printed JSON."""
    Path(path).write_text(json.dumps(design_to_dict(design), indent=2), encoding="utf-8")


def load_design(path: str | Path) -> TrussDesign:
    """Read a design previously written by :func:`save_design`."""
    return design_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def describe_design(design: TrussDesign, precision: int = 3) -> str:
    """Return a compact plain-text description suitable for a VLM prompt.

    The format lists joints (with boundary conditions) then members (with area),
    mirroring how a truss is tabulated in the structural-optimization literature.
    """
    lines = [
        f"Truss: {design.n_joints} joints, {design.n_members} members, "
        f"{design.n_reactions} reaction DOFs.",
        "Joints (id: x, y [support fx,fy] [load Fx,Fy]):",
    ]
    for n in design.node_order:
        x, y = design.coords(n)
        sx, sy = design.support(n)
        lx, ly = design.load(n)
        bc = f" support=({int(sx)},{int(sy)})" if (sx or sy) else ""
        ld = f" load=({lx:.{precision}g},{ly:.{precision}g})" if (lx or ly) else ""
        lines.append(f"  {n}: ({x:.{precision}g}, {y:.{precision}g}){bc}{ld}")
    lines.append("Members (i-j: length, area):")
    for i, j in design.members():
        lines.append(
            f"  {i}-{j}: L={design.member_length(i, j):.{precision}g}, "
            f"A={design.area(i, j):.{precision}g}"
        )
    return "\n".join(lines)
