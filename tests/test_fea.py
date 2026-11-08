"""Analytic checks for the linear truss FEA solver."""

import math

import pytest

from vlmdg.config import Material
from vlmdg.design.truss import TrussDesign
from vlmdg.fea import solve


def test_single_bar_axial_displacement():
    """A single axial bar: u = P L / (E A), member force = P (tension)."""
    material = Material(young_modulus=10.0, stress_allowable=1e9)
    design = TrussDesign.from_spec(
        joints={0: (0.0, 0.0), 1: (2.0, 0.0)},
        members=[(0, 1)],
        supports={0: (True, True), 1: (False, True)},
        loads={1: (3.0, 0.0)},
        areas={(0, 1): 0.5},
    )
    result = solve(design, material)

    assert not result.singular
    assert result.displacements[1][0] == pytest.approx(3.0 * 2.0 / (10.0 * 0.5))
    assert result.member_forces[(0, 1)] == pytest.approx(3.0)
    assert result.member_stress[(0, 1)] == pytest.approx(3.0 / 0.5)


def test_two_bar_truss_statics():
    """Two-bar truss under a tip vertical load matches hand equilibrium."""
    material = Material(young_modulus=1.0, stress_allowable=1e9)
    design = TrussDesign.from_spec(
        joints={0: (0.0, 0.0), 1: (0.0, 1.0), 2: (1.0, 0.0)},
        members=[(0, 2), (1, 2)],
        supports={0: (True, True), 1: (True, True)},
        loads={2: (0.0, -1.0)},
    )
    result = solve(design, material)

    assert not result.singular
    # Diagonal 1-2 carries sqrt(2) tension, horizontal 0-2 carries -1 (compression).
    assert result.member_forces[(1, 2)] == pytest.approx(math.sqrt(2.0), rel=1e-6)
    assert result.member_forces[(0, 2)] == pytest.approx(-1.0, rel=1e-6)


def test_mechanism_is_flagged_singular():
    """An unbraced square (no diagonal) is a mechanism -> singular stiffness."""
    material = Material()
    design = TrussDesign.from_spec(
        joints={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)},
        members=[(0, 1), (1, 2), (2, 3), (3, 0)],
        supports={0: (True, True), 1: (False, True)},
        loads={2: (1.0, 0.0)},
    )
    result = solve(design, material)
    assert result.singular
