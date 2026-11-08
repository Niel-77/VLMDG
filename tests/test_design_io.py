"""Serialization round-trip and structural comparison."""

from vlmdg.design import design_from_dict, design_to_dict
from vlmdg.problems import build_cantilever


def test_dict_round_trip_is_structurally_equal():
    design, _ = build_cantilever()
    clone = design_from_dict(design_to_dict(design))
    assert clone == design
    assert clone.structural_key() == design.structural_key()


def test_relabeled_design_compares_equal():
    design, _ = build_cantilever()
    data = design_to_dict(design)
    shift = 100
    for joint in data["joints"]:
        joint["id"] += shift
    for member in data["members"]:
        member["i"] += shift
        member["j"] += shift
    relabeled = design_from_dict(data)
    assert relabeled == design  # equality is by geometry/topology, not ids
