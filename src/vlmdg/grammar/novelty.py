"""Behavioral duplicate check for proposed grammar rules.

Two rules are duplicates when they produce the same designs, whatever their
code looks like. Rules choose where to act at random, so each rule is sampled
many times on the same probe designs and its set of distinct outcomes is
compared with the outcome sets of the rules already in the grammar.

An outcome is identified by its geometry and connectivity only: joint
coordinates (rounded) and the members between them. Joint ids and bar areas
are ignored, because ids are arbitrary and every candidate is re-sized before
it is scored.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, Optional, Sequence, Set, Tuple

from vlmdg.design.truss import TrussDesign
from vlmdg.grammar.rule import Rule

Point = Tuple[float, float]
Signature = Tuple[FrozenSet[Point], FrozenSet[FrozenSet[Point]]]

#: Coordinates are rounded to this many decimals (1 um for designs in metres).
COORD_DECIMALS = 6
#: Samples of the candidate rule per probe design.
CANDIDATE_SAMPLES = 64
#: Samples of each existing rule per probe design. Larger than the candidate's
#: so that the existing rule's outcome set is covered well.
EXISTING_SAMPLES = 256
#: A candidate is a duplicate when at least this fraction of its distinct
#: outcomes can also be produced by one existing rule.
DUPLICATE_OVERLAP = 0.8


def design_signature(design: TrussDesign) -> Signature:
    """Return a hashable id of a design's geometry and connectivity."""

    def point(joint: int) -> Point:
        x, y = design.coords(joint)
        return (round(x, COORD_DECIMALS), round(y, COORD_DECIMALS))

    joints = frozenset(point(j) for j in design.joints())
    members = frozenset(frozenset((point(i), point(j))) for i, j in design.members())
    return joints, members


def sample_outcomes(
    rule: Rule, probes: Sequence[TrussDesign], n_samples: int, seed: int = 0
) -> Set[Signature]:
    """Apply ``rule`` ``n_samples`` times per probe; return the distinct new designs.

    Applications that fail, return nothing, or leave the geometry and
    connectivity unchanged are not counted as outcomes.
    """
    outcomes: Set[Signature] = set()
    for p_idx, probe in enumerate(probes):
        before = design_signature(probe)
        rng = random.Random(seed * 1_000_003 + p_idx)
        for _ in range(n_samples):
            try:
                result = rule(probe.copy(), rng)
            except Exception:  # pylint: disable=broad-except
                continue
            if result is None:
                continue
            sig = design_signature(result)
            if sig != before:
                outcomes.add(sig)
    return outcomes


@dataclass
class NoveltyReport:
    """Outcome of :func:`check_novelty`."""

    novel: bool
    n_outcomes: int
    duplicate_of: Optional[str] = None
    overlap: float = 0.0
    overlaps: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        """JSON-friendly summary for the run log."""
        return {
            "novel": self.novel,
            "n_outcomes": self.n_outcomes,
            "duplicate_of": self.duplicate_of,
            "overlap": round(self.overlap, 3),
            "overlaps": {k: round(v, 3) for k, v in self.overlaps.items()},
        }


def check_novelty(
    candidate: Rule,
    existing: Iterable[Rule],
    probes: Sequence[TrussDesign],
    *,
    threshold: float = DUPLICATE_OVERLAP,
    candidate_samples: int = CANDIDATE_SAMPLES,
    existing_samples: int = EXISTING_SAMPLES,
    seed: int = 0,
) -> NoveltyReport:
    """Decide whether ``candidate`` adds moves the grammar cannot already make.

    The candidate is rejected when it never changes the probe designs, or when
    at least ``threshold`` of its distinct outcomes are also outcomes of a
    single existing rule. Only rules that change topology or geometry are
    compared; pure sizing rules leave the signature unchanged.
    """
    cand = sample_outcomes(candidate, probes, candidate_samples, seed)
    if not cand:
        return NoveltyReport(
            novel=False,
            n_outcomes=0,
            reason="rule never changed the geometry or connectivity of the design",
        )

    overlaps: dict = {}
    for rule in existing:
        if not rule.changes_topology:
            continue
        other = sample_outcomes(rule, probes, existing_samples, seed + 1)
        if other:
            overlaps[rule.name] = len(cand & other) / len(cand)

    report = NoveltyReport(novel=True, n_outcomes=len(cand), overlaps=overlaps)
    if overlaps:
        name, best = max(overlaps.items(), key=lambda kv: kv[1])
        report.overlap = best
        if best >= threshold:
            report.novel = False
            report.duplicate_of = name
            report.reason = (
                f"duplicates existing rule {name!r}: {best:.0%} of its "
                f"{len(cand)} distinct outcomes are also produced by that rule"
            )
    return report
