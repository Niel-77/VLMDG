"""The rule registry ``G_k`` and per-rule utility statistics."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from vlmdg.grammar.rule import Rule


@dataclass
class RuleStats:
    """Running utility statistics for one rule.

    Attributes
    ----------
    proposed:
        Times the rule was selected by the search.
    applied:
        Times it produced a candidate design (LHS matched, RHS ran).
    admissible:
        Times the candidate passed the admissibility gate.
    accepted:
        Times the Metropolis criterion accepted the candidate.
    improved:
        Times it produced a new incumbent best.
    energy_delta_sum:
        Sum of ``energy_after - energy_before`` over accepted applications
        (negative is good).
    """

    proposed: int = 0
    applied: int = 0
    admissible: int = 0
    accepted: int = 0
    improved: int = 0
    energy_delta_sum: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposals that were accepted."""
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def mean_energy_delta(self) -> float:
        """Mean accepted energy change (0.0 when never accepted)."""
        return self.energy_delta_sum / self.accepted if self.accepted else 0.0


class Grammar:
    """An ordered, growable collection of uniquely named rules.

    The base grammar ``G_0`` is created with :func:`vlmdg.grammar.base_truss_grammar`;
    :meth:`add_rule` appends a VLM- or baseline-generated rule during search.
    """

    def __init__(self, rules: Optional[List[Rule]] = None) -> None:
        self._rules: List[Rule] = []
        self._stats: Dict[str, RuleStats] = {}
        for rule in rules or []:
            self.add_rule(rule)

    # mutation
    def add_rule(self, rule: Rule) -> str:
        """Append ``rule``; if its name clashes, a numeric suffix is appended.

        Returns the (possibly adjusted) rule name.
        """
        name = rule.name
        if name in self._stats:
            suffix = 2
            while f"{name}_{suffix}" in self._stats:
                suffix += 1
            name = f"{name}_{suffix}"
            rule.name = name
        self._rules.append(rule)
        self._stats[name] = RuleStats()
        return name

    # inspection
    @property
    def rules(self) -> List[Rule]:
        """The rules in registration order."""
        return list(self._rules)

    @property
    def size(self) -> int:
        """Number of rules currently in the grammar."""
        return len(self._rules)

    def names(self) -> List[str]:
        """Rule names in registration order."""
        return [r.name for r in self._rules]

    def base_rules(self) -> List[Rule]:
        """Rules that belong to ``G_0`` (``origin == 'base'``)."""
        return [r for r in self._rules if r.origin == "base"]

    def added_rules(self) -> List[Rule]:
        """Rules added after construction (VLM / random baselines)."""
        return [r for r in self._rules if r.origin != "base"]

    def stats(self, name: str) -> RuleStats:
        """Return the :class:`RuleStats` for a rule name."""
        return self._stats[name]

    def all_stats(self) -> Dict[str, RuleStats]:
        """Return a copy of the name -> stats mapping."""
        return dict(self._stats)

    def __iter__(self) -> Iterator[Rule]:
        return iter(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    # sampling
    def sample_rule(
        self, design, rng: random.Random, only_applicable: bool = True
    ) -> Optional[Rule]:
        """Uniformly sample a rule (optionally restricted to those that can fire)."""
        pool = self._rules
        if only_applicable:
            pool = [r for r in self._rules if r.is_applicable(design)]
        if not pool:
            return None
        return rng.choice(pool)

    def record(
        self,
        name: str,
        *,
        proposed: bool = False,
        applied: bool = False,
        admissible: bool = False,
        accepted: bool = False,
        improved: bool = False,
        energy_delta: float = 0.0,
    ) -> None:
        """Increment the utility counters for a rule after one search step."""
        stats = self._stats[name]
        stats.proposed += int(proposed)
        stats.applied += int(applied)
        stats.admissible += int(admissible)
        stats.accepted += int(accepted)
        stats.improved += int(improved)
        if accepted:
            stats.energy_delta_sum += energy_delta

    def utility_table(self) -> List[dict]:
        """Return a list of per-rule dicts for diagnostics / JSON logging."""
        table = []
        for rule in self._rules:
            stats = self._stats[rule.name]
            table.append(
                {
                    "name": rule.name,
                    "origin": rule.origin,
                    "proposed": stats.proposed,
                    "applied": stats.applied,
                    "admissible": stats.admissible,
                    "accepted": stats.accepted,
                    "improved": stats.improved,
                    "acceptance_rate": round(stats.acceptance_rate, 4),
                    "mean_energy_delta": round(stats.mean_energy_delta, 6),
                }
            )
        return table
