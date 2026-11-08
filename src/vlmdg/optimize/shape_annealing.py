"""Shape annealing: simulated annealing whose moves are grammar rules.

Each iteration applies a random rule to the current design (rules that change
nothing are redrawn), checks the result with the admissibility gate, sizes and
scores it, and accepts it by the Metropolis criterion.

If the best design does not improve for ``stall_window`` iterations, a stall is
declared. The optional ``on_stall`` callback may return a new rule, which is
added to the grammar. The search then restarts from the best design, tries the
new rule first, and reheats.
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from vlmdg.config import AnnealConfig, DesignSpec
from vlmdg.design.io import design_to_dict, save_design
from vlmdg.design.plotting import render_truss_figure_png
from vlmdg.design.truss import TrussDesign
from vlmdg.grammar.novelty import design_signature
from vlmdg.grammar.registry import Grammar
from vlmdg.grammar.rule import Rule
from vlmdg.grammar.validate import admissibility_gate
from vlmdg.optimize.diagnostics import SearchDiagnostics, build_diagnostics
from vlmdg.optimize.objective import Evaluation, Objective
from vlmdg.optimize.sizing import FullyStressedSizer, Sizer

StallCallback = Callable[[SearchDiagnostics], Optional[Rule]]
ProgressCallback = Callable[[dict], None]


@dataclass
class AnnealResult:
    """Everything produced by one :meth:`ShapeAnnealer.run` call."""

    best_design: TrussDesign
    best_eval: Evaluation
    grammar: Grammar
    iterations: int
    rules_added: int
    history: List[dict] = field(default_factory=list)
    stalls: List[dict] = field(default_factory=list)
    #: Periodic copies of the current and best designs (for animations).
    snapshots: List[dict] = field(default_factory=list)

    def summary(self) -> dict:
        """Compact JSON-friendly summary of the run outcome."""
        return {
            "iterations": self.iterations,
            "rules_added": self.rules_added,
            "n_stalls": len(self.stalls),
            "best_mass": self.best_eval.mass,
            "best_energy": self.best_eval.energy,
            "best_feasible": self.best_eval.feasible,
            "best_max_violation": self.best_eval.max_violation,
            "grammar_size": self.grammar.size,
            "added_rule_names": [r.name for r in self.grammar.added_rules()],
            "rule_utility": self.grammar.utility_table(),
        }

    def save(self, out_dir: str | Path) -> Path:
        """Write ``best_design.json`` / ``best_design.png`` / ``result.json`` to ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_design(self.best_design, out / "best_design.json")
        render_truss_figure_png(self.best_design, out / "best_design.png")
        payload = {
            "summary": self.summary(),
            "history": self.history,
            "stalls": self.stalls,
        }
        (out / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if self.snapshots:
            (out / "snapshots.json").write_text(json.dumps(self.snapshots), encoding="utf-8")
        return out


def _metropolis_accept(delta: float, temperature: float, rng: random.Random) -> bool:
    """Return ``True`` to accept a move with energy change ``delta`` at ``temperature``."""
    if delta <= 0.0:
        return True
    if not math.isfinite(delta) or temperature <= 0.0:
        return False
    try:
        return rng.random() < math.exp(-delta / temperature)
    except OverflowError:  # pragma: no cover - exp underflow already returns 0.0
        return False


class ShapeAnnealer:
    """Configurable shape-annealing search over a growable grammar."""

    def __init__(
        self,
        spec: DesignSpec,
        objective: Objective,
        config: AnnealConfig,
        *,
        sizer: Optional[Sizer] = None,
        record_every: int = 25,
        snapshot_every: int = 0,
    ) -> None:
        self.spec = spec
        self.objective = objective
        self.config = config
        self.sizer = sizer or FullyStressedSizer(objective.material)
        self.record_every = record_every
        #: Store the current and best designs every this many iterations
        #: (``0`` = never).
        self.snapshot_every = snapshot_every

    def _size(self, design: TrussDesign) -> TrussDesign:
        """Apply the per-topology sizing step, tolerating a misbehaving sizer."""
        try:
            return self.sizer(design)
        except Exception:  # pylint: disable=broad-except
            return design

    @staticmethod
    def _draw_move(current: TrussDesign, grammar: Grammar, rng: random.Random,
                   first: Optional[Rule] = None):
        """Draw rules until one changes ``current``; return ``(rule, candidate)``.

        Base rules report up front whether they can act (their match finder
        checks the joint limit, for example). Generated rules cannot, so a
        rule that raises, returns nothing, or returns the same geometry and
        connectivity is treated as not applicable here, and another rule is
        drawn without spending the iteration. The first draw is exactly the
        one :meth:`Grammar.sample_rule` makes, unless ``first`` is given: that
        rule is tried before any random draw (used for a rule just added at a
        stall). If no rule changes the design, returns ``(last_rule, None)``.
        """
        pool = [r for r in grammar.rules if r.is_applicable(current)]
        before = None
        rule = None
        while pool:
            if first is not None and first in pool:
                rule, first = first, None
            else:
                rule = rng.choice(pool)
            grammar.record(rule.name, proposed=True)
            try:
                candidate = rule(current, rng)
            except Exception:  # pylint: disable=broad-except
                candidate = None  # generated rules may misbehave; treat as no move
            if candidate is not None:
                if before is None:
                    before = design_signature(current)
                if design_signature(candidate) != before:
                    return rule, candidate
            pool.remove(rule)
        return rule, None

    def run(
        self,
        design0: TrussDesign,
        grammar: Grammar,
        on_stall: Optional[StallCallback] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> AnnealResult:
        """Anneal from ``design0`` using ``grammar``; return an :class:`AnnealResult`."""
        cfg = self.config
        rng = random.Random(cfg.seed)

        current = self._size(design0.copy())
        current_eval = self.objective(current)
        best = current.copy()
        best_eval = current_eval
        # Normalize the Metropolis energy change by the starting energy so the
        # temperature schedule is portable between normalized and SI problems.
        energy_scale = (
            current_eval.energy
            if math.isfinite(current_eval.energy) and current_eval.energy > 0.0
            else 1.0
        )

        temperature = cfg.initial_temperature
        since_improvement = 0
        stall_index = 0
        rules_added = 0
        recent_accepted: deque[str] = deque(maxlen=25)
        history: List[dict] = []
        stalls: List[dict] = []
        snapshots: List[dict] = []

        def snapshot(it: int) -> None:
            snapshots.append(
                {
                    "iteration": it,
                    "temperature": temperature,
                    "current_mass": current_eval.mass,
                    "current_feasible": current_eval.feasible,
                    "best_mass": best_eval.mass,
                    "grammar_size": grammar.size,
                    "current": design_to_dict(current),
                    "best": design_to_dict(best),
                }
            )

        if self.snapshot_every:
            snapshot(0)

        history.append(
            {
                "iteration": 0,
                "temperature": temperature,
                "current_energy": current_eval.energy,
                "best_energy": best_eval.energy,
                "best_mass": best_eval.mass,
                "best_feasible": best_eval.feasible,
                "grammar_size": grammar.size,
            }
        )

        iteration = 0
        pending_rule: Optional[Rule] = None  # rule added at the last stall, tried first
        while iteration < cfg.max_iterations:
            iteration += 1
            improved = False

            forced = pending_rule
            rule, candidate = self._draw_move(current, grammar, rng, first=pending_rule)
            pending_rule = None
            first_use: Optional[dict] = None
            if forced is not None:
                first_use = {"iteration": iteration, "changed": False, "admissible": False,
                             "accepted": False, "improved": False, "energy_delta": None}
                stalls[-1]["outcome"]["first_use"] = first_use
            if rule is not None:
                if first_use is not None and rule is not forced:
                    first_use = None  # the new rule did not change the design; another was drawn
                if candidate is not None:
                    if first_use is not None:
                        first_use["changed"] = True
                    grammar.record(rule.name, applied=True)
                    gate = admissibility_gate(candidate, self.spec)
                    if gate.passed:
                        if first_use is not None:
                            first_use["admissible"] = True
                        grammar.record(rule.name, admissible=True)
                        try:
                            candidate = self._size(candidate)
                            cand_eval = self.objective(candidate)
                        except Exception:  # pylint: disable=broad-except
                            cand_eval = None  # analysis failed; reject the move
                        delta = (
                            (cand_eval.energy - current_eval.energy) / energy_scale
                            if cand_eval is not None
                            else math.inf
                        )
                        if first_use is not None and math.isfinite(delta):
                            first_use["energy_delta"] = delta
                        if cand_eval is not None and _metropolis_accept(delta, temperature, rng):
                            if first_use is not None:
                                first_use["accepted"] = True
                            current, current_eval = candidate, cand_eval
                            recent_accepted.append(rule.name)
                            grammar.record(rule.name, accepted=True, energy_delta=delta)
                            if cand_eval.energy < best_eval.energy - 1e-12:
                                best = current.copy()
                                best_eval = cand_eval
                                grammar.record(rule.name, improved=True)
                                improved = True
                                if first_use is not None:
                                    first_use["improved"] = True

            since_improvement = 0 if improved else since_improvement + 1

            if iteration % cfg.steps_per_temperature == 0:
                temperature = max(temperature * cfg.cooling_rate, cfg.min_temperature)

            if improved or iteration % self.record_every == 0:
                point = {
                    "iteration": iteration,
                    "temperature": temperature,
                    "current_energy": current_eval.energy,
                    "best_energy": best_eval.energy,
                    "best_mass": best_eval.mass,
                    "best_feasible": best_eval.feasible,
                    "grammar_size": grammar.size,
                }
                history.append(point)
                if progress is not None:
                    progress(point)

            if self.snapshot_every and iteration % self.snapshot_every == 0:
                snapshot(iteration)

            if since_improvement >= cfg.stall_window:
                stall_index += 1
                diag = build_diagnostics(
                    iteration=iteration,
                    temperature=temperature,
                    stall_index=stall_index,
                    iterations_since_improvement=since_improvement,
                    best_design=best,
                    best_eval=best_eval,
                    rule_utility=grammar.utility_table(),
                    recent_accepted_rules=list(recent_accepted),
                )
                new_rule: Optional[Rule] = on_stall(diag) if on_stall is not None else None
                outcome: dict = {"stall_index": stall_index, "iteration": iteration}
                if new_rule is not None:
                    added_name = grammar.add_rule(new_rule)
                    rules_added += 1
                    outcome["rule_added"] = added_name
                    outcome["rule_origin"] = new_rule.origin
                    # Try the new rule at the next iteration, on the design it was made for.
                    pending_rule = next(
                        (r for r in grammar.rules if r.name == added_name), new_rule
                    )
                else:
                    outcome["rule_added"] = None

                if cfg.restart_from_best:
                    current, current_eval = best.copy(), best_eval
                outcome["restarted_from_best"] = cfg.restart_from_best

                temperature = max(
                    cfg.initial_temperature * cfg.reheat_fraction, cfg.min_temperature
                )
                since_improvement = 0
                stalls.append({"diagnostics": diag.to_dict(), "outcome": outcome})

                if stall_index >= cfg.max_stalls:
                    break

        return AnnealResult(
            best_design=best,
            best_eval=best_eval,
            grammar=grammar,
            iterations=iteration,
            rules_added=rules_added,
            history=history,
            stalls=stalls,
            snapshots=snapshots,
        )
