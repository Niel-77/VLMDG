"""End-to-end VLM rule-generation pipeline used as the ``on_stall`` hook.

Flow (one call to :meth:`RuleGenerator.propose_rule`):

1. render the incumbent best design to a PNG (visual prompting, optional);
2. build the system + user prompt (informed = with diagnostics);
3. query the VLM provider;
4. parse the fenced Python block;
5. **execution gate** - compile + run on probe designs, repairing via the VLM
   up to ``max_repair_attempts`` times (execution only, no correctness signal);
6. **admissibility gate** - apply the rule to the incumbent with up to
   ``SCREEN_DRAWS`` random draws, size each result as the search does, and
   accept the rule when at least one draw changes the design, passes the
   checks for language closure, connectivity, geometry, and stability
   (Maxwell + equilibrium-matrix rank), and has finite energy. The search also
   discards inadmissible draws one at a time, so a rule needs only one valid
   draw to be usable;
7. **novelty gate** - sample the rule on the incumbent and reject it when its
   outcomes can already be produced by an existing rule (or it changes nothing);
8. accept (return the rule for registration) or reject with the reasons. A
   rejected proposal is sent back to the model with the reasons, up to
   ``max_proposal_attempts`` proposals per stall.
"""

from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from vlmdg.config import DesignSpec, VLMConfig
from vlmdg.design.plotting import render_truss_png
from vlmdg.grammar.registry import Grammar
from vlmdg.grammar.rule import FunctionRule
from vlmdg.grammar.novelty import NoveltyReport, check_novelty, design_signature
from vlmdg.grammar.validate import GateReport, admissibility_gate
from vlmdg.optimize.diagnostics import SearchDiagnostics
from vlmdg.optimize.objective import Objective
from vlmdg.optimize.sizing import FullyStressedSizer, Sizer
from vlmdg.rulegen.parser import extract_code_block
from vlmdg.rulegen.prompt import (
    SYSTEM_PROMPT,
    build_rejection_feedback,
    build_repair_prompt,
    build_user_prompt,
)
from vlmdg.rulegen.sandbox import run_execution_gate
from vlmdg.vlm.base import VLMProvider, VLMUsage


#: Random draws of a proposed rule on the incumbent in the admissibility screen.
SCREEN_DRAWS = 16


@dataclass
class RuleProposal:
    """Outcome of one rule-generation attempt."""

    rule: Optional[FunctionRule]
    accepted: bool
    reasons: List[str] = field(default_factory=list)
    gate_report: Optional[GateReport] = None
    vlm_text: str = ""
    source: str = ""
    usage: VLMUsage = field(default_factory=VLMUsage)
    execution_attempts: int = 0
    transcript: List[dict] = field(default_factory=list)
    novelty: Optional[NoveltyReport] = None
    #: Earlier proposals at the same stall that were rejected (``to_dict`` records).
    rejected_attempts: List[dict] = field(default_factory=list)
    #: Outcome counts of the admissibility-screen draws (``valid``, ``unchanged``, ...).
    screen: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-friendly summary for the run log."""
        return {
            "accepted": self.accepted,
            "rule_name": self.rule.name if self.rule else None,
            "reasons": self.reasons,
            "gate_metrics": self.gate_report.metrics if self.gate_report else {},
            "execution_attempts": self.execution_attempts,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cost_usd": self.usage.cost_usd,
                "requests": self.usage.requests,
            },
            "source": self.source,
            "description": self.rule.description if self.rule else "",
            "novelty": self.novelty.to_dict() if self.novelty else {},
            "rejected_attempts": self.rejected_attempts,
            "screen": self.screen,
        }


class RuleGenerator:
    """Turns a search stall into (maybe) a new admissible grammar rule."""

    def __init__(
        self,
        provider: VLMProvider,
        spec: DesignSpec,
        objective: Objective,
        vlm_config: VLMConfig,
        *,
        max_repair_attempts: int = 3,
        max_proposal_attempts: int = 3,
        check_duplicates: bool = True,
        artifact_dir: Optional[str | Path] = None,
        sizer: Optional[Sizer] = None,
    ) -> None:
        self.provider = provider
        # The search sizes every candidate before scoring it, so the screen
        # does the same; otherwise a new member's default area counts as invalid.
        self.sizer = sizer or FullyStressedSizer(objective.material)
        self.spec = spec
        self.objective = objective
        self.vlm_config = vlm_config
        self.max_repair_attempts = max_repair_attempts
        self.max_proposal_attempts = max(1, max_proposal_attempts)
        self.check_duplicates = check_duplicates
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self._counter = 0  # every proposal, for internal labels
        self._accepted = 0  # accepted rules; their names count only these

    # helpers
    def _render_image(self, diag: SearchDiagnostics) -> Optional[Path]:
        if not self.vlm_config.attach_image or diag.best_design is None:
            return None
        base = self.artifact_dir or Path(tempfile.mkdtemp(prefix="vlmdg_rulegen_"))
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"stall_{diag.stall_index:02d}_design.png"
        render_truss_png(
            diag.best_design,
            path,
            title=f"stall #{diag.stall_index}: mass={diag.best_mass:.3g}, "
            f"feasible={diag.best_feasible}",
            annotate_area=True,
        )
        return path

    def _screen(self, rule, diag: SearchDiagnostics):
        """Apply ``rule`` to the incumbent up to ``SCREEN_DRAWS`` times.

        Returns ``(reasons, gate_report, counts)``. ``reasons`` is empty when a
        draw changed the design, passed the admissibility gate, and had finite
        energy after sizing; otherwise it holds the reasons of the first
        failing draw.
        """
        incumbent = diag.best_design
        before = design_signature(incumbent)
        counts = {"draws": 0, "valid": 0, "no_design": 0, "unchanged": 0,
                  "raised": 0, "inadmissible": 0, "infinite_energy": 0}
        first_failure: Optional[tuple] = None
        for k in range(SCREEN_DRAWS):
            counts["draws"] += 1
            rng = random.Random(diag.iteration if k == 0 else diag.iteration * 1000 + k)
            try:
                produced = rule(incumbent, rng)
            except Exception as exc:  # pylint: disable=broad-except
                counts["raised"] += 1
                first_failure = first_failure or ([f"rule raised {type(exc).__name__}: {exc}"], None)
                continue
            if produced is None:
                counts["no_design"] += 1
                continue
            if design_signature(produced) == before:
                counts["unchanged"] += 1
                continue
            report = admissibility_gate(produced, self.spec)
            if not report.passed:
                counts["inadmissible"] += 1
                first_failure = first_failure or (list(report.reasons), report)
                continue
            if not self.objective(self._size(produced)).energy < float("inf"):
                counts["infinite_energy"] += 1
                first_failure = first_failure or (
                    ["resulting design could not be analyzed (infinite energy)"], report)
                continue
            counts["valid"] += 1
            return [], report, counts
        if first_failure is not None:
            return first_failure[0], first_failure[1], counts
        if counts["no_design"] == counts["draws"]:
            return ["rule produced no design when applied to the incumbent"], None, counts
        return [f"rule never changed the geometry or connectivity of the incumbent "
                f"in {counts['draws']} draws"], None, counts

    def _size(self, design):
        """Size ``design`` as the search does, tolerating a misbehaving sizer."""
        try:
            return self.sizer(design.copy())
        except Exception:  # pylint: disable=broad-except
            return design

    def _repair_fn(self, system: str, usage_acc: VLMUsage):
        """Return a ``(code, error) -> new_code`` closure that calls the VLM."""

        def repair(code: str, error: str) -> Optional[str]:
            response = self.provider.generate(
                system,
                build_repair_prompt(code, error),
                images=(),
                max_tokens=self.vlm_config.max_tokens,
                temperature=self.vlm_config.temperature,
            )
            usage_acc.input_tokens += response.usage.input_tokens
            usage_acc.output_tokens += response.usage.output_tokens
            usage_acc.cost_usd += response.usage.cost_usd
            usage_acc.requests += response.usage.requests
            return extract_code_block(response.text)

        return repair

    # main
    def propose_rule(
        self, diag: SearchDiagnostics, grammar: Grammar, *, informed: bool = True
    ) -> RuleProposal:
        """Run the pipeline for one stall, re-prompting after each rejection.

        Every attempt is a fresh request that repeats the stall prompt and adds
        the rejected rules and their reasons, so the model can propose a
        different move. Returns the first accepted proposal, or the last
        rejected one; earlier rejections are kept in ``rejected_attempts``.
        """
        image = self._render_image(diag)
        base_prompt = build_user_prompt(diag, grammar, informed=informed, spec=self.spec)
        rejected: List[RuleProposal] = []
        proposal: Optional[RuleProposal] = None
        for _ in range(self.max_proposal_attempts):
            prompt = base_prompt
            if rejected:
                prompt += "\n\n" + build_rejection_feedback(
                    [(p.source, p.reasons) for p in rejected]
                )
            proposal = self._propose_once(diag, grammar, prompt, image)
            if proposal.accepted:
                break
            rejected.append(proposal)
        assert proposal is not None
        earlier = rejected[:-1] if not proposal.accepted else rejected
        proposal.rejected_attempts = [p.to_dict() for p in earlier]
        return proposal

    def _propose_once(
        self,
        diag: SearchDiagnostics,
        grammar: Grammar,
        user_prompt: str,
        image: Optional[Path],
    ) -> RuleProposal:
        """Query the model once and screen the rule through all three gates."""
        self._counter += 1
        # Provisional label; an accepted rule is renamed vlm_rule_<n> below so
        # that accepted names run 1, 2, 3, ... without gaps.
        rule_name = f"vlm_candidate_{self._counter}"
        usage = VLMUsage()

        response = self.provider.generate(
            SYSTEM_PROMPT,
            user_prompt,
            images=[image] if image else (),
            max_tokens=self.vlm_config.max_tokens,
            temperature=self.vlm_config.temperature,
        )
        usage = usage + response.usage

        code = extract_code_block(response.text)
        if not code:
            return RuleProposal(
                rule=None,
                accepted=False,
                reasons=["no Python code block found in the VLM response"],
                vlm_text=response.text,
                usage=usage,
            )

        probes = [diag.best_design] if diag.best_design is not None else []
        gate = run_execution_gate(
            code,
            rule_name,
            probes=probes,
            max_repairs=self.max_repair_attempts,
            repair_fn=self._repair_fn(SYSTEM_PROMPT, usage),
            seed=diag.iteration,
        )
        if not gate.passed or gate.rule is None:
            return RuleProposal(
                rule=None,
                accepted=False,
                reasons=[f"execution gate failed: {gate.final_error}"],
                vlm_text=response.text,
                source=code,
                usage=usage,
                execution_attempts=gate.attempts,
                transcript=gate.transcript,
            )

        candidate_rule = gate.rule
        candidate_rule.description = (
            candidate_rule.description or f"VLM rule proposed at stall #{diag.stall_index}"
        )

        # Admissibility gate: several draws on the incumbent; one valid draw suffices.
        reasons: List[str] = []
        gate_report: Optional[GateReport] = None
        screen: dict = {}
        if diag.best_design is not None:
            reasons, gate_report, screen = self._screen(candidate_rule, diag)

        # Novelty gate: reject rules whose outcomes the grammar can already produce.
        novelty: Optional[NoveltyReport] = None
        if not reasons and self.check_duplicates and diag.best_design is not None:
            novelty = check_novelty(
                candidate_rule, grammar.rules, [diag.best_design], seed=diag.iteration
            )
            if not novelty.novel:
                reasons.append(novelty.reason)

        accepted = not reasons
        if accepted:
            self._accepted += 1
            candidate_rule.name = f"vlm_rule_{self._accepted}"
        return RuleProposal(
            rule=candidate_rule if accepted else None,
            accepted=accepted,
            reasons=reasons,
            gate_report=gate_report,
            novelty=novelty,
            vlm_text=response.text,
            source=candidate_rule.source,
            usage=usage,
            execution_attempts=gate.attempts,
            transcript=gate.transcript,
            screen=screen,
        )
