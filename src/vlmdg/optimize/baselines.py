"""Stall callbacks for the search modes.

- reheat: add nothing at a stall (fixed grammar).
- random: add a chain of two or three random base rules (growth without
  design knowledge).
- uninformed: ask the VLM (vision language model) for a rule, showing the
  design and the rule list.
- informed: as uninformed, plus the search diagnostics.

make_stall_callback returns the callback for a mode (None for reheat).
vlmdg.rulegen is imported only by the VLM modes.
"""

from __future__ import annotations

import random
from typing import List, Optional

from vlmdg.config import DesignSpec, RunConfig
from vlmdg.grammar.registry import Grammar
from vlmdg.grammar.rule import FunctionRule, Rule
from vlmdg.optimize.diagnostics import SearchDiagnostics
from vlmdg.optimize.objective import Objective
from vlmdg.optimize.shape_annealing import StallCallback

#: Recognized experiment modes.
STALL_CALLBACKS = ("reheat", "anneal", "random", "uninformed", "informed")


def _random_composition_callback(grammar: Grammar, seed: int) -> StallCallback:
    """Return a callback that, per stall, registers a random composed base rule."""
    rng = random.Random(seed ^ 0x5EED)
    topo_rules: List[Rule] = [r for r in grammar.base_rules() if r.changes_topology]

    count = [0]

    def callback(diag: SearchDiagnostics) -> Optional[Rule]:
        count[0] += 1
        k = rng.randint(2, 3)
        chosen = rng.sample(topo_rules, k=min(k, len(topo_rules)))
        names = "+".join(r.name for r in chosen)

        def fn(design, inner_rng, _chosen=tuple(chosen)):
            current = design
            for rule in _chosen:
                stepped = rule(current, inner_rng)
                if stepped is not None:
                    current = stepped
            return current

        return FunctionRule(
            name=f"random_rule_{count[0]}",
            description=f"Random composition of base rules: {names}",
            fn=fn,
            origin="random",
            source=f"# composed from: {names}",
        )

    return callback


def make_stall_callback(
    mode: str,
    *,
    spec: DesignSpec,
    objective: Objective,
    grammar: Grammar,
    run_config: RunConfig,
    provider=None,
    sizer=None,
) -> Optional[StallCallback]:
    """Build the ``on_stall`` callback for ``mode``.

    Parameters
    ----------
    mode:
        One of :data:`STALL_CALLBACKS`.
    spec, objective, grammar, run_config:
        Search context the VLM arms need to screen a proposed rule.
    provider:
        A :class:`~vlmdg.vlm.base.VLMProvider`; required for ``uninformed`` /
        ``informed``. If ``None`` those modes fall back to the offline mock.
    sizer:
        The search's sizer; the VLM arms size a proposed rule's output with it
        before scoring, as the search does. ``None`` uses fully stressed design.
    """
    if mode not in STALL_CALLBACKS:
        raise ValueError(f"unknown mode {mode!r}; expected one of {STALL_CALLBACKS}")

    callback = _make_inner_callback(
        mode, spec=spec, objective=objective, grammar=grammar,
        run_config=run_config, provider=provider, sizer=sizer,
    )
    limit = run_config.max_rule_stalls
    if callback is None or limit is None:
        return callback

    def limited(diag: SearchDiagnostics) -> Optional[Rule]:
        if diag.stall_index > limit:
            return None
        return callback(diag)

    limited.proposals = getattr(callback, "proposals", [])  # type: ignore[attr-defined]
    return limited


def _make_inner_callback(
    mode: str,
    *,
    spec: DesignSpec,
    objective: Objective,
    grammar: Grammar,
    run_config: RunConfig,
    provider=None,
    sizer=None,
) -> Optional[StallCallback]:
    """Build the unrestricted callback for ``mode`` (see :func:`make_stall_callback`)."""
    if mode in ("reheat", "anneal"):
        return None

    if mode == "random":
        return _random_composition_callback(grammar, run_config.anneal.seed)

    # informed / uninformed -> VLM rule generator (lazy import).
    from vlmdg.rulegen.pipeline import RuleGenerator  # pylint: disable=import-outside-toplevel
    from vlmdg.vlm.factory import get_provider  # pylint: disable=import-outside-toplevel

    if provider is None:
        provider = get_provider(run_config.vlm.provider, config=run_config.vlm)

    generator = RuleGenerator(
        provider=provider,
        spec=spec,
        objective=objective,
        vlm_config=run_config.vlm,
        max_repair_attempts=run_config.max_repair_attempts,
        max_proposal_attempts=run_config.max_proposal_attempts,
        check_duplicates=run_config.check_duplicates,
        sizer=sizer,
    )
    informed = mode == "informed"

    def callback(diag: SearchDiagnostics) -> Optional[Rule]:
        proposal = generator.propose_rule(diag, grammar, informed=informed)
        record = proposal.to_dict()
        record["stall_index"] = diag.stall_index
        record["iteration"] = diag.iteration
        callback.proposals.append(record)  # type: ignore[attr-defined]
        return proposal.rule if proposal.accepted else None

    callback.proposals = []  # type: ignore[attr-defined]
    return callback
