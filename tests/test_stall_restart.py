"""At a stall the search restarts from the best design and tries the new rule first."""

from vlmdg.config import AnnealConfig
from vlmdg.grammar.base_truss import base_truss_grammar
from vlmdg.grammar.novelty import design_signature
from vlmdg.grammar.rule import FunctionRule
from vlmdg.optimize.objective import Objective
from vlmdg.optimize.shape_annealing import ShapeAnnealer
from vlmdg.problems import get_material, get_problem


def _setup():
    design, spec = get_problem("ten_bar")
    material = get_material("ten_bar")
    return design, spec, material, base_truss_grammar(spec, material)


def _connect_rule(name):
    def fn(design, rng):
        js = design.joints()
        for k, i in enumerate(js):
            for j in js[k + 1:]:
                if not design.has_member(i, j):
                    design.add_member(i, j)
                    return design
        return design
    return FunctionRule(name=name, description="", fn=fn, origin="vlm", source="")


def _run(on_stall, restart=True, iterations=120):
    design, spec, material, grammar = _setup()
    cfg = AnnealConfig(max_iterations=iterations, stall_window=20, max_stalls=1000,
                       initial_temperature=0.1, restart_from_best=restart, seed=3)
    return ShapeAnnealer(spec, Objective(material), cfg).run(design, grammar, on_stall=on_stall)


def test_new_rule_is_used_at_the_iteration_after_the_stall():
    added = []

    def on_stall(diag):
        if len(added) >= 3:
            return None
        added.append(_connect_rule(f"new_{len(added)}"))
        return added[-1]

    result = _run(on_stall)
    with_rule = [s for s in result.stalls if s["outcome"]["rule_added"]]
    assert len(with_rule) == 3
    for stall in with_rule:
        first = stall["outcome"]["first_use"]
        assert first["iteration"] == stall["outcome"]["iteration"] + 1
        assert first["changed"]
    for stall in result.stalls:
        if not stall["outcome"]["rule_added"]:
            assert "first_use" not in stall["outcome"]


def _record_starts(restart):
    """Return (best design at each stall, design the next iteration starts from)."""
    design, spec, material, grammar = _setup()
    cfg = AnnealConfig(max_iterations=200, stall_window=20, max_stalls=1000,
                       initial_temperature=0.1, restart_from_best=restart, seed=3)
    pairs, state = [], {"best": None}

    class Spy(ShapeAnnealer):
        @staticmethod
        def _draw_move(current, grammar, rng, first=None):
            if state["best"] is not None:
                pairs.append((state["best"], design_signature(current)))
                state["best"] = None
            return ShapeAnnealer._draw_move(current, grammar, rng, first=first)

    def on_stall(diag):
        state["best"] = design_signature(spy_best[0]())
        return None

    annealer = Spy(spec, Objective(material), cfg)
    spy_best = [None]
    orig_size = annealer._size
    # The best design is what the diagnostics are built from.
    import vlmdg.optimize.shape_annealing as sa
    real_build = sa.build_diagnostics

    def build(**kw):
        spy_best[0] = kw["best_design"].copy
        return real_build(**kw)

    sa.build_diagnostics = build
    try:
        annealer.run(design, grammar, on_stall=on_stall)
    finally:
        sa.build_diagnostics = real_build
    return pairs


def test_search_restarts_from_the_best_design():
    pairs = _record_starts(restart=True)
    assert pairs
    assert all(best == start for best, start in pairs)


def test_restart_can_be_turned_off():
    result = _run(lambda d: None, restart=False)
    assert all(not s["outcome"]["restarted_from_best"] for s in result.stalls)
