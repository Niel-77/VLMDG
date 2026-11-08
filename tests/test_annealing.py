"""Shape-annealing search smoke tests."""

from vlmdg.config import AnnealConfig, Material
from vlmdg.grammar.base_truss import base_truss_grammar
from vlmdg.grammar.validate import admissibility_gate
from vlmdg.optimize.objective import Objective
from vlmdg.optimize.shape_annealing import ShapeAnnealer
from vlmdg.problems import build_cantilever


def test_baseline_run_improves_and_stays_admissible():
    design, spec = build_cantilever()
    objective = Objective(Material())
    grammar = base_truss_grammar(spec, Material())
    config = AnnealConfig(max_iterations=120, stall_window=999, seed=0)

    annealer = ShapeAnnealer(spec, objective, config)
    result = annealer.run(design, grammar)

    start_energy = result.history[0]["best_energy"]
    assert result.best_eval.energy <= start_energy  # best-so-far never worsens
    assert result.iterations == 120
    assert result.rules_added == 0
    assert admissibility_gate(result.best_design, spec).passed


def test_history_is_monotone_in_best_energy():
    design, spec = build_cantilever()
    objective = Objective(Material())
    grammar = base_truss_grammar(spec, Material())
    result = ShapeAnnealer(
        spec, objective, AnnealConfig(max_iterations=150, stall_window=999, seed=3)
    ).run(design, grammar)

    best = [h["best_energy"] for h in result.history]
    assert all(later <= earlier + 1e-9 for earlier, later in zip(best, best[1:]))
