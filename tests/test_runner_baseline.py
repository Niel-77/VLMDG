"""End-to-end smoke test of the shape-annealing-only baseline runner."""

import json

from vlmdg.runner import run_baseline


def test_run_baseline_ten_bar_writes_outputs_and_stays_feasible(tmp_path):
    out = tmp_path / "baseline"
    result = run_baseline(
        iterations=6, problem="ten_bar", seed=0, out_dir=out, verbose=False
    )

    for name in ("initial_design.png", "best_design.png", "best_design.json",
                 "convergence.png", "result.json", "run_config.json"):
        assert (out / name).is_file(), name

    # The sized 10-bar starting point is feasible, and best-so-far never worsens.
    assert result.iterations == 6
    assert result.rules_added == 0
    assert result.best_eval.feasible
    assert result.history[0]["best_energy"] >= result.best_eval.energy

    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["problem"] == "ten_bar"
    assert cfg["material"]["stress_allowable"] == 172e6
