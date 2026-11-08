"""End-to-end experiment runner.

``run_experiment`` wires a problem, the base grammar, the objective, an optional
VLM-backed stall callback, and the shape annealer together, then writes a run
directory containing:

* ``run_config.json``    - the exact configuration used;
* ``initial_design.png`` - the starting truss ``D_0``;
* ``best_design.json`` / ``best_design.png`` - the optimized result ``D_N``;
* ``convergence.png``    - current and best-so-far energy vs iteration;
* ``result.json``        - summary, full history, and per-stall diagnostics.
"""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from vlmdg.config import RunConfig, load_env
from vlmdg.design.plotting import render_truss_figure_png
from vlmdg.grammar.base_truss import base_truss_grammar
from vlmdg.optimize.baselines import make_stall_callback
from vlmdg.optimize.objective import Objective
from vlmdg.optimize.shape_annealing import AnnealResult, ShapeAnnealer
from vlmdg.optimize.sizing import make_sizer
from vlmdg.problems import DEFAULT_PROBLEM, get_material, get_problem
from vlmdg.reporting import plot_convergence


def _timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_experiment(
    config: RunConfig,
    out_dir: Optional[str | Path] = None,
    *,
    verbose: bool = True,
) -> AnnealResult:
    """Run one experiment arm described by ``config`` and write its run directory."""
    load_env()

    if out_dir is None:
        out_dir = Path(config.output_dir) / f"{config.problem}_{config.mode}_{_timestamp()}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    material = config.material or get_material(config.problem)
    design0, spec = get_problem(config.problem)
    objective = Objective(material, penalty_weight=config.anneal.penalty_weight)
    grammar = base_truss_grammar(spec, material)
    if config.extra_rules_path:
        for rule in load_rule_pool(config.extra_rules_path):
            grammar.add_rule(rule)
    sizer = make_sizer(material, config.sizer, spec)

    on_stall = make_stall_callback(
        config.mode,
        spec=spec,
        objective=objective,
        grammar=grammar,
        run_config=config,
        sizer=sizer,
    )

    design0_sized = sizer(design0) if sizer is not None else design0
    render_truss_figure_png(design0_sized, out_dir / "initial_design.png")

    if verbose:
        on_stall = _logged_stall_callback(on_stall, out_dir)
        print(f"{_now()} start problem={config.problem} mode={config.mode} "
              f"seed={config.anneal.seed} iterations={config.anneal.max_iterations}", flush=True)
    progress = _make_printer() if verbose else None
    record_every = max(1, config.anneal.max_iterations // 100)
    annealer = ShapeAnnealer(
        spec, objective, config.anneal, sizer=sizer, record_every=record_every,
        snapshot_every=config.snapshot_every,
    )
    result = annealer.run(design0, grammar, on_stall=on_stall, progress=progress)

    result.save(out_dir)
    proposals = getattr(on_stall, "proposals", None)
    if proposals:
        (out_dir / "rulegen_proposals.json").write_text(
            json.dumps(proposals, indent=2), encoding="utf-8"
        )
    # The cantilever is in normalized units; the other problems are in newtons.
    if config.problem == "cantilever":
        plot_convergence(result, out_dir / "convergence.png", unit=None, scale=1.0)
    else:
        plot_convergence(result, out_dir / "convergence.png")
    cfg_dict = _config_to_dict(config)
    cfg_dict["material"] = _config_to_dict(material)  # the resolved material
    (out_dir / "run_config.json").write_text(
        json.dumps(cfg_dict, indent=2), encoding="utf-8"
    )

    if verbose:
        summary = result.summary()
        print(f"\n{_now()} [vlmdg] run complete -> {out_dir}")
        print(f"[vlmdg] best mass={summary['best_mass']:.4g} "
              f"feasible={summary['best_feasible']} "
              f"rules_added={summary['rules_added']} stalls={summary['n_stalls']}")
    return result


def load_rule_pool(path: str | Path) -> list:
    """Compile a JSON rule pool (``[{"name", "source"}, ...]``) into rules.

    The rules keep their generated docstrings as descriptions and are marked
    ``origin="pooled"`` so their utility is reported separately from ``G_0``.
    """
    from vlmdg.rulegen.sandbox import compile_rule  # pylint: disable=import-outside-toplevel

    rules = []
    for entry in json.loads(Path(path).read_text(encoding="utf-8")):
        rule = compile_rule(entry["source"], entry["name"])
        rule.origin = "pooled"
        rules.append(rule)
    return rules


def run_baseline(
    *,
    iterations: int = 50,
    problem: str = DEFAULT_PROBLEM,
    seed: int = 0,
    out_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> AnnealResult:
    """Shape-annealing-only baseline: no VLM, no rule added, ``iterations`` steps.

    Produces the convergence plot and the optimized-truss image of the
    fixed-grammar baseline.
    """
    config = RunConfig(problem=problem, mode="reheat")
    config = dataclasses.replace(
        config,
        anneal=dataclasses.replace(
            config.anneal,
            max_iterations=iterations,
            seed=seed,
            # keep the stall window above the budget so the 50-step run is pure SA
            stall_window=max(iterations + 1, config.anneal.stall_window),
        ),
    )
    if out_dir is None:
        out_dir = Path(config.output_dir) / f"baseline_{problem}_{iterations}it_seed{seed}"
    return run_experiment(config, out_dir=out_dir, verbose=verbose)


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _make_printer():
    """Return a progress callback that prints every recorded history point."""
    def printer(point: dict) -> None:
        print(
            f"{_now()} it={point['iteration']:>5}  T={point['temperature']:.3g}  "
            f"current_energy={point['current_energy']:.4g}  "
            f"best_energy={point['best_energy']:.4g}  best_mass={point['best_mass']:.4g}  "
            f"feasible={point['best_feasible']}  |G|={point['grammar_size']}",
            flush=True,
        )

    return printer


def _logged_stall_callback(on_stall, out_dir: Path):
    """Wrap ``on_stall`` so each stall and each rule proposal is logged as it happens.

    Also rewrites ``rulegen_proposals.json`` after every stall, so the proposals
    so far can be read while the run is going. Returns what ``on_stall`` returns.
    """
    proposals = getattr(on_stall, "proposals", None)

    def logged(diag):
        print(f"{_now()} STALL #{diag.stall_index} at it={diag.iteration}  "
              f"best_mass={diag.best_mass:.4g}  feasible={diag.best_feasible}", flush=True)
        if on_stall is None:
            print(f"{_now()}   no rule added (reheat only)", flush=True)
            return None
        n_before = len(proposals) if proposals is not None else 0
        t0 = time.time()
        rule = on_stall(diag)
        elapsed = time.time() - t0
        if proposals is not None and len(proposals) > n_before:
            record = proposals[-1]
            attempts = list(record.get("rejected_attempts") or []) + [record]
            for k, att in enumerate(attempts, start=1):
                verdict = "ACCEPTED" if att.get("accepted") else "rejected"
                reason = "; ".join(att.get("reasons") or [])[:200]
                screen = att.get("screen") or {}
                counts = ", ".join(f"{key}={val}" for key, val in screen.items() if val)
                print(f"{_now()}   proposal {k}: {verdict}"
                      + (f" - {reason}" if reason else "")
                      + (f"  [screen: {counts}]" if counts else ""), flush=True)
            usage = record.get("usage", {})
            print(f"{_now()}   {len(attempts)} proposal(s), {elapsed:.0f} s, "
                  f"tokens in/out of last={usage.get('input_tokens', 0)}/"
                  f"{usage.get('output_tokens', 0)}", flush=True)
            (out_dir / "rulegen_proposals.json").write_text(
                json.dumps(proposals, indent=2), encoding="utf-8"
            )
        name = getattr(rule, "name", None)
        print(f"{_now()}   rule added: {name}" if name else f"{_now()}   no rule added",
              flush=True)
        return rule

    if proposals is not None:
        logged.proposals = proposals  # type: ignore[attr-defined]
    return logged


def _config_to_dict(config: RunConfig) -> dict:
    """Convert a (nested, frozen) :class:`RunConfig` to plain JSON-friendly data."""
    def convert(obj):
        if dataclasses.is_dataclass(obj):
            return {f.name: convert(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    return convert(config)
