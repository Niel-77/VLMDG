"""Command-line entry point for :mod:`vlmdg`.

Examples
--------
Shape-annealing-only baseline (no VLM), 50 iterations::

    vlmdg baseline --iterations 50 --seed 0

Full run with the informed VLM rule generator (offline mock provider)::

    vlmdg run --mode informed --provider mock --iterations 4000
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from typing import Sequence

from vlmdg.config import AnnealConfig, RunConfig, VLMConfig
from vlmdg.optimize.baselines import STALL_CALLBACKS
from vlmdg.optimize.sizing import SIZER_KINDS
from vlmdg.problems import DEFAULT_PROBLEM


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--problem", default=DEFAULT_PROBLEM, help="registered problem name")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", default=None, help="output run directory")
    parser.add_argument("--quiet", action="store_true", help="suppress per-step logging")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="vlmdg", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_base = sub.add_parser("baseline", help="shape-annealing-only baseline run")
    _add_common(p_base)
    p_base.add_argument("--iterations", type=int, default=50, help="number of SA iterations")

    p_run = sub.add_parser("run", help="full run (optionally VLM-backed)")
    _add_common(p_run)
    p_run.add_argument("--mode", choices=STALL_CALLBACKS, default="informed")
    p_run.add_argument("--provider", default="mock", help="VLM provider: mock|tamu|anthropic")
    p_run.add_argument("--model", default=VLMConfig().model, help="VLM model id")
    p_run.add_argument("--iterations", type=int, default=AnnealConfig().max_iterations)
    p_run.add_argument("--stall-window", type=int, default=AnnealConfig().stall_window)
    p_run.add_argument("--max-stalls", type=int, default=AnnealConfig().max_stalls)
    p_run.add_argument("--initial-temperature", type=float,
                       default=AnnealConfig().initial_temperature)
    p_run.add_argument(
        "--cooling-rate", default=str(AnnealConfig().cooling_rate),
        help="temperature factor per 20 iterations, or 'auto' to reach "
        "--min-temperature at the end of the budget",
    )
    p_run.add_argument("--min-temperature", type=float, default=AnnealConfig().min_temperature)
    p_run.add_argument("--reheat-fraction", type=float, default=AnnealConfig().reheat_fraction,
                       help="reheat to this fraction of the initial temperature at a stall")
    p_run.add_argument("--max-rule-stalls", type=int, default=None,
                       help="generate rules only at the first N stalls; later stalls reheat")
    p_run.add_argument("--extra-rules", default=None,
                       help="JSON rule pool added to the initial grammar (frozen grammar runs)")
    p_run.add_argument("--snapshot-every", type=int, default=0,
                       help="store current and best designs every N iterations")
    p_run.add_argument(
        "--sizer", choices=SIZER_KINDS, default="auto",
        help="per-topology sizer: auto/fsd (areas, fast), slsqp (areas), "
        "shape (areas + free joint positions, slowest)",
    )
    p_run.add_argument(
        "--max-proposals", type=int, default=RunConfig().max_proposal_attempts,
        help="rule proposals per stall; a rejected rule is sent back with its reasons",
    )
    p_run.add_argument(
        "--allow-duplicate-rules", action="store_true",
        help="skip the novelty gate that rejects rules an existing rule already covers",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the requested command."""
    args = build_parser().parse_args(argv)
    # Import here so ``--help`` stays fast and import-error-free.
    from vlmdg.runner import run_baseline, run_experiment  # pylint: disable=import-outside-toplevel

    if args.command == "baseline":
        run_baseline(
            iterations=args.iterations,
            problem=args.problem,
            seed=args.seed,
            out_dir=args.out,
            verbose=not args.quiet,
        )
        return 0

    if args.command == "run":
        steps = AnnealConfig().steps_per_temperature
        if args.cooling_rate != "auto":
            cooling = float(args.cooling_rate)
        else:
            # Geometric schedule from the initial to the minimum temperature
            # over the iteration budget.
            cooling = (args.min_temperature / args.initial_temperature) ** (
                steps / max(1, args.iterations)
            )
        config = RunConfig(
            problem=args.problem,
            mode=args.mode,
            anneal=dataclasses.replace(
                AnnealConfig(),
                max_iterations=args.iterations,
                stall_window=args.stall_window,
                max_stalls=args.max_stalls,
                seed=args.seed,
                initial_temperature=args.initial_temperature,
                cooling_rate=cooling,
                min_temperature=args.min_temperature,
                reheat_fraction=args.reheat_fraction,
            ),
            vlm=dataclasses.replace(VLMConfig(), provider=args.provider, model=args.model),
            sizer=args.sizer,
            max_proposal_attempts=args.max_proposals,
            check_duplicates=not args.allow_duplicate_rules,
            max_rule_stalls=args.max_rule_stalls,
            extra_rules_path=args.extra_rules,
            snapshot_every=args.snapshot_every,
        )
        run_experiment(config, out_dir=args.out, verbose=not args.quiet)
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
