"""Run one condition of the study for several seeds in parallel.

Each run writes to runs/study_<problem>/<condition>/seed<k>/, with a live
run.log. All conditions use the same annealing; they differ only in what
happens at a stall:

  baseline     fixed grammar; reheat only
  random       add a chain of 2-3 base rules
  uninformed   the VLM writes a rule from the design image and rule list
  informed     the VLM also sees how each rule has performed
  expanded     fixed grammar plus a pool of VLM rules (--pool); reheat only

Run from vlmdg/:

    python studies/run_study.py --condition baseline --seeds 0-9
    python studies/run_study.py --condition expanded --pool runs/study_ten_bar/rule_pool.json
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

#: Annealing protocol shared by every condition.
PROTOCOL = {
    "ten_bar": [
        "--iterations", "1000", "--stall-window", "50", "--max-stalls", "1000000",
        "--initial-temperature", "0.1", "--cooling-rate", "auto",
        "--reheat-fraction", "0.5", "--max-rule-stalls", "10", "--snapshot-every", "30",
    ],
    # Same schedule; planar bridge (see vlmdg.problems.bridge).
    "bridge": [
        "--iterations", "1000", "--stall-window", "50", "--max-stalls", "1000000",
        "--initial-temperature", "0.1", "--cooling-rate", "auto",
        "--reheat-fraction", "0.5", "--max-rule-stalls", "10", "--snapshot-every", "30",
    ],
}
MODES = {"baseline": "reheat", "random": "random", "uninformed": "uninformed",
         "informed": "informed", "expanded": "reheat"}


def parse_seeds(text: str) -> list[int]:
    """Parse ``0-9`` or ``0,3,5`` into a list of seeds."""
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problem", default="ten_bar")
    parser.add_argument("--condition", choices=MODES, required=True)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pool", default=None, help="rule pool JSON for 'expanded'")
    parser.add_argument("--tag", default=None, help="output name (default: condition)")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--provider", default="tamu")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="seconds before a hung run is killed and restarted once")
    args = parser.parse_args()

    out_root = Path(args.out_root or f"runs/study_{args.problem}")
    out_root.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.condition
    if args.condition == "expanded" and not args.pool:
        parser.error("--pool is required for the expanded condition")

    def run(seed: int) -> str:
        out = out_root / tag / f"seed{seed}"
        if (out / "result.json").exists():
            return f"{tag} seed{seed} skipped (done)"
        cmd = ["vlmdg", "run", "--problem", args.problem, "--mode", MODES[args.condition],
               "--provider", args.provider, "--model", args.model, "--seed", str(seed),
               "--out", str(out), *PROTOCOL[args.problem]]
        if args.pool:
            cmd += ["--extra-rules", args.pool]
        for attempt in (1, 2):
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            with open(out / "run.log", "w", encoding="utf-8") as fh:
                try:
                    code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                          timeout=args.timeout,
                                          env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
                except subprocess.TimeoutExpired:
                    code = None
            if code is not None:
                return f"{tag} seed{seed} exit {code}"
            print(f"{tag} seed{seed} timed out after {args.timeout} s (attempt {attempt})",
                  flush=True)
        return f"{tag} seed{seed} exit timeout"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(run, parse_seeds(args.seeds)):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
