"""Run the whole study: every condition and seed, the analysis, and the plots.

For each problem (in parallel):

  1. fixed grammar, random rules, uninformed VLM and informed VLM runs
  2. the rule pool from the accepted VLM rules
  3. expanded grammar runs with that pool
  4. analysis: summary tables, convergence plot, best designs, and GIFs

Results go to runs/study_<problem>/, the analysis to runs/study_<problem>/analysis/,
and progress to runs/run_all.log (--runs picks another folder). Finished runs are
skipped, so rerunning the command resumes the study; --fresh deletes earlier
results first. A run that hangs (for example on a model call) is restarted once.

Run from vlmdg/:

    python -m vlmdg.run_all
    python -m vlmdg.run_all --problems bridge --seeds 0-4
    python -m vlmdg.run_all --provider mock --seeds 0-1 --no-gif   (offline check)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[2]
STUDIES = ROOT / "studies"
DEFAULT_RUNS = ROOT / "runs"
RUNS = DEFAULT_RUNS  # set from --runs in main
GROWING = ("baseline", "random", "uninformed", "informed")

_log_lock = Lock()


def log(message: str) -> None:
    """Print a time-stamped line and append it to runs/run_all.log."""
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    with _log_lock:
        print(line, flush=True)
        with open(RUNS / "run_all.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def step(name: str, cmd: list[str], logfile: Path) -> bool:
    """Run one command with its output in logfile; report and return success."""
    with open(logfile, "w", encoding="utf-8") as fh:
        code = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    log(f"{name}: {'done' if code == 0 else f'FAILED (exit {code}, see {logfile.name})'}")
    return code == 0


def run_problem(problem: str, args: argparse.Namespace) -> bool:
    """Run every condition of one problem, then build the pool and analyze."""
    py = sys.executable
    out = RUNS / f"study_{problem}"
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    common = ["--problem", problem, "--seeds", args.seeds, "--workers", str(args.workers),
              "--provider", args.provider, "--model", args.model,
              "--timeout", str(args.timeout), "--out-root", str(out)]

    def study(cond: str, extra: list[str] | None = None) -> bool:
        cmd = [py, str(STUDIES / "run_study.py"), "--condition", cond, *common, *(extra or [])]
        return step(f"{problem} {cond}", cmd, RUNS / f"study_{problem}.{cond}.driver.log")

    log(f"{problem}: start")
    with ThreadPoolExecutor(max_workers=len(GROWING)) as pool:
        ok = all(pool.map(study, GROWING))

    pool_file = out / "rule_pool.json"
    ok &= step(f"{problem} rule pool",
               [py, str(STUDIES / "build_pool.py"), "--problem", problem, "--root", str(out)],
               RUNS / f"study_{problem}.pool.log")
    if pool_file.exists():
        ok &= study("expanded", ["--pool", str(pool_file)])

    cmd = [py, str(STUDIES / "analyze.py"), "--problem", problem, "--root", str(out)]
    if args.no_gif:
        cmd.append("--no-gif")
    ok &= step(f"{problem} analysis", cmd, RUNS / f"study_{problem}.analysis.log")
    log(f"{problem}: {'finished' if ok else 'finished with failures'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problems", default="ten_bar,bridge",
                        help="comma-separated problems (default: ten_bar,bridge)")
    parser.add_argument("--seeds", default="0-9", help="seed range, e.g. 0-9 (default)")
    parser.add_argument("--workers", type=int, default=5,
                        help="parallel runs per condition (default: 5)")
    parser.add_argument("--provider", default="tamu", help="tamu (default), openai, or mock")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="seconds before a hung run is restarted (default: 3600)")
    parser.add_argument("--runs", default=str(DEFAULT_RUNS),
                        help="output folder (default: runs/)")
    parser.add_argument("--no-gif", action="store_true", help="skip the GIFs")
    parser.add_argument("--fresh", action="store_true",
                        help="delete earlier results of these problems first")
    args = parser.parse_args()

    if not STUDIES.is_dir():
        parser.error(f"studies/ not found next to src/ (looked in {ROOT})")
    global RUNS
    RUNS = Path(args.runs).resolve()
    RUNS.mkdir(parents=True, exist_ok=True)
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    log(f"study start: problems={problems} seeds={args.seeds} "
        f"provider={args.provider} model={args.model}")
    with ThreadPoolExecutor(max_workers=len(problems)) as pool:
        ok = all(pool.map(lambda p: run_problem(p, args), problems))
    log("all done" if ok else "all done, with failures (see the logs above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
