"""Collect every accepted VLM rule of a study into one rule pool.

Reads runs/study_<problem>/{uninformed,informed}/seed*/rulegen_proposals.json
and writes runs/study_<problem>/rule_pool.json, used by the expanded condition.

Run from vlmdg/:

    python studies/build_pool.py --problem ten_bar
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problem", default="ten_bar")
    parser.add_argument("--conditions", default="uninformed,informed")
    parser.add_argument("--root", default=None, help="study folder (default: runs/study_<problem>)")
    args = parser.parse_args()

    root = Path(args.root or f"runs/study_{args.problem}")
    pool = []
    for condition in args.conditions.split(","):
        for path in sorted((root / condition).glob("seed*/rulegen_proposals.json")):
            seed = int(re.search(r"seed(\d+)", path.parent.name).group(1))
            for proposal in json.loads(path.read_text()):
                if not proposal["accepted"]:
                    continue
                pool.append(
                    {
                        "name": f"{condition[0]}{seed}_s{proposal['stall_index']}",
                        "source": proposal["source"],
                        "description": proposal.get("description", ""),
                        "condition": condition,
                        "seed": seed,
                        "stall": proposal["stall_index"],
                    }
                )
    out = root / "rule_pool.json"
    out.write_text(json.dumps(pool, indent=2))
    print(f"{len(pool)} rules -> {out}")


if __name__ == "__main__":
    main()
