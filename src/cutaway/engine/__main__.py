"""Generate a synthetic match: python -m cutaway.engine --seed 17 --out data/"""

import argparse
import json
from pathlib import Path

from cutaway.analytics.insights import summarize

from .simulator import simulate_match


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    match = simulate_match(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"match-{args.seed}.json").write_text(match.model_dump_json())
    summary = summarize(match)
    (args.out / f"insights-{args.seed}.json").write_text(json.dumps(summary))

    last = match.events[-1]
    print(f"{match.home.name} {last.home_score}-{last.away_score} {match.away.name}  ({len(match.events)} events)")
    for m in summary["moments"]:
        if m["kind"] in ("goal", "tactical_shift", "momentum_shift", "big_save"):
            print(f"  {m['clock']:>8}  {m['kind']:<16} {m['team_id']}  importance={m['importance']}")


if __name__ == "__main__":
    main()
