"""Produce a reel from the command line.

python -m cutaway.agents --seed 17 --mode casual --lang el --club NGR
"""

import argparse
import asyncio

from cutaway.engine import simulate_match

from .crew import Producer
from .reel import Persona


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--mode", choices=["casual", "analyst", "player"], default="casual")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--club")
    ap.add_argument("--player")
    ap.add_argument("--tone", choices=["neutral", "passionate", "playful"], default="neutral")
    ap.add_argument("--seconds", type=int, default=60)
    args = ap.parse_args()

    persona = Persona(mode=args.mode, language=args.lang, club_id=args.club, player_id=args.player,
                      tone=args.tone, target_seconds=args.seconds)
    producer = Producer()
    reel = asyncio.run(producer.produce(simulate_match(args.seed), persona))
    print(f"{reel.title}   [{reel.provider}, verified={reel.verified}, {reel.screen_seconds:.0f}s]")
    print(f"  {reel.intro}")
    for s in reel.segments:
        print(f"  {s.clock:>8} {s.kind:<18} {s.camera:<12} {s.screen_seconds:4.1f}s  {s.narration}")
    print(f"  {reel.outro}")
    for i in reel.issues:
        print(f"  ! {i.moment_id}: {i.problem}")
    print(f"  model calls: {producer.llm.calls}, cache hits: {producer.llm.cache_hits}")


if __name__ == "__main__":
    main()
