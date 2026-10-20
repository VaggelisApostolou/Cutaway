import asyncio

import pytest

from cutaway.agents.crew import Producer, verify
from cutaway.agents.llm import LLM, LLMSettings
from cutaway.agents.reel import Persona
from cutaway.engine import simulate_match


@pytest.fixture(scope="module")
def match():
    return simulate_match(17)


def produce(match, **persona):
    producer = Producer(LLM(LLMSettings("offline", "offline")))
    return asyncio.run(producer.produce(match, Persona(**persona)))


def test_offline_reel_is_verified_and_fits_budget(match):
    reel = produce(match, mode="casual", target_seconds=60)
    assert reel.verified and not reel.issues
    assert 3 <= len(reel.segments) and reel.screen_seconds <= 60 * 1.15
    assert all(s.narration for s in reel.segments)
    goals = [e for e in match.events if e.type.value == "goal"]
    assert sum(s.kind == "goal" for s in reel.segments) == len(goals)


def test_segments_are_chronological_with_evidence(match):
    reel = produce(match, mode="analyst", language="el", club_id="NGR", target_seconds=90)
    ts = [s.start_t for s in reel.segments]
    assert ts == sorted(ts)
    ids = {e.id for e in match.events}
    for s in reel.segments:
        assert s.evidence and set(s.evidence) <= ids
        for o in s.overlays:
            assert set(o.evidence) <= ids


def test_player_mode_focuses_player(match):
    reel = produce(match, mode="player", player_id="NGR-9", target_seconds=45)
    assert all(s.focus_player_id == "NGR-9" for s in reel.segments if s.kind != "big_save")


def test_verifier_catches_invented_numbers(match):
    reel = produce(match, mode="casual")
    seg = reel.segments[0]
    seg.narration = "An incredible 97% chance at 150 km/h."
    problems = verify(reel.segments, reel.intro, reel.outro, match)
    assert any(p.moment_id == seg.moment_id and "97" in p.problem for p in problems)
