from cutaway.analytics.insights import key_moments, summarize, team_stats
from cutaway.analytics.metrics import expected_goals, pass_difficulty
from cutaway.engine import EventType, simulate_match
from cutaway.engine.models import PITCH_LENGTH, PITCH_WIDTH

import pytest


@pytest.fixture(scope="module")
def match():
    return simulate_match(7)


def test_same_seed_same_match(match):
    again = simulate_match(7)
    assert [e.model_dump() for e in again.events] == [e.model_dump() for e in match.events]


def test_time_is_monotonic(match):
    ts = [e.t for e in match.events]
    assert ts == sorted(ts)


def test_coordinates_on_pitch(match):
    for e in match.events:
        for x, y in ((e.x, e.y), (e.end_x, e.end_y)):
            if x is not None:
                assert 0 <= x <= PITCH_LENGTH and 0 <= y <= PITCH_WIDTH, e


def test_score_matches_goal_events(match):
    goals = [e for e in match.events if e.type == EventType.GOAL]
    last = match.events[-1]
    assert last.home_score == sum(g.team_id == match.home.id for g in goals)
    assert last.away_score == sum(g.team_id == match.away.id for g in goals)


def test_every_goal_is_a_key_moment(match):
    goal_shots = {e.id for e in match.events if e.type == EventType.SHOT and e.outcome == "goal"}
    moment_goals = {int(m.id.split("-")[1]) for m in key_moments(match) if m.kind == "goal"}
    assert goal_shots == moment_goals


def test_moment_evidence_exists(match):
    ids = {e.id for e in match.events}
    for m in key_moments(match):
        assert m.event_ids and set(m.event_ids) <= ids


def test_team_stats_realistic(match):
    for s in team_stats(match).values():
        assert 0.6 < s["pass_accuracy"] < 0.95
        assert 3 <= s["shots"] <= 30


def test_summary_serialises(match):
    import json
    json.dumps(summarize(match))


def test_xg_behaves():
    close = expected_goals(99, 34).value
    far = expected_goals(80, 34).value
    wide = expected_goals(99, 10).value
    assert 0 < far < close < 1
    assert wide < close
    assert expected_goals(99, 34, header=True).value < close


def test_pass_difficulty_components_sum():
    d = pass_difficulty((40, 34), (80, 30), [(55, 33), (60, 36), (79, 31)])
    assert d.lines_broken == 3
    parts = d.distance_pts + d.receiver_tightness_pts + d.passer_pressure_pts + d.lines_broken_pts
    assert d.total == pytest.approx(min(100, parts), abs=0.2)
    assert pass_difficulty((40, 34), (45, 34), []).total < d.total
