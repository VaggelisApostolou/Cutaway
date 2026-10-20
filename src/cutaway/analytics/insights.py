"""Match-level intelligence: momentum, control vs chaos, key moments, player stats.

Everything here is deterministic and cites event ids, so agents can explain a moment
and a verifier can check each claim against the source events.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from cutaway.engine.models import ON_BALL_TYPES, EventType, Match, MatchEvent

from .metrics import in_box, to_rel

FINAL_THIRD_X = 70.0
SPRINT_KMH = 30.0
ROCKET_KMH = 110.0
BIG_CHANCE_XG = 0.25


def _rel_end(e: MatchEvent) -> tuple[float, float] | None:
    if e.end_x is None or e.attack_dir is None:
        return None
    return to_rel(e.end_x, e.end_y, e.attack_dir)


def _rel_start(e: MatchEvent) -> tuple[float, float] | None:
    if e.x is None or e.attack_dir is None:
        return None
    return to_rel(e.x, e.y, e.attack_dir)


def threat(e: MatchEvent) -> float:
    """Attacking threat contributed by one event (used for momentum)."""
    if e.type == EventType.SHOT:
        return e.xg or 0.0
    if e.type == EventType.CORNER:
        return 0.03
    if e.type in (EventType.PASS, EventType.CARRY, EventType.DRIBBLE) and e.outcome != "incomplete":
        end = _rel_end(e)
        if end is None or e.outcome == "fail":
            return 0.0
        if in_box(*end):
            return 0.05
        if end[0] >= FINAL_THIRD_X:
            return 0.02
    return 0.0


# ---------------------------------------------------------------- momentum
def momentum_timeline(match: Match) -> list[dict]:
    """Per real-time minute: threat per team and a smoothed momentum in [-1, 1] (+ = home)."""
    last_t = match.events[-1].t
    n = int(last_t // 60) + 1
    home = [0.0] * n
    away = [0.0] * n
    clocks: dict[int, str] = {}
    for e in match.events:
        b = int(e.t // 60)
        clocks.setdefault(b, e.clock)
        if not e.team_id:
            continue
        val = threat(e)
        if e.team_id == match.home.id:
            home[b] += val
        else:
            away[b] += val
    out, ema = [], 0.0
    for b in range(n):
        ema = 0.6 * ema + 0.4 * (home[b] - away[b])
        out.append({
            "bucket": b,
            "clock": clocks.get(b, ""),
            "home_threat": round(home[b], 3),
            "away_threat": round(away[b], 3),
            "momentum": round(math.tanh(ema * 5), 3),
        })
    return out


# ---------------------------------------------------------- control vs chaos
def possession_chains(match: Match) -> list[dict]:
    """Consecutive on-ball events by one team."""
    chains: list[dict] = []
    for e in match.events:
        if e.type not in ON_BALL_TYPES or not e.team_id:
            continue
        if chains and chains[-1]["team_id"] == e.team_id and e.type != EventType.KICK_OFF:
            chains[-1]["event_ids"].append(e.id)
            chains[-1]["end_t"] = e.t
        else:
            chains.append({"team_id": e.team_id, "event_ids": [e.id], "start_t": e.t, "end_t": e.t})
    return chains


def control_chaos(match: Match, window_s: float = 300.0) -> list[dict]:
    """Rolling 5-minute index. 0 = calm control, 100 = end-to-end chaos.

    chaos = 50 * min(turnovers_per_min / 4, 1) + 50 * min((1 - pass_completion) / 0.4, 1)
    """
    chains = possession_chains(match)
    passes = [e for e in match.events if e.type == EventType.PASS]
    last_t = match.events[-1].t
    out = []
    for b in range(int(last_t // 60) + 1):
        hi = (b + 1) * 60.0
        lo = max(0.0, hi - window_s)
        span_min = (hi - lo) / 60
        in_win = [c for c in chains if lo <= c["start_t"] < hi]
        turnovers = max(0, len(in_win) - 1)
        win_passes = [p for p in passes if lo <= p.t < hi]
        completion = (
            sum(p.outcome == "complete" for p in win_passes) / len(win_passes) if win_passes else 1.0
        )
        tpm = turnovers / span_min
        chaos = 50 * min(tpm / 4, 1) + 50 * min((1 - completion) / 0.4, 1)
        home_time = sum(c["end_t"] - c["start_t"] + 2 for c in in_win if c["team_id"] == match.home.id)
        all_time = sum(c["end_t"] - c["start_t"] + 2 for c in in_win) or 1
        out.append({
            "bucket": b,
            "chaos_index": round(chaos, 1),
            "label": "control" if chaos < 40 else "contested" if chaos < 60 else "chaos",
            "turnovers_per_min": round(tpm, 2),
            "pass_completion": round(completion, 3),
            "home_possession": round(home_time / all_time, 3),
        })
    return out


# ------------------------------------------------------------- player stats
def player_stats(match: Match) -> dict[str, dict]:
    s: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for e in match.events:
        pid = e.player_id
        if not pid:
            continue
        p = s[pid]
        if e.type == EventType.PASS:
            p["passes"] += 1
            p["passes_completed"] += e.outcome == "complete"
            p["difficulty_sum"] += e.pass_difficulty or 0
            p["lines_broken"] += (e.lines_broken or 0) if e.outcome == "complete" else 0
        elif e.type == EventType.SHOT:
            p["shots"] += 1
            p["xg"] += e.xg or 0
            p["shots_on_target"] += e.outcome in ("goal", "saved")
            p["top_shot_kmh"] = max(p["top_shot_kmh"], e.ball_speed_kmh or 0)
        elif e.type == EventType.GOAL:
            p["goals"] += 1
            if e.related_player_id:
                s[e.related_player_id]["assists"] += 1
        elif e.type == EventType.CARRY:
            p["carry_m"] += e.distance_m or 0
            p["top_speed_kmh"] = max(p["top_speed_kmh"], e.ball_speed_kmh or 0)
        elif e.type == EventType.DRIBBLE:
            p["dribbles"] += 1
            p["dribbles_won"] += e.outcome == "complete"
        elif e.type in (EventType.TACKLE, EventType.INTERCEPTION):
            p["ball_wins"] += 1
        elif e.type == EventType.PRESSURE:
            p["pressures"] += 1
        elif e.type == EventType.SAVE:
            p["saves"] += 1
        p["touches"] += e.type in ON_BALL_TYPES
    out = {}
    for pid, p in s.items():
        d = {k: round(v, 3) if isinstance(v, float) else v for k, v in p.items()}
        if p["passes"]:
            d["pass_accuracy"] = round(p["passes_completed"] / p["passes"], 3)
            d["avg_pass_difficulty"] = round(p["difficulty_sum"] / p["passes"], 1)
        d.pop("difficulty_sum", None)
        out[pid] = d
    return out


def team_stats(match: Match) -> dict[str, dict]:
    chains = possession_chains(match)
    total = sum(c["end_t"] - c["start_t"] + 2 for c in chains) or 1
    out = {}
    for team in (match.home, match.away):
        ev = [e for e in match.events if e.team_id == team.id]
        passes = [e for e in ev if e.type == EventType.PASS]
        shots = [e for e in ev if e.type == EventType.SHOT]
        out[team.id] = {
            "goals": sum(e.type == EventType.GOAL for e in ev),
            "xg": round(sum(e.xg or 0 for e in shots), 2),
            "shots": len(shots),
            "shots_on_target": sum(e.outcome in ("goal", "saved") for e in shots),
            "passes": len(passes),
            "pass_accuracy": round(sum(p.outcome == "complete" for p in passes) / max(1, len(passes)), 3),
            "possession": round(
                sum(c["end_t"] - c["start_t"] + 2 for c in chains if c["team_id"] == team.id) / total, 3
            ),
            "corners": sum(e.type == EventType.CORNER for e in ev),
            "fouls": sum(e.type == EventType.FOUL for e in ev),
        }
    return out


# -------------------------------------------------------------- key moments
@dataclass
class Moment:
    id: str
    kind: str
    t: float
    clock: str
    team_id: str | None
    importance: float
    event_ids: list[int]
    player_ids: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _chain_lookup(match: Match) -> dict[int, list[int]]:
    lookup = {}
    for c in possession_chains(match):
        for eid in c["event_ids"]:
            lookup[eid] = c["event_ids"]
    return lookup


def key_moments(match: Match) -> list[Moment]:
    """Ranked moments, each with the event ids that prove it."""
    by_id = {e.id: e for e in match.events}
    chain_of = _chain_lookup(match)
    moments: list[Moment] = []

    def add(kind, e: MatchEvent, importance, event_ids, players, **facts):
        moments.append(Moment(
            id=f"{kind}-{e.id}", kind=kind, t=e.t, clock=e.clock, team_id=e.team_id,
            importance=round(min(100.0, importance), 1), event_ids=sorted(set(event_ids)),
            player_ids=[p for p in players if p], facts=facts,
        ))

    for e in match.events:
        if e.type == EventType.SHOT:
            chain = chain_of.get(e.id, [e.id])
            build_up = [i for i in chain if i <= e.id][-6:]
            if e.outcome == "goal":
                goal_ev = by_id[e.id + 1]
                add("goal", e, 90 + (e.xg < 0.1) * 10, build_up + [goal_ev.id],
                    [e.player_id, goal_ev.related_player_id],
                    xg=e.xg, shot_speed_kmh=e.ball_speed_kmh, distance_m=e.distance_m,
                    score=f"{goal_ev.home_score}-{goal_ev.away_score}", detail=e.detail,
                    build_up_passes=sum(by_id[i].type == EventType.PASS for i in build_up))
            elif (e.xg or 0) >= BIG_CHANCE_XG:
                add("big_chance_missed", e, 55 + e.xg * 50, build_up, [e.player_id],
                    xg=e.xg, outcome=e.outcome, distance_m=e.distance_m)
            if e.outcome != "goal" and (e.ball_speed_kmh or 0) >= ROCKET_KMH:
                add("rocket_shot", e, 35, [e.id], [e.player_id], shot_speed_kmh=e.ball_speed_kmh,
                    outcome=e.outcome)
        elif e.type == EventType.SAVE:
            shot = by_id[e.id - 1]
            if (shot.xg or 0) >= 0.15:
                add("big_save", e, 50 + shot.xg * 60, [shot.id, e.id], [e.player_id, shot.player_id],
                    xg_denied=shot.xg, shot_speed_kmh=shot.ball_speed_kmh)
        elif e.type == EventType.PASS and e.outcome == "complete" and (e.pass_difficulty or 0) >= 35:
            chain = chain_of.get(e.id, [])
            later_shot = next((by_id[i] for i in chain if i > e.id and by_id[i].type == EventType.SHOT), None)
            if (later_shot and e.pass_difficulty >= 45) or (e.lines_broken or 0) >= 3:
                add("line_breaking_pass", e, 30 + (e.pass_difficulty - 35) + (15 if later_shot else 0),
                    [e.id] + ([later_shot.id] if later_shot else []), [e.player_id, e.receiver_id],
                    pass_difficulty=e.pass_difficulty, lines_broken=e.lines_broken,
                    distance_m=e.distance_m, led_to_shot=bool(later_shot))
        elif e.type == EventType.CARRY and (e.ball_speed_kmh or 0) >= SPRINT_KMH and (e.distance_m or 0) >= 10:
            add("speed_burst", e, 30, [e.id], [e.player_id], speed_kmh=e.ball_speed_kmh,
                distance_m=e.distance_m)
        elif e.type == EventType.TACKLE:
            start = _rel_start(e)
            if start and start[0] <= 16.5:  # defender's own box (team frame)
                add("last_ditch_tackle", e, 45, [e.id], [e.player_id, e.related_player_id])
        elif e.type in (EventType.TACTICAL_SHIFT, EventType.SUBSTITUTION):
            add(e.type.value, e, 25 if e.type == EventType.SUBSTITUTION else 40, [e.id],
                [e.player_id, e.related_player_id], detail=e.detail)

    moments = _cap(moments, "speed_burst", 6, key=lambda m: m.facts["speed_kmh"])
    moments = _cap(moments, "line_breaking_pass", 8, key=lambda m: m.importance)
    moments.extend(_momentum_shifts(match))
    moments.sort(key=lambda m: m.t)
    return moments


def _cap(moments: list[Moment], kind: str, n: int, key) -> list[Moment]:
    """Keep only the top ``n`` moments of a frequent kind."""
    keep = {id(m) for m in sorted((m for m in moments if m.kind == kind), key=key, reverse=True)[:n]}
    return [m for m in moments if m.kind != kind or id(m) in keep]


def _momentum_shifts(match: Match) -> list[Moment]:
    """Momentum swings, linked to plausible causes (tactical shift, sub, goal) just before."""
    tl = momentum_timeline(match)
    out: list[Moment] = []
    last_shift = -99
    for i in range(5, len(tl)):
        before, now = tl[i - 5]["momentum"], tl[i]["momentum"]
        if abs(now - before) < 0.9 or (before * now) > 0 or i - last_shift < 8:
            continue
        last_shift = i
        lo, hi = (i - 5) * 60, (i + 1) * 60
        team_id = match.home.id if now > 0 else match.away.id
        window = [e for e in match.events if lo <= e.t < hi and e.team_id == team_id and threat(e) > 0]
        top = sorted(window, key=threat, reverse=True)[:5]
        causes = [e for e in match.events
                  if (i - 10) * 60 <= e.t < hi
                  and e.type in (EventType.TACTICAL_SHIFT, EventType.SUBSTITUTION, EventType.GOAL)]
        anchor = top[0] if top else next(e for e in match.events if e.t >= lo)
        out.append(Moment(
            id=f"momentum_shift-{i}", kind="momentum_shift", t=anchor.t, clock=anchor.clock,
            team_id=team_id, importance=round(50 + abs(now - before) * 15, 1),
            event_ids=sorted({e.id for e in top + causes}),
            player_ids=[],
            facts={"momentum_before": before, "momentum_after": now,
                   "possible_causes": [{"event_id": c.id, "type": c.type.value, "team_id": c.team_id,
                                        "detail": c.detail} for c in causes]},
        ))
    return out


def summarize(match: Match) -> dict:
    return {
        "team_stats": team_stats(match),
        "player_stats": player_stats(match),
        "momentum": momentum_timeline(match),
        "control_chaos": control_chaos(match),
        "moments": [m.as_dict() for m in key_moments(match)],
    }
