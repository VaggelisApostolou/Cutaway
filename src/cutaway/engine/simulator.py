"""Seeded, football-realistic synthetic match simulator.

The model is a possession-chain Markov process over a 2D pitch: the ball carrier chooses
to pass, carry, dribble, shoot or clear based on location, pressure and attributes; the
outcome depends on computed metrics (pass difficulty, xG) plus team form. Team shape is
generated from formation templates that shift with the ball, which gives every event a
22-player snapshot for rendering.

Latent drivers (form random-walk, tactical shifts, substitutions) create real momentum
swings, so the "why" behind a turning point is actually present in the data.
"""

from __future__ import annotations

import math
import random

from cutaway.analytics.metrics import (
    ball_speed_kmh,
    distance,
    expected_goals,
    pass_difficulty,
    shot_angle,
    to_rel,
)

from .models import EventType, Match, MatchEvent, PlayerPosition, Team
from .teams import FORMATIONS, away_team, home_team

PERIOD_SECONDS = 45 * 60
ROLE_GROUP = {
    "GK": "GK",
    "RB": "DF", "LB": "DF", "RCB": "DF", "LCB": "DF",
    "DM": "MF", "RDM": "MF", "LDM": "MF", "RCM": "MF", "LCM": "MF", "CAM": "MF",
    "ST": "FW", "RW": "FW", "LW": "FW", "RAM": "FW", "LAM": "FW",
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class MatchSimulator:
    def __init__(self, seed: int = 7, home: Team | None = None, away: Team | None = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.home = home or home_team()
        self.away = away or away_team()
        self.match_id = f"syn-{seed}"
        self.teams = {self.home.id: self.home, self.away.id: self.away}
        self.players = {p.id: p for t in self.teams.values() for p in t.players}
        self.role = {p.id: p.position for p in self.players.values()}
        self.on_pitch = {t.id: [p.id for p in t.players if p.starter] for t in self.teams.values()}
        self.bench = {t.id: [p.id for p in t.players if not p.starter] for t in self.teams.values()}
        self.score = {self.home.id: 0, self.away.id: 0}
        self.form = {self.home.id: 0.0, self.away.id: 0.0}
        self.press = {self.home.id: 0.5, self.away.id: 0.45}
        self.shifted = {self.home.id: False, self.away.id: False}
        self.sub_plan = {
            tid: sorted(self.rng.sample(range(58, 86), 3)) for tid in self.teams
        }
        self.events: list[MatchEvent] = []
        self.period = 1
        self.t = 0.0
        self.period_start_t = 0.0
        self.next_minute_t = 60.0
        self.possession = self.home.id
        self.carrier: str | None = None
        self.ball = (52.5, 34.0)
        self.last_pass: tuple[str, str, float] | None = None  # (passer, receiver, t)

    # ------------------------------------------------------------------ helpers
    def _opp(self, tid: str) -> str:
        return self.away.id if tid == self.home.id else self.home.id

    def _dir(self, tid: str) -> int:
        home_dir = 1 if self.period == 1 else -1
        return home_dir if tid == self.home.id else -home_dir

    def _role_of(self, tid: str, role: str) -> str:
        return next(p for p in self.on_pitch[tid] if self.role[p] == role)

    def _gk(self, tid: str) -> str:
        return self._role_of(tid, "GK")

    def _clock(self) -> tuple[str, int]:
        ps = self.t - self.period_start_t
        base = 45 * (self.period - 1)
        if ps < PERIOD_SECONDS:
            m = int(ps // 60)
            return f"{base + m:02d}:{int(ps % 60):02d}", base + m + 1
        extra = ps - PERIOD_SECONDS
        return f"{base + 45}+{int(extra // 60)}:{int(extra % 60):02d}", base + 45

    def _positions(self) -> dict[str, tuple[float, float]]:
        pos: dict[str, tuple[float, float]] = {}
        for tid, team in self.teams.items():
            d = self._dir(tid)
            bx, by = to_rel(*self.ball, d)
            attacking = tid == self.possession
            shape = FORMATIONS[team.formation]
            for pid in self.on_pitch[tid]:
                rx, ry = shape[self.role[pid]]
                if self.role[pid] == "GK":
                    x, y = 4 + max(0.0, bx - 40) * 0.1, 34 + (by - 34) * 0.15
                elif attacking:
                    x = rx + (bx - 45) * 0.6
                    y = 34 + (ry - 34) * 1.1 + (by - 34) * 0.25
                else:
                    x = rx * 0.85 + (bx - 52.5) * 0.5 + self.press[tid] * 8 - 4
                    y = 34 + (ry - 34) * 0.8 + (by - 34) * 0.35
                x = _clamp(x + self.rng.gauss(0, 1.5), 1, 104)
                y = _clamp(y + self.rng.gauss(0, 1.5), 1, 67)
                pos[pid] = to_rel(x, y, d)  # to_rel is its own inverse
        if self.carrier:
            pos[self.carrier] = self.ball
        return pos

    def _nearest(self, tid: str, point: tuple[float, float], pos, exclude=()) -> tuple[float, str]:
        return min(
            (distance(*pos[p], *point), p)
            for p in self.on_pitch[tid]
            if p not in exclude
        )

    def _emit(self, etype: EventType, *, team_id: str | None = None, player_id: str | None = None,
              at: tuple[float, float] | None = None, end: tuple[float, float] | None = None,
              pos: dict | None = None, **fields) -> MatchEvent:
        clock, minute = self._clock()
        ev = MatchEvent(
            id=len(self.events) + 1,
            match_id=self.match_id,
            period=self.period,
            t=round(self.t, 2),
            clock=clock,
            minute=minute,
            type=etype,
            team_id=team_id,
            player_id=player_id,
            attack_dir=self._dir(team_id) if team_id else None,
            x=round(at[0], 2) if at else None,
            y=round(at[1], 2) if at else None,
            end_x=round(end[0], 2) if end else None,
            end_y=round(end[1], 2) if end else None,
            home_score=self.score[self.home.id],
            away_score=self.score[self.away.id],
            snapshot=[PlayerPosition(id=p, x=round(x, 1), y=round(y, 1)) for p, (x, y) in pos.items()]
            if pos else None,
            **fields,
        )
        self.events.append(ev)
        return ev

    def _give_ball(self, tid: str, player: str, at: tuple[float, float]) -> None:
        self.possession = tid
        self.carrier = player
        self.ball = (_clamp(at[0], 0.5, 104.5), _clamp(at[1], 0.5, 67.5))
        self.last_pass = None

    # --------------------------------------------------------------- match flow
    def run(self) -> Match:
        for period in (1, 2):
            self._play_period(period)
        return Match(
            id=self.match_id, seed=self.seed, home=self.home, away=self.away,
            venue="Harbourside Arena", events=self.events,
        )

    def _play_period(self, period: int) -> None:
        self.period = period
        self.period_start_t = self.t
        self.next_minute_t = self.t + 60
        end_t = self.t + PERIOD_SECONDS + self.rng.randint(60, 300)
        self._emit(EventType.PERIOD_START, detail=f"period_{period}")
        self._kick_off(self.home.id if period == 1 else self.away.id)
        while self.t < end_t:
            if self.t >= self.next_minute_t:
                self._minute_tick()
                self.next_minute_t += 60
            self._step()
        self._emit(EventType.PERIOD_END, detail=f"period_{period}")

    def _kick_off(self, tid: str) -> None:
        striker = self._role_of(tid, "ST")
        self._give_ball(tid, striker, (52.5, 34.0))
        self._emit(EventType.KICK_OFF, team_id=tid, player_id=striker, at=self.ball, pos=self._positions())
        self.t += 1.0

    def _minute_tick(self) -> None:
        _, minute = self._clock()
        for tid in self.teams:
            self.form[tid] = _clamp(self.form[tid] * 0.95 + self.rng.gauss(0, 0.06), -0.4, 0.4)
            diff = self.score[tid] - self.score[self._opp(tid)]
            if self.period == 2 and not self.shifted[tid]:
                if diff < 0 and minute >= 55:
                    self.shifted[tid] = True
                    self.press[tid] = min(1.0, self.press[tid] + 0.35)
                    self.form[tid] = min(0.4, self.form[tid] + 0.12)
                    self._emit(EventType.TACTICAL_SHIFT, team_id=tid, detail="high_press")
                elif diff > 0 and minute >= 65 and self.rng.random() < 0.08:
                    self.shifted[tid] = True
                    self.press[tid] = max(0.1, self.press[tid] - 0.25)
                    self._emit(EventType.TACTICAL_SHIFT, team_id=tid, detail="low_block")
            if self.period == 2 and self.sub_plan[tid] and minute >= self.sub_plan[tid][0] and self.bench[tid]:
                self.sub_plan[tid].pop(0)
                self._substitute(tid)

    def _substitute(self, tid: str) -> None:
        on = self.bench[tid].pop(0)
        group = ROLE_GROUP[self.players[on].position]
        options = [p for p in self.on_pitch[tid] if ROLE_GROUP[self.role[p]] == group and p != self.carrier]
        off = self.rng.choice(options)
        self.on_pitch[tid][self.on_pitch[tid].index(off)] = on
        self.role[on] = self.role[off]
        self.form[tid] = min(0.4, self.form[tid] + 0.03)
        self._emit(EventType.SUBSTITUTION, team_id=tid, player_id=on, related_player_id=off,
                   detail=self.role[on])

    # ------------------------------------------------------------- on-ball step
    def _step(self) -> None:
        tid = self.possession
        opp = self._opp(tid)
        d = self._dir(tid)
        carrier = self.players[self.carrier]
        pos = self._positions()
        rx, ry = to_rel(*self.ball, d)
        nd, presser = self._nearest(opp, self.ball, pos)
        under_pressure = nd < 3.5

        if nd < 6 and self.rng.random() < 0.18 + 0.3 * self.press[opp]:
            self._emit(EventType.PRESSURE, team_id=opp, player_id=presser, at=pos[presser],
                       related_player_id=carrier.id, pos=pos)
            self.t += 0.6
            under_pressure = True
            if self.rng.random() < 0.08 + 0.14 * self.press[opp] - (carrier.passing - 70) / 400:
                self._tackle(opp, presser, carrier.id, pos)
                return

        role = self.role[carrier.id]
        dist_goal = distance(rx, ry, 105, 34)
        p_shot = 0.03 + 0.5 * max(0.0, (30 - dist_goal) / 30) ** 1.5 if dist_goal < 30 else 0.0
        if shot_angle(rx, ry) < 0.15:
            p_shot *= 0.3
        p_clear = 0.35 if (rx < 25 and under_pressure and ROLE_GROUP[role] in ("GK", "DF")) else 0.0
        p_dribble = 0.0 if role == "GK" else (0.05 + (carrier.pace - 60) / 400 if rx > 40 and nd < 8 else 0.02)
        p_carry = 0.0 if role == "GK" else (0.22 if nd > 7 else 0.08)

        r = self.rng.random()
        if r < p_shot:
            self._shot(pos, under_pressure)
        elif r < p_shot + p_clear:
            self._clearance(pos)
        elif r < p_shot + p_clear + p_dribble:
            self._dribble(pos)
        elif r < p_shot + p_clear + p_dribble + p_carry:
            self._carry(pos, nd)
        else:
            self._pass(pos, under_pressure)

    def _pass(self, pos, under_pressure: bool, targets: list[str] | None = None,
              end_override: tuple[float, float] | None = None, detail: str | None = None) -> bool:
        tid, opp, d = self.possession, self._opp(self.possession), self._dir(self.possession)
        passer = self.players[self.carrier]
        rx, ry = to_rel(*self.ball, d)
        opp_rel = [to_rel(*pos[o], d) for o in self.on_pitch[opp]]

        cands = []
        for p in targets or [p for p in self.on_pitch[tid] if p != passer.id]:
            tx, ty = to_rel(*pos[p], d)
            dist = distance(rx, ry, tx, ty)
            if not targets and (dist < 4 or dist > 60):
                continue
            gap = min(distance(tx, ty, ox, oy) for ox, oy in opp_rel)
            w = math.exp((tx - rx) / 20) * (0.3 + min(gap, 10) / 10) / (1 + (dist / 28) ** 2)
            if self.role[p] == "GK":
                w *= 0.15
            cands.append((w, p, tx, ty))
        if not cands:
            self._carry(pos, 10.0)
            return False
        _, receiver, tx, ty = self.rng.choices(cands, weights=[c[0] for c in cands])[0]
        if end_override:
            ex, ey = end_override
        else:
            ex = _clamp(tx + self.rng.uniform(0, 4), 1, 104)
            ey = _clamp(ty + self.rng.gauss(0, 1.5), 1, 67)
        diff = pass_difficulty((rx, ry), (ex, ey), opp_rel)
        length = distance(rx, ry, ex, ey)
        speed = _clamp(11 + length * 0.25 + self.rng.gauss(0, 1.5), 8, 30)
        dur = length / speed
        p_ok = 0.985 - diff.total / 100 * 0.6 + (passer.passing - 70) / 250 + self.form[tid] * 0.08 - 0.05 * self.press[opp]
        if detail == "corner":
            p_ok = 0.35
        ok = self.rng.random() < _clamp(p_ok, 0.2, 0.98)
        end_abs = to_rel(ex, ey, d)
        self._emit(
            EventType.PASS, team_id=tid, player_id=passer.id, at=self.ball, end=end_abs, pos=pos,
            outcome="complete" if ok else "incomplete", receiver_id=receiver, duration=round(dur, 2),
            distance_m=round(length, 1), ball_speed_kmh=ball_speed_kmh(length, dur),
            pass_difficulty=diff.total, lines_broken=diff.lines_broken, under_pressure=under_pressure,
            detail=detail,
        )
        self.t += dur
        if ok:
            self.carrier = receiver
            self.ball = end_abs
            self.last_pass = (passer.id, receiver, self.t)
            self.t += self.rng.uniform(0.8, 2.5)
            return True

        if detail != "corner" and self.rng.random() < 0.25:
            out = (_clamp(end_abs[0], 0.5, 104.5), 0.5 if end_abs[1] < 34 else 67.5)
            self._emit(EventType.BALL_OUT, team_id=tid, player_id=passer.id, at=out)
            self.t += self.rng.uniform(5, 12)
            _, taker = self._nearest(opp, out, pos)
            self._give_ball(opp, taker, out)
        else:
            mid = (self.ball[0] * 0.3 + end_abs[0] * 0.7, self.ball[1] * 0.3 + end_abs[1] * 0.7)
            _, interceptor = self._nearest(opp, mid, pos)
            self._emit(EventType.INTERCEPTION, team_id=opp, player_id=interceptor, at=mid,
                       related_player_id=passer.id, pos=pos)
            self.t += 0.5
            self._give_ball(opp, interceptor, mid)
        return False

    def _carry(self, pos, nd: float) -> None:
        tid, d = self.possession, self._dir(self.possession)
        carrier = self.players[self.carrier]
        rx, ry = to_rel(*self.ball, d)
        ang = self.rng.gauss(0, 0.5)
        length = self.rng.uniform(4, 14) * (0.6 + carrier.pace / 200)
        ex = _clamp(rx + math.cos(ang) * length, 1, 104)
        ey = _clamp(ry + math.sin(ang) * length, 1, 67)
        speed = 4.0 + carrier.pace / 100 * 4 + (1.0 if length > 10 else 0.0) + self.rng.gauss(0, 0.6)
        speed = _clamp(speed, 3.0, 10.0)
        real_len = distance(rx, ry, ex, ey)
        dur = real_len / speed
        end_abs = to_rel(ex, ey, d)
        self._emit(EventType.CARRY, team_id=tid, player_id=carrier.id, at=self.ball, end=end_abs, pos=pos,
                   duration=round(dur, 2), distance_m=round(real_len, 1),
                   ball_speed_kmh=ball_speed_kmh(real_len, dur))
        self.t += dur
        self.ball = end_abs

    def _dribble(self, pos) -> None:
        tid, opp, d = self.possession, self._opp(self.possession), self._dir(self.possession)
        carrier = self.players[self.carrier]
        _, defender_id = self._nearest(opp, self.ball, pos)
        defender = self.players[defender_id]
        p_win = _clamp(0.5 + (carrier.pace - defender.defending) / 200 + self.form[tid] * 0.1, 0.15, 0.85)
        rx, ry = to_rel(*self.ball, d)
        if self.rng.random() < p_win:
            ex, ey = _clamp(rx + self.rng.uniform(5, 8), 1, 104), _clamp(ry + self.rng.gauss(0, 2), 1, 67)
            end_abs = to_rel(ex, ey, d)
            self._emit(EventType.DRIBBLE, team_id=tid, player_id=carrier.id, at=self.ball, end=end_abs,
                       pos=pos, outcome="complete", related_player_id=defender_id, duration=2.0,
                       distance_m=round(distance(rx, ry, ex, ey), 1))
            self.t += 2.0
            self.ball = end_abs
        else:
            self._emit(EventType.DRIBBLE, team_id=tid, player_id=carrier.id, at=self.ball, pos=pos,
                       outcome="fail", related_player_id=defender_id, duration=1.5)
            self.t += 1.5
            self._tackle(opp, defender_id, carrier.id, pos)

    def _tackle(self, tid: str, defender: str, victim: str, pos) -> None:
        if self.rng.random() < 0.14:
            self._emit(EventType.FOUL, team_id=tid, player_id=defender, at=self.ball,
                       related_player_id=victim)
            if self.rng.random() < 0.2:
                self._emit(EventType.CARD, team_id=tid, player_id=defender, detail="yellow")
            self.t += self.rng.uniform(15, 35)
            return  # free kick: fouled team keeps the ball
        self._emit(EventType.TACKLE, team_id=tid, player_id=defender, at=self.ball, outcome="won",
                   related_player_id=victim, pos=pos)
        self.t += 0.8
        self._give_ball(tid, defender, self.ball)

    def _clearance(self, pos) -> None:
        tid, opp, d = self.possession, self._opp(self.possession), self._dir(self.possession)
        rx, _ = to_rel(*self.ball, d)
        land = to_rel(self.rng.uniform(50, 80), self.rng.uniform(8, 60), d)
        length = distance(*self.ball, *land)
        dur = length / 22
        self._emit(EventType.CLEARANCE, team_id=tid, player_id=self.carrier, at=self.ball, end=land,
                   pos=pos, duration=round(dur, 2), distance_m=round(length, 1),
                   ball_speed_kmh=ball_speed_kmh(length, dur))
        self.t += dur + 1.0
        winner_team = opp if self.rng.random() < 0.55 else tid
        _, player = self._nearest(winner_team, land, pos)
        self._give_ball(winner_team, player, land)

    def _shot(self, pos, under_pressure: bool, header: bool = False, set_piece: str | None = None) -> None:
        tid, opp, d = self.possession, self._opp(self.possession), self._dir(self.possession)
        shooter = self.players[self.carrier]
        rx, ry = to_rel(*self.ball, d)
        xg = expected_goals(rx, ry, header=header, under_pressure=under_pressure)
        gk = self._gk(opp)
        keeper = self.players[gk]
        p_goal = _clamp(
            xg.value * (0.78 + (shooter.finishing - keeper.defending) / 120 + self.form[tid] * 0.1),
            0.005, 0.95,
        )
        if header:
            speed = _clamp(45 + shooter.finishing * 0.1 + self.rng.gauss(0, 6), 30, 75)
        else:
            speed = _clamp(62 + shooter.finishing * 0.4 + self.rng.gauss(0, 8), 45, 125)
        nd, blocker = self._nearest(opp, self.ball, pos, exclude=(gk,))

        r = self.rng.random()
        if r < p_goal:
            outcome = "goal"
            target = (105.0, 34 + self.rng.uniform(-3.3, 3.3))
        elif nd < 2.5 and self.rng.random() < 0.5:
            outcome = "blocked"
            target = (rx + 1.5, ry)
        elif self.rng.random() < 0.45:
            outcome = "saved"
            target = (104.0, 34 + self.rng.uniform(-3, 3))
        else:
            outcome = "off_target"
            target = (105.0, 34 + self.rng.choice((-1, 1)) * self.rng.uniform(4, 9))
        length = distance(rx, ry, *target)
        dur = length / (speed / 3.6)
        detail = "header" if header else None
        if set_piece:
            detail = f"{detail}_{set_piece}" if detail else set_piece
        self._emit(EventType.SHOT, team_id=tid, player_id=shooter.id, at=self.ball, end=to_rel(*target, d),
                   pos=pos, outcome=outcome, xg=xg.value, distance_m=xg.distance_m,
                   ball_speed_kmh=round(speed, 1), duration=round(dur, 2), under_pressure=under_pressure,
                   detail=detail)
        self.t += dur

        if outcome == "goal":
            self.score[tid] += 1
            assist = None
            if self.last_pass and self.last_pass[1] == shooter.id and self.t - self.last_pass[2] < 12:
                assist = self.last_pass[0]
            self._emit(EventType.GOAL, team_id=tid, player_id=shooter.id, related_player_id=assist,
                       at=to_rel(*target, d))
            self.form[tid] = min(0.4, self.form[tid] + 0.05)
            self.t += self.rng.uniform(40, 70)
            self._kick_off(opp)
        elif outcome == "saved":
            self._emit(EventType.SAVE, team_id=opp, player_id=gk, at=to_rel(*target, d),
                       related_player_id=shooter.id)
            self.t += 1.0
            if self.rng.random() < 0.35:
                self._corner(tid)
            else:
                self._give_ball(opp, gk, to_rel(*target, d))
        elif outcome == "blocked":
            if self.rng.random() < 0.45:
                self._corner(tid)
            else:
                self._give_ball(opp, blocker, self.ball)
        else:
            self.t += self.rng.uniform(6, 12)
            self._give_ball(opp, gk, to_rel(6, 34, self._dir(opp)))

    def _corner(self, tid: str) -> None:
        d = self._dir(tid)
        side_y = self.rng.choice((0.5, 67.5))
        spot = to_rel(104.5, side_y, d)
        wide = [p for p in self.on_pitch[tid] if self.role[p] in ("RW", "LW", "RAM", "LAM", "CAM", "LCM")]
        taker = max(wide or self.on_pitch[tid], key=lambda p: self.players[p].passing)
        self._give_ball(tid, taker, spot)
        self._emit(EventType.CORNER, team_id=tid, player_id=taker, at=spot)
        self.t += self.rng.uniform(20, 35)
        pos = self._positions()
        targets = [p for p in self.on_pitch[tid] if self.role[p] in ("RCB", "LCB", "ST") and p != taker]
        end = (self.rng.uniform(96, 101), 34 + self.rng.uniform(-7, 7))
        if self._pass(pos, False, targets=targets, end_override=end, detail="corner"):
            self._shot(self._positions(), under_pressure=True, header=True, set_piece="corner")
        elif self.possession != tid:
            pos = self._positions()
            self._clearance(pos)


def simulate_match(seed: int = 7) -> Match:
    return MatchSimulator(seed).run()
