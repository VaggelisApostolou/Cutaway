"""Deterministic football metrics.

Every number shown to a viewer is computed here, in code, never by a language model.
Functions return their components so an agent can explain (and a verifier can recompute)
where a value came from.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from cutaway.engine.models import PITCH_LENGTH, PITCH_WIDTH

GOAL_X = PITCH_LENGTH
GOAL_Y = PITCH_WIDTH / 2
HALF_GOAL_WIDTH = 7.32 / 2


def to_rel(x: float, y: float, attack_dir: int) -> tuple[float, float]:
    """Absolute pitch coordinates -> acting team's frame (attacking towards x=105)."""
    return (x, y) if attack_dir > 0 else (PITCH_LENGTH - x, PITCH_WIDTH - y)


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def shot_angle(x: float, y: float) -> float:
    """Angle (radians) subtended by the goal mouth from a shot location (team frame)."""
    dx = GOAL_X - x
    dy = y - GOAL_Y
    angle = math.atan2(2 * HALF_GOAL_WIDTH * dx, dx * dx + dy * dy - HALF_GOAL_WIDTH**2)
    return angle + math.pi if angle < 0 else angle


@dataclass(frozen=True)
class XGBreakdown:
    value: float
    distance_m: float
    angle_deg: float
    header: bool
    under_pressure: bool

    def as_dict(self) -> dict:
        return asdict(self)


def expected_goals(x: float, y: float, *, header: bool = False, under_pressure: bool = False) -> XGBreakdown:
    """Logistic xG on distance and goal-mouth angle (team frame)."""
    d = distance(x, y, GOAL_X, GOAL_Y)
    ang = shot_angle(x, y)
    z = -0.8 - 0.11 * d + 1.6 * ang
    if header:
        z -= 0.9
    if under_pressure:
        z -= 0.35
    value = 1 / (1 + math.exp(-z))
    return XGBreakdown(round(value, 3), round(d, 1), round(math.degrees(ang), 1), header, under_pressure)


@dataclass(frozen=True)
class PassDifficulty:
    """Pass difficulty on a 0-100 scale, as the sum of four explainable components."""

    total: float
    distance_pts: float
    receiver_tightness_pts: float
    passer_pressure_pts: float
    lines_broken_pts: float
    lines_broken: int

    def as_dict(self) -> dict:
        return asdict(self)


def pass_difficulty(
    start: tuple[float, float],
    end: tuple[float, float],
    opponents: list[tuple[float, float]],
) -> PassDifficulty:
    """All coordinates in the passing team's frame. ``opponents`` excludes nobody."""
    sx, sy = start
    ex, ey = end
    length = distance(sx, sy, ex, ey)
    distance_pts = min(length / 45, 1.0) * 35

    receiver_gap = min((distance(ex, ey, ox, oy) for ox, oy in opponents), default=10.0)
    receiver_pts = max(0.0, (6 - receiver_gap) / 6) * 25

    passer_gap = min((distance(sx, sy, ox, oy) for ox, oy in opponents), default=10.0)
    passer_pts = 15.0 if passer_gap < 3 else 8.0 if passer_gap < 5 else 0.0

    lines = 0
    if ex - sx > 5:
        for ox, oy in opponents:
            if sx < ox < ex:
                lane_y = sy + (ey - sy) * (ox - sx) / (ex - sx)
                if abs(oy - lane_y) < 9:
                    lines += 1
    lines_pts = min(lines, 5) * 5

    total = min(100.0, distance_pts + receiver_pts + passer_pts + lines_pts)
    return PassDifficulty(
        round(total, 1), round(distance_pts, 1), round(receiver_pts, 1),
        passer_pts, float(lines_pts), lines,
    )


def ball_speed_kmh(length_m: float, duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return round(length_m / duration_s * 3.6, 1)


def in_box(x: float, y: float) -> bool:
    """Opponent penalty area, team frame."""
    return x >= PITCH_LENGTH - 16.5 and 13.85 <= y <= PITCH_WIDTH - 13.85
