"""Data models for synthetic match data.

Coordinates are absolute pitch metres: x in [0, 105] (length), y in [0, 68] (width),
origin at the bottom-left corner. Each on-ball event carries ``attack_dir`` (+1 when the
acting team attacks towards x=105, -1 otherwise) so any consumer can convert to the
acting team's frame without knowing the period/side rules.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


class EventType(str, Enum):
    PERIOD_START = "period_start"
    PERIOD_END = "period_end"
    KICK_OFF = "kick_off"
    PASS = "pass"
    CARRY = "carry"
    DRIBBLE = "dribble"
    PRESSURE = "pressure"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    CLEARANCE = "clearance"
    BALL_OUT = "ball_out"
    SHOT = "shot"
    SAVE = "save"
    GOAL = "goal"
    CORNER = "corner"
    FOUL = "foul"
    CARD = "card"
    SUBSTITUTION = "substitution"
    TACTICAL_SHIFT = "tactical_shift"


# Event types where the acting team has the ball. Used to derive possession.
ON_BALL_TYPES = {
    EventType.KICK_OFF,
    EventType.PASS,
    EventType.CARRY,
    EventType.DRIBBLE,
    EventType.SHOT,
    EventType.CLEARANCE,
    EventType.CORNER,
}


class Player(BaseModel):
    id: str
    name: str
    number: int
    position: str
    team_id: str
    pace: int = Field(ge=1, le=99)
    passing: int = Field(ge=1, le=99)
    finishing: int = Field(ge=1, le=99)
    defending: int = Field(ge=1, le=99)
    starter: bool = True


class Team(BaseModel):
    id: str
    name: str
    short_name: str
    color: str
    formation: str
    players: list[Player]


class PlayerPosition(BaseModel):
    id: str
    x: float
    y: float


class MatchEvent(BaseModel):
    id: int
    match_id: str
    period: int
    t: float = Field(description="Elapsed match seconds, monotonic across periods")
    clock: str = Field(description="Football clock, e.g. '45+2:10'")
    minute: int
    type: EventType
    team_id: str | None = None
    player_id: str | None = None
    attack_dir: int | None = None
    x: float | None = None
    y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    duration: float = 0.0
    outcome: str | None = None
    receiver_id: str | None = None
    related_player_id: str | None = None
    detail: str | None = None
    # Derived metrics, computed in code so every number can be re-verified.
    distance_m: float | None = None
    ball_speed_kmh: float | None = None
    pass_difficulty: float | None = None
    lines_broken: int | None = None
    xg: float | None = None
    under_pressure: bool = False
    home_score: int = 0
    away_score: int = 0
    snapshot: list[PlayerPosition] | None = None


class Match(BaseModel):
    id: str
    seed: int
    home: Team
    away: Team
    venue: str
    events: list[MatchEvent]

    def team(self, team_id: str) -> Team:
        return self.home if team_id == self.home.id else self.away

    def player(self, player_id: str) -> Player:
        for team in (self.home, self.away):
            for p in team.players:
                if p.id == player_id:
                    return p
        raise KeyError(player_id)
