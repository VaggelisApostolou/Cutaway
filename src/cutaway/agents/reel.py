"""Reel spec: the timed, machine-readable output that a player or a broadcast graphics
engine renders. Every overlay value and every number in the narration traces back to
event ids in the source match."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["casual", "analyst", "player"]
Camera = Literal["wide", "follow_ball", "box_zoom", "player_focus"]


class Persona(BaseModel):
    mode: Mode = "casual"
    language: str = Field("en", description="BCP-47 code, e.g. en, el, es, fr")
    club_id: str | None = None
    player_id: str | None = None
    tone: Literal["neutral", "passionate", "playful"] = "neutral"
    target_seconds: int = Field(60, ge=20, le=180)


class Overlay(BaseModel):
    kind: str
    label: str
    value: str
    at: float = Field(0.0, description="Seconds from segment start")
    evidence: list[int] = Field(default_factory=list)


class Segment(BaseModel):
    moment_id: str
    kind: str
    clock: str
    team_id: str | None
    start_t: float = Field(description="Match time (s) where the replay starts")
    end_t: float = Field(description="Match time (s) where the replay ends")
    speed: float = 1.0
    camera: Camera = "follow_ball"
    focus_player_id: str | None = None
    overlays: list[Overlay] = Field(default_factory=list)
    facts: dict = Field(default_factory=dict, description="Verified facts the narration may use")
    narration: str = ""
    evidence: list[int] = Field(default_factory=list)

    @property
    def screen_seconds(self) -> float:
        return (self.end_t - self.start_t) / self.speed


class VerificationIssue(BaseModel):
    moment_id: str
    problem: str


class ReelSpec(BaseModel):
    match_id: str
    seed: int
    persona: Persona
    title: str
    intro: str = ""
    outro: str = ""
    segments: list[Segment]
    verified: bool = False
    issues: list[VerificationIssue] = Field(default_factory=list)
    provider: str = "offline"
    trace: list[dict] = Field(default_factory=list, description="What each agent did, for the UI")

    @property
    def screen_seconds(self) -> float:
        return sum(s.screen_seconds for s in self.segments)
