"""The Cutaway production crew.

Producer orchestrates four specialists, mirroring a TV highlights team:

* Editor (LLM)       picks and orders the moments that matter to this viewer
* Director (code)    turns each moment into a timed replay: window, camera, speed, overlays
* Scriptwriter (LLM) writes narration in the viewer's language, tone and mode
* Verifier (code)    rejects any number not backed by the facts; failures go back to the
                     Scriptwriter with the exact problems, then fall back to a safe template

Language models choose and write; code measures and checks. With no model configured the
Editor and Scriptwriter run deterministic fallbacks, so the pipeline always produces a reel.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from cutaway.analytics.insights import Moment, key_moments
from cutaway.engine.models import Match

from .llm import LLM
from .reel import Overlay, Persona, ReelSpec, Segment, VerificationIssue

LANGUAGE_NAMES = {"en": "English", "el": "Greek", "es": "Spanish", "fr": "French", "de": "German",
                  "it": "Italian", "pt": "Portuguese", "ar": "Arabic", "sw": "Swahili"}

BASE_WEIGHT = {
    "goal": 100, "big_save": 60, "big_chance_missed": 55, "momentum_shift": 45,
    "line_breaking_pass": 40, "tactical_shift": 35, "last_ditch_tackle": 35,
    "rocket_shot": 30, "speed_burst": 30, "substitution": 5,
}
MODE_BOOST = {
    "casual": {"goal": 1.3, "big_save": 1.1, "rocket_shot": 1.2, "momentum_shift": 0.6,
               "tactical_shift": 0.5, "line_breaking_pass": 0.6, "substitution": 0.2},
    "analyst": {"momentum_shift": 1.6, "tactical_shift": 1.8, "line_breaking_pass": 1.5,
                "speed_burst": 0.6, "rocket_shot": 0.7},
    "player": {},
}
SECONDS_PER_SEGMENT = 10.0
WORDS_PER_SECOND = 3.0


# ---------------------------------------------------------------- helpers
def minute_label(clock: str) -> str:
    """'73:11' -> "74'", '45+2:10' -> "45+3'"."""
    main = clock.split(":")[0]
    if "+" in main:
        base, extra = main.split("+")
        return f"{base}+{int(extra) + 1}'"
    return f"{int(main) + 1}'"


def _name(match: Match, pid: str | None) -> str | None:
    return match.player(pid).name if pid else None


# ------------------------------------------------------------------ Editor
class EditorPick(BaseModel):
    moment_id: str
    why: str = Field(description="One short reason this moment matters to this viewer")


class EditorDecision(BaseModel):
    picks: list[EditorPick]


EDITOR_INSTRUCTIONS = """You are the Editor of a football highlights team.
You receive candidate moments from a match (already measured by code) and a viewer profile.
Pick the moments that tell the best story FOR THIS VIEWER within the time budget, and order
them as they happened. Always include every goal unless the viewer is in player mode and the
goal does not involve their player. Prefer variety; avoid two moments that show the same play.
Use only moment_id values from the candidate list."""


def persona_score(m: Moment, persona: Persona) -> float:
    score = BASE_WEIGHT.get(m.kind, 20) * MODE_BOOST[persona.mode].get(m.kind, 1.0) + m.importance * 0.3
    if persona.club_id and m.team_id == persona.club_id:
        score *= 1.25
    if persona.player_id:
        if persona.player_id in m.player_ids:
            score += 80
        elif persona.mode == "player":
            score *= 0.8 if m.kind == "goal" else 0.4
    return round(score, 1)


def candidate_moments(match: Match, persona: Persona, limit: int = 20) -> list[Moment]:
    """Greedy pick by relevance; repeats of the same kind lose 30% each for variety."""
    pool = list(key_moments(match))
    picked: list[Moment] = []
    kinds: dict[str, int] = {}
    while pool and len(picked) < limit:
        best = max(pool, key=lambda m: persona_score(m, persona) * 0.7 ** kinds.get(m.kind, 0))
        pool.remove(best)
        if any(abs(best.t - p.t) < 12 for p in picked):
            continue  # same passage of play
        picked.append(best)
        kinds[best.kind] = kinds.get(best.kind, 0) + 1
    return picked


def budget(persona: Persona) -> int:
    return max(3, min(12, round(persona.target_seconds / SECONDS_PER_SEGMENT)))


def editor_offline(cands: list[Moment], persona: Persona) -> list[EditorPick]:
    chosen = cands[: budget(persona)]  # already ranked for relevance and variety
    chosen.sort(key=lambda m: m.t)
    return [EditorPick(moment_id=m.id, why=f"{m.kind.replace('_', ' ')} (score {persona_score(m, persona)})")
            for m in chosen]


def describe_candidate(match: Match, m: Moment, persona: Persona) -> dict:
    return {
        "moment_id": m.id, "kind": m.kind, "minute": minute_label(m.clock),
        "team": match.team(m.team_id).name if m.team_id else None,
        "players": [_name(match, p) for p in m.player_ids],
        "importance": m.importance, "relevance": persona_score(m, persona),
        "facts": {k: v for k, v in m.facts.items() if k != "possible_causes"},
    }


def persona_brief(match: Match, persona: Persona) -> dict:
    return {
        "mode": persona.mode,
        "language": LANGUAGE_NAMES.get(persona.language, persona.language),
        "favourite_club": match.team(persona.club_id).name if persona.club_id else None,
        "favourite_player": _name(match, persona.player_id),
        "tone": persona.tone,
        "target_seconds": persona.target_seconds,
        "max_moments": budget(persona),
    }


# ---------------------------------------------------------------- Director
def direct(match: Match, m: Moment, persona: Persona) -> Segment:
    """Deterministic shot list for one moment: replay window, camera, speed, overlays, facts."""
    by_id = {e.id: e for e in match.events}
    ev = [by_id[i] for i in m.event_ids if i in by_id]
    first_t = min((e.t for e in ev), default=m.t)
    main_player = m.player_ids[0] if m.player_ids else None
    team = match.team(m.team_id).name if m.team_id else None
    f = m.facts
    facts: dict = {"minute": minute_label(m.clock), "team": team}
    overlays: list[Overlay] = []
    camera, speed = "follow_ball", 1.0
    start, end = m.t - 4, m.t + 3

    def ov(kind, label, value, at=0.0, evidence=None):
        overlays.append(Overlay(kind=kind, label=label, value=str(value), at=at, evidence=evidence or m.event_ids))

    if m.kind == "goal":
        start, end, camera, speed = max(first_t, m.t - 10), m.t + 4, "box_zoom", 0.8
        goal_ev = ev[-1]
        facts.update(scorer=_name(match, m.player_ids[0]), assist=_name(match, goal_ev.related_player_id),
                     xg=f["xg"], chance_pct=round(f["xg"] * 100), shot_speed_kmh=f["shot_speed_kmh"],
                     distance_m=f["distance_m"], score_after=f["score"], build_up_passes=f["build_up_passes"],
                     header=bool(f.get("detail") and "header" in f["detail"]))
        ov("score", "GOAL", f"{match.home.short_name} {f['score']} {match.away.short_name}", at=(m.t - start) / speed)
        ov("xg", "xG", f"{f['xg']:.2f}", at=max(0.0, (m.t - start) / speed - 2))
        ov("shot_speed", "Shot speed", f"{f['shot_speed_kmh']:.0f} km/h", at=(m.t - start) / speed)
    elif m.kind in ("big_chance_missed", "big_save", "rocket_shot"):
        start, end, camera = max(first_t, m.t - 8), m.t + 3, "box_zoom"
        xg = f.get("xg", f.get("xg_denied"))
        facts.update(shooter=_name(match, m.player_ids[-1] if m.kind == "big_save" else main_player),
                     keeper=_name(match, main_player) if m.kind == "big_save" else None,
                     outcome=f.get("outcome", "saved"), shot_speed_kmh=f.get("shot_speed_kmh"))
        if xg is not None:
            facts.update(xg=xg, chance_pct=round(xg * 100))
            ov("xg", "xG", f"{xg:.2f}", at=(m.t - start) - 1)
        if f.get("shot_speed_kmh"):
            ov("shot_speed", "Shot speed", f"{f['shot_speed_kmh']:.0f} km/h", at=m.t - start)
    elif m.kind == "line_breaking_pass":
        start, end, speed = m.t - 3, m.t + 6, 0.75
        facts.update(passer=_name(match, m.player_ids[0]), receiver=_name(match, m.player_ids[1]),
                     pass_difficulty=f["pass_difficulty"], difficulty_scale=100, lines_broken=f["lines_broken"],
                     distance_m=f["distance_m"], led_to_shot=f["led_to_shot"])
        ov("pass_difficulty", "Pass difficulty", f"{f['pass_difficulty']:.0f}/100", at=2)
        ov("lines_broken", "Lines broken", f["lines_broken"], at=3)
    elif m.kind == "speed_burst":
        start, end, camera = m.t - 1, m.t + 5, "player_focus"
        facts.update(player=_name(match, main_player), speed_kmh=f["speed_kmh"], distance_m=f["distance_m"])
        ov("speed", "Top speed", f"{f['speed_kmh']:.1f} km/h", at=2)
    elif m.kind == "last_ditch_tackle":
        start, end, camera = m.t - 5, m.t + 2, "box_zoom"
        facts.update(defender=_name(match, main_player), attacker=_name(match, m.player_ids[1] if len(m.player_ids) > 1 else None))
    elif m.kind == "momentum_shift":
        start, end, camera = m.t - 5, m.t + 5, "wide"
        before, after = f["momentum_before"], f["momentum_after"]
        causes = []
        for c in f.get("possible_causes", []):
            if c["type"] == "tactical_shift":
                causes.append(f"{match.team(c['team_id']).name} switched to a {c['detail'].replace('_', ' ')}")
            elif c["type"] == "substitution":
                causes.append(f"{match.team(c['team_id']).name} made a substitution")
            elif c["type"] == "goal":
                causes.append(f"a goal by {match.team(c['team_id']).name}")
        facts.update(momentum_before=before, momentum_after=after, now_on_top=team, possible_causes=causes)
        ov("momentum", "Momentum", f"{before:+.2f} → {after:+.2f}", at=1)
    elif m.kind == "tactical_shift":
        start, end, camera = m.t, m.t + 8, "wide"
        facts.update(change=f.get("detail", "").replace("_", " "))
        ov("tactic", "Tactical change", facts["change"].title(), at=1)
    elif m.kind == "substitution":
        facts.update(on=_name(match, m.player_ids[0]), off=_name(match, m.player_ids[1] if len(m.player_ids) > 1 else None))

    focus = persona.player_id if persona.player_id in m.player_ids else main_player
    if persona.mode == "player" and persona.player_id in m.player_ids:
        camera = "player_focus"
    if focus:
        p = match.player(focus)
        ov("player_tag", match.team(p.team_id).short_name, f"#{p.number} {p.name}", at=0.5, evidence=m.event_ids[:1])
    return Segment(
        moment_id=m.id, kind=m.kind, clock=m.clock, team_id=m.team_id,
        start_t=round(max(0.0, start), 2), end_t=round(end, 2), speed=speed, camera=camera,
        focus_player_id=focus, overlays=overlays,
        facts={k: v for k, v in facts.items() if v is not None}, evidence=m.event_ids,
    )


# ------------------------------------------------------------ Scriptwriter
class ScriptLine(BaseModel):
    moment_id: str
    narration: str


class Script(BaseModel):
    title: str
    intro: str
    lines: list[ScriptLine]
    outro: str


SCRIPT_INSTRUCTIONS = """You are the Scriptwriter for a football highlights reel.
Write voice-over narration for each segment, in the requested language, tone and viewer mode:
- casual: vivid and simple, no jargon, explain any stat in plain words.
- analyst: precise, tactical, cause and effect (why the moment mattered).
- player: keep the favourite player at the centre of every line.
If the viewer has a favourite club, narrate as their companion, but never change the facts.

HARD RULES
1. Use ONLY the facts given for each segment. Every number you write must appear in that
   segment's facts (you may show a 0-1 value as the given chance_pct percentage).
2. Never invent names, minutes, scores, distances or speeds.
3. Keep each line within its word budget (max_words); it is read aloud over the replay.
4. Explain WHY the moment matters, not only what happened.
5. Return one line per segment, with the same moment_id values, plus a short title, a one
   sentence intro and a one sentence outro (they may only use the final score numbers)."""


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}" if v < 1 else f"{v:.0f}"
    return str(v)


TEMPLATES = {
    "en": {
        "goal": "{minute}: {scorer} scores for {team}. A {chance_pct}% chance, struck at {shot_speed_kmh} km/h. It's {score_after}.",
        "big_save": "{minute}: {keeper} denies {shooter}. That chance was worth {chance_pct}%.",
        "big_chance_missed": "{minute}: {shooter} should have scored. A {chance_pct}% chance goes begging.",
        "rocket_shot": "{minute}: {shooter} lets fly at {shot_speed_kmh} km/h.",
        "line_breaking_pass": "{minute}: {passer} finds {receiver} through {lines_broken} defenders. Difficulty {pass_difficulty} out of 100.",
        "speed_burst": "{minute}: {player} hits {speed_kmh} km/h, carrying the ball {distance_m} metres.",
        "last_ditch_tackle": "{minute}: {defender} wins it back in the box just in time.",
        "momentum_shift": "{minute}: the match tilts towards {now_on_top}.",
        "tactical_shift": "{minute}: {team} change the plan: {change}.",
        "substitution": "{minute}: {on} comes on for {off}.",
        "intro": "{home} against {away}. Here is your match.",
        "outro": "Full time: {home} {hs}, {away} {as_}.",
        "title": "{home} {hs}-{as_} {away}: your highlights",
    },
    "el": {
        "goal": "{minute}: Γκολ του {scorer} για την {team}. Ευκαιρία {chance_pct}%, σουτ στα {shot_speed_kmh} χλμ/ώρα. Το σκορ γίνεται {score_after}.",
        "big_save": "{minute}: Ο {keeper} σταματά τον {shooter}. Η ευκαιρία είχε πιθανότητα {chance_pct}%.",
        "big_chance_missed": "{minute}: Ο {shooter} χάνει τεράστια ευκαιρία, πιθανότητα {chance_pct}%.",
        "rocket_shot": "{minute}: Κεραυνός του {shooter} στα {shot_speed_kmh} χλμ/ώρα.",
        "line_breaking_pass": "{minute}: Ο {passer} βρίσκει τον {receiver} περνώντας {lines_broken} αντιπάλους. Δυσκολία {pass_difficulty} στα 100.",
        "speed_burst": "{minute}: Ο {player} πιάνει {speed_kmh} χλμ/ώρα με την μπάλα για {distance_m} μέτρα.",
        "last_ditch_tackle": "{minute}: Ο {defender} κλέβει την μπάλα μέσα στην περιοχή την τελευταία στιγμή.",
        "momentum_shift": "{minute}: Ο αγώνας γέρνει προς την {now_on_top}.",
        "tactical_shift": "{minute}: Η {team} αλλάζει πλάνο: {change}.",
        "substitution": "{minute}: Μπαίνει ο {on} στη θέση του {off}.",
        "intro": "{home} απέναντι στην {away}. Αυτός είναι ο δικός σου αγώνας.",
        "outro": "Τελικό σκορ: {home} {hs}, {away} {as_}.",
        "title": "{home} {hs}-{as_} {away}: τα highlights σου",
    },
}


class _Default(dict):
    def __missing__(self, key):
        return "?"


def template_line(seg: Segment, lang: str) -> str:
    t = TEMPLATES.get(lang, TEMPLATES["en"])
    values = _Default({k: _fmt(v) for k, v in seg.facts.items()})
    return t.get(seg.kind, "{minute}").format_map(values)


def template_frame(match: Match, lang: str) -> tuple[str, str, str]:
    t = TEMPLATES.get(lang, TEMPLATES["en"])
    last = match.events[-1]
    v = dict(home=match.home.name, away=match.away.name, hs=last.home_score, as_=last.away_score)
    return t["title"].format(**v), t["intro"].format(**v), t["outro"].format(**v)


# ---------------------------------------------------------------- Verifier
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _allowed_numbers(obj, out: set[str]) -> set[str]:
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        v = float(obj)
        for form in (v, round(v, 1), round(v, 2), round(v), int(v)):
            out.add(_norm(str(form)))
        if 0 < v < 1:
            out.add(_norm(str(round(v * 100))))
    elif isinstance(obj, str):
        for n in NUMBER.findall(obj):
            out.add(_norm(n))
    elif isinstance(obj, dict):
        for x in obj.values():
            _allowed_numbers(x, out)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _allowed_numbers(x, out)
    return out


def _norm(n: str) -> str:
    n = n.replace(",", ".")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n


def verify(segments: list[Segment], intro: str, outro: str, match: Match) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    last = match.events[-1]
    frame_ok = _allowed_numbers([last.home_score, last.away_score], set())
    for text, label in ((intro, "intro"), (outro, "outro")):
        bad = [n for n in NUMBER.findall(text) if _norm(n) not in frame_ok]
        if bad:
            issues.append(VerificationIssue(moment_id=label, problem=f"unsupported numbers {bad}"))
    for seg in segments:
        if not seg.narration.strip():
            issues.append(VerificationIssue(moment_id=seg.moment_id, problem="missing narration"))
            continue
        allowed = _allowed_numbers(seg.facts, set())
        bad = [n for n in NUMBER.findall(seg.narration) if _norm(n) not in allowed]
        if bad:
            issues.append(VerificationIssue(
                moment_id=seg.moment_id,
                problem=f"numbers {bad} are not in the facts {sorted(allowed)}"))
        max_words = int(seg.screen_seconds * WORDS_PER_SECOND) + 4
        if len(seg.narration.split()) > max_words:
            issues.append(VerificationIssue(moment_id=seg.moment_id,
                                            problem=f"too long: {len(seg.narration.split())} words, max {max_words}"))
    return issues


# ---------------------------------------------------------------- Producer
class Producer:
    def __init__(self, llm: LLM | None = None, max_rewrites: int = 2):
        self.llm = llm or LLM()
        self.max_rewrites = max_rewrites

    async def produce(self, match: Match, persona: Persona) -> ReelSpec:
        trace: list[dict] = []
        cands = candidate_moments(match, persona)
        by_id = {m.id: m for m in cands}

        # 1. Editor
        picks = None
        if self.llm.online:
            prompt = json.dumps({"viewer": persona_brief(match, persona),
                                 "candidates": [describe_candidate(match, m, persona) for m in cands]},
                                ensure_ascii=False)
            try:
                decision = await self.llm.run("editor", EDITOR_INSTRUCTIONS, prompt, EditorDecision)
                picks = [p for p in decision.picks if p.moment_id in by_id][: budget(persona) + 2]
            except Exception as exc:  # model or network failure: keep the reel coming
                trace.append({"agent": "editor", "action": "fallback", "detail": str(exc)[:200]})
        if not picks:
            picks = editor_offline(cands, persona)
        picks.sort(key=lambda p: by_id[p.moment_id].t)
        trace.append({"agent": "editor", "action": "selected",
                      "detail": [{"moment_id": p.moment_id, "why": p.why} for p in picks]})

        # 2. Director, then trim to the time budget (lowest relevance first, goals stay)
        segments = [direct(match, by_id[p.moment_id], persona) for p in picks]
        limit = persona.target_seconds * 1.15
        while sum(s.screen_seconds for s in segments) > limit and len(segments) > 3:
            droppable = [s for s in segments if s.kind != "goal"] or segments
            drop = min(droppable, key=lambda s: persona_score(by_id[s.moment_id], persona))
            segments.remove(drop)
            trace.append({"agent": "director", "action": "trimmed", "detail": drop.moment_id})
        trace.append({"agent": "director", "action": "shot list",
                      "detail": [{"moment_id": s.moment_id, "camera": s.camera, "speed": s.speed,
                                  "seconds": round(s.screen_seconds, 1), "overlays": len(s.overlays)}
                                 for s in segments]})

        # 3 + 4. Scriptwriter <-> Verifier loop
        title, intro, outro = template_frame(match, persona.language)
        issues: list[VerificationIssue] = []
        if self.llm.online:
            feedback: list[VerificationIssue] = []
            for attempt in range(self.max_rewrites + 1):
                try:
                    script = await self.llm.run("scriptwriter", SCRIPT_INSTRUCTIONS,
                                                self._script_prompt(match, persona, segments, feedback), Script)
                except Exception as exc:
                    trace.append({"agent": "scriptwriter", "action": "fallback", "detail": str(exc)[:200]})
                    break
                lines = {l.moment_id: l.narration for l in script.lines}
                for s in segments:
                    s.narration = lines.get(s.moment_id, "")
                title, intro, outro = script.title, script.intro, script.outro
                issues = verify(segments, intro, outro, match)
                trace.append({"agent": "verifier", "action": f"check {attempt + 1}",
                              "detail": [i.model_dump() for i in issues] or "all claims verified"})
                if not issues:
                    break
                feedback = issues

        # Safe fallback for anything still unverified (or when offline).
        bad = {i.moment_id for i in issues}
        for s in segments:
            if not self.llm.online or s.moment_id in bad or not s.narration:
                s.narration = template_line(s, persona.language)
        if not self.llm.online or {"intro", "outro"} & bad:
            title, intro, outro = template_frame(match, persona.language)
        final_issues = verify(segments, intro, outro, match)
        trace.append({"agent": "verifier", "action": "final",
                      "detail": [i.model_dump() for i in final_issues] or "all claims verified"})

        return ReelSpec(
            match_id=match.id, seed=match.seed, persona=persona, title=title, intro=intro, outro=outro,
            segments=segments, verified=not final_issues, issues=final_issues,
            provider=f"{self.llm.settings.provider}/{self.llm.settings.model}", trace=trace,
        )

    @staticmethod
    def _script_prompt(match: Match, persona: Persona, segments: list[Segment],
                       feedback: list[VerificationIssue]) -> str:
        last = match.events[-1]
        body = {
            "viewer": persona_brief(match, persona),
            "match": {"home": match.home.name, "away": match.away.name,
                      "final_score": f"{last.home_score}-{last.away_score}"},
            "segments": [{"moment_id": s.moment_id, "kind": s.kind, "facts": s.facts,
                          "max_words": int(s.screen_seconds * WORDS_PER_SECOND)} for s in segments],
        }
        if feedback:
            body["verifier_feedback"] = [f.model_dump() for f in feedback]
            body["instruction"] = "Your previous draft failed verification. Fix exactly these problems."
        return json.dumps(body, ensure_ascii=False)
