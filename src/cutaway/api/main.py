"""HTTP API: synthetic matches, computed insights and a real-time event stream."""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from cutaway.agents import Persona, Producer, ReelSpec
from cutaway.analytics.insights import summarize
from cutaway.engine import Match, simulate_match

app = FastAPI(title="Cutaway API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_SEED = 10_000
producer = Producer()


@lru_cache(maxsize=32)
def get_match(seed: int) -> Match:
    if not 0 <= seed <= MAX_SEED:
        raise HTTPException(404, "Unknown match")
    return simulate_match(seed)


@lru_cache(maxsize=32)
def get_insights(seed: int) -> dict:
    return summarize(get_match(seed))


def _meta(match: Match) -> dict:
    last = match.events[-1]
    return {
        "id": match.id,
        "seed": match.seed,
        "venue": match.venue,
        "home": match.home.model_dump(),
        "away": match.away.model_dump(),
        "final_score": {"home": last.home_score, "away": last.away_score},
        "event_count": len(match.events),
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/matches/{seed}")
def match(seed: int, snapshots: bool = False) -> dict:
    m = get_match(seed)
    exclude = None if snapshots else {"snapshot"}
    return {"match": _meta(m), "events": [e.model_dump(mode="json", exclude=exclude) for e in m.events]}


@app.get("/api/matches/{seed}/insights")
def insights(seed: int) -> dict:
    return get_insights(seed)


@app.get("/api/matches/{seed}/stream")
async def stream(seed: int, speed: float = 20.0) -> StreamingResponse:
    """Server-sent events replaying the match in (scaled) real time."""
    m = get_match(seed)
    speed = max(1.0, min(speed, 600.0))

    async def gen():
        prev_t = 0.0
        for e in m.events:
            await asyncio.sleep(max(0.0, (e.t - prev_t) / speed))
            prev_t = e.t
            yield f"event: match_event\ndata: {json.dumps(e.model_dump(mode='json', exclude={'snapshot'}))}\n\n"
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/matches/{seed}/reels")
async def create_reel(seed: int, persona: Persona) -> ReelSpec:
    """Run the agent crew and return a verified, personalised reel spec."""
    return await producer.produce(get_match(seed), persona)
