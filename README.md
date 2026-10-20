# Cutaway

**Every fan gets their own highlight reel, produced by AI agents and fact-checked by code.**

Cutaway turns a live stream of synthetic, football-realistic match events into personalised,
auto-produced highlight reels: an animated 2D pitch, broadcast-style overlays (xG, shot speed,
pass difficulty, momentum) and voice commentary in the viewer's language. The same match becomes
a 60-second reel for a casual fan of the home side in Greek, and a tactical breakdown for an
analyst in English.

Built for the *Inside the Game: Developer Hackathon* (Microsoft x Premier League) on the
*Synthetic Match Insights Engine* challenge.

> All data is synthetic. Clubs, players and venues are fictional. No real league data is used.

## How it works: one pipeline, five stages

| Stage | What happens | Where |
|---|---|---|
| 1. Ingest | Seeded simulator streams events (passes, carries, dribbles, pressure, tackles, shots, saves, corners, subs, tactical shifts) with 22-player snapshots | `src/cutaway/engine` |
| 2. Interpret | Deterministic metrics: xG, pass difficulty (4 explainable components), lines broken, ball and shot speed, possession, momentum, control vs chaos | `src/cutaway/analytics` |
| 3. Explain | Key moments ranked by importance, each citing the event ids that prove it; momentum swings linked to likely causes (tactical shift, substitution, goal) | `analytics/insights.py` |
| 4. Render | Agents turn moments into a timed reel spec (replay windows, camera, speed, overlays, narration); the web player renders it in sync | `agents/`, `web/` *(player in progress)* |
| 5. Personalise | Persona (casual / analyst / player-focus), favourite club and player, language and tone shape the selection, script and voice | `agents/crew.py` |

### The agent crew

| Agent | Kind | Job |
|---|---|---|
| Producer | orchestrator | Runs the crew, enforces the time budget, records a trace of every decision |
| Editor | LLM (Microsoft Agent Framework) | Picks and orders the moments that matter to *this* viewer |
| Director | code | Turns each moment into a replay: window, camera, slow-motion, overlays with event-id evidence |
| Scriptwriter | LLM (Microsoft Agent Framework) | Writes narration in the viewer's language, tone and mode, using only verified facts |
| Verifier | code | Rejects any number not in the facts and any line over its word budget; sends exact problems back to the Scriptwriter, then falls back to a safe template |

Models run through Microsoft Foundry (Azure OpenAI) or GitHub Models, chosen in `.env`.
Every answer is cached, and with no model configured the crew runs deterministic fallbacks,
so the pipeline always produces a verified reel.

**Numbers come from code, words come from AI.** Language models never invent statistics: every
figure in a script is pulled from the analytics layer, and a verifier recomputes it before the
reel is published.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -e ".[dev]"

python -m cutaway.engine --seed 17      # generate a match + insights into data/
pytest                                  # run the tests
python -m cutaway.agents --seed 17 --mode analyst --lang el --club NGR   # produce a reel
uvicorn cutaway.api.main:app --reload   # API on http://localhost:8000/docs
```

### API

| Endpoint | Description |
|---|---|
| `GET /api/matches/{seed}` | Match metadata and events (`?snapshots=true` adds 22-player positions) |
| `GET /api/matches/{seed}/insights` | Team/player stats, momentum, control vs chaos, key moments |
| `GET /api/matches/{seed}/stream?speed=20` | Server-sent events replaying the match in scaled real time |
| `POST /api/matches/{seed}/reels` | Run the agent crew for a persona and return a verified reel spec |

## Synthetic data model

The simulator is a possession-chain model on a 105 x 68 m pitch. The ball carrier chooses to
pass, carry, dribble, shoot or clear from location, pressure and player attributes; outcomes
depend on the same metrics the analytics layer reports (pass difficulty, xG) plus latent team
form. Formation templates shift with the ball to give every event a 22-player snapshot.

Latent drivers create real stories: a trailing team switches to a high press after 55 minutes,
substitutions refresh form, and goals shift momentum. That is what lets Cutaway explain *why*
a match turned, with evidence, instead of only reporting *that* it did.

Over 20 simulated matches the engine produces about 25 shots, 1.5 xG per team and 83% pass
completion per match, all in realistic ranges.

## Roadmap

- [x] Synthetic match engine, metrics, insights, API, tests
- [x] Agent crew on Microsoft Agent Framework: Editor, Director, Scriptwriter, Verifier
- [x] Reel spec format (timed, machine-readable overlay cues with evidence)
- [ ] Web player: animated pitch, overlays, synced voice-over (Azure AI Speech)
- [ ] Deploy to Azure Container Apps

## License

MIT
