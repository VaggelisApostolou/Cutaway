from fastapi.testclient import TestClient

import os

os.environ["CUTAWAY_LLM_PROVIDER"] = "offline"

from cutaway.api.main import app  # noqa: E402

client = TestClient(app)


def test_match_and_insights():
    r = client.get("/api/matches/17")
    assert r.status_code == 200
    body = r.json()
    assert body["match"]["home"]["name"] == "Harbour City"
    assert "snapshot" not in body["events"][0]
    ins = client.get("/api/matches/17/insights").json()
    assert {"team_stats", "player_stats", "momentum", "control_chaos", "moments"} <= ins.keys()


def test_unknown_match():
    assert client.get("/api/matches/999999").status_code == 404


def test_reel_endpoint():
    r = client.post("/api/matches/17/reels", json={"mode": "casual", "language": "en"})
    assert r.status_code == 200
    reel = r.json()
    assert reel["verified"] and reel["segments"]
