from fastapi.testclient import TestClient

from cutaway.api.main import app

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
