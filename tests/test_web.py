from __future__ import annotations

from web_app import app


def test_health():
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json["status"] == "ok"


def test_run_dashboard():
    client = app.test_client()
    resp = client.post("/run", data={"config_path": "config/default.yaml"})
    assert resp.status_code == 200


def test_api_run():
    client = app.test_client()
    resp = client.post("/api/run", json={"config_path": "config/default.yaml"})
    assert resp.status_code == 200
    payload = resp.json
    assert "recommendations" in payload
    assert "risk_report" in payload

