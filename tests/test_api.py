from fastapi.testclient import TestClient


def test_api_end_to_end(tmp_path, monkeypatch):
    import perkpilot.api as api
    import perkpilot.config as config
    import perkpilot.db as db

    path = tmp_path / "t.sqlite"
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(db, "DB_PATH", path)
    monkeypatch.setattr(api, "DB_PATH", path)
    from perkpilot.build import build

    build(db_path=path, mode="offline")
    c = TestClient(api.app)
    credits = c.get("/api/credits").json()
    assert any(x["credit_id"] == "amex_gold.resy_credit" and x["remaining"] == 20 for x in credits)
    sim = c.post("/api/simulate", json={"merchant": "Bistro", "amount": 50, "merchant_category": "dining"}).json()
    assert sim["ranked"][0]["card"] == "amex_gold"
    ans = c.post("/api/ask", json={"question": "What is the annual fee on the Amex Platinum?"}).json()
    assert ans["value"] == 695 and ans["claims"][0]["citations"][0]["clause_id"] == "AP-1"
    cl = c.get("/api/clause/AP-1").json()
    assert cl["text"].startswith("[AP-1]")
    assert "perkpilot" in c.get("/").text
