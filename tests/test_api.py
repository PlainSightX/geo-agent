"""API 契约测试：用假 Agent 替换真实 Agent，不需要 API key 或数据库。"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from geo_agent import api
from geo_agent.agent import AgentRun, Step
from geo_agent.db import QueryResult


class FakeAgent:
    def run(self, question: str, qid: str = "") -> AgentRun:
        run = AgentRun(
            question=question, status="answered", answer="最近的是 大行宫（105 米）", model_calls=2
        )
        pt = json.dumps({"type": "Point", "coordinates": [118.7895, 32.0436]})
        run.final_result = QueryResult(
            ["name", "dist_m", "geojson"], [["大行宫", 104.7, pt]], False, 1.0, "SELECT ..."
        )
        run.steps = [Step(tool="nearest_k", args={"k": 1}, ref="R1", sql="SELECT ...", rows=1)]
        return run


def test_ask_returns_bound_answer_and_geojson(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_agent", lambda kind="round1": FakeAgent())
    client = TestClient(api.app)
    r = client.post("/ask", json={"question": "离南京图书馆最近的地铁站？"})
    assert r.status_code == 200
    body = r.json()
    assert body["columns"] == ["name", "dist_m"]  # geojson 列不进表格
    assert body["rows"] == [["大行宫", 104.7]]
    assert body["geojson"]["features"][0]["properties"]["name"] == "大行宫"
    assert body["steps"][0]["tool"] == "nearest_k"


def test_ask_validates_input(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_agent", lambda kind="round1": FakeAgent())
    client = TestClient(api.app)
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 301}).status_code == 422


def test_index_served() -> None:
    client = TestClient(api.app)
    r = client.get("/")
    assert r.status_code == 200 and "leaflet" in r.text


def test_ask_rejects_unknown_agent(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_agent", lambda kind="round1": FakeAgent())
    client = TestClient(api.app)
    assert client.post("/ask", json={"question": "测试问题", "agent": "round2"}).status_code == 200
    assert client.post("/ask", json={"question": "测试问题", "agent": "other"}).status_code == 422
