"""FastAPI 服务：POST /ask 返回绑定后的答案、结果表与 GeoJSON；/ 提供 Leaflet 演示页。

安全说明：该服务没有鉴权，只用于本机演示（默认监听 127.0.0.1）。
数据库侧只用只读角色 geo_reader，SQL 经 AST 白名单 + 只读事务 + 语句超时三层约束；
但每次 /ask 都会消耗 DeepSeek API 额度，若要暴露到公网，必须先加鉴权与限流。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import GeoAgent

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="geo-agent", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=300)


class AskResponse(BaseModel):
    status: str
    answer: str
    columns: list[str]
    rows: list[list[Any]]
    geojson: dict[str, Any]
    steps: list[dict[str, Any]]
    model_calls: int
    cost_usd: float
    latency_s: float


@lru_cache(maxsize=1)
def get_agent() -> GeoAgent:
    return GeoAgent(tag="api")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    agent = get_agent()
    try:
        run = agent.run(req.question.strip(), qid="api")
    except Exception as err:  # 对外只返回类型名，细节写服务端日志
        raise HTTPException(status_code=502, detail=f"agent failed: {type(err).__name__}") from err
    res = run.final_result
    cols = [c for c in (res.columns if res else []) if c != "geojson"]
    idx = [res.columns.index(c) for c in cols] if res else []
    rows = [[r[i] for i in idx] for r in res.rows] if res else []
    return AskResponse(
        status=run.status,
        answer=run.answer,
        columns=cols,
        rows=rows,
        geojson=GeoAgent.geojson(run),
        steps=[
            {"tool": s.tool, "args": s.args, "sql": s.sql, "rows": s.rows, "problems": s.problems}
            for s in run.steps
        ],
        model_calls=run.model_calls,
        cost_usd=round(run.usage.cost(), 5),
        latency_s=round(run.latency_s, 2),
    )
