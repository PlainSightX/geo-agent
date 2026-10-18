"""FastAPI 服务：POST /ask 返回绑定后的答案、结果表与 GeoJSON；/ 提供 Leaflet 演示页。

安全说明：该服务没有鉴权，只用于本机演示（默认监听 127.0.0.1）。
数据库侧只用只读角色 geo_reader，SQL 经 AST 白名单 + 只读事务 + 语句超时三层约束；
但每次 /ask 都会消耗 DeepSeek API 额度，若要暴露到公网，必须先加鉴权与限流。
"""

from __future__ import annotations

import json
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
    # round1 = 第一轮 Agent（默认，行为不变）；round2 = 第二轮 Agent（路网、条文检索、合规工具）
    agent: str = Field(default="round1", pattern="^(round1|round2)$")


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


@lru_cache(maxsize=2)
def get_agent(kind: str = "round1") -> GeoAgent:
    if kind == "round2":
        from .r2_agent import R2Agent

        return R2Agent(tag="api_r2")
    return GeoAgent(tag="api")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/demo/life-circle")
def demo_life_circle(residential: str = "南湾营融康苑", minutes: float = 12.5) -> dict[str, Any]:
    """确定性演示（不调用模型）：小区步行等时圈 + 圈内中学 + 按 GB 50180-2018 初中服务半径的合规判定。

    用于截图与离线演示；等时圈、可达设施与合规结论都由 routing / r2_tools 的参数化查询计算。
    """
    from . import r2_tools as X
    from . import routing as RT
    from .db import connect
    from .tools import R2TargetFilter

    origin = RT.Place(layer="residential_areas", name=residential)
    school = R2TargetFilter(layer="pois", subtypes=["school"], name_like="中学")
    with connect() as conn:
        try:
            iso = RT.isochrone(origin, minutes, conn)
            fac = RT.reachable_facilities(origin, school, minutes, conn)
            ok = X.check_compliance(residential, "GB50180-2018:C.0.1:初中", school, conn)
        except Exception as err:
            raise HTTPException(status_code=400, detail=str(err)[:200]) from err
    features = [
        {
            "type": "Feature",
            "geometry": json.loads(iso.rows[0][-1]),
            "properties": {"name": f"步行 {minutes:g} 分钟"},
        }
    ]
    for name, walk_m, _walk_min, gj in fac.rows:
        features.append(
            {
                "type": "Feature",
                "geometry": json.loads(gj),
                "properties": {"name": f"{name}（{walk_m:.0f} m）"},
            }
        )
    compliant, clause_id, radius_m, nearest_m, nearest_name = ok.rows[0]
    verdict = "符合" if compliant else "不符合"
    answer = (
        f"{residential}：按 {clause_id}（初中服务半径不宜大于 {radius_m} m），沿步行路网{verdict}"
        + (f"，最近中学 {nearest_name} 步行 {nearest_m:.0f} m。" if compliant else "。")
        + f" 图中蓝色为步行 {minutes:g} 分钟（4.8 km/h）等时圈，面积 {iso.rows[0][3]:.2f} km²。"
    )
    return {
        "status": "answered",
        "answer": answer,
        "columns": ["name", "walk_m"],
        "rows": [[r[0], round(r[1], 1)] for r in fac.rows],
        "geojson": {"type": "FeatureCollection", "features": features, "map_source": "demo"},
        "steps": [
            {"tool": "isochrone", "args": {"minutes": minutes}, "sql": iso.sql, "rows": 1, "problems": []},
            {
                "tool": "reachable",
                "args": {"target": "中学"},
                "sql": fac.sql,
                "rows": len(fac.rows),
                "problems": [],
            },
            {
                "tool": "check_compliance",
                "args": {"clause_id": clause_id},
                "sql": ok.sql,
                "rows": 1,
                "problems": [],
            },
        ],
        "model_calls": 0,
        "cost_usd": 0.0,
        "latency_s": 0.0,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    agent = get_agent(req.agent)
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
