"""第二轮 Agent：复用第一轮 LangGraph 图（model ⇄ tools，校验 + 有限修复 + 数字绑定），替换工具集与提示词。

工具（OpenAI function calling）：
- 自省：find_entity（含别名与小区/公交站图层）、distinct_values、tag_values
- 直线 GIS（第一轮）：within_distance、nearest_k、count_by_district、measure、spatial_relation、distance_between
- 路网（pgRouting）：walk_route、isochrone、reachable
- 标准（RAG）：retrieve_standard、check_compliance
- 通用：run_sql（第二轮白名单，允许 pgRouting 函数；仍经 AST 守卫、单位静态检查、只读事务）
- 收尾：final_answer

消融臂（ablation=True）：去掉 walk_route、isochrone、reachable、retrieve_standard、check_compliance，
其余（含 run_sql 写 pgRouting、精简 schema、规则、校验修复）完全相同。

第二轮新增的校验：合规类最终答案必须引用一个存在于 standard_clauses 的 clause_id，
且若结果含 radius_m，必须与该条文的 max_radius_m 相等——条文数字被程序约束，模型无法手填。
"""

from __future__ import annotations

import json
import time
from typing import Any

import psycopg
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from . import r2_tools as X
from . import routing as RT
from . import tools as T
from .agent import TOOL_SPECS as R1_SPECS
from .agent import AgentRun, GeoAgent, State, Step, _fn, _inline, _preview, validate
from .config import Settings, get_settings
from .db import QueryError, QueryResult, run_select
from .llm import LLMClient
from .r2_prompts import R2_AGENT_SYSTEM
from .r2_schema import FUNCTION_NOTES, describe_lean_schema
from .sql_guard import SqlRejected
from .units import lint_units

_R2_LAYERS = [
    "pois",
    "metro_stations",
    "districts",
    "water_areas",
    "parks",
    "waterways",
    "roads",
    "residential_areas",
    "bus_stops",
]
_PLACE = _inline(RT.Place.model_json_schema())
_R2_TARGET = _inline(T.R2TargetFilter.model_json_schema())

_R1_KEEP = {
    "distinct_values",
    "within_distance",
    "nearest_k",
    "count_by_district",
    "measure",
    "spatial_relation",
    "distance_between",
    "final_answer",
}

R2_SPECS: list[dict[str, Any]] = [
    _fn(
        "find_entity",
        "Fuzzy-find features by name or alias (alt_name/old_name/short_name/name:en); returns layer, exact name, "
        "category, district and how it matched.",
        {"name": {"type": "string"}, "layer": {"type": "string", "enum": _R2_LAYERS}},
        ["name"],
    ),
    _fn(
        "tag_values",
        "Raw OSM tags of pois. Without key: the 60 most common tag keys. With key: values and counts "
        "(optionally filtered by substring `like`).",
        {"key": {"type": "string"}, "like": {"type": "string"}},
        [],
    ),
    *[s for s in R1_SPECS if s["function"]["name"] in _R1_KEEP and s["function"]["name"] != "final_answer"],
    _fn(
        "walk_route",
        "Shortest walking path on the pedestrian network between two named features: walk_m (network meters), "
        "walk_min (at 4.8 km/h), straight_m, snap distances, GeoJSON line.",
        {"a": _PLACE, "b": _PLACE},
        ["a", "b"],
    ),
    _fn(
        "isochrone",
        "Walking isochrone: area_km2 of the 50 m buffer around edges reachable within `minutes` (4.8 km/h), "
        "GeoJSON polygon.",
        {"origin": _PLACE, "minutes": {"type": "number"}},
        ["origin", "minutes"],
    ),
    _fn(
        "reachable",
        "Target features reachable within `minutes` of walking (network distance <= 80 m/min * minutes), sorted "
        "by walk_m. count_only=true returns n.",
        {
            "origin": _PLACE,
            "target": _R2_TARGET,
            "minutes": {"type": "number"},
            "count_only": {"type": "boolean"},
        },
        ["origin", "target", "minutes"],
    ),
    _fn(
        "retrieve_standard",
        "Semantic search over GB 50180-2018 clause excerpts (local embedding model). Returns clause_id, title, "
        "facility, circle_min, max_radius_m, walk_m_min/max, score and body text.",
        {"query": {"type": "string"}, "k": {"type": "integer"}},
        ["query"],
    ),
    _fn(
        "check_compliance",
        "Apply a clause to a residential area: the program reads max_radius_m from the clause and checks whether "
        "a target facility is reachable on the walking network within it. Returns compliant, clause_id, "
        "radius_m, nearest_walk_m, nearest_name.",
        {"residential": {"type": "string"}, "clause_id": {"type": "string"}, "target": _R2_TARGET},
        ["residential", "clause_id", "target"],
    ),
    _fn(
        "run_sql",
        "Run one read-only PostGIS/pgRouting SELECT (max 200 rows, 5 s). Add ST_AsGeoJSON(geom) AS geojson "
        "to show features on the map.",
        {"sql": {"type": "string"}},
        ["sql"],
    ),
    next(s for s in R1_SPECS if s["function"]["name"] == "final_answer"),
]
ABLATION_DROP = {"walk_route", "isochrone", "reachable", "retrieve_standard", "check_compliance"}


def compliance_problems(res: QueryResult, conn: psycopg.Connection) -> list[str]:
    """合规答案的确定性校验：引用的 clause_id 必须存在；radius_m 必须与条文一致。"""
    if not res.rows:
        return []
    row = res.rows[0]
    cols = [c.lower() for c in res.columns]
    cited = [v for v in row if isinstance(v, str) and v.startswith("GB50180-2018:")]
    if not cited:
        return []
    found = run_select(
        "SELECT clause_id, max_radius_m FROM standard_clauses WHERE clause_id = '"
        + cited[0].replace("'", "''")
        + "'",
        conn=conn,
        r2=True,
    )
    if not found.rows:
        return [f"clause_id {cited[0]} does not exist; use retrieve_standard"]
    if "radius_m" in cols and found.rows[0][1] is not None:
        v = row[cols.index("radius_m")]
        if v != found.rows[0][1]:
            return [f"radius_m {v} differs from clause {cited[0]} max_radius_m {found.rows[0][1]}"]
    return []


class R2Agent(GeoAgent):
    def __init__(
        self,
        settings: Settings | None = None,
        llm: Any | None = None,
        schema_text: str | None = None,
        tag: str = "dev",
        ablation: bool = False,
    ) -> None:
        s = settings or get_settings()
        # 第二轮：最多 10 次模型调用（新增了检索与合规两步）；修复额度仍为 3
        s = Settings(**{**s.__dict__, "max_tool_steps": 10})
        self.ablation = ablation
        self.specs = [x for x in R2_SPECS if not (ablation and x["function"]["name"] in ABLATION_DROP)]
        super().__init__(
            settings=s,
            llm=llm or LLMClient(s, tag=tag),
            schema_text=schema_text or describe_lean_schema(),
        )
        self.arm = "r2_ablation" if ablation else "r2_agent"

    def system_prompt(self) -> str:
        return R2_AGENT_SYSTEM + FUNCTION_NOTES + "\nDatabase schema:\n" + self.schema_text

    # 覆盖：模型节点使用第二轮工具集
    def _model(self, state: State, config: RunnableConfig) -> dict[str, Any]:
        run = state["run"]
        qid = config["configurable"].get("qid", "")
        msg, usage = self.llm.chat(
            state["messages"], tools=self.specs, qid=qid, arm=self.arm, max_tokens=1500
        )
        run.model_calls += 1
        run.usage.add(usage)
        m: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            m["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages = state["messages"] + [m]
        if not msg.tool_calls:
            messages.append(
                {
                    "role": "user",
                    "content": "Call final_answer (or another tool). Plain-text answers are not accepted.",
                }
            )
        return {"messages": messages}

    def _exec_tool(
        self,
        name: str,
        args: dict[str, Any],
        step: Step,
        results: dict[str, QueryResult],
        run: AgentRun,
        conn: psycopg.Connection,
    ) -> str:
        if self.ablation and name in ABLATION_DROP:
            step.problems.append(f"unknown tool {name}")
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            if name == "run_sql":
                sql_text = str(args.get("sql", ""))
                rec = {"sql": sql_text, "valid": False, "error": None}
                run.raw_sqls.append(rec)
                issues = lint_units(sql_text)
                if issues:
                    rec["error"] = "unit: " + "; ".join(map(str, issues))
                    step.problems.append(
                        "unit/CRS error: "
                        + "; ".join(map(str, issues))
                        + ". geom is EPSG:4326 degrees; use ::geography for meters"
                    )
                    return json.dumps({"error": "rejected before execution"}, ensure_ascii=False)
                res = run_select(sql_text, conn=conn, settings=self.s, r2=True)
                rec["valid"] = True
            elif name == "find_entity":
                res = X.find_entity_r2(args["name"], conn, args.get("layer"))
            elif name == "tag_values":
                res = X.tag_values(conn, args.get("key"), args.get("like"))
            elif name == "walk_route":
                res = RT.walk_route(RT.Place(**args["a"]), RT.Place(**args["b"]), conn)
            elif name == "isochrone":
                res = RT.isochrone(RT.Place(**args["origin"]), float(args["minutes"]), conn)
            elif name == "reachable":
                res = RT.reachable_facilities(
                    RT.Place(**args["origin"]),
                    T.R2TargetFilter(**args["target"]),
                    float(args["minutes"]),
                    conn,
                    bool(args.get("count_only", False)),
                )
            elif name == "retrieve_standard":
                res = X.retrieve_standard(str(args["query"]), conn, int(args.get("k", 5)))
            elif name == "check_compliance":
                res = X.check_compliance(
                    str(args["residential"]), str(args["clause_id"]), T.R2TargetFilter(**args["target"]), conn
                )
            else:
                # 第一轮直线 GIS 工具与 distinct_values：沿用父类实现
                return super()._exec_tool(name, args, step, results, run, conn)
        except (SqlRejected, QueryError, T.ToolError) as err:
            if name == "run_sql":
                run.raw_sqls[-1]["error"] = str(err)
            step.problems.append(str(err))
            return json.dumps({"error": str(err)}, ensure_ascii=False)
        except (KeyError, TypeError, ValueError, ValidationError) as err:
            step.problems.append(f"bad arguments: {err}")
            return json.dumps({"error": f"bad arguments: {err}"}, ensure_ascii=False)
        ref = f"R{len(results) + 1}"
        results[ref] = res
        run.all_results[ref] = res
        step.ref, step.sql, step.rows = ref, res.sql, len(res.rows)
        if name not in ("find_entity", "tag_values", "retrieve_standard"):
            # 合规结论 false 与可达计数 0 都是合法答案，不按“空/0 结果”要求修复
            if name not in ("check_compliance",) and not (name == "reachable" and args.get("count_only")):
                step.problems.extend(validate(name, res))
            step.problems.extend(compliance_problems(res, conn))
        return _preview(ref, res)

    def run(self, question: str, qid: str = "", conn: psycopg.Connection | None = None) -> AgentRun:
        # 与父类相同，只替换系统提示词
        from .db import connect

        run = AgentRun(question=question)
        own = conn is None
        c = conn or connect(self.s)
        t0 = time.perf_counter()
        try:
            state: State = {
                "messages": [
                    {"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": question},
                ],
                "run": run,
                "results": {},
                "done": False,
            }
            self.graph.invoke(
                state,
                {"configurable": {"conn": c, "qid": qid}, "recursion_limit": 4 * self.s.max_tool_steps + 4},
            )
        finally:
            run.latency_s = time.perf_counter() - t0
            if own:
                c.close()
        if run.status == "running":
            run.status = "no_answer"
        return run


__all__ = ["R2Agent", "R2_SPECS", "ABLATION_DROP"]
