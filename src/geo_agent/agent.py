"""LangGraph 空间问答 Agent：plan → 工具/SQL → 确定性校验 → 有限修复 → 绑定答案。

图结构（两个节点 + 条件边）：

    START → model ──有 tool_calls──→ tools ──未结束──→ model
              │                        │
              └──无 tool_calls（提醒）─┘  └─final_answer 或步数用尽─→ END

- model 节点：调用 deepseek-chat，携带工具定义；
- tools 节点：执行工具，给结果编号 R1, R2...，并运行校验器（守卫拒绝、执行错误、单位/CRS、空结果、截断）。
  校验发现问题即消耗一次修复额度，问题描述随工具结果回传给模型；额度用尽后只提示“请直接作答”。
- 终止：模型调用 final_answer（引用某个结果 id）或达到最大步数。答案中的数字由 answer.bind 从结果中填入。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from . import tools as T
from .answer import BoundAnswer, bind, to_feature_collection
from .config import Settings, get_settings
from .db import QueryError, QueryResult, connect, describe_schema, run_select
from .llm import LLMClient, Usage
from .prompts import AGENT_SYSTEM, REPAIR_HINT
from .sql_guard import SqlRejected
from .units import lint_units

# ---------------------------------------------------------------- 工具定义（OpenAI function calling 格式）

_ANCHOR = T.Anchor.model_json_schema()
_TARGET = T.TargetFilter.model_json_schema()
_LAYERS = ["pois", "metro_stations", "districts", "water_areas", "parks", "waterways", "roads"]


def _fn(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _inline(schema: dict[str, Any]) -> dict[str, Any]:
    # pydantic 生成的 schema 含 $defs/$ref，部分模型不支持，这里展开为纯对象
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


TOOL_SPECS: list[dict[str, Any]] = [
    _fn(
        "find_entity",
        "Fuzzy-find features by name; returns layer, exact name, category, district.",
        {"name": {"type": "string"}, "layer": {"type": "string", "enum": _LAYERS}},
        ["name"],
    ),
    _fn(
        "distinct_values",
        "List category values with counts for a layer (pois.subtype, waterways.waterway, ...).",
        {"layer": {"type": "string", "enum": _LAYERS}, "prefix": {"type": "string"}},
        ["layer"],
    ),
    _fn(
        "within_distance",
        "Features of target within radius_m meters of the anchor (geography distance). "
        "count_only=true returns a single count n.",
        {
            "anchor": _inline(_ANCHOR),
            "target": _inline(_TARGET),
            "radius_m": {"type": "number"},
            "count_only": {"type": "boolean"},
        },
        ["anchor", "target", "radius_m"],
    ),
    _fn(
        "nearest_k",
        "k nearest target features to the anchor with distance in meters.",
        {"anchor": _inline(_ANCHOR), "target": _inline(_TARGET), "k": {"type": "integer"}},
        ["anchor", "target", "k"],
    ),
    _fn(
        "count_by_district",
        "Count point features per district (ST_Within), with per_km2 density and area_km2.",
        {"target": _inline(_TARGET)},
        ["target"],
    ),
    _fn(
        "measure",
        "Area in km2 (polygon layers) or length in km (line layers) of the named feature(s), summed.",
        {
            "layer": {"type": "string", "enum": _LAYERS},
            "name": {"type": "string"},
            "metric": {"type": "string", "enum": ["area_km2", "length_km"]},
        },
        ["layer", "name", "metric"],
    ),
    _fn(
        "spatial_relation",
        "Features of left_layer that intersect/touch/are within/contain the anchor feature.",
        {
            "left_layer": {"type": "string", "enum": _LAYERS},
            "relation": {"type": "string", "enum": ["intersects", "touches", "within", "contains"]},
            "right": _inline(_ANCHOR),
        },
        ["left_layer", "relation", "right"],
    ),
    _fn(
        "distance_between",
        "Shortest distance in meters and km between two named features.",
        {"a": _inline(_ANCHOR), "b": _inline(_ANCHOR)},
        ["a", "b"],
    ),
    _fn(
        "run_sql",
        "Run one read-only PostGIS SELECT (max 200 rows, 5 s). Use ::geography for meters. "
        "Add ST_AsGeoJSON(geom) AS geojson to show features on the map.",
        {"sql": {"type": "string"}},
        ["sql"],
    ),
    _fn(
        "final_answer",
        "Finish. result_ref is the result id (e.g. R3) whose rows answer the question. "
        "If only some rows answer it (e.g. the top 1 of a nearest_k list, or one district of count_by_district), "
        "set result_rows to those 0-based row indexes, e.g. [0]. "
        "If the question asks 'how many' and the result lists the matching features, set result_as='count' "
        "(the program counts the rows). "
        "answer uses placeholders {R3.n}, {R3.count}, {R3.rows[0].col}, {R3.rows}; never type numbers. "
        "Set refuse=true when the data cannot answer.",
        {
            "result_ref": {"type": "string"},
            "result_rows": {"type": "array", "items": {"type": "integer"}},
            "result_as": {"type": "string", "enum": ["rows", "count"]},
            "answer": {"type": "string"},
            "refuse": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        ["answer"],
    ),
]


# ---------------------------------------------------------------- 运行记录


@dataclass
class Step:
    tool: str
    args: dict[str, Any]
    ref: str | None = None
    sql: str | None = None
    rows: int | None = None
    problems: list[str] = field(default_factory=list)
    elapsed_ms: float | None = None


@dataclass
class AgentRun:
    question: str
    status: str = "running"  # answered | refused | no_answer | error
    answer: str = ""
    result_ref: str | None = None
    final_result: QueryResult | None = None
    refuse_reason: str | None = None
    steps: list[Step] = field(default_factory=list)
    model_calls: int = 0
    repairs_used: int = 0
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    raw_sqls: list[dict[str, Any]] = field(default_factory=list)  # 每条模型写的 SQL 及其是否有效
    bound: BoundAnswer | None = None
    # 本次运行产生的全部工具结果（地图兜底渲染用；评测只看 final_result）
    all_results: dict[str, QueryResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "question": self.question,
            "status": self.status,
            "answer": self.answer,
            "result_ref": self.result_ref,
            "refuse_reason": self.refuse_reason,
            "steps": [asdict(s) for s in self.steps],
            "model_calls": self.model_calls,
            "repairs_used": self.repairs_used,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "latency_s": round(self.latency_s, 2),
            "raw_sqls": self.raw_sqls,
            "unbound_numbers": self.bound.unbound_numbers if self.bound else [],
        }
        if self.final_result is not None:
            d["final_columns"] = self.final_result.columns
            d["final_rows"] = [[v for v in r] for r in self.final_result.rows]
            d["final_sql"] = self.final_result.sql
            d["result_as"] = next(
                (st.args.get("result_as") for st in reversed(self.steps) if st.tool == "final_answer"), None
            )
            d["result_rows"] = next(
                (st.args.get("result_rows") for st in reversed(self.steps) if st.tool == "final_answer"), None
            )
        return d


class State(TypedDict):
    messages: list[dict[str, Any]]
    run: AgentRun
    results: dict[str, QueryResult]
    done: bool


# ---------------------------------------------------------------- 校验器


def validate(tool: str, res: QueryResult) -> list[str]:
    """对成功执行的结果做确定性检查；返回问题列表（空列表表示通过）。"""
    problems: list[str] = []
    if not res.rows:
        problems.append("result is empty; verify exact names (find_entity) and categories (distinct_values)")
    elif (
        tool in ("run_sql", "within_distance")
        and len(res.rows) == 1
        and len(res.columns) == 1
        and res.rows[0][0] in (0, None)
    ):
        problems.append("count is 0/NULL; verify the anchor name and category values before answering")
    if res.truncated:
        problems.append("result truncated at 200 rows; aggregate or add a filter")
    return problems


def _preview(ref: str, res: QueryResult, limit: int = 15) -> str:
    cols = [c for c in res.columns if c != "geojson"]
    idx = [res.columns.index(c) for c in cols]
    rows = [[r[i] for i in idx] for r in res.rows[:limit]]
    more = f" (showing {limit} of {len(res.rows)})" if len(res.rows) > limit else ""
    return (
        json.dumps(
            {
                "id": ref,
                "columns": cols,
                "rows": rows,
                "row_count": len(res.rows),
                "has_geojson": "geojson" in res.columns,
            },
            ensure_ascii=False,
            default=str,
        )
        + more
    )


# ---------------------------------------------------------------- Agent


class GeoAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: Any | None = None,
        schema_text: str | None = None,
        tag: str = "dev",
    ) -> None:
        self.s = settings or get_settings()
        self.llm = llm or LLMClient(self.s, tag=tag)
        self.schema_text = schema_text or describe_schema(settings=self.s)
        self.graph = self._build()

    # 节点：调用模型
    def _model(self, state: State, config: RunnableConfig) -> dict[str, Any]:
        run = state["run"]
        qid = config["configurable"].get("qid", "")
        msg, usage = self.llm.chat(state["messages"], tools=TOOL_SPECS, qid=qid, arm="agent", max_tokens=1200)
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
            # 模型直接输出文本：提醒它通过 final_answer 收尾（文本中的数字不被采信）
            messages.append(
                {
                    "role": "user",
                    "content": "Call final_answer (or another tool). Plain-text answers are not accepted.",
                }
            )
        return {"messages": messages}

    # 节点：执行工具 + 校验
    def _tools(self, state: State, config: RunnableConfig) -> dict[str, Any]:
        run, results = state["run"], dict(state["results"])
        conn: psycopg.Connection = config["configurable"]["conn"]
        last = state["messages"][-1]
        messages = list(state["messages"])
        done = False
        for tc in last.get("tool_calls", []):
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            step = Step(tool=name, args=args)
            run.steps.append(step)
            if name == "final_answer":
                content, done = self._finish(run, results, args, step)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
                continue
            t0 = time.perf_counter()
            content = self._exec_tool(name, args, step, results, run, conn)
            step.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            if step.problems:
                left = self.s.max_repairs - run.repairs_used
                if left > 0:
                    run.repairs_used += 1
                    content += "\n" + REPAIR_HINT.format(problems="; ".join(step.problems), left=left - 1)
                else:
                    content += (
                        "\nValidator: "
                        + "; ".join(step.problems)
                        + ". No repairs left: call final_answer now."
                    )
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
        return {"messages": messages, "results": results, "done": done}

    def _exec_tool(
        self,
        name: str,
        args: dict[str, Any],
        step: Step,
        results: dict[str, QueryResult],
        run: AgentRun,
        conn: psycopg.Connection,
    ) -> str:
        try:
            if name == "run_sql":
                sql_text = str(args.get("sql", ""))
                rec = {"sql": sql_text, "valid": False, "error": None}
                run.raw_sqls.append(rec)
                issues = lint_units(sql_text)
                if issues:
                    # 单位错误在执行前拦截：这类 SQL 能跑出“看似合理”的错误数字
                    rec["error"] = "unit: " + "; ".join(map(str, issues))
                    step.problems.append(
                        "unit/CRS error: "
                        + "; ".join(map(str, issues))
                        + ". geom is EPSG:4326 degrees; use ::geography for meters"
                    )
                    return json.dumps({"error": "rejected before execution"}, ensure_ascii=False)
                res = run_select(sql_text, conn=conn, settings=self.s)
                rec["valid"] = True
            elif name == "find_entity":
                res = T.find_entity(args["name"], conn, args.get("layer"))
            elif name == "distinct_values":
                res = T.distinct_values(args["layer"], conn, args.get("prefix"))
            elif name == "within_distance":
                res = T.within_distance(
                    T.Anchor(**args["anchor"]),
                    T.TargetFilter(**args["target"]),
                    float(args["radius_m"]),
                    conn,
                    bool(args.get("count_only", False)),
                )
            elif name == "nearest_k":
                res = T.nearest_k(
                    T.Anchor(**args["anchor"]), T.TargetFilter(**args["target"]), int(args["k"]), conn
                )
            elif name == "count_by_district":
                res = T.count_by_district(T.TargetFilter(**args["target"]), conn)
            elif name == "measure":
                res = T.measure(args["layer"], args["name"], args["metric"], conn)
            elif name == "spatial_relation":
                res = T.spatial_relation(
                    args["left_layer"], args["relation"], T.Anchor(**args["right"]), conn
                )
            elif name == "distance_between":
                res = T.distance_between(T.Anchor(**args["a"]), T.Anchor(**args["b"]), conn)
            else:
                step.problems.append(f"unknown tool {name}")
                return json.dumps({"error": f"unknown tool {name}"})
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
        # 自省工具（find_entity/distinct_values）的空结果是正常信息，不消耗修复额度
        if name not in ("find_entity", "distinct_values"):
            step.problems.extend(validate(name, res))
        return _preview(ref, res)

    def _finish(
        self, run: AgentRun, results: dict[str, QueryResult], args: dict[str, Any], step: Step
    ) -> tuple[str, bool]:
        if args.get("refuse"):
            run.status, run.refuse_reason = "refused", str(args.get("reason") or args.get("answer") or "")
            run.answer = "无法用当前数据回答：" + run.refuse_reason
            return "ok", True
        ref = args.get("result_ref")
        if ref not in results:
            step.problems.append(f"result_ref {ref!r} does not exist; valid ids: {sorted(results)}")
            # 无效引用且还有步数时让模型重试；否则以 no_answer 结束
            return json.dumps({"error": step.problems[-1]}), False
        bound = bind(str(args.get("answer", "")), results, run.question)
        run.bound = bound
        final = results[ref]
        rows_sel = args.get("result_rows")
        if isinstance(rows_sel, list) and rows_sel:
            # 行选择由程序执行：被评分/渲染的结果就是答案所引用的那几行
            try:
                picked = [final.rows[int(i)] for i in rows_sel]
            except (IndexError, ValueError, TypeError):
                step.problems.append(
                    f"result_rows {rows_sel} out of range for {ref} ({len(final.rows)} rows)"
                )
                return json.dumps({"error": step.problems[-1]}), False
            final = QueryResult(
                final.columns, picked, False, final.elapsed_ms, final.sql, final.geojson_columns
            )
        if args.get("result_as") == "count":
            # 计数由程序完成：最终结果变为单行单列 n
            final = QueryResult(["n"], [[len(final.rows)]], False, final.elapsed_ms, final.sql, [])
        run.status, run.result_ref, run.final_result, run.answer = "answered", ref, final, bound.text
        return "ok", True

    def _route(self, state: State) -> str:
        run = state["run"]
        if state["done"]:
            return END
        if run.model_calls >= self.s.max_tool_steps:
            run.status = "no_answer"
            return END
        last = state["messages"][-1]
        return "tools" if last.get("role") == "assistant" and last.get("tool_calls") else "model"

    def _build(self) -> Any:
        g = StateGraph(State)
        g.add_node("model", self._model)
        g.add_node("tools", self._tools)
        g.add_edge(START, "model")
        g.add_conditional_edges("model", self._route, {"tools": "tools", "model": "model", END: END})
        g.add_conditional_edges("tools", self._route, {"model": "model", "tools": "tools", END: END})
        return g.compile()

    def run(self, question: str, qid: str = "", conn: psycopg.Connection | None = None) -> AgentRun:
        run = AgentRun(question=question)
        own = conn is None
        c = conn or connect(self.s)
        t0 = time.perf_counter()
        try:
            state: State = {
                "messages": [
                    {"role": "system", "content": AGENT_SYSTEM + self.schema_text},
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

    @staticmethod
    def geojson(run: AgentRun) -> dict[str, Any]:
        """优先渲染最终结果；若最终结果没有几何列，退回到最近一个带 geojson 的工具结果，并在 map_source 中注明。"""
        fc = to_feature_collection(run.final_result)
        fc["map_source"] = run.result_ref
        if not fc["features"] and run.status == "answered":
            for ref in reversed(list(run.all_results)):
                alt = to_feature_collection(run.all_results[ref])
                if alt["features"]:
                    alt["map_source"] = ref
                    return alt
        return fc
