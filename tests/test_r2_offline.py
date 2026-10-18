"""第二轮组件的离线单元测试：不需要数据库、API key 或向量模型文件。

覆盖：守卫对 pgRouting 内层查询的递归检查、条文数值字段与原文一致、步速换算、
合规答案校验、Agent 工具集（消融臂确实去掉了路网/RAG 工具）、基线 JSON 解析、精简 schema 不含标签映射。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from geo_agent import routing as RT
from geo_agent.config import R2_TABLES, Settings
from geo_agent.llm import Usage
from geo_agent.r2_agent import ABLATION_DROP, R2_SPECS, R2Agent
from geo_agent.r2_baseline import R2Baseline, first_json
from geo_agent.r2_schema import FUNCTION_NOTES, LEAN_COLUMN_NOTES, LEAN_TABLE_NOTES
from geo_agent.sql_guard import SqlRejected, check_sql
from geo_agent.tools import ToolError

ROOT = Path(__file__).resolve().parents[1]
EDGES = "'SELECT id, source, target, cost_m AS cost, cost_m AS reverse_cost FROM walk_edges'"


# ---------------------------------------------------------------- 守卫


def test_pgr_allowed_only_with_flag() -> None:
    sql = f"SELECT max(agg_cost) FROM pgr_dijkstra({EDGES}, 1, 2, false)"
    with pytest.raises(SqlRejected):
        check_sql(sql)  # 第一轮默认：不放行
    assert "walk_edges" in check_sql(sql, R2_TABLES, allow_pgr=True).tables


@pytest.mark.parametrize(
    "inner",
    [
        "'DELETE FROM walk_edges'",
        "'SELECT id, source, target, 1 AS cost FROM osm_raw.lines'",
        "'SELECT pg_sleep(10)'",
        "'SELECT 1; DROP TABLE pois'",
        "'SELECT id, source, target, cost_m AS cost FROM pg_shadow'",
    ],
)
def test_pgr_inner_query_is_guarded(inner: str) -> None:
    with pytest.raises(SqlRejected) as exc:
        check_sql(f"SELECT * FROM pgr_drivingDistance({inner}, 1, 800, false)", R2_TABLES, allow_pgr=True)
    assert "inner query rejected" in str(exc.value)


def test_vector_operator_and_r2_tables() -> None:
    check_sql(
        "SELECT clause_id FROM standard_clauses ORDER BY embedding <=> '[0.1,0.2]'::vector LIMIT 3",
        R2_TABLES,
        allow_pgr=True,
    )
    with pytest.raises(SqlRejected):
        check_sql("SELECT * FROM walk_edges")  # 第一轮白名单不含第二轮表


def test_r2_gold_sql_passes_guard() -> None:
    for name in ("questions.json", "dev_questions.json"):
        for q in json.loads((ROOT / "eval" / "r2" / name).read_text(encoding="utf-8")):
            if q["gold_sql"]:
                check_sql(q["gold_sql"], R2_TABLES, allow_pgr=True)


# ---------------------------------------------------------------- 条文与口径


def test_clause_numbers_appear_in_body() -> None:
    # 条文的结构化数值必须能在原文中找到（防止转录时数字字段与原文不一致）
    clauses = json.loads((ROOT / "data" / "standards" / "clauses.json").read_text(encoding="utf-8"))
    assert len({c["clause_id"] for c in clauses}) == len(clauses)
    for c in clauses:
        for k in ("max_radius_m", "walk_m_min", "walk_m_max"):
            if c[k] is not None:
                assert re.search(rf"(?<!\d){c[k]}m", c["body"]), (c["clause_id"], k)


def test_frozen_compliance_clauses_exist() -> None:
    ids = {
        c["clause_id"] for c in json.loads((ROOT / "data" / "standards" / "clauses.json").read_text("utf-8"))
    }
    for q in json.loads((ROOT / "eval" / "r2" / "questions.json").read_text(encoding="utf-8")):
        if q.get("compliance"):
            assert q["compliance"] in ids


def test_walk_speed() -> None:
    assert RT.WALK_M_PER_MIN == pytest.approx(80.0)
    assert RT.minutes_to_m(15) == pytest.approx(1200.0)
    with pytest.raises(ToolError):
        RT.minutes_to_m(0)
    with pytest.raises(ToolError):
        RT.minutes_to_m(31)


def test_lean_schema_has_no_tag_mapping() -> None:
    # 精简 schema 不能出现第一轮那样的“中文=标签”映射
    text = " ".join(LEAN_TABLE_NOTES.values()) + " ".join(LEAN_COLUMN_NOTES.values())
    for leak in ("医院", "hospital", "银行", "bank=", "肯德基", "buddhist", "brand="):
        assert leak not in text
    assert set(LEAN_TABLE_NOTES) == set(R2_TABLES)
    assert "pgr_dijkstra" in FUNCTION_NOTES and "80 m per minute" in FUNCTION_NOTES


# ---------------------------------------------------------------- 两臂


def test_ablation_drops_routing_and_rag_tools() -> None:
    names = {s["function"]["name"] for s in R2_SPECS}
    assert ABLATION_DROP <= names
    llm = SimpleNamespace(chat=None)
    full = R2Agent(settings=Settings(), llm=llm, schema_text="S")
    abl = R2Agent(settings=Settings(), llm=llm, schema_text="S", ablation=True)
    assert {s["function"]["name"] for s in abl.specs} == names - ABLATION_DROP
    assert full.system_prompt() == abl.system_prompt()  # 消融只差工具，不差提示词
    assert full.s.max_tool_steps == 10


def test_both_arms_share_rules_and_schema() -> None:
    llm = SimpleNamespace(chat=None)
    a = R2Baseline(settings=Settings(), llm=llm, schema_text="SCHEMA-X").system_prompt()
    b = R2Agent(settings=Settings(), llm=llm, schema_text="SCHEMA-X").system_prompt()
    from geo_agent.r2_prompts import R2_COMMON_RULES

    for p in (a, b):
        assert p.startswith(R2_COMMON_RULES) and FUNCTION_NOTES in p and p.endswith("SCHEMA-X")


def test_first_json_ignores_trailing_text() -> None:
    assert first_json('{"sql": "SELECT 1"}\nWait, actually {"sql": "SELECT 2"}') == {"sql": "SELECT 1"}
    assert first_json("no json") is None


class _OneShot:
    def __init__(self, content: str) -> None:
        self.content = content

    def chat(self, messages, **_: Any):
        return SimpleNamespace(content=self.content, tool_calls=None), Usage(10, 5, 1)


def test_baseline_refusal_offline() -> None:
    r = R2Baseline(
        settings=Settings(), llm=_OneShot('{"refuse": true, "reason": "no data"}'), schema_text="S"
    ).run("人口？")
    assert r.status == "refused" and r.usage.calls == 1


def test_baseline_bad_output_keeps_raw_text() -> None:
    r = R2Baseline(settings=Settings(), llm=_OneShot("sorry, cannot"), schema_text="S").run("x")
    assert r.status == "bad_output" and r.error == "sorry, cannot"
