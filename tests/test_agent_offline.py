"""Agent 循环的离线测试：用脚本化的假 LLM 驱动 LangGraph，验证修复、绑定、拒答与步数上限。

数据库相关测试在 PostGIS 不可用时自动跳过（CI 中只跑纯逻辑部分）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from geo_agent.agent import GeoAgent
from geo_agent.config import Settings
from geo_agent.llm import Usage


class ScriptedLLM:
    """按顺序返回预设的工具调用；记录模型看到的每一轮消息。"""

    def __init__(self, script: list[list[tuple[str, dict[str, Any]]] | str]) -> None:
        self.script = list(script)
        self.seen: list[list[dict[str, Any]]] = []

    def chat(self, messages, **_: Any):
        self.seen.append(list(messages))
        step = self.script.pop(0) if self.script else "done"
        if isinstance(step, str):
            return SimpleNamespace(content=step, tool_calls=None), Usage(100, 10, 1)
        calls = [
            SimpleNamespace(
                id=f"c{len(self.seen)}_{i}",
                function=SimpleNamespace(name=n, arguments=json.dumps(a, ensure_ascii=False)),
            )
            for i, (n, a) in enumerate(step)
        ]
        return SimpleNamespace(content="", tool_calls=calls), Usage(100, 10, 1)


def _db_available() -> bool:
    try:
        from geo_agent.db import connect

        with connect() as c:
            c.execute("SELECT 1")
        return True
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _db_available(), reason="PostGIS container not running")


def test_refusal_without_db() -> None:
    llm = ScriptedLLM([[("final_answer", {"refuse": True, "answer": "", "reason": "no population data"})]])
    agent = GeoAgent(settings=Settings(), llm=llm, schema_text="TABLE pois ...")

    class NoConn:
        def close(self) -> None:
            pass

    run = agent.run("南京各区人口？", conn=NoConn())  # type: ignore[arg-type]
    assert run.status == "refused"
    assert run.model_calls == 1
    assert run.final_result is None


def test_step_limit_ends_with_no_answer() -> None:
    llm = ScriptedLLM(["thinking..."] * 20)
    agent = GeoAgent(settings=Settings(max_tool_steps=3), llm=llm, schema_text="x")

    class NoConn:
        def close(self) -> None:
            pass

    run = agent.run("问题", conn=NoConn())  # type: ignore[arg-type]
    assert run.status == "no_answer"
    assert run.model_calls == 3


@needs_db
def test_unit_error_is_repaired_and_numbers_bound() -> None:
    bad = "SELECT count(*) AS n FROM pois p, metro_stations m WHERE m.name = '新街口' AND p.subtype = 'restaurant' AND ST_DWithin(p.geom, m.geom, 0.005)"
    good = bad.replace(
        "ST_DWithin(p.geom, m.geom, 0.005)", "ST_DWithin(p.geom::geography, m.geom::geography, 500)"
    )
    llm = ScriptedLLM(
        [
            [("run_sql", {"sql": bad})],
            [("run_sql", {"sql": good})],
            [("final_answer", {"result_ref": "R1", "answer": "共有 {R1.n} 家，约 99 家"})],
        ]
    )
    agent = GeoAgent(settings=Settings(), llm=llm, schema_text="x")
    run = agent.run("新街口地铁站 500 米内有多少家餐厅？")
    assert run.status == "answered"
    assert run.repairs_used == 1
    assert run.raw_sqls[0]["valid"] is False and "unit" in run.raw_sqls[0]["error"]
    assert run.raw_sqls[1]["valid"] is True
    assert run.final_result is not None and run.final_result.rows == [[32]]
    # 99 是模型自己写的数字，不来自任何结果：必须被剔除
    assert "99" not in run.answer and "32" in run.answer
    assert run.bound is not None and run.bound.unbound_numbers == ["99"]
    # 校验器的修复提示确实回传给了模型
    assert any("unit/CRS error" in m.get("content", "") for m in llm.seen[1] if m["role"] == "tool")


@needs_db
def test_guard_rejection_counts_as_invalid_sql() -> None:
    llm = ScriptedLLM(
        [
            [("run_sql", {"sql": "DELETE FROM pois"})],
            [("final_answer", {"refuse": True, "answer": "", "reason": "x"})],
        ]
    )
    run = GeoAgent(settings=Settings(), llm=llm, schema_text="x").run("删除所有兴趣点")
    assert run.raw_sqls == [
        {"sql": "DELETE FROM pois", "valid": False, "error": "only SELECT queries are allowed, got Delete"}
    ]


@needs_db
def test_result_rows_and_count_selection() -> None:
    llm = ScriptedLLM(
        [
            [
                (
                    "nearest_k",
                    {
                        "anchor": {"layer": "pois", "name": "南京博物院"},
                        "target": {"layer": "metro_stations"},
                        "k": 3,
                    },
                )
            ],
            [
                (
                    "final_answer",
                    {"result_ref": "R1", "result_rows": [0], "answer": "最近的是 {R1.rows[0].name}"},
                )
            ],
        ]
    )
    run = GeoAgent(settings=Settings(), llm=llm, schema_text="x").run("离南京博物院最近的地铁站？")
    assert run.final_result is not None and len(run.final_result.rows) == 1
    assert run.final_result.rows[0][0] == "明故宫"
    assert run.answer == "最近的是 明故宫"
    fc = GeoAgent.geojson(run)
    assert fc["features"][0]["geometry"]["type"] == "Point"

    llm2 = ScriptedLLM(
        [
            [
                (
                    "within_distance",
                    {
                        "anchor": {"layer": "pois", "name": "德基广场"},
                        "target": {"layer": "metro_stations"},
                        "radius_m": 1000,
                    },
                )
            ],
            [("final_answer", {"result_ref": "R1", "result_as": "count", "answer": "{R1.count} 个"})],
        ]
    )
    run2 = GeoAgent(settings=Settings(), llm=llm2, schema_text="x").run("德基广场 1 公里内有多少个地铁站？")
    assert run2.final_result is not None and run2.final_result.rows == [[4]]
