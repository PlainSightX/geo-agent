"""MCP 服务器测试。

离线部分（CI 中运行，不需要数据库）：工具注册表与第二轮 Agent 工具集一致、参数 schema 是合法 JSON Schema、
run_sql 经 MCP 调用时 DML/DDL 被同一 sqlglot 守卫拒绝（守卫在建立数据库连接之前执行）。
集成部分：用 SDK 的进程内客户端会话列出工具并调用 find_entity、isochrone；没有第二轮数据库时跳过。
"""

from __future__ import annotations

import json

import anyio
import pytest
from jsonschema import Draft202012Validator
from mcp import Client

from geo_agent import mcp_server as M
from geo_agent import r2_tools as X
from geo_agent.db import QueryResult
from geo_agent.r2_agent import R2_SPECS

try:
    from geo_agent.db import connect

    with connect() as _c:
        _c.execute("SELECT 1 FROM walk_edges LIMIT 1")
    HAVE_DB = True
except Exception:
    HAVE_DB = False

R2_TOOL_NAMES = {s["function"]["name"] for s in R2_SPECS} - {"final_answer"}


def _list_tools() -> list:
    return anyio.run(M.mcp.list_tools)


def _call(name: str, args: dict):
    async def go():
        async with Client(M.mcp) as client:
            return await client.call_tool(name, args)

    return anyio.run(go)


# ---------------------------------------------------------------- 离线


def test_registry_matches_r2_tool_set() -> None:
    names = [t.name for t in _list_tools()]
    assert len(names) == len(set(names))
    assert set(names) == R2_TOOL_NAMES  # final_answer 属于 Agent 编排层，不暴露


def test_required_params_match_r2_specs() -> None:
    specs = {s["function"]["name"]: s["function"]["parameters"] for s in R2_SPECS}
    for tool in _list_tools():
        spec = specs[tool.name]
        assert set(tool.input_schema["properties"]) == set(spec["properties"]), tool.name
        assert set(tool.input_schema.get("required", [])) == set(spec["required"]), tool.name


def test_schemas_are_valid_json_schema_and_chinese() -> None:
    for tool in _list_tools():
        Draft202012Validator.check_schema(tool.input_schema)
        Draft202012Validator.check_schema(tool.output_schema)
        assert tool.input_schema["type"] == "object"
        assert set(tool.output_schema["required"]) >= {"columns", "rows", "truncated"}
        assert any("一" <= ch <= "鿿" for ch in tool.description), tool.name
        assert tool.annotations.read_only_hint and not tool.annotations.destructive_hint


def test_find_entity_layers_match_r2_tools() -> None:
    schema = next(t for t in _list_tools() if t.name == "find_entity").input_schema
    layer = schema["properties"]["layer"]
    enum = next(x["enum"] for x in layer["anyOf"] if "enum" in x)
    assert set(enum) == set(X._FIND_LAYERS)


def test_reused_pydantic_models_validate_arguments() -> None:
    # 参数模型来自 routing.Place / tools.R2TargetFilter：非法图层在进入工具前被拒绝
    res = _call("isochrone", {"origin": {"layer": "roads", "name": "x"}, "minutes": 10})
    assert res.is_error
    res = _call("isochrone", {"origin": {"layer": "residential_areas", "name": "x"}, "minutes": 45})
    assert res.is_error


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM pois",
        "UPDATE pois SET name = 'x'",
        "INSERT INTO pois (name) VALUES ('x')",
        "DROP TABLE pois",
        "CREATE TABLE t AS SELECT 1",
        "SELECT 1; DELETE FROM pois",
        "SELECT * INTO t FROM pois",
        "SELECT * FROM pg_shadow",
    ],
)
def test_run_sql_rejects_writes_through_guard(sql: str) -> None:
    res = _call("run_sql", {"sql": sql})
    assert res.is_error
    text = res.content[0].text
    assert any(
        k in text
        for k in ("only SELECT", "forbidden construct", "exactly one statement", "table not allowed")
    ), text


def test_run_sql_rejects_degree_distance_before_execution() -> None:
    res = _call("run_sql", {"sql": "SELECT ST_Distance(a.geom, b.geom) FROM pois a, pois b LIMIT 1"})
    assert res.is_error and "unit/CRS" in res.content[0].text


def test_to_tool_result_splits_geojson() -> None:
    pt = json.dumps({"type": "Point", "coordinates": [118.78, 32.04]})
    r = M.to_tool_result(QueryResult(["name", "dist_m", "geojson"], [["大行宫", 104.7, pt]], False, 1.0, "S"))
    assert r.columns == ["name", "dist_m"] and r.rows == [["大行宫", 104.7]]
    assert r.geojson["features"][0]["properties"] == {"name": "大行宫", "dist_m": 104.7}
    assert M.to_tool_result(QueryResult(["n"], [[3]], True, 1.0, "S")).geojson is None


def test_http_transport_is_loopback_only(monkeypatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(M.mcp, "run", lambda transport, **kw: seen.update(transport=transport, **kw))
    M.main(["--transport", "streamable-http", "--port", "9999"])
    assert seen == {"transport": "streamable-http", "host": "127.0.0.1", "port": 9999}
    with pytest.raises(SystemExit):
        M.main(["--host", "0.0.0.0"])


def test_resources_and_prompt_registered() -> None:
    uris = {str(r.uri) for r in anyio.run(M.mcp.list_resources)}
    assert uris == {M.SCHEMA_URI, M.CLAUSES_URI}
    prompts = anyio.run(M.mcp.list_prompts)
    assert [p.name for p in prompts] == ["compliance_check"]


# ---------------------------------------------------------------- 集成（需要第二轮数据库）


@pytest.mark.db
@pytest.mark.skipif(not HAVE_DB, reason="round-2 database not available")
def test_client_session_lists_tools_and_calls_db_tools() -> None:
    async def go():
        async with Client(M.mcp) as client:
            tools = await client.list_tools()
            found = await client.call_tool("find_entity", {"name": "融康苑", "layer": "residential_areas"})
            iso = await client.call_tool(
                "isochrone",
                {"origin": {"layer": "residential_areas", "name": "南湾营融康苑"}, "minutes": 12.5},
            )
            missing = await client.call_tool(
                "isochrone", {"origin": {"layer": "residential_areas", "name": "不存在的小区"}, "minutes": 5}
            )
            schema = await client.read_resource(M.SCHEMA_URI)
            clauses = await client.read_resource(M.CLAUSES_URI)
            return tools, found, iso, missing, schema, clauses

    tools, found, iso, missing, schema, clauses = anyio.run(go)
    assert {t.name for t in tools.tools} == R2_TOOL_NAMES

    assert not found.is_error
    f = found.structured_content
    assert f["columns"][:2] == ["layer", "name"]
    assert ["residential_areas", "南湾营融康苑"] in [r[:2] for r in f["rows"]]

    assert not iso.is_error
    s = iso.structured_content
    assert s["row_count"] == 1 and not s["truncated"]
    row = dict(zip(s["columns"], s["rows"][0], strict=True))
    assert row["network_m"] == pytest.approx(1000.0)
    assert 0.1 < row["area_km2"] < 5
    geom = s["geojson"]["features"][0]["geometry"]
    assert geom["type"] in ("Polygon", "MultiPolygon")

    assert missing.is_error and "find_entity" in missing.content[0].text
    assert "TABLE walk_edges" in schema.contents[0].text
    assert json.loads(clauses.contents[0].text)["count"] >= 30
