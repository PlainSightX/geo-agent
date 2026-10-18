"""单位/CRS 静态检查与答案绑定（数字只能来自结果）的单元测试。"""

import json

import pytest

from geo_agent.answer import bind, to_feature_collection
from geo_agent.db import QueryResult
from geo_agent.units import lint_units, uses_measure

UNIT_BAD = [
    "SELECT ST_Distance(a.geom, b.geom) * 111000 FROM pois a, pois b",
    "SELECT count(*) FROM pois a, pois b WHERE ST_DWithin(a.geom, b.geom, 0.005)",
    "SELECT ST_Area(geom) FROM districts",
    "SELECT ST_Length(geom) FROM waterways",
    "SELECT ST_Area(ST_Transform(geom, 3857)) FROM districts",
    "SELECT ST_Buffer(geom, 500) FROM pois",
    "SELECT ST_Length(ST_Intersection(r.geom, d.geom)) FROM roads r, districts d",
]
UNIT_OK = [
    "SELECT ST_Distance(a.geom::geography, b.geom::geography) FROM pois a, pois b",
    "SELECT ST_Area(CAST(geom AS geography)) FROM districts",
    "SELECT ST_Area(geography(geom)) FROM districts",
    "SELECT ST_Area(ST_Transform(geom, 32650)) FROM districts",
    "SELECT ST_Length(ST_Transform(geom, 4549)) FROM waterways",
    "SELECT ST_DistanceSphere(a.geom, b.geom) FROM pois a, pois b",
    "SELECT ST_Length(ST_Intersection(r.geom, d.geom)::geography) FROM roads r, districts d",
    "SELECT ST_Buffer(geom::geography, 500) FROM pois",
    "SELECT name FROM pois WHERE ST_Within(geom, (SELECT geom FROM districts WHERE name = '鼓楼区'))",
]


@pytest.mark.parametrize("sql", UNIT_BAD)
def test_unit_errors_flagged(sql: str) -> None:
    assert lint_units(sql), sql


@pytest.mark.parametrize("sql", UNIT_OK)
def test_unit_ok(sql: str) -> None:
    assert lint_units(sql) == [], [str(i) for i in lint_units(sql)]


def test_uses_measure() -> None:
    assert uses_measure("SELECT ST_Area(geom::geography) FROM districts")
    assert not uses_measure("SELECT count(*) FROM pois")


def _res(cols, rows) -> QueryResult:
    return QueryResult(columns=cols, rows=rows, truncated=False, elapsed_ms=1.0)


def test_bind_numbers_come_from_results() -> None:
    results = {"R1": _res(["n"], [[32]]), "R2": _res(["name", "dist_m"], [["大行宫", 104.70092964]])}
    out = bind("共有 {R1.n} 家；最近的是 {R2.rows[0].name}（{R2.rows[0].dist_m} 米）", results)
    assert out.text == "共有 32 家；最近的是 大行宫（105 米）"
    assert out.used_refs == ["R1", "R2"]
    assert out.unbound_numbers == [] and out.errors == []


def test_bind_strips_invented_numbers_but_keeps_question_numbers() -> None:
    results = {"R1": _res(["n"], [[3]])}
    out = bind("500 米内有 {R1.n} 家，另外大约 12 家在建", results, question="500 米内有多少家？")
    assert "500" in out.text
    assert "12" not in out.text
    assert out.unbound_numbers == ["12"]


def test_bind_count_and_n_semantics() -> None:
    detail = _res(["name", "dist_m"], [["A", 1.0], ["B", 2.0], ["C", 3.0]])
    out = bind("{R1.count} 个；{R1.n} 个", {"R1": detail})
    assert out.text == "3 个；3 个"


def test_bind_invalid_placeholder_reported() -> None:
    out = bind("共 {R1.row_count} 条，{R9.n}", {"R1": _res(["n"], [[1]])})
    assert "?" in out.text
    assert any("invalid placeholder" in e for e in out.errors)
    assert any("unknown result R9" in e for e in out.errors)


def test_feature_collection() -> None:
    res = _res(
        ["name", "dist_m", "geojson"],
        [["A", 1.5, json.dumps({"type": "Point", "coordinates": [118.78, 32.04]})]],
    )
    fc = to_feature_collection(res)
    assert fc["features"][0]["geometry"]["type"] == "Point"
    assert fc["features"][0]["properties"] == {"name": "A", "dist_m": 1.5}
    assert to_feature_collection(_res(["n"], [[1]]))["features"] == []
