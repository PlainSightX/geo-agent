"""第二轮集成测试：路网、条文检索、合规工具与金标口径一致性。需要本地数据库（含第二轮表），否则跳过；
条文检索还需要本地向量模型文件（scripts/fetch_model.sh），否则跳过。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geo_agent import routing as RT
from geo_agent.tools import R2TargetFilter, ToolError

ROOT = Path(__file__).resolve().parents[1]

try:
    from geo_agent.db import connect, run_select

    with connect() as _c:
        _c.execute("SELECT 1 FROM walk_edges LIMIT 1")
    HAVE_DB = True
except Exception:
    HAVE_DB = False

try:
    from geo_agent.embed import get_embedder

    get_embedder()
    HAVE_MODEL = True
except Exception:
    HAVE_MODEL = False

pytestmark = [pytest.mark.db, pytest.mark.skipif(not HAVE_DB, reason="round-2 database not available")]
SCHOOL = R2TargetFilter(layer="pois", subtypes=["school"], name_like="中学")


@pytest.fixture()
def conn():
    with connect() as c:
        yield c


def _gold(qid: str) -> dict:
    return next(
        x for x in json.loads((ROOT / "eval/r2/gold.json").read_text("utf-8"))["items"] if x["id"] == qid
    )


def test_network_stats_match_frozen(conn) -> None:
    stats = json.loads((ROOT / "eval/r2/gold.json").read_text("utf-8"))["data"]["walk_network"]
    row = run_select(
        "SELECT edges, vertices, components FROM walk_network_stats", conn=conn, guard=False
    ).rows[0]
    assert row == [stats["edges"], stats["vertices"], stats["components"]]
    assert stats["main_length_share"] > 0.95


def test_walk_route_matches_gold_and_returns_line(conn) -> None:
    r = RT.walk_route(
        RT.Place(layer="metro_stations", name="珠江路"), RT.Place(layer="metro_stations", name="鼓楼"), conn
    )
    walk_m, walk_min, straight_m = r.rows[0][:3]
    assert walk_m == pytest.approx(_gold("D01")["rows"][0][0], rel=1e-9)
    assert walk_min == pytest.approx(walk_m / 80)
    assert walk_m >= straight_m  # 网络距离不小于直线距离
    assert json.loads(r.rows[0][-1])["type"] in ("LineString", "MultiLineString")


def test_isochrone_matches_gold_polygon(conn) -> None:
    r = RT.isochrone(RT.Place(layer="metro_stations", name="新街口"), 15, conn)
    assert r.rows[0][3] == pytest.approx(_gold("D09")["rows"][0][0], rel=1e-9)
    assert json.loads(r.rows[0][-1])["type"] in ("Polygon", "MultiPolygon")
    assert r.elapsed_ms < 5000


def test_reachable_matches_gold(conn) -> None:
    r = RT.reachable_facilities(
        RT.Place(layer="residential_areas", name="龙岸花园幸福苑"),
        R2TargetFilter(layer="pois", subtypes=["kindergarten"]),
        5,
        conn,
    )
    assert sorted(x[0] for x in r.rows) == sorted(x[0] for x in _gold("D11")["rows"])
    assert all(x[1] <= 400 for x in r.rows)


def test_unknown_place_is_tool_error(conn) -> None:
    with pytest.raises(ToolError):
        RT.walk_route(
            RT.Place(layer="pois", name="不存在的地方"), RT.Place(layer="metro_stations", name="鼓楼"), conn
        )


@pytest.mark.parametrize(("qid", "resid"), [("E01", "南湾营融康苑"), ("E02", "百水芊城")])
def test_check_compliance_matches_gold(conn, qid: str, resid: str) -> None:
    from geo_agent.r2_tools import check_compliance

    r = check_compliance(resid, "GB50180-2018:C.0.1:初中", SCHOOL, conn)
    assert r.rows[0][:3] == _gold(qid)["rows"][0]


def test_check_compliance_rejects_clause_without_radius(conn) -> None:
    from geo_agent.r2_tools import check_compliance

    with pytest.raises(ToolError):
        check_compliance("百水芊城", "GB50180-2018:2.0.2", SCHOOL, conn)


def test_alias_lookup(conn) -> None:
    from geo_agent.r2_tools import find_entity_r2

    rows = find_entity_r2("瞻园", conn).rows
    assert any(r[1] == "太平天国历史博物馆" and r[4].startswith("tag alt_name") for r in rows)


@pytest.mark.skipif(not HAVE_MODEL, reason="embedding model files not downloaded")
@pytest.mark.parametrize(
    ("query", "clause"),
    [
        ("初中服务半径", "GB50180-2018:C.0.1:初中"),
        ("幼儿园离家多远", "GB50180-2018:C.0.2:幼儿园"),
        ("派出所", "GB50180-2018:C.0.1:派出所"),
    ],
)
def test_retrieve_standard_top1(conn, query: str, clause: str) -> None:
    from geo_agent.r2_tools import retrieve_standard

    r = retrieve_standard(query, conn, 3)
    assert r.rows[0][0] == clause
