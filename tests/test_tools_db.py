"""类型化 GIS 工具与只读执行层的集成测试；需要本地 PostGIS（docker compose up），否则跳过。"""

from __future__ import annotations

import pytest

from geo_agent import tools as T

try:
    from geo_agent.db import QueryError, connect, run_select

    _conn = connect()
    _conn.close()
    HAVE_DB = True
except Exception:
    HAVE_DB = False

pytestmark = [pytest.mark.db, pytest.mark.skipif(not HAVE_DB, reason="PostGIS container not running")]


@pytest.fixture()
def conn():
    with connect() as c:
        yield c


def test_within_distance_count(conn) -> None:
    r = T.within_distance(
        T.Anchor(layer="metro_stations", name="新街口"),
        T.TargetFilter(layer="pois", subtypes=["restaurant"]),
        500,
        conn,
        count_only=True,
    )
    assert r.rows == [[32]]


def test_within_distance_excludes_anchor_itself(conn) -> None:
    r = T.within_distance(
        T.Anchor(layer="pois", name="江苏省人民医院"),
        T.TargetFilter(layer="pois", subtypes=["hospital"]),
        2000,
        conn,
        count_only=True,
    )
    assert r.rows == [[10]]


def test_nearest_k_uses_geography_order(conn) -> None:
    r = T.nearest_k(
        T.Anchor(layer="water_areas", name="玄武湖"),
        T.TargetFilter(layer="pois", subtypes=["university"]),
        3,
        conn,
    )
    dists = [row[1] for row in r.rows]
    assert dists == sorted(dists)
    assert r.rows[0][0] == "中国药科大学(玄武门校区)"
    assert "geojson" in r.columns


def test_count_by_district_single(conn) -> None:
    r = T.count_by_district(T.TargetFilter(layer="pois", subtypes=["bank"], district="栖霞区"), conn)
    assert len(r.rows) == 1 and r.rows[0][:2] == ["栖霞区", 37]


def test_measure_and_relation(conn) -> None:
    area = T.measure("districts", "建邺区", "area_km2", conn).rows[0][1]
    assert 81.0 < area < 81.6
    names = [
        r[0]
        for r in T.spatial_relation(
            "districts", "touches", T.Anchor(layer="districts", name="玄武区"), conn
        ).rows
    ]
    assert names == sorted(["栖霞区", "江宁区", "秦淮区", "鼓楼区"])


def test_bad_anchor_message(conn) -> None:
    with pytest.raises(T.ToolError, match="find_entity"):
        T.nearest_k(T.Anchor(layer="pois", name="不存在的地方"), T.TargetFilter(layer="pois"), 1, conn)


def test_injection_in_filter_is_bound_as_value(conn) -> None:
    r = T.within_distance(
        T.Anchor(layer="metro_stations", name="新街口"),
        T.TargetFilter(layer="pois", name_like="'; DROP TABLE pois; --"),
        500,
        conn,
        count_only=True,
    )
    assert r.rows == [[0]]


def test_read_only_role_blocks_writes_even_without_guard(conn) -> None:
    with pytest.raises(QueryError):
        run_select("CREATE TABLE should_fail (a int)", conn=conn, guard=False)


def test_statement_timeout(conn) -> None:
    with pytest.raises(QueryError, match="(?i)cancel|timeout"):
        run_select("SELECT pg_sleep(7)", conn=conn, guard=False)


def test_row_limit_truncates(conn) -> None:
    r = run_select("SELECT id FROM pois", conn=conn)
    assert len(r.rows) == 200 and r.truncated
