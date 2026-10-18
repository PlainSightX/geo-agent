"""步行路网工具（pgRouting）：最短路径、等时圈、步行可达设施。

统一口径（预注册，金标与工具共用本模块的 SQL 片段，见 eval/r2/PREREGISTRATION.md）：
- 步行速度 WALK_SPEED_KMH = 4.8 km/h = 80 m/min；N 分钟步行 = 80·N 米网络距离。
- 吸附：任一要素先取代表点（点要素即自身；面要素 ST_PointOnSurface；同名多要素取 id 最小的一个），
  再吸附到最大连通分量（walk_vertices.in_main）中 KNN 最近的顶点。吸附段的直线距离不计入路网距离，
  但作为 snap_m 返回，便于判断“起点离路网很远”的情况。
- 最短路径：pgr_dijkstra 无向图，cost = reverse_cost = 边长（米）。
- 可达设施：设施吸附顶点的网络距离 ≤ 阈值（pgr_drivingDistance 求出起点阈值内的顶点集合）。
- 等时圈：两端点都在阈值内的边，在 UTM 50N 上各自简化（2 m）并缓冲 BUFFER_M=50 m（quad_segs=2）后求并；
  面积用 geography 计算（平方公里）。这是“边缓冲”等时圈，不是凹包；先缓冲后求并比先求并后缓冲快约 16 倍，
  面积差 < 1%（新街口 15 分钟：1.466 vs 1.474 km²）。
所有几何以 GeoJSON 输出（geojson 列），前端直接渲染。
"""

from __future__ import annotations

from typing import Literal

import psycopg
from psycopg import sql as pgsql
from pydantic import BaseModel, Field

from .db import QueryResult
from .tools import R2TargetFilter, ToolError, _execute, _target_where, _where_sql

WALK_SPEED_KMH = 4.8
WALK_M_PER_MIN = WALK_SPEED_KMH * 1000 / 60  # 80 m/min
BUFFER_M = 50
MAX_MINUTES = 30
EDGES_SQL = "SELECT id, source, target, cost_m AS cost, cost_m AS reverse_cost FROM walk_edges"

# 可作为起终点的图层：点图层取自身，面图层取面内点
PlaceLayer = Literal["pois", "metro_stations", "residential_areas", "parks", "bus_stops", "districts"]
_POLY_LAYERS = {"residential_areas", "parks", "districts"}
# 可达性查询的目标图层（点图层或可取面内点的面图层）
_TARGET_LAYERS = {"pois", "metro_stations", "bus_stops", "parks", "residential_areas"}


class Place(BaseModel):
    layer: PlaceLayer = Field(description="layer of the start/end feature")
    name: str = Field(description="exact feature name")


def minutes_to_m(minutes: float) -> float:
    if not 0 < minutes <= MAX_MINUTES:
        raise ToolError(f"minutes must be in (0, {MAX_MINUTES}]")
    return minutes * WALK_M_PER_MIN


def rep_point_sql(layer: str) -> pgsql.Composable:
    """要素代表点表达式（作用于别名 t）。"""
    return pgsql.SQL("ST_PointOnSurface(t.geom)" if layer in _POLY_LAYERS else "t.geom")


def _place_cte(place: Place, alias: str, params: dict) -> pgsql.Composable:
    """CTE：<alias>(pt, vid, snap_m)——代表点、吸附顶点、吸附距离。同名多要素取 id 最小者。"""
    if alias not in ("a", "b", "o"):
        raise ValueError("internal alias must be a, b or o")
    params[f"{alias}_name"] = place.name
    return pgsql.SQL(
        "{ap} AS (SELECT {rp} AS pt FROM {tbl} t WHERE t.name = {ph} ORDER BY t.id LIMIT 1), "
        "{a} AS (SELECT p.pt, v.id AS vid, ST_Distance(p.pt::geography, v.geom::geography) AS snap_m "
        "FROM {ap} p CROSS JOIN LATERAL (SELECT id, geom FROM walk_vertices WHERE in_main "
        "ORDER BY geom <-> p.pt LIMIT 1) v)"
    ).format(
        a=pgsql.SQL(alias),
        ap=pgsql.SQL(alias + "_p"),
        rp=rep_point_sql(place.layer),
        tbl=pgsql.Identifier(place.layer),
        ph=pgsql.Placeholder(f"{alias}_name"),
    )


def _check_place(place: Place, conn: psycopg.Connection) -> int:
    q = pgsql.SQL("SELECT count(*) FROM {} WHERE name = %(n)s").format(pgsql.Identifier(place.layer))
    n = int(_execute(q, {"n": place.name}, conn, r2=True).rows[0][0])
    if n == 0:
        raise ToolError(f"'{place.name}' not found in {place.layer}; call find_entity for the exact name")
    return n


def walk_route(a: Place, b: Place, conn: psycopg.Connection) -> QueryResult:
    """两地之间的最短步行路径：网络距离（米）、按 4.8 km/h 的步行分钟数、路径 GeoJSON 线。"""
    _check_place(a, conn)
    _check_place(b, conn)
    params: dict = {"edges": EDGES_SQL, "speed": WALK_M_PER_MIN}
    q = pgsql.SQL(
        "WITH {ca}, {cb}, r AS (SELECT * FROM pgr_dijkstra(%(edges)s, (SELECT vid FROM a), (SELECT vid FROM b), false)) "
        "SELECT (SELECT coalesce(max(agg_cost), 0) FROM r) AS walk_m, "
        "(SELECT coalesce(max(agg_cost), 0) FROM r) / %(speed)s AS walk_min, "
        "ST_Distance(a.pt::geography, b.pt::geography) AS straight_m, a.snap_m AS snap_a_m, b.snap_m AS snap_b_m, "
        "ST_AsGeoJSON(ST_LineMerge((SELECT ST_Collect(e.geom ORDER BY r.seq) FROM r JOIN walk_edges e ON e.id = r.edge)), 6) AS geojson "
        "FROM a, b"
    ).format(ca=_place_cte(a, "a", params), cb=_place_cte(b, "b", params))
    res = _execute(q, params, conn, r2=True)
    if res.rows and res.rows[0][0] == 0 and res.rows[0][2] > 1:
        raise ToolError("no walking path found between the two places")
    return res


def _reach_cte(origin: Place, meters: float, params: dict) -> pgsql.Composable:
    """CTE：o（起点）与 reach(node, agg_cost)——阈值内可达顶点。"""
    params["edges"] = EDGES_SQL
    params["dist"] = meters
    return pgsql.SQL(
        "{co}, reach AS (SELECT node, agg_cost FROM pgr_drivingDistance(%(edges)s, (SELECT vid FROM o), %(dist)s, false))"
    ).format(co=_place_cte(origin, "o", params))


def isochrone(origin: Place, minutes: float, conn: psycopg.Connection) -> QueryResult:
    """步行等时圈：N 分钟（80 m/min）内可达边的 50 m 缓冲面，返回面积（km²）与 GeoJSON 面。"""
    _check_place(origin, conn)
    meters = minutes_to_m(minutes)
    params: dict = {"buf": BUFFER_M}
    q = pgsql.SQL(
        "WITH {rc}, edges_in AS (SELECT e.geom FROM walk_edges e JOIN reach r1 ON r1.node = e.source "
        "JOIN reach r2 ON r2.node = e.target), "
        "poly AS (SELECT ST_Transform(ST_Union(ST_Buffer(ST_Simplify(ST_Transform(geom, 32650), 2), %(buf)s, 'quad_segs=2')), 4326) AS g "
        "FROM edges_in) "
        "SELECT %(mins)s::float AS minutes, %(dist)s::float AS network_m, (SELECT count(*) FROM reach) AS reachable_nodes, "
        "ST_Area(poly.g::geography) / 1e6 AS area_km2, o.snap_m AS snap_m, ST_AsGeoJSON(poly.g, 6) AS geojson FROM poly, o"
    ).format(rc=_reach_cte(origin, meters, params))
    params["mins"] = minutes
    return _execute(q, params, conn, r2=True)


def reachable_facilities(
    origin: Place, target: R2TargetFilter, minutes: float, conn: psycopg.Connection, count_only: bool = False
) -> QueryResult:
    """N 分钟步行可达的目标要素：目标吸附顶点的网络距离 ≤ 80·N 米；按网络距离排序。

    target 与第一轮工具同构（layer + subtypes + name_like + named_only），不预设“中学=某标签”之类映射：
    例如中学需要模型自己表达为 subtypes=['school'], name_like='中学'。
    """
    if target.layer not in _TARGET_LAYERS:
        raise ToolError(f"target layer must be one of {sorted(_TARGET_LAYERS)}")
    _check_place(origin, conn)
    meters = minutes_to_m(minutes)
    params: dict = {"speed": WALK_M_PER_MIN}
    conds = _target_where(target.model_copy(update={"district": None}), "t", params)
    rp = rep_point_sql(target.layer)
    conds.append(pgsql.SQL("ST_DWithin({}::geography, (SELECT pt FROM o)::geography, %(dist)s)").format(rp))
    body = pgsql.SQL(
        "WITH {rc}, fac AS (SELECT t.id, t.name, {rp} AS pt FROM {tbl} t WHERE {cond}), "
        "snapped AS (SELECT f.id, f.name, f.pt, v.id AS vid FROM fac f CROSS JOIN LATERAL "
        "(SELECT id FROM walk_vertices WHERE in_main ORDER BY geom <-> f.pt LIMIT 1) v), "
        "hit AS (SELECT s.id, s.name, s.pt, r.agg_cost AS walk_m FROM snapped s JOIN reach r ON r.node = s.vid) "
    ).format(
        rc=_reach_cte(origin, meters, params),
        rp=rp,
        tbl=pgsql.Identifier(target.layer),
        cond=_where_sql(conds),
    )
    if count_only:
        tail = pgsql.SQL("SELECT count(*) AS n FROM hit")
    else:
        tail = pgsql.SQL(
            "SELECT name, walk_m, walk_m / %(speed)s AS walk_min, ST_AsGeoJSON(pt, 6) AS geojson "
            "FROM hit ORDER BY walk_m, id"
        )
    return _execute(body + tail, params, conn, r2=True)
