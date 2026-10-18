"""第二轮金标 SQL 生成器：把预注册的路网口径（routing.py 文档字符串）写成纯 SQL 文本。

金标必须是“作者自己写的、确定性的 SQL”，冻结时执行并存下结果；
这里用 Python 字符串模板只是为了避免 30 道路网题手写重复的吸附/可达 CTE。
口径与 src/geo_agent/routing.py 完全一致（同一吸附规则、80 m/min、无向图、同名取 id 最小），
tests/test_r2_gold.py 在有数据库时检查两者对同一输入给出相同数值。
"""

from __future__ import annotations

EDGES = "'SELECT id, source, target, cost_m AS cost, cost_m AS reverse_cost FROM walk_edges'"
POLY = {"residential_areas", "parks", "districts"}


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def place(layer: str, name: str) -> tuple[str, str]:
    return layer, name


def _pt(layer: str, alias: str = "t") -> str:
    return f"ST_PointOnSurface({alias}.geom)" if layer in POLY else f"{alias}.geom"


def snap_cte(alias: str, p: tuple[str, str]) -> str:
    layer, name = p
    return (
        f"{alias}_p AS (SELECT {_pt(layer)} AS pt FROM {layer} t WHERE t.name = {_q(name)} ORDER BY t.id LIMIT 1), "
        f"{alias} AS (SELECT p.pt, v.id AS vid FROM {alias}_p p CROSS JOIN LATERAL "
        f"(SELECT id FROM walk_vertices WHERE in_main ORDER BY geom <-> p.pt LIMIT 1) v)"
    )


def route_sql(a: tuple[str, str], b: tuple[str, str]) -> str:
    """最短步行距离（米）与分钟数。"""
    return (
        f"WITH {snap_cte('a', a)}, {snap_cte('b', b)} "
        f"SELECT max(agg_cost) AS walk_m, max(agg_cost) / 80.0 AS walk_min "
        f"FROM pgr_dijkstra({EDGES}, (SELECT vid FROM a), (SELECT vid FROM b), false)"
    )


# 设施过滤：(表, 条件)；条件只引用别名 t
FAC = {
    "middle_school": ("pois", "t.subtype = 'school' AND t.name LIKE '%中学%'"),
    "primary_school": ("pois", "t.subtype = 'school' AND t.name LIKE '%小学%'"),
    "kindergarten": ("pois", "t.subtype = 'kindergarten'"),
    "hospital": ("pois", "t.subtype = 'hospital'"),
    "pharmacy": ("pois", "t.subtype = 'pharmacy'"),
    "supermarket": ("pois", "t.subtype = 'supermarket'"),
    "convenience": ("pois", "t.subtype = 'convenience'"),
    "bank": ("pois", "t.subtype = 'bank'"),
    "post_office": ("pois", "t.subtype = 'post_office'"),
    "police": ("pois", "t.subtype = 'police'"),
    "restaurant": ("pois", "t.subtype = 'restaurant'"),
    "cafe": ("pois", "t.subtype = 'cafe'"),
    "toilets": ("pois", "t.subtype = 'toilets'"),
    "mall": ("pois", "t.subtype = 'mall'"),
    "metro_station": ("metro_stations", "TRUE"),
    "bus_stop": ("bus_stops", "TRUE"),
    "park": ("parks", "TRUE"),
}


def fac(kind: str) -> tuple[str, str]:
    return FAC[kind]


def reach_sql(origin: tuple[str, str], facility: tuple[str, str], meters: float, mode: str = "list") -> str:
    """步行 meters 米网络距离内可达的设施。

    mode: list（name, walk_m）| count（n）| nearest（最近 1 个的 name, walk_m）| exists（has: bool）
    """
    layer, cond = facility
    body = (
        f"WITH {snap_cte('o', origin)}, "
        f"reach AS (SELECT node, agg_cost FROM pgr_drivingDistance({EDGES}, (SELECT vid FROM o), {meters}, false)), "
        f"fac AS (SELECT t.id, t.name, {_pt(layer)} AS pt FROM {layer} t WHERE {cond} "
        f"AND ST_DWithin({_pt(layer)}::geography, (SELECT pt FROM o)::geography, {meters})), "
        f"snapped AS (SELECT f.id, f.name, v.id AS vid FROM fac f CROSS JOIN LATERAL "
        f"(SELECT id FROM walk_vertices WHERE in_main ORDER BY geom <-> f.pt LIMIT 1) v), "
        f"hit AS (SELECT s.id, s.name, r.agg_cost AS walk_m FROM snapped s JOIN reach r ON r.node = s.vid) "
    )
    if mode == "count":
        return body + "SELECT count(*) AS n FROM hit"
    if mode == "exists":
        return body + "SELECT count(*) > 0 AS has FROM hit"
    if mode == "nearest":
        return body + "SELECT name, walk_m FROM hit ORDER BY walk_m, id LIMIT 1"
    return body + "SELECT name, walk_m FROM hit ORDER BY walk_m, id"


def reach_margin_sql(origin: tuple[str, str], facility: tuple[str, str], meters: float) -> str:
    """边界裕度检查用：所有同类设施（不限阈值，取 2 倍半径内）的网络距离。"""
    layer, cond = facility
    big = meters * 2 + 500
    return (
        f"WITH {snap_cte('o', origin)}, "
        f"reach AS (SELECT node, agg_cost FROM pgr_drivingDistance({EDGES}, (SELECT vid FROM o), {big}, false)), "
        f"fac AS (SELECT t.id, {_pt(layer)} AS pt FROM {layer} t WHERE {cond} "
        f"AND ST_DWithin({_pt(layer)}::geography, (SELECT pt FROM o)::geography, {big})), "
        f"snapped AS (SELECT f.id, v.id AS vid FROM fac f CROSS JOIN LATERAL "
        f"(SELECT id FROM walk_vertices WHERE in_main ORDER BY geom <-> f.pt LIMIT 1) v) "
        f"SELECT r.agg_cost AS d FROM snapped s JOIN reach r ON r.node = s.vid"
    )


def iso_area_sql(origin: tuple[str, str], meters: float) -> str:
    """等时圈面积（km²）：可达边各自 2 m 简化、50 m 缓冲（quad_segs=2）后求并。"""
    return (
        f"WITH {snap_cte('o', origin)}, "
        f"reach AS (SELECT node FROM pgr_drivingDistance({EDGES}, (SELECT vid FROM o), {meters}, false)), "
        f"e AS (SELECT e.geom FROM walk_edges e JOIN reach r1 ON r1.node = e.source JOIN reach r2 ON r2.node = e.target) "
        f"SELECT ST_Area(ST_Transform(ST_Union(ST_Buffer(ST_Simplify(ST_Transform(geom, 32650), 2), 50, 'quad_segs=2')), 4326)::geography) / 1e6 AS area_km2 FROM e"
    )


def compliance_sql(origin: tuple[str, str], facility: tuple[str, str], clause_id: str) -> str:
    """合规题金标：按条文 max_radius_m 取阈值，返回 (compliant, clause_id, radius_m, nearest_walk_m)。

    nearest_walk_m 为阈值内最近设施的网络距离；无可达设施时为 NULL、compliant=false。
    """
    layer, cond = facility
    return (
        f"WITH cl AS (SELECT clause_id, max_radius_m AS r FROM standard_clauses WHERE clause_id = {_q(clause_id)}), "
        f"{snap_cte('o', origin)}, "
        f"reach AS (SELECT node, agg_cost FROM pgr_drivingDistance({EDGES}, (SELECT vid FROM o), (SELECT r FROM cl), false)), "
        f"fac AS (SELECT t.id, {_pt(layer)} AS pt FROM {layer} t WHERE {cond} "
        f"AND ST_DWithin({_pt(layer)}::geography, (SELECT pt FROM o)::geography, (SELECT r FROM cl))), "
        f"snapped AS (SELECT f.id, v.id AS vid FROM fac f CROSS JOIN LATERAL "
        f"(SELECT id FROM walk_vertices WHERE in_main ORDER BY geom <-> f.pt LIMIT 1) v), "
        f"hit AS (SELECT min(r.agg_cost) AS m FROM snapped s JOIN reach r ON r.node = s.vid) "
        f"SELECT (hit.m IS NOT NULL) AS compliant, cl.clause_id, cl.r AS radius_m FROM hit, cl"
    )
