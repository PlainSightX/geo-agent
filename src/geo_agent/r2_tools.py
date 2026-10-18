"""第二轮新增的 Agent 工具：别名感知的实体查找、原始标签自省、条文检索（RAG）、合规判定。

- find_entity_r2：在第一轮图层之外还查 residential_areas、bus_stops，并按 poi_tags 中的
  alt_name / old_name / short_name / official_name / name:en 匹配别名；
- tag_values：列出 poi_tags 中某个键的取值与计数（或列出常见键），让模型自己发现“品牌=brand”这类映射；
- retrieve_standard：本地 bge-small-zh 计算查询向量，pgvector 余弦距离取 top-k 条文；返回 clause_id 供答案引用；
- check_compliance：给定小区、条文 id 与目标设施过滤条件，**阈值由程序从条文 max_radius_m 读取**，模型不能自填数字，
  再用路网可达性判定；结果列 compliant / clause_id / radius_m / nearest_walk_m / nearest_name。
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg import sql as pgsql

from .db import QueryResult
from .embed import get_embedder, to_pgvector
from .routing import EDGES_SQL, Place, _check_place, _place_cte, rep_point_sql
from .tools import _CATEGORY_COLUMN, R2TargetFilter, ToolError, _execute, _target_where, _where_sql

ALIAS_KEYS = ("alt_name", "old_name", "short_name", "official_name", "name:en", "name:zh")
_FIND_LAYERS = (
    "metro_stations",
    "pois",
    "residential_areas",
    "districts",
    "water_areas",
    "parks",
    "waterways",
    "bus_stops",
)


def find_entity_r2(name: str, conn: psycopg.Connection, layer: str | None = None) -> QueryResult:
    """名称模糊匹配 + POI 别名匹配；返回 layer, 精确 name, category, district, matched_by, same_name_count。"""
    if layer is not None and layer not in _FIND_LAYERS:
        raise ToolError(f"layer must be one of {list(_FIND_LAYERS)}")
    layers = [layer] if layer else list(_FIND_LAYERS)
    params: dict[str, Any] = {"pat": f"%{name}%", "keys": list(ALIAS_KEYS)}
    parts: list[pgsql.Composable] = []
    for lyr in layers:
        col = _CATEGORY_COLUMN.get(lyr)
        cat = pgsql.Identifier(col) if col else pgsql.SQL("NULL")
        pt = rep_point_sql(lyr)
        parts.append(
            pgsql.SQL(
                "SELECT {lit} AS layer, t.name, {cat}::text AS category, "
                "(SELECT d.name FROM districts d WHERE ST_Within({pt}, d.geom) LIMIT 1) AS district, "
                "'name' AS matched_by, count(*) OVER (PARTITION BY t.name) AS same_name_count "
                "FROM {tbl} t WHERE t.name LIKE %(pat)s"
            ).format(lit=pgsql.Literal(lyr), cat=cat, pt=pt, tbl=pgsql.Identifier(lyr))
        )
    if "pois" in layers:
        parts.append(
            pgsql.SQL(
                "SELECT 'pois', t.name, t.subtype::text, "
                "(SELECT d.name FROM districts d WHERE ST_Within(t.geom, d.geom) LIMIT 1), "
                "'tag ' || g.key || '=' || g.value, 1::bigint "
                "FROM pois t JOIN poi_tags g ON g.poi_id = t.id "
                "WHERE g.key = ANY(%(keys)s) AND g.value LIKE %(pat)s"
            )
        )
    q = pgsql.SQL("SELECT * FROM ({}) u ORDER BY length(name), name LIMIT 25").format(
        pgsql.SQL(" UNION ALL ").join(parts)
    )
    return _execute(q, params, conn, r2=True)


def tag_values(conn: psycopg.Connection, key: str | None = None, like: str | None = None) -> QueryResult:
    """key 为空：列出最常见的 60 个标签键；否则列出该键的取值与计数（like 可过滤取值子串）。"""
    params: dict[str, Any] = {}
    if not key:
        q = pgsql.SQL("SELECT key, count(*) AS n FROM poi_tags GROUP BY key ORDER BY n DESC LIMIT 60")
    else:
        params["k"] = key
        params["like"] = f"%{like or ''}%"
        q = pgsql.SQL(
            "SELECT value, count(*) AS n FROM poi_tags WHERE key = %(k)s AND value LIKE %(like)s "
            "GROUP BY value ORDER BY n DESC LIMIT 60"
        )
    return _execute(q, params, conn, r2=True)


def retrieve_standard(query: str, conn: psycopg.Connection, k: int = 5) -> QueryResult:
    """向量检索条文：bge-small-zh 查询向量（带检索指令前缀）与条文向量的余弦距离，取前 k 条。"""
    if not query.strip():
        raise ToolError("query must not be empty")
    k = max(1, min(int(k), 10))
    vec = get_embedder().encode([query], query=True)[0]
    q = pgsql.SQL(
        "SELECT clause_id, title, facility, circle_min, max_radius_m, walk_m_min, walk_m_max, "
        "1 - (embedding <=> %(v)s::vector) AS score, body "
        "FROM standard_clauses ORDER BY embedding <=> %(v)s::vector, clause_id LIMIT %(k)s"
    )
    return _execute(q, {"v": to_pgvector(vec), "k": k}, conn, r2=True)


def check_compliance(
    residential: str, clause_id: str, target: R2TargetFilter, conn: psycopg.Connection
) -> QueryResult:
    """合规判定：阈值 = 条文 max_radius_m（程序读取）；小区沿步行路网在阈值内是否有目标设施。"""
    cl = _execute(
        pgsql.SQL("SELECT max_radius_m FROM standard_clauses WHERE clause_id = %(c)s"),
        {"c": clause_id},
        conn,
        r2=True,
    )
    if not cl.rows:
        raise ToolError(f"clause {clause_id!r} not found; call retrieve_standard first")
    radius = cl.rows[0][0]
    if radius is None:
        raise ToolError(f"clause {clause_id} states no service radius; choose a clause with max_radius_m")
    origin = Place(layer="residential_areas", name=residential)
    _check_place(origin, conn)
    params: dict[str, Any] = {
        "edges": EDGES_SQL,
        "dist": float(radius),
        "clause": clause_id,
        "radius": radius,
    }
    conds = _target_where(target.model_copy(update={"district": None}), "t", params)
    rp = rep_point_sql(target.layer)
    conds.append(pgsql.SQL("ST_DWithin({}::geography, (SELECT pt FROM o)::geography, %(dist)s)").format(rp))
    q = pgsql.SQL(
        "WITH {co}, reach AS (SELECT node, agg_cost FROM pgr_drivingDistance(%(edges)s, (SELECT vid FROM o), %(dist)s, false)), "
        "fac AS (SELECT t.id, t.name, {rp} AS pt FROM {tbl} t WHERE {cond}), "
        "snapped AS (SELECT f.id, f.name, v.id AS vid FROM fac f CROSS JOIN LATERAL "
        "(SELECT id FROM walk_vertices WHERE in_main ORDER BY geom <-> f.pt LIMIT 1) v), "
        "hit AS (SELECT s.name, r.agg_cost AS walk_m FROM snapped s JOIN reach r ON r.node = s.vid ORDER BY walk_m LIMIT 1) "
        "SELECT EXISTS (SELECT 1 FROM hit) AS compliant, %(clause)s::text AS clause_id, %(radius)s::int AS radius_m, "
        "(SELECT walk_m FROM hit) AS nearest_walk_m, (SELECT name FROM hit) AS nearest_name"
    ).format(
        co=_place_cte(origin, "o", params), rp=rp, tbl=pgsql.Identifier(target.layer), cond=_where_sql(conds)
    )
    return _execute(q, params, conn, r2=True)
