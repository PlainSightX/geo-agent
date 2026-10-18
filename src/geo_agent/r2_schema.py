"""第二轮“精简 schema”（lean schema）：两个评测臂看到的是同一份文本。

为什么精简：第一轮 schema 注释里直接写了“医院=hospital、银行=bank …”这样的中文→OSM 标签映射，
等于把题目最难的一步提前告诉了单次 text-to-SQL；真实项目的库注释通常只有字段含义、单位与坐标系。
第二轮的规则（预注册，见 eval/r2/PREREGISTRATION.md）：
- 保留：表名、列名、类型、行数、几何类型与 SRID、单位、主外键关系、简短的字段含义（与真实库中常见的注释同级）；
- 去掉：中文概念→OSM 标签值的映射表、具体取值示例、“如何写 SQL”的提示（如“须用 ::geography”）；
- 函数说明（pgRouting、pgvector、PostGIS 度量）单独作为 FUNCTION_NOTES 给两臂，等价于一个称职的提示工程师
  会写给模型的“可用函数清单”；它说明函数签名与返回单位，但不说明某道题该怎么解。
第一轮代码（describe_schema / BUSINESS_TABLES / build_tables.sql 中的注释）一律不改。
"""

from __future__ import annotations

import psycopg
from psycopg import sql as pgsql

from .config import R2_TABLES
from .db import connect

# 表/列的精简注释：只描述含义与单位，不给取值映射
LEAN_TABLE_NOTES: dict[str, str] = {
    "districts": "Nanjing municipal districts (OSM admin_level=6)",
    "pois": "points of interest from OSM amenity/shop/tourism features; polygons reduced to an interior point",
    "roads": "OSM roads for motor vehicles, clipped to the city boundary",
    "waterways": "OSM linear waterways",
    "water_areas": "OSM natural=water polygons",
    "parks": "OSM leisure=park polygons",
    "metro_stations": "metro/light-rail stations, one row per station name",
    "poi_tags": "all remaining raw OSM tags of pois, one row per (poi, key)",
    "residential_areas": "named OSM landuse=residential polygons (housing estates / 小区)",
    "bus_stops": "OSM highway=bus_stop points (not merged; one stop often has two rows)",
    "walk_edges": "pedestrian network edges for pgRouting (undirected)",
    "walk_vertices": "pedestrian network vertices",
    "standard_clauses": "excerpts of the planning standard GB 50180-2018 with embeddings",
}
LEAN_COLUMN_NOTES: dict[tuple[str, str], str] = {
    ("pois", "category"): "OSM key the feature came from: amenity | shop | tourism",
    ("pois", "subtype"): "raw OSM value of that key (English, lower case)",
    ("pois", "cuisine"): "raw OSM cuisine tag",
    ("pois", "osm_type"): "node | way | relation",
    ("poi_tags", "poi_id"): "references pois.id",
    ("poi_tags", "key"): "raw OSM tag key, e.g. brand, operator, opening_hours, name:en",
    ("poi_tags", "value"): "raw OSM tag value",
    ("roads", "highway"): "raw OSM highway value",
    ("waterways", "waterway"): "raw OSM waterway value",
    ("water_areas", "water_type"): "raw OSM water value",
    ("metro_stations", "station_type"): "subway | light_rail",
    ("metro_stations", "osm_node_count"): "number of OSM station nodes merged into this row",
    ("walk_edges", "source"): "walk_vertices.id",
    ("walk_edges", "target"): "walk_vertices.id",
    ("walk_edges", "cost_m"): "edge length in meters",
    ("walk_edges", "highway"): "raw OSM highway value of the way",
    ("walk_vertices", "component"): "connected component id",
    ("walk_vertices", "in_main"): "true if the vertex is in the largest connected component",
    ("standard_clauses", "clause_id"): "citable clause id",
    ("standard_clauses", "facility"): "facility the clause is about (Chinese), null for general clauses",
    ("standard_clauses", "circle_min"): "life-circle level in minutes (5/10/15) the clause belongs to",
    ("standard_clauses", "max_radius_m"): "service radius upper bound in meters stated by the clause",
    ("standard_clauses", "walk_m_min"): "walking distance lower bound in meters (table 3.0.4)",
    ("standard_clauses", "walk_m_max"): "walking distance upper bound in meters (table 3.0.4)",
    ("standard_clauses", "strength"): "definition | should | should_not_exceed",
    ("standard_clauses", "embedding"): "vector(512), BAAI/bge-small-zh-v1.5, L2-normalised",
}

FUNCTION_NOTES = """\
Spatial facts and available functions:
- Every geom column is EPSG:4326 (lon/lat degrees). PostGIS 3.5 is available: ST_Distance/ST_DWithin/ST_Area/
  ST_Length return degrees on geometry and meters on geography (geom::geography); ST_Transform(geom, 32650)
  is UTM zone 50N in meters.
- pgRouting 3.8 is available. Edge query for walking:
    'SELECT id, source, target, cost_m AS cost, cost_m AS reverse_cost FROM walk_edges'
  pgr_dijkstra(edges_sql, start_vid, end_vid, directed => false) -> (seq, path_seq, node, edge, cost, agg_cost)
  pgr_drivingDistance(edges_sql, start_vid, distance, directed => false) -> (seq, depth, start_vid, pred, node, edge,
    cost, agg_cost) — all vertices within `distance` network meters.
  To route from a feature, snap it to the nearest walk_vertices row with in_main = true
  (ORDER BY v.geom <-> point LIMIT 1). Polygon features use ST_PointOnSurface(geom) as their point.
- Walking speed for this project: 4.8 km/h = 80 m per minute.
- pgvector 0.8 is available. standard_clauses.embedding <=> query_vector is cosine distance. You cannot compute
  embeddings in SQL; filter standard_clauses with ordinary predicates (facility, circle_min, title LIKE ...) instead.
"""


def describe_lean_schema(conn: psycopg.Connection | None = None) -> str:
    """生成第二轮精简 schema 文本（两臂共用）。不读取数据库中的列注释，只用上面的精简注释。"""
    own = conn is None
    c = conn or connect()
    lines: list[str] = []
    try:
        with c.cursor() as cur:
            for table in R2_TABLES:
                cur.execute(pgsql.SQL("SELECT count(*) FROM {}").format(pgsql.Identifier(table)))
                n = cur.fetchone()[0]
                lines.append(f"TABLE {table} ({n} rows) -- {LEAN_TABLE_NOTES[table]}")
                cur.execute(
                    """
                    SELECT a.attname, format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    WHERE a.attrelid = %s::regclass AND a.attnum > 0 AND NOT a.attisdropped
                    ORDER BY a.attnum
                    """,
                    (table,),
                )
                for name, typ in cur.fetchall():
                    note = LEAN_COLUMN_NOTES.get((table, name))
                    lines.append(f"  {name} {typ}" + (f" -- {note}" if note else ""))
        c.rollback()
    finally:
        if own:
            c.close()
    return "\n".join(lines)
