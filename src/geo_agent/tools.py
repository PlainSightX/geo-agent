"""类型化 GIS 工具：把常见空间操作封装成参数化查询，模型只填参数，不写 SQL。

设计要点：
- 所有距离/面积/长度一律在 geography（WGS84 椭球，单位米）上计算，单位由工具保证而不是由模型记住；
- 表名、类别等标识符走白名单，值走绑定参数（psycopg 参数化），杜绝注入；
- 每个工具返回 QueryResult，与 run_sql 结果同构，评测与前端都按同一结构处理；
- 返回的几何统一输出为 GeoJSON 列 geojson，前端直接渲染。

“锚点”（anchor）是距离类问题的参照对象，统一描述为 {layer, name}，
layer ∈ {pois, metro_stations, districts, water_areas, parks, waterways}；同名多要素时按并集几何处理。
"""

from __future__ import annotations

from typing import Any, Literal

import psycopg
from psycopg import sql as pgsql
from pydantic import BaseModel, Field

from .db import QueryResult, run_select

LayerName = Literal["pois", "metro_stations", "districts", "water_areas", "parks", "waterways", "roads"]
# 各图层可用于过滤的类别列
_CATEGORY_COLUMN: dict[str, str | None] = {
    "pois": "subtype",
    "metro_stations": "station_type",
    "districts": None,
    "water_areas": "water_type",
    "parks": None,
    "waterways": "waterway",
    "roads": "highway",
}


class ToolError(ValueError):
    """参数不合法或锚点找不到；message 用英文，返回给模型用于修复。"""


class Anchor(BaseModel):
    layer: LayerName = Field(description="layer holding the reference feature")
    name: str = Field(description="exact feature name, e.g. 新街口 or 南京博物院")


class TargetFilter(BaseModel):
    layer: LayerName = Field(description="layer to search")
    subtypes: list[str] | None = Field(
        default=None, description="category values (pois.subtype, waterways.waterway, ...), e.g. ['bank']"
    )
    name_like: str | None = Field(default=None, description="optional substring filter on name")
    named_only: bool = Field(default=False, description="only features with a non-null name")
    district: str | None = Field(
        default=None, description="restrict to features within this district, e.g. 鼓楼区"
    )


def _execute(sql_obj: pgsql.Composable, params: dict[str, Any], conn: psycopg.Connection) -> QueryResult:
    # 工具生成的 SQL 已参数化，但仍在只读事务中执行；先在客户端渲染成字符串以便记录与复现
    text = sql_obj.as_string(conn)
    rendered = psycopg.ClientCursor(conn).mogrify(text, params)
    return run_select(rendered, conn=conn, guard=True)


def _target_where(t: TargetFilter, alias: str, params: dict[str, Any]) -> list[pgsql.Composable]:
    conds: list[pgsql.Composable] = []
    a = pgsql.Identifier(alias)
    col = _CATEGORY_COLUMN[t.layer]
    if t.subtypes:
        if col is None:
            raise ToolError(f"layer {t.layer} has no category column; drop subtypes")
        params[f"{alias}_sub"] = list(t.subtypes)
        conds.append(
            pgsql.SQL("{}.{} = ANY({})").format(a, pgsql.Identifier(col), pgsql.Placeholder(f"{alias}_sub"))
        )
    if t.name_like:
        params[f"{alias}_like"] = f"%{t.name_like}%"
        conds.append(pgsql.SQL("{}.name LIKE {}").format(a, pgsql.Placeholder(f"{alias}_like")))
    if t.named_only:
        conds.append(pgsql.SQL("{}.name IS NOT NULL").format(a))
    if t.district:
        params[f"{alias}_dist"] = t.district
        conds.append(
            pgsql.SQL(
                "EXISTS (SELECT 1 FROM districts dd WHERE dd.name = {} AND ST_Within({}.geom, dd.geom))"
            ).format(pgsql.Placeholder(f"{alias}_dist"), a)
        )
    return conds


def _anchor_cte(anchor: Anchor, params: dict[str, Any]) -> pgsql.Composable:
    params["anchor_name"] = anchor.name
    return pgsql.SQL(
        "anchor AS (SELECT ST_Union(geom) AS geom FROM {} WHERE name = {} HAVING count(*) > 0)"
    ).format(pgsql.Identifier(anchor.layer), pgsql.Placeholder("anchor_name"))


def _check_anchor(anchor: Anchor, conn: psycopg.Connection) -> int:
    q = pgsql.SQL("SELECT count(*) FROM {} WHERE name = %(n)s").format(pgsql.Identifier(anchor.layer))
    res = _execute(q, {"n": anchor.name}, conn)
    n = int(res.rows[0][0])
    if n == 0:
        raise ToolError(
            f"anchor '{anchor.name}' not found in {anchor.layer}; call find_entity to get the exact name"
        )
    return n


def _where_sql(conds: list[pgsql.Composable]) -> pgsql.Composable:
    return pgsql.SQL(" AND ").join(conds) if conds else pgsql.SQL("TRUE")


# ---------------------------------------------------------------- 自省工具


def find_entity(name: str, conn: psycopg.Connection, layer: LayerName | None = None) -> QueryResult:
    """按名称模糊查找要素，返回所在图层、精确名称、类别和所属区——用于确定锚点的准确写法。"""
    layers = (
        [layer] if layer else ["metro_stations", "pois", "districts", "water_areas", "parks", "waterways"]
    )
    parts = []
    for lyr in layers:
        col = _CATEGORY_COLUMN[lyr]
        cat = pgsql.Identifier(col) if col else pgsql.SQL("NULL")
        parts.append(
            pgsql.SQL(
                "SELECT {lit} AS layer, t.name, {cat}::text AS category, "
                "(SELECT d.name FROM districts d WHERE ST_Intersects(d.geom, ST_PointOnSurface(t.geom)) LIMIT 1) AS district, "
                "count(*) OVER (PARTITION BY t.name) AS same_name_count "
                "FROM {tbl} t WHERE t.name LIKE %(pat)s"
            ).format(lit=pgsql.Literal(lyr), cat=cat, tbl=pgsql.Identifier(lyr))
        )
    q = pgsql.SQL("SELECT * FROM ({}) u ORDER BY length(name), name LIMIT 25").format(
        pgsql.SQL(" UNION ALL ").join(parts)
    )
    return _execute(q, {"pat": f"%{name}%"}, conn)


def distinct_values(layer: LayerName, conn: psycopg.Connection, prefix: str | None = None) -> QueryResult:
    """列出某图层类别列的取值与计数（如 pois.subtype），用于把中文概念映射到 OSM 标签。"""
    col = _CATEGORY_COLUMN[layer]
    if col is None:
        raise ToolError(f"layer {layer} has no category column")
    params: dict[str, Any] = {"p": f"{prefix or ''}%"}
    q = pgsql.SQL(
        "SELECT {c} AS value, count(*) AS n FROM {t} WHERE {c} LIKE %(p)s GROUP BY 1 ORDER BY 2 DESC LIMIT 60"
    ).format(c=pgsql.Identifier(col), t=pgsql.Identifier(layer))
    return _execute(q, params, conn)


# ---------------------------------------------------------------- 空间分析工具


def within_distance(
    anchor: Anchor, target: TargetFilter, radius_m: float, conn: psycopg.Connection, count_only: bool = False
) -> QueryResult:
    """缓冲区查询：返回距锚点 radius_m 米以内的目标要素（geography 距离）。"""
    if not 0 < radius_m <= 50000:
        raise ToolError("radius_m must be in (0, 50000]")
    _check_anchor(anchor, conn)
    params: dict[str, Any] = {"r": radius_m}
    conds = _target_where(target, "t", params)
    # 锚点与目标同图层时排除锚点自身（例如“某医院附近的其他医院”）
    if target.layer == anchor.layer:
        conds.append(pgsql.SQL("t.name IS DISTINCT FROM %(anchor_name)s"))
    conds.append(pgsql.SQL("ST_DWithin(t.geom::geography, anchor.geom::geography, %(r)s)"))
    if count_only:
        select = pgsql.SQL("count(*) AS n")
        tail = pgsql.SQL("")
    else:
        select = pgsql.SQL(
            "t.name, ST_Distance(t.geom::geography, anchor.geom::geography) AS dist_m, ST_AsGeoJSON(t.geom, 6) AS geojson"
        )
        tail = pgsql.SQL(" ORDER BY dist_m")
    q = pgsql.SQL("WITH {a} SELECT {s} FROM {tbl} t, anchor WHERE {w}{tail}").format(
        a=_anchor_cte(anchor, params),
        s=select,
        tbl=pgsql.Identifier(target.layer),
        w=_where_sql(conds),
        tail=tail,
    )
    return _execute(q, params, conn)


def nearest_k(anchor: Anchor, target: TargetFilter, k: int, conn: psycopg.Connection) -> QueryResult:
    """最近邻：返回距锚点最近的 k 个目标及其 geography 距离（米）。

    先用 KNN 索引运算符 <-> 在度坐标上取 max(10k, 50) 个候选，再按 geography 精确距离重排，
    避免“度距离排序 ≠ 米距离排序”的问题（经纬度在南京纬度上东西向被压缩约 15%）。
    """
    if not 1 <= k <= 50:
        raise ToolError("k must be in [1, 50]")
    _check_anchor(anchor, conn)
    params: dict[str, Any] = {"k": k, "cand": max(10 * k, 50)}
    conds = _target_where(target, "t", params)
    if target.layer == anchor.layer:
        conds.append(pgsql.SQL("t.name IS DISTINCT FROM %(anchor_name)s"))
    q = pgsql.SQL(
        "WITH {a}, cand AS (SELECT t.name, t.geom FROM {tbl} t, anchor WHERE {w} ORDER BY t.geom <-> anchor.geom LIMIT %(cand)s) "
        "SELECT c.name, ST_Distance(c.geom::geography, anchor.geom::geography) AS dist_m, ST_AsGeoJSON(c.geom, 6) AS geojson "
        "FROM cand c, anchor ORDER BY dist_m LIMIT %(k)s"
    ).format(a=_anchor_cte(anchor, params), tbl=pgsql.Identifier(target.layer), w=_where_sql(conds))
    return _execute(q, params, conn)


def count_by_district(target: TargetFilter, conn: psycopg.Connection) -> QueryResult:
    """按区统计：点在面内（ST_Within）计数，并给出每平方公里密度；无要素的区计 0。"""
    if target.layer not in ("pois", "metro_stations"):
        raise ToolError("count_by_district supports point layers: pois, metro_stations")
    params: dict[str, Any] = {}
    conds = _target_where(target.model_copy(update={"district": None}), "t", params)
    conds.append(pgsql.SQL("ST_Within(t.geom, d.geom)"))
    # 指定 district 时只返回该区一行（否则模型容易引用错行）
    dfilter = pgsql.SQL("")
    if target.district:
        params["only_d"] = target.district
        dfilter = pgsql.SQL(" WHERE d.name = %(only_d)s")
    q = pgsql.SQL(
        "SELECT d.name AS district, count(t.*) AS n, "
        "count(t.*) / (ST_Area(d.geom::geography) / 1e6) AS per_km2, ST_Area(d.geom::geography) / 1e6 AS area_km2 "
        "FROM districts d LEFT JOIN {tbl} t ON {w}{df} GROUP BY d.name, d.geom ORDER BY n DESC"
    ).format(tbl=pgsql.Identifier(target.layer), w=_where_sql(conds), df=dfilter)
    res = _execute(q, params, conn)
    if target.district and not res.rows:
        raise ToolError(f"district '{target.district}' not found; valid names end with 区, e.g. 鼓楼区")
    return res


def measure(
    layer: LayerName, name: str, metric: Literal["area_km2", "length_km"], conn: psycopg.Connection
) -> QueryResult:
    """面积（平方公里）或长度（公里）：同名要素求和，geography 椭球计算。"""
    if metric == "area_km2" and layer not in ("districts", "water_areas", "parks"):
        raise ToolError("area_km2 needs a polygon layer: districts, water_areas, parks")
    if metric == "length_km" and layer not in ("waterways", "roads"):
        raise ToolError("length_km needs a line layer: waterways, roads")
    fn = (
        pgsql.SQL("ST_Area(geom::geography) / 1e6")
        if metric == "area_km2"
        else pgsql.SQL("ST_Length(geom::geography) / 1000")
    )
    q = pgsql.SQL(
        "SELECT name, sum({fn}) AS {m}, count(*) AS parts, ST_AsGeoJSON(ST_Union(geom), 5) AS geojson "
        "FROM {tbl} WHERE name = %(n)s GROUP BY name"
    ).format(fn=fn, m=pgsql.Identifier(metric), tbl=pgsql.Identifier(layer))
    res = _execute(q, {"n": name}, conn)
    if not res.rows:
        raise ToolError(f"'{name}' not found in {layer}; call find_entity first")
    return res


def spatial_relation(
    left_layer: LayerName,
    relation: Literal["intersects", "touches", "within", "contains"],
    right: Anchor,
    conn: psycopg.Connection,
) -> QueryResult:
    """拓扑关系：返回 left_layer 中与锚点要素满足 relation 的要素（如与玄武区相邻的区、长江流经的区）。"""
    _check_anchor(right, conn)
    fn = {
        "intersects": "ST_Intersects",
        "touches": "ST_Touches",
        "within": "ST_Within",
        "contains": "ST_Contains",
    }[relation]
    params: dict[str, Any] = {}
    excl = (
        pgsql.SQL(" AND l.name IS DISTINCT FROM %(anchor_name)s")
        if left_layer == right.layer
        else pgsql.SQL("")
    )
    q = pgsql.SQL(
        "WITH {a} SELECT l.name, ST_AsGeoJSON(ST_SimplifyPreserveTopology(l.geom, 0.0005), 5) AS geojson "
        "FROM {tbl} l, anchor WHERE {fn}(l.geom, anchor.geom){excl} ORDER BY l.name"
    ).format(a=_anchor_cte(right, params), tbl=pgsql.Identifier(left_layer), fn=pgsql.SQL(fn), excl=excl)
    return _execute(q, params, conn)


def distance_between(a: Anchor, b: Anchor, conn: psycopg.Connection) -> QueryResult:
    """两个命名要素之间的最短 geography 距离（米与公里）。"""
    _check_anchor(a, conn)
    _check_anchor(b, conn)
    params = {"an": a.name, "bn": b.name}
    q = pgsql.SQL(
        "WITH a AS (SELECT ST_Union(geom) g FROM {ta} WHERE name = %(an)s), b AS (SELECT ST_Union(geom) g FROM {tb} WHERE name = %(bn)s) "
        "SELECT ST_Distance(a.g::geography, b.g::geography) AS dist_m, ST_Distance(a.g::geography, b.g::geography) / 1000 AS dist_km, "
        "ST_AsGeoJSON(ST_ShortestLine(a.g, b.g), 6) AS geojson FROM a, b"
    ).format(ta=pgsql.Identifier(a.layer), tb=pgsql.Identifier(b.layer))
    return _execute(q, params, conn)
