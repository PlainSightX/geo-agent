"""MCP 服务器：把第二轮已测试的类型化空间工具原样暴露给 MCP 客户端（Claude Desktop、Cursor、MCP Inspector 等）。

与仓库内 LangGraph Agent 的关系：
- 工具实现完全复用 tools / routing / r2_tools（同一份参数化 SQL、同一口径、同一组 pydantic 参数模型），这里不重写任何查询；
- Agent 循环里的“校验 + 有限修复 + 数字绑定 + final_answer”属于本项目的编排层，不在 MCP 中暴露：
  MCP 只提供工具、资源和一个提示模板，调用顺序与答案组织由客户端里的模型自己决定。

安全边界（与 api.py 相同）：
- 只用只读角色 geo_reader（GEO_AGENT_DSN）；每次查询都过 sqlglot AST 守卫 → READ ONLY 事务 → 5 s 语句超时 → 200 行上限；
- run_sql 与第二轮 Agent 的 run_sql 走同一路径：先做单位/CRS 静态检查，再进守卫，DDL/DML 一律拒绝；
- 服务器**没有鉴权**。推荐 stdio 传输（由客户端以子进程启动，不监听端口）；streamable-http 只绑定 127.0.0.1。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as MCPToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from . import r2_tools as X
from . import routing as RT
from . import tools as T
from .answer import to_feature_collection
from .config import get_settings
from .db import QueryError, QueryResult, connect, run_select
from .r2_schema import FUNCTION_NOTES, describe_lean_schema
from .sql_guard import SqlRejected
from .units import lint_units

LOOPBACK = "127.0.0.1"
SCHEMA_URI = "geo-agent://schema/lean"
CLAUSES_URI = "geo-agent://standards/clauses"

# find_entity 可查的图层（与 r2_tools._FIND_LAYERS 一致；测试检查两者不漂移）
FindLayer = Literal[
    "metro_stations",
    "pois",
    "residential_areas",
    "districts",
    "water_areas",
    "parks",
    "waterways",
    "bus_stops",
]

# 所有工具都只读、可重复调用、只访问本地库
_READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)


# 结果单元格：数据库值已由 db._to_jsonable 转成 JSON 原生类型（一维数组列为 list，json 列为 dict）
JsonScalar = str | int | float | bool | None
JsonCell = JsonScalar | list[JsonScalar] | dict[str, Any]


class ToolResult(BaseModel):
    """统一返回结构：与 QueryResult 同构；geojson 列从表格中拆出，组装成 FeatureCollection。"""

    columns: list[str] = Field(description="结果列名（不含 geojson 列）")
    rows: list[list[JsonCell]] = Field(description="结果行，与 columns 一一对应")
    truncated: bool = Field(description="是否超过行数上限（200）被截断")
    row_count: int = Field(description="返回的行数")
    elapsed_ms: float = Field(description="数据库执行耗时（毫秒）")
    sql: str = Field(description="实际执行的只读 SQL（参数已渲染），便于复现与审计")
    geojson: dict[str, Any] | None = Field(
        default=None, description="几何结果的 GeoJSON FeatureCollection（EPSG:4326）；工具不产生几何时为 null"
    )


def to_tool_result(res: QueryResult) -> ToolResult:
    """QueryResult → ToolResult：表格去掉 geojson 列，几何放入 FeatureCollection（其余列作 properties）。"""
    if "geojson" in res.columns:
        gi = res.columns.index("geojson")
        cols = [c for i, c in enumerate(res.columns) if i != gi]
        rows = [[v for i, v in enumerate(r) if i != gi] for r in res.rows]
        fc: dict[str, Any] | None = to_feature_collection(res, max_features=len(res.rows))
    else:
        cols, rows, fc = list(res.columns), [list(r) for r in res.rows], None
    return ToolResult(
        columns=cols,
        rows=rows,
        truncated=res.truncated,
        row_count=len(rows),
        elapsed_ms=res.elapsed_ms,
        sql=res.sql,
        geojson=fc,
    )


def _call(fn: Callable[[Any], QueryResult]) -> ToolResult:
    """每次调用独立建立只读连接；把项目内可预期的错误转成 MCP ToolError（客户端模型可读到原因并自行修正）。"""
    try:
        with connect() as conn:
            return to_tool_result(fn(conn))
    except (SqlRejected, QueryError, T.ToolError) as err:
        raise MCPToolError(str(err)) from err


def guarded_select(sql: str) -> QueryResult:
    """run_sql 的执行路径（与第二轮 Agent 相同）：单位/CRS 静态检查 → AST 守卫（第二轮白名单）→ 只读事务执行。"""
    issues = lint_units(sql)
    if issues:
        raise T.ToolError(
            "unit/CRS error: "
            + "; ".join(map(str, issues))
            + ". geom is EPSG:4326 degrees; use ::geography for meters"
        )
    return run_select(sql, settings=get_settings(), r2=True)


mcp = MCPServer(
    name="geo-agent",
    title="南京 OSM 空间工具（PostGIS / pgRouting / pgvector）",
    version="0.1.0",
    instructions=(
        "南京 OpenStreetMap 快照上的只读空间工具。名称不确定时先调用 find_entity 取精确名称；"
        "距离/面积一律为米制（geography）；步行工具按 4.8 km/h = 80 m/min 计；"
        "合规判定先 retrieve_standard 找条文，再 check_compliance（半径由程序从条文读取）。"
        f"表结构见资源 {SCHEMA_URI}，条文清单见 {CLAUSES_URI}。"
    ),
)


def _tool(description: str) -> Callable[[Callable[..., ToolResult]], Callable[..., ToolResult]]:
    return mcp.tool(description=description, annotations=_READ_ONLY, structured_output=True)


# ---------------------------------------------------------------- 自省


@_tool(
    "按名称或别名（alt_name / old_name / short_name / official_name / name:en）模糊查找要素，"
    "返回图层、精确名称、类别、所属区、匹配方式与同名要素数。其他工具需要的 name 应取这里返回的精确值。"
)
def find_entity(
    name: Annotated[str, Field(min_length=1, description="名称或别名片段，如 新街口、南京博物院")],
    layer: Annotated[FindLayer | None, Field(description="只在该图层中查找；省略则查全部图层")] = None,
) -> ToolResult:
    return _call(lambda c: X.find_entity_r2(name, c, layer))


@_tool(
    "列出某图层类别列的取值与计数（如 pois.subtype、waterways.waterway），用于把中文概念对应到 OSM 标签值。"
)
def distinct_values(
    layer: Annotated[T.LayerName, Field(description="图层名")],
    prefix: Annotated[str | None, Field(description="只列出以此前缀开头的取值")] = None,
) -> ToolResult:
    return _call(lambda c: T.distinct_values(layer, c, prefix))


@_tool(
    "查看 pois 的原始 OSM 标签。不给 key：列出最常见的 60 个标签键；给 key：列出该键的取值与计数"
    "（like 按子串过滤取值），用于发现 brand、operator、opening_hours 等映射。"
)
def tag_values(
    key: Annotated[str | None, Field(description="标签键，如 brand、operator、name:en")] = None,
    like: Annotated[str | None, Field(description="取值子串过滤")] = None,
) -> ToolResult:
    return _call(lambda c: X.tag_values(c, key, like))


# ---------------------------------------------------------------- 直线 GIS（geography 米制）


@_tool(
    "缓冲区查询：锚点 radius_m 米以内（geography 直线距离）的目标要素，按距离排序并带 GeoJSON；count_only=true 只返回数量 n。"
)
def within_distance(
    anchor: Annotated[T.Anchor, Field(description="参照要素 {layer, name}")],
    target: Annotated[T.TargetFilter, Field(description="目标要素过滤条件")],
    radius_m: Annotated[float, Field(gt=0, le=50000, description="半径（米）")],
    count_only: Annotated[bool, Field(description="只返回数量")] = False,
) -> ToolResult:
    return _call(lambda c: T.within_distance(anchor, target, radius_m, c, count_only))


@_tool("最近邻：距锚点最近的 k 个目标要素及其 geography 直线距离（米），带 GeoJSON 点。")
def nearest_k(
    anchor: Annotated[T.Anchor, Field(description="参照要素 {layer, name}")],
    target: Annotated[T.TargetFilter, Field(description="目标要素过滤条件")],
    k: Annotated[int, Field(ge=1, le=50, description="返回个数")],
) -> ToolResult:
    return _call(lambda c: T.nearest_k(anchor, target, k, c))


@_tool(
    "按区统计点要素（ST_Within）：每区数量 n、每平方公里密度 per_km2 与区面积 area_km2；目标 district 非空时只返回该区。"
)
def count_by_district(
    target: Annotated[T.TargetFilter, Field(description="目标要素过滤条件（图层限 pois、metro_stations）")],
) -> ToolResult:
    return _call(lambda c: T.count_by_district(target, c))


@_tool(
    "量算同名要素：面图层的面积（area_km2，平方公里）或线图层的长度（length_km，公里），geography 椭球计算，带 GeoJSON。"
)
def measure(
    layer: Annotated[
        T.LayerName, Field(description="图层名：面积用 districts/water_areas/parks，长度用 waterways/roads")
    ],
    name: Annotated[str, Field(min_length=1, description="要素精确名称")],
    metric: Annotated[Literal["area_km2", "length_km"], Field(description="量算指标")],
) -> ToolResult:
    return _call(lambda c: T.measure(layer, name, metric, c))


@_tool(
    "拓扑关系：left_layer 中与锚点要素满足 intersects / touches / within / contains 的要素（如与玄武区相邻的区）。"
)
def spatial_relation(
    left_layer: Annotated[T.LayerName, Field(description="被筛选的图层")],
    relation: Annotated[
        Literal["intersects", "touches", "within", "contains"], Field(description="空间关系")
    ],
    right: Annotated[T.Anchor, Field(description="参照要素 {layer, name}")],
) -> ToolResult:
    return _call(lambda c: T.spatial_relation(left_layer, relation, right, c))


@_tool("两个命名要素之间的最短 geography 直线距离（米与公里），带最短连线 GeoJSON。")
def distance_between(
    a: Annotated[T.Anchor, Field(description="要素 A {layer, name}")],
    b: Annotated[T.Anchor, Field(description="要素 B {layer, name}")],
) -> ToolResult:
    return _call(lambda c: T.distance_between(a, b, c))


# ---------------------------------------------------------------- 步行路网（pgRouting）


@_tool(
    "两地之间沿步行路网的最短路径（pgr_dijkstra）：网络距离 walk_m、按 4.8 km/h 的步行分钟 walk_min、"
    "直线距离 straight_m、两端吸附距离，以及路径 GeoJSON 线。"
)
def walk_route(
    a: Annotated[RT.Place, Field(description="起点 {layer, name}")],
    b: Annotated[RT.Place, Field(description="终点 {layer, name}")],
) -> ToolResult:
    return _call(lambda c: RT.walk_route(a, b, c))


@_tool(
    "步行等时圈：起点 minutes 分钟（80 m/min）内可达路网边的 50 m 缓冲面，返回面积 area_km2、"
    "可达顶点数、吸附距离与 GeoJSON 面。"
)
def isochrone(
    origin: Annotated[RT.Place, Field(description="起点 {layer, name}，如小区 residential_areas")],
    minutes: Annotated[float, Field(gt=0, le=RT.MAX_MINUTES, description="步行分钟数（0, 30]")],
) -> ToolResult:
    return _call(lambda c: RT.isochrone(origin, minutes, c))


@_tool(
    "N 分钟步行可达的目标要素：目标吸附顶点的网络距离 ≤ 80·minutes 米，按 walk_m 排序并带 GeoJSON；"
    "count_only=true 只返回数量 n。"
)
def reachable(
    origin: Annotated[RT.Place, Field(description="起点 {layer, name}")],
    target: Annotated[T.R2TargetFilter, Field(description="目标过滤：layer + subtypes + name_like 等")],
    minutes: Annotated[float, Field(gt=0, le=RT.MAX_MINUTES, description="步行分钟数（0, 30]")],
    count_only: Annotated[bool, Field(description="只返回数量")] = False,
) -> ToolResult:
    return _call(lambda c: RT.reachable_facilities(origin, target, minutes, c, count_only))


# ---------------------------------------------------------------- 标准条文（RAG）与合规


@_tool(
    "在 GB 50180-2018 条文摘录中做语义检索（本地 bge-small-zh 向量 + pgvector 余弦）：返回 clause_id、标题、设施、"
    "生活圈级别、服务半径 max_radius_m、步行距离区间、相似度与条文正文。"
)
def retrieve_standard(
    query: Annotated[str, Field(min_length=1, description="检索语句，如 初中服务半径")],
    k: Annotated[int, Field(ge=1, le=10, description="返回条数")] = 5,
) -> ToolResult:
    return _call(lambda c: X.retrieve_standard(query, c, k))


@_tool(
    "按条文判定小区的配套合规性：程序从条文读取服务半径 max_radius_m（不接受手填数字），沿步行路网检查半径内"
    "是否有目标设施。返回 compliant、clause_id、radius_m、nearest_walk_m、nearest_name。"
    "结论只是按推荐性服务半径的空间核查，不是规划审批结论。"
)
def check_compliance(
    residential: Annotated[str, Field(min_length=1, description="小区精确名称（residential_areas）")],
    clause_id: Annotated[str, Field(min_length=1, description="条文 id，如 GB50180-2018:C.0.1:初中")],
    target: Annotated[
        T.R2TargetFilter, Field(description="目标设施过滤，如 中学 = pois/school + name_like 中学")
    ],
) -> ToolResult:
    return _call(lambda c: X.check_compliance(residential, clause_id, target, c))


# ---------------------------------------------------------------- 受限 SQL


@_tool(
    "执行一条只读 PostGIS / pgRouting SELECT：先做单位/CRS 静态检查，再经 sqlglot 语法树白名单（禁止 DDL/DML/COPY/SET 等），"
    "在只读事务中执行，语句超时 5 秒，最多返回 200 行。需要上图时加 ST_AsGeoJSON(geom) AS geojson。"
)
def run_sql(
    sql: Annotated[str, Field(min_length=1, max_length=20000, description="单条 SELECT 语句")],
) -> ToolResult:
    try:
        return to_tool_result(guarded_select(sql))
    except (SqlRejected, QueryError, T.ToolError) as err:
        raise MCPToolError(str(err)) from err


# ---------------------------------------------------------------- 资源与提示模板


@mcp.resource(
    SCHEMA_URI,
    name="lean_schema",
    title="第二轮精简 schema",
    description="第二轮评测两臂共用的精简表结构（表、列、类型、行数、简短含义）与可用函数说明。",
    mime_type="text/plain",
)
def lean_schema() -> str:
    return describe_lean_schema() + "\n\n" + FUNCTION_NOTES


@mcp.resource(
    CLAUSES_URI,
    name="standard_clauses",
    title="GB 50180-2018 条文清单",
    description="已入库的条文摘录元数据（不含正文与向量）：clause_id、标题、设施、生活圈级别、服务半径、页码。",
    mime_type="application/json",
)
def standard_clauses() -> str:
    res = run_select(
        "SELECT clause_id, title, facility, circle_min, max_radius_m, walk_m_min, walk_m_max, strength, page "
        "FROM standard_clauses ORDER BY clause_id",
        r2=True,
    )
    items = [dict(zip(res.columns, r, strict=True)) for r in res.rows]
    return json.dumps({"count": len(items), "clauses": items}, ensure_ascii=False, indent=1)


@mcp.prompt(
    name="compliance_check",
    title="小区配套合规核查",
    description="按 GB 50180-2018 服务半径核查某小区步行可达的某类设施。",
)
def compliance_check(residential: str, facility: str = "初中") -> str:
    return (
        f"请核查小区「{residential}」的「{facility}」配套是否满足 GB 50180-2018 的服务半径要求。步骤：\n"
        "1. 用 find_entity（layer=residential_areas）确认小区的精确名称；\n"
        f"2. 用 retrieve_standard 检索「{facility}服务半径」，选出带 max_radius_m 的条文，记下 clause_id；\n"
        "3. 必要时用 distinct_values / tag_values 确定设施对应的 OSM 类别（如 中学 = pois.subtype school 且名称含 中学）；\n"
        "4. 调用 check_compliance，半径由程序从条文读取，不要自己填写数字；\n"
        "5. 回答时给出 compliant、clause_id、radius_m 和最近设施的步行距离，数字只引用工具结果；"
        "并说明这只是按推荐性服务半径的空间核查，不是规划审批结论。"
    )


# ---------------------------------------------------------------- 入口


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="geo-agent-mcp",
        description="geo-agent MCP 服务器（只读空间工具，无鉴权；推荐 stdio）",
    )
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--port", type=int, default=8766, help="streamable-http 端口（固定绑定 127.0.0.1）")
    args = parser.parse_args(argv)
    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        # 没有鉴权：HTTP 只监听回环地址，不提供改绑其他地址的选项
        mcp.run("streamable-http", host=LOOPBACK, port=args.port)


if __name__ == "__main__":
    main()
