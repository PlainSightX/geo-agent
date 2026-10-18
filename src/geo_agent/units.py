"""空间单位/CRS 静态检查。

本库所有几何列都是 EPSG:4326（单位：度）。对其直接调用 ST_Distance / ST_Area / ST_Length /
ST_DWithin / ST_Buffer 得到的是“度”或“平方度”，乘以任何常数都不是米——这是 LLM 生成空间 SQL
最常见的静默错误：SQL 能执行、结果非空，但数值错了几个数量级。

判定规则：度量函数的几何参数必须“米制化”，即满足以下之一：
- 显式转 geography（::geography / CAST AS geography / geography(...)）；
- 经 ST_Transform 投影到米制 CRS（本项目认可 32650 UTM 50N、4547/4549 CGCS2000 高斯-克吕格、3857 不认可）；
- 使用本身返回米的函数（ST_DistanceSphere / ST_DistanceSpheroid / ST_LengthSpheroid）。
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

METRIC_SRIDS = {32650, 4547, 4549, 4548, 32651}
# 3857 Web 墨卡托在南京纬度（~32°N）长度放大约 1/cos(32°)≈1.18，面积约 1.39 倍，视为单位错误
DISTORTED_SRIDS = {3857, 900913}
MEASURE_FUNCS = {"st_distance", "st_area", "st_length", "st_dwithin", "st_buffer", "st_perimeter"}
METER_FUNCS = {"st_distancesphere", "st_distancespheroid", "st_lengthspheroid", "st_distance_sphere"}


@dataclass(frozen=True)
class UnitIssue:
    function: str
    argument: str
    reason: str

    def __str__(self) -> str:
        return f"{self.function}({self.argument}): {self.reason}"


def _fname(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return node.name.lower()
    if isinstance(node, exp.Func):
        return node.sql_name().lower().replace("st_distance", "st_distance") if node.sql_name() else ""
    return ""


def _args(node: exp.Expression) -> list[exp.Expression]:
    if isinstance(node, exp.Anonymous):
        return list(node.expressions)
    return [a for a in (node.this, node.args.get("expression")) if isinstance(a, exp.Expression)]


def _is_metric(arg: exp.Expression) -> tuple[bool, str]:
    """判断一个几何表达式是否已处于米制单位；返回 (是否米制, 原因)。"""
    node = arg
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Cast):
        if node.to.is_type("geography"):
            return True, "geography cast"
        if node.to.is_type("geometry"):
            return _is_metric(node.this)
        return False, f"cast to {node.to.sql('postgres')}"
    name = _fname(node)
    if name == "geography":
        return True, "geography()"
    if name == "st_transform":
        args = _args(node)
        if len(args) >= 2 and isinstance(args[1], exp.Literal) and args[1].is_int:
            srid = int(args[1].this)
            if srid in METRIC_SRIDS:
                return True, f"EPSG:{srid}"
            if srid in DISTORTED_SRIDS:
                return False, f"EPSG:{srid} is Web Mercator (scale error ~18% length at 32N)"
            return False, f"EPSG:{srid} not in accepted metric CRS list"
        return False, "ST_Transform with non-literal SRID"
    if name in {
        "st_buffer",
        "st_intersection",
        "st_union",
        "st_centroid",
        "st_pointonsurface",
        "st_collect",
        "st_makevalid",
        "st_multi",
        "st_envelope",
        "st_convexhull",
        "st_unaryunion",
        "st_closestpoint",
        # 第二轮开发集发现的漏项（误报：ST_Buffer(ST_Simplify(ST_Transform(g, 32650), 2), 50) 被判为度坐标）
        "st_simplify",
        "st_simplifypreservetopology",
        "st_snaptogrid",
        "st_linemerge",
    }:
        # 这些函数保持输入的单位：任一几何参数米制即视为米制（如 ST_Intersection(a::geography, ...) 不合法，但 transform 合法）
        inner = [a for a in _args(node) if not isinstance(a, exp.Literal)]
        if inner and all(_is_metric(a)[0] for a in inner[:2] if not isinstance(a, exp.Literal)):
            return True, f"{name} of metric input"
        return False, f"{name} of EPSG:4326 input"
    return False, "EPSG:4326 geometry (degrees)"


def lint_units(sql: str) -> list[UnitIssue]:
    """返回所有“在 4326 度坐标上做米制度量”的问题；SQL 解析失败时返回空列表（交给守卫报错）。"""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.ParseError:
        return []
    issues: list[UnitIssue] = []
    for node in tree.find_all(exp.Func):
        name = _fname(node)
        if name not in MEASURE_FUNCS:
            continue
        geom_args = _args(node)
        if name in {"st_dwithin"}:
            geom_args = geom_args[:2]
        elif name in {"st_buffer"}:
            geom_args = geom_args[:1]
        elif name in {"st_distance"}:
            geom_args = geom_args[:2]
        else:
            geom_args = geom_args[:1]
        for a in geom_args:
            ok, why = _is_metric(a)
            if not ok:
                issues.append(UnitIssue(name, a.sql("postgres")[:80], why))
    return issues


def uses_measure(sql: str) -> bool:
    """SQL 是否包含度量函数（评测里用来确定“单位相关题”的范围）。"""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.ParseError:
        return False
    return any(_fname(n) in MEASURE_FUNCS | METER_FUNCS for n in tree.find_all(exp.Func))
