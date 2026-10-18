"""答案绑定：模型只写带占位符的模板，数字由程序从工具结果中填入。

占位符语法（只允许这四种）：
- {R3.n}               计数：有 n 列取第一行的 n；单值结果取该值；多行明细取行数
- {R3.rows[0].name}    结果 R3 第 0 行的 name 列
- {R3.count}           结果 R3 的行数（由程序计数）
- {R3.rows}            结果 R3 的所有行，渲染为“名称（数值）”列表

绑定后再做一次“未绑定数字”检查：模板中出现的阿拉伯数字（问题里本来就有的数字除外）视为模型自行编造，
直接剔除并记录在 unbound_numbers 中——这条规则让“答案中的数字均来自查询结果”成为可测试的性质。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .db import QueryResult

_PLACEHOLDER = re.compile(r"\{(R\d+)\.(n|count|rows(?:\[(\d+)\]\.([A-Za-z_][A-Za-z0-9_]*))?)\}")
_BAD_PLACEHOLDER = re.compile(r"\{R\d+[^}]*\}")
_NUMBER = re.compile(r"(?<![A-Za-z_\d.])\d+(?:\.\d+)?")


def format_value(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.4f}"
    return "（无名称）" if v is None else str(v)


def _render_rows(res: QueryResult, limit: int = 20) -> str:
    cols = [c for c in res.columns if c != "geojson"]
    idx = [res.columns.index(c) for c in cols]
    items = []
    for row in res.rows[:limit]:
        vals = [row[i] for i in idx]
        head, rest = vals[0], vals[1:]
        items.append(format_value(head) + (f"（{'，'.join(format_value(v) for v in rest)}）" if rest else ""))
    more = f" 等 {len(res.rows)} 条" if len(res.rows) > limit else ""
    return "、".join(items) + more


@dataclass
class BoundAnswer:
    text: str
    used_refs: list[str] = field(default_factory=list)
    unbound_numbers: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def bind(template: str, results: dict[str, QueryResult], question: str = "") -> BoundAnswer:
    """把模板中的占位符替换为结果值；无法解析的占位符记为错误并替换为“?”。"""
    used: list[str] = []
    errors: list[str] = []

    # 先剔除模板中的裸数字（问题原文出现过的数字允许保留，如“500 米内”）
    allowed = set(_NUMBER.findall(question))
    stripped: list[str] = []

    def drop_numbers(segment: str) -> str:
        def repl(m: re.Match[str]) -> str:
            if m.group(0) in allowed:
                return m.group(0)
            stripped.append(m.group(0))
            return "□"

        return _NUMBER.sub(repl, segment)

    out: list[str] = []
    pos = 0
    for m in _PLACEHOLDER.finditer(template):
        out.append(drop_numbers(template[pos : m.start()]))
        pos = m.end()
        ref, kind, row_i, col = m.group(1), m.group(2), m.group(3), m.group(4)
        res = results.get(ref)
        if res is None:
            errors.append(f"unknown result {ref}")
            out.append("?")
            continue
        used.append(ref)
        try:
            if kind == "n":
                # .n 的确定性口径：有 n 列取第一行 n；1×1 结果取该值；否则取行数
                if not res.rows:
                    out.append("0")
                elif "n" in res.columns:
                    out.append(format_value(res.rows[0][res.columns.index("n")]))
                elif len(res.rows) == 1 and len([c for c in res.columns if c != "geojson"]) == 1:
                    out.append(format_value(res.rows[0][0]))
                else:
                    out.append(str(len(res.rows)))
            elif kind == "count":
                out.append(str(len(res.rows)))
            elif kind == "rows":
                out.append(_render_rows(res) if res.rows else "（无结果）")
            else:
                out.append(format_value(res.rows[int(row_i)][res.columns.index(col)]))
        except (IndexError, ValueError):
            errors.append(f"placeholder {m.group(0)} out of range")
            out.append("?")
    out.append(drop_numbers(template[pos:]))
    text = "".join(out)
    # 语法不合法的占位符（如 {R1.row_count}）不能原样输出，记为错误
    for bad in _BAD_PLACEHOLDER.findall(text):
        errors.append(f"invalid placeholder {bad}")
    text = _BAD_PLACEHOLDER.sub("?", text)
    return BoundAnswer(text, sorted(set(used)), stripped, errors)


def to_feature_collection(res: QueryResult | None, max_features: int = 500) -> dict[str, Any]:
    """把结果中的 geojson 列组装成 FeatureCollection，其余列作为 properties。"""
    fc: dict[str, Any] = {"type": "FeatureCollection", "features": []}
    if res is None or "geojson" not in res.columns:
        return fc
    gi = res.columns.index("geojson")
    for row in res.rows[:max_features]:
        if row[gi] is None:
            continue
        props = {c: row[i] for i, c in enumerate(res.columns) if i != gi}
        fc["features"].append({"type": "Feature", "geometry": json.loads(row[gi]), "properties": props})
    return fc
