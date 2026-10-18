"""第二轮判分规则（预注册，见 eval/r2/PREREGISTRATION.md“判分规则”）。

- 一般题：沿用第一轮 matcher.compare（数值 0.5%）；
- 路网数值题：数值容差放宽到 2%（计数、名称仍精确——整数计数在 2% 下只对 > 50 的计数生效，本题集路网计数都 < 30）；
- 合规题：布尔结论 + 条文 id 都要对；若提供 radius_m 列，必须等于条文数值。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import matcher as M

ROUTE_REL_TOL = 0.02
_TRUE = {"true", "t", "1", "yes", "y", "是", "符合", "满足", "有", "合规", "达标"}
_FALSE = {"false", "f", "0", "no", "n", "否", "不符合", "不满足", "没有", "无", "不合规", "不达标"}


def compare_tol(
    gold_columns: Sequence[str],
    gold_rows: Sequence[Sequence[Any]],
    pred_columns: Sequence[str],
    pred_rows: Sequence[Sequence[Any]],
    *,
    order_matters: bool,
    rel_tol: float,
) -> M.MatchResult:
    """在指定相对容差下调用第一轮比对器（临时替换模块级容差，调用结束后恢复）。"""
    old = M.REL_TOL
    M.REL_TOL = rel_tol
    try:
        return M.compare(gold_columns, gold_rows, pred_columns, pred_rows, order_matters=order_matters)
    finally:
        M.REL_TOL = old


def as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    return None


def compare_compliance(
    gold_row: Sequence[Any], pred_columns: Sequence[str], pred_rows: Sequence[Sequence[Any]]
) -> M.MatchResult:
    """gold_row = (compliant, clause_id, radius_m)。预测须为 1 行，含与金标一致的布尔列与条文 id 列。"""
    g_ok, g_clause, g_radius = gold_row
    if len(pred_rows) != 1:
        return M.MatchResult(False, f"compliance answer must be 1 row, got {len(pred_rows)}")
    row = pred_rows[0]
    if not any(isinstance(v, str) and v.strip() == g_clause for v in row):
        cited = [v for v in row if isinstance(v, str) and v.startswith("GB50180")]
        return M.MatchResult(False, f"clause id {g_clause} not cited (cited: {cited})")
    bools = [b for b in (as_bool(v) for v in row) if b is not None]
    if not bools:
        return M.MatchResult(False, "no boolean conclusion column")
    if bools[0] != g_ok:
        return M.MatchResult(False, f"conclusion {bools[0]} != gold {g_ok}")
    cols = [c.lower() for c in pred_columns]
    if "radius_m" in cols:
        v = row[cols.index("radius_m")]
        if not M.values_equal(v, g_radius):
            return M.MatchResult(False, f"radius_m {v} != clause value {g_radius}")
    return M.MatchResult(True, "match")


def score_item(gold: dict, pred_columns: Sequence[str], pred_rows: Sequence[Sequence[Any]]) -> M.MatchResult:
    """按层选择比对规则。"""
    if gold["tier"] == "compliance":
        return compare_compliance(gold["rows"][0], pred_columns, pred_rows)
    tol = ROUTE_REL_TOL if gold["tier"] == "route" else M.REL_TOL
    return compare_tol(
        gold["columns"],
        gold["rows"],
        pred_columns,
        pred_rows,
        order_matters=gold.get("order_matters", False),
        rel_tol=tol,
    )
