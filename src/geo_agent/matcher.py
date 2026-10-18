"""结果集比对器：执行准确率（execution accuracy）的判定规则，评测前已预注册。

规则（与 eval/PREREGISTRATION.md 一致）：
1. 预测结果可以多出列（如几何列、id 列），但金标的每一列都必须能单射映射到预测结果的某一列；
2. 行按多重集合比较（默认忽略顺序；order_matters=True 时按顺序逐行比较）；
3. 数值比较：两者都是数值时，相对误差 ≤ 0.5% 或绝对误差 ≤ 1e-6 视为相等；整数计数同样适用该规则，
   但计数通常是精确整数，0.5% 只在 > 200 的计数上才可能放宽；
4. 字符串去首尾空白后精确比较；NULL 只与 NULL 相等。
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

REL_TOL = 0.005
ABS_TOL = 1e-6


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def values_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if _is_num(a) and _is_num(b):
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            return False
        return math.isclose(fa, fb, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    # 字符串形式的数字（如 numeric 被转成 str）也按数值比较
    if _is_num(a) and isinstance(b, str) or _is_num(b) and isinstance(a, str):
        try:
            return values_equal(float(a), float(b))
        except ValueError:
            return False
    return str(a).strip() == str(b).strip()


def _rows_match(gold_rows: list[tuple], pred_rows: list[tuple], order_matters: bool) -> bool:
    if len(gold_rows) != len(pred_rows):
        return False
    if order_matters:
        return all(
            all(values_equal(g, p) for g, p in zip(gr, pr, strict=True))
            for gr, pr in zip(gold_rows, pred_rows, strict=True)
        )
    # 多重集合匹配：容差比较不满足传递性，不能排序后比较，这里用贪心 + 回溯的二分匹配
    remaining = list(range(len(pred_rows)))

    def match(i: int) -> bool:
        if i == len(gold_rows):
            return True
        for j in list(remaining):
            if all(values_equal(g, p) for g, p in zip(gold_rows[i], pred_rows[j], strict=True)):
                remaining.remove(j)
                if match(i + 1):
                    return True
                remaining.append(j)
        return False

    return match(0)


@dataclass(frozen=True)
class MatchResult:
    correct: bool
    reason: str
    mapping: tuple[int, ...] | None = None


def compare(
    gold_columns: Sequence[str],
    gold_rows: Sequence[Sequence[Any]],
    pred_columns: Sequence[str],
    pred_rows: Sequence[Sequence[Any]],
    *,
    order_matters: bool = False,
    max_mappings: int = 5000,
) -> MatchResult:
    """判断预测结果是否与金标一致；返回原因便于写入失败案例。"""
    k, m = len(gold_columns), len(pred_columns)
    if k == 0:
        return MatchResult(False, "gold has no columns")
    if m < k:
        return MatchResult(False, f"pred has {m} columns, gold needs {k}")
    if len(gold_rows) != len(pred_rows):
        return MatchResult(False, f"row count {len(pred_rows)} != gold {len(gold_rows)}")
    if not gold_rows:
        return MatchResult(True, "both empty", tuple(range(k)))

    # 先按列内容筛出每个金标列可能对应的预测列，缩小排列搜索空间
    candidates: list[list[int]] = []
    for gi in range(k):
        gold_col = [r[gi] for r in gold_rows]
        cands = []
        for pj in range(m):
            pred_col = [r[pj] for r in pred_rows]
            if order_matters:
                ok = all(values_equal(a, b) for a, b in zip(gold_col, pred_col, strict=True))
            else:
                ok = _rows_match([(v,) for v in gold_col], [(v,) for v in pred_col], False)
            if ok:
                cands.append(pj)
        if not cands:
            return MatchResult(False, f"no pred column matches gold column '{gold_columns[gi]}'")
        candidates.append(cands)

    tried = 0
    for combo in itertools.product(*candidates):
        if len(set(combo)) != k:
            continue
        tried += 1
        if tried > max_mappings:
            break
        projected = [tuple(r[j] for j in combo) for r in pred_rows]
        if _rows_match([tuple(r) for r in gold_rows], projected, order_matters):
            return MatchResult(True, "match", combo)
    return MatchResult(False, "columns match individually but rows do not align")


def canonical_hash(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """金标结果的规范化哈希：浮点保留 6 位有效数字，避免平台差异导致哈希漂移。"""

    def norm(v: Any) -> Any:
        if isinstance(v, float):
            return float(f"{v:.6g}")
        return v

    payload = json.dumps(
        {"columns": list(columns), "rows": [[norm(v) for v in r] for r in rows]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
