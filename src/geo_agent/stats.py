"""评测统计：Wilson 置信区间与精确 McNemar 检验（只依赖标准库）。"""

from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """二项比例的 Wilson 得分区间；n=0 时返回 (0, 1)。"""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(b: int, c: int) -> float:
    """精确 McNemar（双侧）：b = 仅 A 对，c = 仅 B 对；在 b+c 次不一致中检验 p=0.5。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
