"""结果比对器与统计函数的单元测试。"""

import math

import pytest

from geo_agent.matcher import canonical_hash, compare, values_equal
from geo_agent.stats import mcnemar_exact, wilson


def test_float_tolerance() -> None:
    assert values_equal(100.0, 100.4)  # 0.4% < 0.5%
    assert not values_equal(100.0, 101.0)  # 1%
    assert values_equal(0, 0.0)
    assert values_equal(None, None)
    assert not values_equal(None, 0)
    assert values_equal("32", 32)


def test_order_insensitive_and_extra_columns() -> None:
    gold_cols, gold_rows = ["name", "n"], [["鼓楼区", 17], ["玄武区", 15]]
    pred_cols = ["district", "geojson", "count"]
    pred_rows = [["玄武区", "{}", 15], ["鼓楼区", "{}", 17]]
    m = compare(gold_cols, gold_rows, pred_cols, pred_rows)
    assert m.correct, m.reason
    assert m.mapping == (0, 2)


def test_order_matters() -> None:
    gold = [["江宁区", 59], ["建邺区", 36]]
    assert compare(["name", "n"], gold, ["name", "n"], gold, order_matters=True).correct
    assert not compare(["name", "n"], gold, ["name", "n"], gold[::-1], order_matters=True).correct


def test_row_count_mismatch() -> None:
    m = compare(["n"], [[3]], ["n"], [[3], [4]])
    assert not m.correct and "row count" in m.reason


def test_missing_column() -> None:
    m = compare(["name", "dist_m"], [["大行宫", 104.7]], ["name"], [["大行宫"]])
    assert not m.correct


def test_unit_error_detected_as_wrong() -> None:
    # 距离以公里返回（0.1047）而金标是米（104.7）：判错
    m = compare(["name", "dist_m"], [["大行宫", 104.7]], ["name", "dist_km"], [["大行宫", 0.1047]])
    assert not m.correct


def test_duplicate_values_multiset() -> None:
    # 多重集合：两个 None 名称必须对应两个 None
    gold = [[None, 1.0], [None, 2.0], ["A", 3.0]]
    pred = [["A", 3.0], [None, 2.0], [None, 1.0]]
    assert compare(["name", "d"], gold, ["name", "d"], pred).correct
    pred_bad = [["A", 3.0], ["A", 2.0], [None, 1.0]]
    assert not compare(["name", "d"], gold, ["name", "d"], pred_bad).correct


def test_canonical_hash_stable_under_float_noise() -> None:
    a = canonical_hash(["d"], [[104.70092964]])
    b = canonical_hash(["d"], [[104.70092964000001]])
    assert a == b


def test_wilson_known_values() -> None:
    lo, hi = wilson(8, 10)
    assert math.isclose(lo, 0.4902, abs_tol=1e-3) and math.isclose(hi, 0.9433, abs_tol=1e-3)
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(0, 9)
    assert lo == 0.0 and 0.29 < hi < 0.30


@pytest.mark.parametrize(("b", "c", "p"), [(0, 0, 1.0), (0, 6, 0.03125), (2, 8, 0.109375), (5, 5, 1.0)])
def test_mcnemar_exact(b: int, c: int, p: float) -> None:
    assert math.isclose(mcnemar_exact(b, c), p, rel_tol=1e-9)
