"""第二轮判分规则单元测试（离线）。"""

from geo_agent import matcher as M
from geo_agent.matcher_r2 import as_bool, compare_compliance, score_item

CL = "GB50180-2018:C.0.1:初中"


def test_route_tolerance_is_two_percent_and_restored() -> None:
    g = {"tier": "route", "columns": ["walk_m"], "rows": [[904.5]]}
    assert score_item(g, ["d"], [[920.0]]).correct  # 1.7%
    assert not score_item(g, ["d"], [[930.0]]).correct  # 2.8%
    assert M.REL_TOL == 0.005  # 临时容差不泄漏到第一轮比对器


def test_non_route_keeps_half_percent() -> None:
    g = {"tier": "crs", "columns": ["km"], "rows": [[2.983]]}
    assert score_item(g, ["x"], [[2.990]]).correct
    assert not score_item(g, ["x"], [[3.02]]).correct


def test_compliance_requires_conclusion_and_clause() -> None:
    gold = [True, CL, 1000]
    assert compare_compliance(gold, ["ok", "clause_id"], [["符合", CL]]).correct
    assert compare_compliance(gold, ["ok", "clause_id", "radius_m"], [[True, CL, 1000]]).correct
    assert not compare_compliance(gold, ["ok"], [[True]]).correct  # 未引用条文
    assert not compare_compliance(gold, ["ok", "c"], [[False, CL]]).correct  # 结论错
    assert not compare_compliance(gold, ["ok", "c", "radius_m"], [[True, CL, 800]]).correct  # 条文数字不一致
    assert not compare_compliance(gold, ["ok", "c"], [[True, "GB50180-2018:C.0.1:小学"]]).correct
    assert not compare_compliance(gold, ["ok", "c"], [[True, CL], [True, CL]]).correct


def test_as_bool() -> None:
    assert as_bool("是") is True and as_bool("不符合") is False and as_bool(0) is False
    assert as_bool("maybe") is None and as_bool(1000) is None
