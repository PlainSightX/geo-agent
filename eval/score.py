"""离线判分：读取运行输出与冻结金标，计算预注册指标，输出 JSON 汇总与 Markdown 报告片段。

用法：uv run python eval/score.py --run test1 [--set test]
不调用模型；可以反复运行。判分规则见 eval/PREREGISTRATION.md 与 src/geo_agent/matcher.py。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from geo_agent.config import get_settings
from geo_agent.matcher import compare
from geo_agent.stats import mcnemar_exact, wilson
from geo_agent.units import lint_units, uses_measure

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("baseline", "agent")


def load_jsonl(p: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    if not p.exists():
        return recs
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["id"]] = r  # 补跑记录覆盖原记录（rerun=true 会在报告中披露）
    return recs


def power_of_ten_off(pred: Any, gold: Any) -> bool:
    """数值是否与金标相差 10 的整数次幂（±2%）——典型的米/公里、平方米/平方公里混淆。"""
    try:
        p, g = float(pred), float(gold)
    except (TypeError, ValueError):
        return False
    if p == 0 or g == 0 or math.isclose(p, g, rel_tol=0.005):
        return False
    r = abs(p / g)
    e = round(math.log10(r))
    return e != 0 and math.isclose(r, 10.0**e, rel_tol=0.02)


def score_one(rec: dict | None, gold: dict, gold_sql: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {"id": gold["id"], "tier": gold["tier"]}
    if rec is None:
        out.update(correct=False, reason="missing run record", status="missing")
        return out
    status = rec.get("status")
    out["status"] = status
    if not gold["answerable"]:
        out["correct"] = status == "refused"
        out["reason"] = "refused" if out["correct"] else f"should refuse, got {status}"
        return out
    if status != "answered" or "final_columns" not in rec:
        out.update(
            correct=False,
            reason=f"no result ({status}): {rec.get('error') or rec.get('refuse_reason') or ''}"[:200],
        )
    else:
        m = compare(
            gold["columns"],
            gold["rows"],
            rec["final_columns"],
            rec["final_rows"],
            order_matters=gold.get("order_matters", False),
        )
        out.update(correct=m.correct, reason=m.reason)
    # 单位/CRS 错误：仅在金标含度量函数的题上统计
    if gold_sql and uses_measure(gold_sql):
        out["unit_scope"] = True
        final_sql = rec.get("final_sql") or rec.get("sql") or ""
        static = bool(final_sql) and bool(lint_units(final_sql))
        scale = False
        if "final_rows" in rec and rec.get("final_rows") and gold["rows"]:
            scale = any(
                power_of_ten_off(v, gold["rows"][0][j])
                for j, gc in enumerate(gold["columns"])
                if isinstance(gold["rows"][0][j], (int, float))
                for v in (rec["final_rows"][0] if rec["final_rows"] else [])
                if isinstance(v, (int, float))
            )
        out["unit_error"] = static or scale
    return out


def summarize(scored: list[dict], recs: dict[str, dict]) -> dict[str, Any]:
    s = get_settings()
    n = len(scored)
    k = sum(r["correct"] for r in scored)
    ans = [r for r in scored if r["tier"] != "unanswerable"]
    una = [r for r in scored if r["tier"] == "unanswerable"]
    unit = [r for r in scored if r.get("unit_scope")]
    raw = [x for r in recs.values() for x in (r.get("raw_sqls") or [])]
    calls = [r.get("model_calls", 0) for r in recs.values()]
    pin = sum(r.get("prompt_tokens", 0) for r in recs.values())
    pout = sum(r.get("completion_tokens", 0) for r in recs.values())
    lat = sorted(r.get("latency_s", 0.0) for r in recs.values())
    cost = pin / 1e6 * s.price_in_per_m + pout / 1e6 * s.price_out_per_m

    def prop(kk: int, nn: int) -> dict[str, Any]:
        lo, hi = wilson(kk, nn)
        return {
            "k": kk,
            "n": nn,
            "p": round(kk / nn, 4) if nn else None,
            "ci95": [round(lo, 4), round(hi, 4)],
        }

    tiers: dict[str, dict] = {}
    for t in sorted({r["tier"] for r in scored}):
        rows = [r for r in scored if r["tier"] == t]
        tiers[t] = prop(sum(r["correct"] for r in rows), len(rows))
    return {
        "overall": prop(k, n),
        "execution_accuracy_answerable": prop(sum(r["correct"] for r in ans), len(ans)),
        "refusal_correct_unanswerable": prop(sum(r["correct"] for r in una), len(una)),
        "false_refusals_on_answerable": sum(r.get("status") == "refused" for r in ans),
        "valid_sql_rate": prop(sum(1 for x in raw if x.get("valid")), len(raw)),
        "unit_error_rate": prop(sum(r.get("unit_error", False) for r in unit), len(unit)),
        "by_tier": tiers,
        "calls_per_question": round(sum(calls) / max(len(calls), 1), 2),
        "tokens_in": pin,
        "tokens_out": pout,
        "cost_usd": round(cost, 4),
        "cost_per_question_usd": round(cost / max(len(recs), 1), 5),
        "latency_s_median": lat[len(lat) // 2] if lat else None,
        "latency_s_p90": lat[int(0.9 * (len(lat) - 1))] if lat else None,
        "reruns": sorted(r["id"] for r in recs.values() if r.get("rerun")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", choices=["dev", "test"], default="test")
    args = ap.parse_args()
    gold_file = ROOT / "eval" / ("gold.json" if args.set == "test" else "dev_gold.json")
    q_file = ROOT / "eval" / ("questions.json" if args.set == "test" else "dev_questions.json")
    gold = json.loads(gold_file.read_text(encoding="utf-8"))
    gold_sql = {q["id"]: q["gold_sql"] for q in json.loads(q_file.read_text(encoding="utf-8"))}
    run_dir = ROOT / "eval" / "runs" / args.run

    per_arm: dict[str, list[dict]] = {}
    summary: dict[str, Any] = {
        "run": args.run,
        "set": args.set,
        "gold_frozen_at": gold["frozen_at"],
        "questions_sha256": gold["questions_sha256"],
        "arms": {},
    }
    for arm in ARMS:
        recs = load_jsonl(run_dir / f"{arm}.jsonl")
        if not recs:
            continue
        scored = [score_one(recs.get(g["id"]), g, gold_sql[g["id"]]) for g in gold["items"]]
        per_arm[arm] = scored
        summary["arms"][arm] = summarize(scored, recs)

    if len(per_arm) == 2:
        a = {r["id"]: r["correct"] for r in per_arm["baseline"]}
        b = {r["id"]: r["correct"] for r in per_arm["agent"]}
        only_base = sorted(i for i in a if a[i] and not b[i])
        only_agent = sorted(i for i in a if b[i] and not a[i])
        summary["paired"] = {
            "both_correct": sum(a[i] and b[i] for i in a),
            "both_wrong": sum(not a[i] and not b[i] for i in a),
            "only_baseline_correct": only_base,
            "only_agent_correct": only_agent,
            "mcnemar_exact_p": round(mcnemar_exact(len(only_base), len(only_agent)), 5),
        }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (run_dir / "per_question.json").write_text(
        json.dumps({arm: rows for arm, rows in per_arm.items()}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
