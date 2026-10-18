"""第二轮离线判分：读取 eval/runs/<run>/<arm>.jsonl 与冻结金标，按 matcher_r2 判分，输出汇总。

用法：
  uv run python eval/r2/score.py --runs r2run1,r2run2,r2run3 [--set test]
输出：eval/runs/r2_summary_<set>.json（逐次、合并、配对检验）与 eval/runs/<run>/r2_per_question.json。
不调用模型。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from geo_agent.config import get_settings
from geo_agent.matcher import values_equal
from geo_agent.matcher_r2 import score_item
from geo_agent.stats import mcnemar_exact, wilson
from geo_agent.units import lint_units, uses_measure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ("baseline", "agent", "ablation")


def load_jsonl(p: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["id"]] = r
    return recs


def power_of_ten_off(pred: Any, gold: Any) -> bool:
    try:
        p, g = float(pred), float(gold)
    except (TypeError, ValueError):
        return False
    if p == 0 or g == 0 or math.isclose(p, g, rel_tol=0.005):
        return False
    r = abs(p / g)
    e = round(math.log10(r))
    return e != 0 and math.isclose(r, 10.0**e, rel_tol=0.02)


def failure_type(rec: dict | None, correct: bool, answerable: bool) -> str:
    if correct:
        return "ok"
    if rec is None:
        return "missing"
    st = rec.get("status")
    if not answerable:
        return "answered_unanswerable"
    if st == "refused":
        return "false_refusal"
    if st in ("bad_output", "invalid_sql", "exec_error", "no_answer", "error"):
        err = str(rec.get("error") or "")
        if "statement timeout" in err or "canceling statement" in err:
            return "timeout"
        return st
    return "wrong_result"


def score_one(rec: dict | None, gold: dict, gold_sql: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {"id": gold["id"], "tier": gold["tier"]}
    if rec is None:
        out.update(correct=False, reason="missing", status="missing", failure="missing")
        return out
    out["status"] = rec.get("status")
    if not gold["answerable"]:
        out["correct"] = rec.get("status") == "refused"
        out["reason"] = "refused" if out["correct"] else f"should refuse, got {rec.get('status')}"
    elif rec.get("status") != "answered" or "final_columns" not in rec:
        out.update(
            correct=False,
            reason=f"no result ({rec.get('status')}): {str(rec.get('error') or rec.get('refuse_reason') or '')[:200]}",
        )
    else:
        m = score_item(gold, rec["final_columns"], rec["final_rows"])
        out.update(correct=m.correct, reason=m.reason)
    if gold_sql and uses_measure(gold_sql) and gold["tier"] in ("crs", "join", "tag"):
        out["unit_scope"] = True
        final_sql = rec.get("final_sql") or rec.get("sql") or ""
        static = bool(final_sql) and bool(lint_units(final_sql))
        scale = False
        if not out["correct"] and rec.get("final_rows") and gold.get("rows"):
            pred = [
                v for v in rec["final_rows"][0] if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            for gv in gold["rows"][0]:
                if isinstance(gv, (int, float)) and not isinstance(gv, bool):
                    if not any(values_equal(v, gv) for v in pred) and any(
                        power_of_ten_off(v, gv) for v in pred
                    ):
                        scale = True
        out["unit_error"] = static or scale
    out["failure"] = failure_type(rec, out["correct"], gold["answerable"])
    return out


def prop(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


def summarize(scored: list[dict], recs: dict[str, dict]) -> dict[str, Any]:
    s = get_settings()
    tiers = sorted({r["tier"] for r in scored})
    raw = [x for r in recs.values() for x in (r.get("raw_sqls") or [])]
    pin = sum(r.get("prompt_tokens", 0) for r in recs.values())
    pout = sum(r.get("completion_tokens", 0) for r in recs.values())
    lat = sorted(r.get("latency_s", 0.0) for r in recs.values())
    calls = [r.get("model_calls", 0) for r in recs.values()]
    cost = pin / 1e6 * s.price_in_per_m + pout / 1e6 * s.price_out_per_m
    fails: dict[str, int] = {}
    for r in scored:
        if not r["correct"]:
            fails[r["failure"]] = fails.get(r["failure"], 0) + 1
    unit = [r for r in scored if r.get("unit_scope")]
    return {
        "overall": prop(sum(r["correct"] for r in scored), len(scored)),
        "by_tier": {
            t: prop(sum(r["correct"] for r in scored if r["tier"] == t), sum(r["tier"] == t for r in scored))
            for t in tiers
        },
        "valid_sql_rate": prop(sum(1 for x in raw if x.get("valid")), len(raw)),
        "unit_error_rate": prop(sum(r.get("unit_error", False) for r in unit), len(unit)),
        "failures": dict(sorted(fails.items(), key=lambda kv: -kv[1])),
        "calls_per_question": round(sum(calls) / max(len(calls), 1), 2),
        "tokens_in": pin,
        "tokens_out": pout,
        "cost_usd": round(cost, 4),
        "cost_per_question_usd": round(cost / max(len(recs), 1), 5),
        "latency_s_median": lat[len(lat) // 2] if lat else None,
        "latency_s_p90": lat[int(0.9 * (len(lat) - 1))] if lat else None,
        "reruns": sorted(r["id"] for r in recs.values() if r.get("rerun")),
    }


def paired(a: dict[str, bool], b: dict[str, bool]) -> dict[str, Any]:
    only_a = sorted(i for i in a if a[i] and not b.get(i, False))
    only_b = sorted(i for i in a if b.get(i, False) and not a[i])
    return {
        "both_correct": sum(a[i] and b.get(i, False) for i in a),
        "both_wrong": sum(not a[i] and not b.get(i, False) for i in a),
        "only_first": only_a,
        "only_second": only_b,
        "mcnemar_exact_p": round(mcnemar_exact(len(only_a), len(only_b)), 5),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--set", choices=["dev", "test"], default="test")
    ap.add_argument("--root", default=str(ROOT / "eval" / "runs"))
    args = ap.parse_args()
    gold = json.loads(
        (HERE / ("gold.json" if args.set == "test" else "dev_gold.json")).read_text(encoding="utf-8")
    )
    qs = json.loads(
        (HERE / ("questions.json" if args.set == "test" else "dev_questions.json")).read_text(
            encoding="utf-8"
        )
    )
    gold_sql = {q["id"]: q["gold_sql"] for q in qs}
    runs = args.runs.split(",")
    root = Path(args.root)
    per: dict[str, dict[str, list[dict]]] = {}
    summary: dict[str, Any] = {
        "set": args.set,
        "gold_frozen_at": gold["frozen_at"],
        "questions_sha256": gold["questions_sha256"],
        "runs": {},
    }
    for run in runs:
        summary["runs"][run] = {}
        per[run] = {}
        for arm in ARMS:
            recs = load_jsonl(root / run / f"{arm}.jsonl")
            if not recs:
                continue
            scored = [score_one(recs.get(g["id"]), g, gold_sql[g["id"]]) for g in gold["items"]]
            per[run][arm] = scored
            summary["runs"][run][arm] = summarize(scored, recs)
        (root / run / "r2_per_question.json").write_text(
            json.dumps(per[run], ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )

    # 合并：各臂 3 次运行的均值、逐次正确率、合并区间、按层均值；题级一致率
    pooled: dict[str, Any] = {}
    for arm in ARMS:
        rs = [r for r in runs if arm in per[r]]
        if not rs:
            continue
        accs = [summary["runs"][r][arm]["overall"]["p"] for r in rs]
        k = sum(summary["runs"][r][arm]["overall"]["k"] for r in rs)
        n = sum(summary["runs"][r][arm]["overall"]["n"] for r in rs)
        ids = [g["id"] for g in gold["items"]]
        agree = sum(
            len({next(x["correct"] for x in per[r][arm] if x["id"] == i) for r in rs}) == 1 for i in ids
        )
        tiers = sorted({g["tier"] for g in gold["items"]})
        pooled[arm] = {
            "runs": rs,
            "per_run_accuracy": accs,
            "mean_accuracy": round(statistics.mean(accs), 4),
            "sd_accuracy": round(statistics.stdev(accs), 4) if len(accs) > 1 else 0.0,
            "pooled": prop(k, n),
            "by_tier_mean": {
                t: round(statistics.mean(summary["runs"][r][arm]["by_tier"][t]["p"] for r in rs), 4)
                for t in tiers
            },
            "question_level_agreement": prop(agree, len(ids)),
            "cost_usd_total": round(sum(summary["runs"][r][arm]["cost_usd"] for r in rs), 4),
        }
    summary["pooled"] = pooled
    first = runs[0]
    pairs = {}
    for a, b in (("baseline", "agent"), ("ablation", "agent"), ("baseline", "ablation")):
        if a in per[first] and b in per[first]:
            pairs[f"{a}_vs_{b}"] = paired(
                {x["id"]: x["correct"] for x in per[first][a]}, {x["id"]: x["correct"] for x in per[first][b]}
            )
    summary["paired_run1"] = pairs
    name = f"r2_summary_{args.set}.json"
    (root / name).write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pooled": pooled, "paired_run1": pairs}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
