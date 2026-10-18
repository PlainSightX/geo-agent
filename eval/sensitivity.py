"""事后敏感性分析（非预注册，报告中单独标注）：基线的 JSON 解析是否不公平地拖累了基线？

test1 中基线有 3 题状态为 bad_output：模型先输出了合法的 {"sql": ...}，随后又追加了自我纠正的文字，
贪婪正则 + json.loads 因此解析失败。这里用 json.JSONDecoder.raw_decode 只取第一个 JSON 对象，
离线执行恢复出的 SQL（只读库，不调用模型），重新判分。主结果仍以预注册口径为准。
"""

from __future__ import annotations

import json
from pathlib import Path

from geo_agent.db import QueryError, connect, run_select
from geo_agent.matcher import compare
from geo_agent.sql_guard import SqlRejected
from geo_agent.stats import mcnemar_exact, wilson

ROOT = Path(__file__).resolve().parents[1]


def first_json(text: str) -> dict | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def main(run: str = "test1") -> int:
    gold = {
        g["id"]: g for g in json.loads((ROOT / "eval" / "gold.json").read_text(encoding="utf-8"))["items"]
    }
    per_q = json.loads((ROOT / "eval" / "runs" / run / "per_question.json").read_text(encoding="utf-8"))
    base = {r["id"]: r["correct"] for r in per_q["baseline"]}
    agent = {r["id"]: r["correct"] for r in per_q["agent"]}
    recs = [
        json.loads(line)
        for line in (ROOT / "eval" / "runs" / run / "baseline.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recovered = {}
    with connect() as conn:
        for r in recs:
            if r["status"] != "bad_output":
                continue
            obj = first_json(r.get("error") or "")
            g = gold[r["id"]]
            outcome = {"recovered_sql": None, "correct": False, "reason": "no JSON object"}
            if obj and isinstance(obj.get("sql"), str):
                outcome["recovered_sql"] = obj["sql"]
                try:
                    res = run_select(obj["sql"], conn=conn)
                    m = compare(
                        g["columns"],
                        g["rows"],
                        res.columns,
                        res.rows,
                        order_matters=g.get("order_matters", False),
                    )
                    outcome.update(correct=m.correct, reason=m.reason)
                except (SqlRejected, QueryError) as err:
                    outcome["reason"] = str(err)
            recovered[r["id"]] = outcome
            base[r["id"]] = outcome["correct"]
    k, n = sum(base.values()), len(base)
    # 原始输出在记录中被截断为 300 字符（日志缺陷，已修复），无法恢复的题给出“全部算对”的上界
    unrecoverable = sorted(i for i, o in recovered.items() if o["recovered_sql"] is None)
    upper_k = k + len(unrecoverable)
    only_b = sorted(i for i in base if base[i] and not agent[i])
    only_a = sorted(i for i in base if agent[i] and not base[i])
    lo, hi = wilson(k, n)
    out = {
        "note": "post-hoc, not pre-registered; baseline parser relaxed to first JSON object",
        "recovered": recovered,
        "baseline_overall": {"k": k, "n": n, "p": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)]},
        "unrecoverable_truncated_log": unrecoverable,
        "baseline_overall_upper_bound": {"k": upper_k, "n": n, "p": round(upper_k / n, 4)},
        "only_baseline_correct": only_b,
        "only_agent_correct": only_a,
        "mcnemar_exact_p": round(mcnemar_exact(len(only_b), len(only_a)), 5),
    }
    (ROOT / "eval" / "runs" / run / "sensitivity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
