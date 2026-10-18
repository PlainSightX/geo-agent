"""Qwen3-8B（本地 vLLM）与 DeepSeek（r2run1..3）在同一 72 题冻结集上的对比：逐臂正确率、按层、Wilson 区间、配对 McNemar、代价。

前提：两边都已用 eval/r2/score.py 判分（各 run 目录下有 r2_per_question.json）。
主分析与预注册一致用 run1 配对；另报告 3 次合并与“逐题多数票”（≥2/3 次答对）配对作为稳健性检查。
用法：uv run python eval/local_llm/compare_deepseek.py → eval/runs/local/compare_deepseek.json
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from geo_agent.stats import mcnemar_exact, wilson

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "eval" / "runs"
LOC = DS / "local"
SYSTEMS = {
    "deepseek": (DS, [f"r2run{i}" for i in (1, 2, 3)]),
    "qwen3-8b": (LOC, [f"local_qwen3-8b_run{i}" for i in (1, 2, 3)]),
    "qwen3-8b_think": (LOC, ["local_qwen3-8b_think_run1"]),
}


def load(root: Path, run: str, arm: str) -> tuple[dict[str, bool], dict[str, str], list[dict]]:
    p = root / run / "r2_per_question.json"
    if not p.exists():
        return {}, {}, []
    per = json.loads(p.read_text("utf-8")).get(arm, [])
    recs = []
    f = root / run / f"{arm}.jsonl"
    if f.exists():
        recs = [json.loads(x) for x in f.read_text("utf-8").splitlines() if x.strip()]
    return {x["id"]: x["correct"] for x in per}, {x["id"]: x["tier"] for x in per}, recs


def prop(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": round(k / n, 4), "ci95": [round(lo * 100, 1), round(hi * 100, 1)]}


def pair(a: dict[str, bool], b: dict[str, bool]) -> dict:
    oa = sorted(i for i in a if a[i] and not b[i])
    ob = sorted(i for i in a if b[i] and not a[i])
    return {
        "only_first": len(oa),
        "only_second": len(ob),
        "only_first_ids": oa,
        "only_second_ids": ob,
        "mcnemar_exact_p": float(f"{mcnemar_exact(len(oa), len(ob)):.3g}"),
    }


def cost(recs: list[dict]) -> dict:
    lat = sorted(r.get("latency_s", 0.0) for r in recs)
    n = max(len(recs), 1)
    return {
        "calls_per_q": round(sum(r.get("model_calls", 0) for r in recs) / n, 2),
        "repairs_per_q": round(sum(r.get("repairs_used", 0) for r in recs) / n, 2),
        "prompt_tokens_per_q": round(sum(r.get("prompt_tokens", 0) for r in recs) / n),
        "completion_tokens_per_q": round(sum(r.get("completion_tokens", 0) for r in recs) / n),
        "latency_median_s": lat[len(lat) // 2] if lat else None,
        "latency_p90_s": lat[int(0.9 * (len(lat) - 1))] if lat else None,
        "no_answer": sum(r.get("status") == "no_answer" for r in recs),
        "refused": sum(r.get("status") == "refused" for r in recs),
    }


def main() -> int:
    out: dict = {"systems": {}, "paired": {}}
    first: dict[tuple[str, str], dict[str, bool]] = {}
    major: dict[tuple[str, str], dict[str, bool]] = {}
    for sys_name, (root, runs) in SYSTEMS.items():
        for arm in ("baseline", "agent", "ablation"):
            per_run, tiers, recs_all = [], {}, []
            for run in runs:
                c, t, recs = load(root, run, arm)
                if c:
                    per_run.append(c)
                    tiers = t
                    recs_all.append(recs)
            if not per_run:
                continue
            ids = sorted(per_run[0])
            k = sum(sum(c.values()) for c in per_run)
            n = sum(len(c) for c in per_run)
            by_tier = {}
            for tier in sorted(set(tiers.values())):
                tk = sum(c[i] for c in per_run for i in ids if tiers[i] == tier)
                tn = sum(1 for c in per_run for i in ids if tiers[i] == tier)
                by_tier[tier] = prop(tk, tn)
            out["systems"][f"{sys_name}:{arm}"] = {
                "runs": len(per_run),
                "run1": prop(sum(per_run[0].values()), len(ids)),
                "per_run_accuracy": [round(sum(c.values()) / len(c), 4) for c in per_run],
                "mean": round(statistics.mean(sum(c.values()) / len(c) for c in per_run), 4),
                "pooled": prop(k, n),
                "by_tier_pooled": by_tier,
                "cost_run1": cost(recs_all[0]),
                "cost_all_runs": cost([r for rs in recs_all for r in rs]),
            }
            first[(sys_name, arm)] = per_run[0]
            major[(sys_name, arm)] = {i: sum(c[i] for c in per_run) * 2 > len(per_run) for i in ids}
    for a, b in [
        (("deepseek", "agent"), ("qwen3-8b", "agent")),
        (("qwen3-8b", "baseline"), ("qwen3-8b", "agent")),
        (("qwen3-8b", "ablation"), ("qwen3-8b", "agent")),
        (("deepseek", "baseline"), ("qwen3-8b", "baseline")),
        (("deepseek", "agent"), ("qwen3-8b_think", "agent")),
        (("qwen3-8b", "agent"), ("qwen3-8b_think", "agent")),
        (("qwen3-8b", "baseline"), ("qwen3-8b_think", "baseline")),
    ]:
        if a in first and b in first:
            key = f"{a[0]}:{a[1]} vs {b[0]}:{b[1]}"
            out["paired"][key] = {"run1": pair(first[a], first[b])}
            if a[0] != "qwen3-8b_think" and b[0] != "qwen3-8b_think":
                out["paired"][key]["majority_of_3"] = pair(major[a], major[b])
    p = LOC / "compare_deepseek.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for k, v in out["systems"].items():
        print(k, v["run1"], v["per_run_accuracy"], v["pooled"]["ci95"], v["cost_run1"])
    for k, v in out["paired"].items():
        print(k, {kk: (vv["only_first"], vv["only_second"], vv["mcnemar_exact_p"]) for kk, vv in v.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
