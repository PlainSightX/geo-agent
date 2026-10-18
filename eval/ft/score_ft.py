"""微调评测判分与汇总（判定规则见 eval/ft/PREREGISTRATION_FT.md）。不调用模型。

  uv run python eval/ft/score_ft.py  → eval/runs/ft/summary.json（并打印主表）
各 run 目录写 per_question.json；DeepSeek 对照读 eval/runs/r2run{1,2,3}/r2_per_question.json（已判分）。
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eval" / "r2"))
from score import score_one  # noqa: E402  与 r2 完全相同的逐题判分

from geo_agent.stats import mcnemar_exact, wilson  # noqa: E402

OUT = ROOT / "eval" / "runs" / "ft"
GOLD = {"heldout": HERE / "heldout" / "gold.json", "r2": ROOT / "eval" / "r2" / "gold.json"}
QS = {"heldout": HERE / "heldout" / "questions.json", "r2": ROOT / "eval" / "r2" / "questions.json"}
ARG_ERR = re.compile(
    r"bad arguments|must be one of|needs a (polygon|line)|supports point layers|unknown tool"
)


def prop(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "p": round(k / n, 4) if n else None,
        "ci95": [round(lo * 100, 1), round(hi * 100, 1)],
    }


def pair(a: dict[str, bool], b: dict[str, bool]) -> dict:
    """a = 第一臂，b = 第二臂；only_second = 仅第二臂答对。"""
    oa = sorted(i for i in a if a[i] and not b[i])
    ob = sorted(i for i in a if b[i] and not a[i])
    return {"only_first": len(oa), "only_second": len(ob), "only_first_ids": oa, "only_second_ids": ob,
            "mcnemar_exact_p": float(f"{mcnemar_exact(len(oa), len(ob)):.3g}")}  # fmt: skip


def load(p: Path) -> dict[str, dict]:
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def text_rate(set_name: str, model_alias: str) -> dict:
    """纯文本工具调用率：没有结构化 tool_calls 的模型调用占比（calls.jsonl）。"""
    n = t = 0
    f = OUT / "calls.jsonl"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            parts = r["tag"].split(":")
            if (
                len(parts) >= 4
                and parts[1] == set_name
                and parts[2].startswith(model_alias + "_")
                and parts[3] == "agent"
            ):
                n += 1
                t += r["tool_calls"] == 0
    return {"calls": n, "text_only": t, "rate": round(t / n, 4) if n else None}


def costs(recs: list[dict]) -> dict:
    lat = sorted(r.get("latency_s", 0.0) for r in recs)
    n = max(len(recs), 1)
    argerr = sum(
        1 for r in recs for s in r.get("steps") or [] for p in s.get("problems") or [] if ARG_ERR.search(p)
    )
    return {"calls_per_q": round(sum(r.get("model_calls", 0) for r in recs) / n, 2),
            "repairs_per_q": round(sum(r.get("repairs_used", 0) for r in recs) / n, 2),
            "arg_errors_per_q": round(argerr / n, 3), "no_answer": sum(r.get("status") == "no_answer" for r in recs),
            "refused": sum(r.get("status") == "refused" for r in recs), "errors": sum(r.get("status") == "error" for r in recs),
            "latency_median_s": lat[len(lat) // 2] if lat else None,
            "latency_p90_s": lat[int(0.9 * (len(lat) - 1))] if lat else None,
            "prompt_tokens_per_q": round(sum(r.get("prompt_tokens", 0) for r in recs) / n)}  # fmt: skip


def score_set(set_name: str, arm: str = "agent") -> dict:
    gold = json.loads(GOLD[set_name].read_text(encoding="utf-8"))
    gsql = {q["id"]: q["gold_sql"] for q in json.loads(QS[set_name].read_text(encoding="utf-8"))}
    ids = [g["id"] for g in gold["items"]]
    tiers = {g["id"]: g["tier"] for g in gold["items"]}
    systems: dict[str, list[dict[str, bool]]] = {}
    out: dict = {
        "gold_questions_sha256": gold["questions_sha256"],
        "systems": {},
        "paired_run1": {},
        "paired_majority": {},
    }
    for alias in ("base", "lora"):
        per_run, recs_all = [], []
        for run_dir in sorted((OUT / set_name).glob(f"{alias}_run*")):
            recs = load(run_dir / f"{arm}.jsonl")
            if len(recs) < len(ids):
                continue
            scored = [score_one(recs.get(g["id"]), g, gsql[g["id"]]) for g in gold["items"]]
            (run_dir / "per_question.json").write_text(
                json.dumps(scored, ensure_ascii=False, indent=1) + "\n", "utf-8"
            )
            per_run.append({x["id"]: x["correct"] for x in scored})
            recs_all.append(list(recs.values()))
        if per_run:
            systems[alias] = per_run
            out["systems"][alias] = summarize(per_run, recs_all, ids, tiers) | {
                "text_tool_calls": text_rate(set_name, alias)
            }
    if set_name == "r2":
        ds = []
        for i in (1, 2, 3):
            per = json.loads(
                (ROOT / "eval" / "runs" / f"r2run{i}" / "r2_per_question.json").read_text("utf-8")
            )["agent"]
            ds.append({x["id"]: x["correct"] for x in per})
        recs = [list(load(ROOT / "eval" / "runs" / f"r2run{i}" / "agent.jsonl").values()) for i in (1, 2, 3)]
        systems["deepseek"] = ds
        out["systems"]["deepseek"] = summarize(ds, recs, ids, tiers)
    for a, b in (("base", "lora"), ("deepseek", "lora"), ("deepseek", "base")):
        if a in systems and b in systems:
            out["paired_run1"][f"{a}_vs_{b}"] = pair(systems[a][0], systems[b][0])
            maj = [{i: sum(c[i] for c in systems[s]) * 2 > len(systems[s]) for i in ids} for s in (a, b)]
            out["paired_majority"][f"{a}_vs_{b}"] = pair(*maj)
    return out


def summarize(
    per_run: list[dict[str, bool]], recs_all: list[list[dict]], ids: list[str], tiers: dict[str, str]
) -> dict:
    k = sum(sum(c.values()) for c in per_run)
    by_tier = {}
    for t in sorted(set(tiers.values())):
        by_tier[t] = prop(
            sum(c[i] for c in per_run for i in ids if tiers[i] == t),
            sum(tiers[i] == t for i in ids) * len(per_run),
        )
    agree = sum(len({c[i] for c in per_run}) == 1 for i in ids)
    return {"runs": len(per_run), "run1": prop(sum(per_run[0].values()), len(ids)),
            "per_run": [sum(c.values()) for c in per_run], "pooled": prop(k, len(ids) * len(per_run)),
            "mean": round(statistics.mean(sum(c.values()) / len(ids) for c in per_run), 4),
            "agreement": round(agree / len(ids), 4), "by_tier_pooled": by_tier,
            "cost_run1": costs(recs_all[0]), "cost_all": costs([r for rs in recs_all for r in rs])}  # fmt: skip


def decide(h: dict, r: dict) -> dict:
    """预注册判定：heldout run1 配对显著（LoRA 更好）且 r2 run1 LoRA 正确数 ≥ 基座 → 有效。"""
    hp = h["paired_run1"].get("base_vs_lora")
    if not hp or "lora" not in r["systems"] or "base" not in r["systems"]:
        return {"verdict": "incomplete"}
    sig = hp["only_second"] > hp["only_first"] and hp["mcnemar_exact_p"] < 0.05
    r2_ok = r["systems"]["lora"]["run1"]["k"] >= r["systems"]["base"]["run1"]["k"]
    verdict = "微调有效" if sig and r2_ok else "对模板分布有效、对未见题面不泛化" if sig else "未证明有效"
    return {
        "heldout_significant": sig,
        "heldout_p": hp["mcnemar_exact_p"],
        "r2_not_worse": r2_ok,
        "verdict": verdict,
    }


def main() -> int:
    arm = "ablation" if "--ablation" in sys.argv else "agent"
    res = {s: score_set(s, arm) for s in ("heldout", "r2")}
    res["decision"] = (
        decide(res["heldout"], res["r2"])
        if arm == "agent"
        else {"verdict": "ablation (not part of decision)"}
    )
    name = "summary.json" if arm == "agent" else "summary_ablation.json"
    (OUT / name).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for s in ("heldout", "r2"):
        for k, v in res[s]["systems"].items():
            c = v["cost_all"]
            print(s, k, "run1", v["run1"]["k"], "/", v["run1"]["n"], v["run1"]["ci95"], "per_run", v["per_run"],
                  "pooled", v["pooled"]["k"], v["pooled"]["ci95"], "text", (v.get("text_tool_calls") or {}).get("rate"),
                  "argerr/q", c["arg_errors_per_q"], "no_answer", c["no_answer"], "calls/q", c["calls_per_q"],
                  "lat", c["latency_median_s"], c["latency_p90_s"])  # fmt: skip
        for k, v in res[s]["paired_run1"].items():
            print(s, k, v["only_first"], "vs", v["only_second"], "p", v["mcnemar_exact_p"])
    print(json.dumps(res["decision"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
