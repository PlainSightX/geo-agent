"""事后失败分类（非预注册）：对 Agent / 消融臂的每道错题给一个主要原因，Qwen 本地运行与 DeepSeek r2run1..3 用同一规则。

规则按顺序取第一个命中的类别（每题次只计一类）：
  text_tool_call     no_answer 且“模型调用次数 − 实际执行的工具步数 ≥ 3”（多数轮次没有结构化 tool_call，见 diagnose_text_toolcalls.py）
  tool_loop          no_answer 且同一 (工具, 参数) 重复 ≥ 3 次
  steps_exhausted    其余 no_answer
  count_misuse       final_answer 用 result_as='count'，而被引用的结果本身只有 1 行（聚合值/单行结果被再数一遍，得 n=1）
  name_resolution    拒答理由或最后的校验问题是“找不到实体”（not found / 无法找到 / 未找到 / 不在）
  false_refusal      其余可回答题拒答
  answered_unanswerable  不可回答题给出了结果
  wrong_result       其余（选错工具/参数、口径、列名不符等）
同时统计每题次的工具参数校验错误（pydantic/图层/度量类型不符）次数。
用法：uv run python eval/local_llm/failure_taxonomy.py  → eval/runs/local/failure_taxonomy.json
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOT_FOUND = re.compile(r"not found|无法找到|未找到|没有找到|不在|找不到|未在数据中")
ARG_ERR = re.compile(
    r"bad arguments|must be one of|needs a (polygon|line)|supports point layers|unknown tool"
)


def classify(rec: dict, scored: dict) -> str:
    st, steps = rec.get("status"), rec.get("steps") or []
    if st == "no_answer":
        if rec.get("model_calls", 0) - len(steps) >= 3:
            return "text_tool_call"
        sig = collections.Counter(
            json.dumps([s["tool"], s["args"]], sort_keys=True, ensure_ascii=False) for s in steps
        )
        return "tool_loop" if sig and max(sig.values()) >= 3 else "steps_exhausted"
    if scored["failure"] == "answered_unanswerable":
        return "answered_unanswerable"
    fin = next((s for s in reversed(steps) if s["tool"] == "final_answer"), None)
    if st == "answered" and fin and fin["args"].get("result_as") == "count":
        ref = fin["args"].get("result_ref")
        src = next((s for s in steps if s.get("ref") == ref), None)
        if src is not None and src.get("rows") == 1:
            return "count_misuse"
    if st == "refused":
        probs = " ".join(p for s in steps for p in (s.get("problems") or []))
        if NOT_FOUND.search(str(rec.get("refuse_reason") or "")) or (steps and NOT_FOUND.search(probs)):
            return "name_resolution"
        return "false_refusal"
    return "wrong_result"


def analyse(root: Path, runs: list[str], arm: str) -> dict:
    cats: collections.Counter = collections.Counter()
    arg_errs = calls = repairs = n = 0
    for run in runs:
        per = json.loads((root / run / "r2_per_question.json").read_text("utf-8"))
        scored = {x["id"]: x for x in per[arm]}
        for line in (root / run / f"{arm}.jsonl").read_text("utf-8").splitlines():
            rec = json.loads(line)
            n += 1
            calls += rec.get("model_calls", 0)
            repairs += rec.get("repairs_used", 0)
            arg_errs += sum(
                1 for s in rec.get("steps") or [] for p in (s.get("problems") or []) if ARG_ERR.search(p)
            )
            if not scored[rec["id"]]["correct"]:
                cats[classify(rec, scored[rec["id"]])] += 1
    return {
        "question_runs": n,
        "wrong": sum(cats.values()),
        "by_cause": dict(cats.most_common()),
        "tool_argument_errors": arg_errs,
        "tool_argument_errors_per_question": round(arg_errs / n, 2),
        "calls_per_question": round(calls / n, 2),
        "repairs_per_question": round(repairs / n, 2),
    }


def main() -> int:
    out = {}
    local = ROOT / "eval" / "runs" / "local"
    ds = ROOT / "eval" / "runs"
    qr = [f"local_qwen3-8b_run{i}" for i in (1, 2, 3)]
    dr = [f"r2run{i}" for i in (1, 2, 3)]
    for arm in ("agent", "ablation"):
        out[f"qwen3-8b:{arm}"] = analyse(local, qr, arm)
        out[f"deepseek:{arm}"] = analyse(ds, dr, arm)
    for extra in ("local_qwen3-8b_think_run1", "diag_qwen3-8b_salvage"):
        if (local / extra / "r2_per_question.json").exists():
            per = json.loads((local / extra / "r2_per_question.json").read_text("utf-8"))
            for arm in per:
                if arm != "baseline":
                    out[f"{extra}:{arm}"] = analyse(local, [extra], arm)
    p = local / "failure_taxonomy.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
