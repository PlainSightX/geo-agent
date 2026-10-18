"""微调评测（预注册附录见 eval/ft/PREREGISTRATION_FT.md）：同一 vLLM 端点上的基座 Qwen3-8B（served name qwen3-8b）
与 LoRA（--lora-modules geo=...），在 heldout（eval/ft/heldout，训练前冻结）与 r2 冻结集（eval/r2，72 题）上各跑 3 次 Agent 臂。

与上一轮本地评测相同：同一 R2Agent、提示词、工具定义、校验器；温度 0；非思考模式（run_vllm_arms.LocalLLM）。
额外记录每次模型调用是否带结构化 tool_calls（eval/runs/ft/calls.jsonl），用于“纯文本工具调用率”。
  uv run python eval/ft/run_eval_ft.py --base-url http://127.0.0.1:PORT/v1 --models qwen3-8b,geo --sets heldout,r2 --runs 3
输出：eval/runs/ft/<set>/<base|lora>_run<i>/agent.jsonl（与 eval/r2/run_eval.py 同一记录格式；断点续跑）。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eval" / "local_llm"))
import run_vllm_arms as rva  # noqa: E402

from geo_agent.db import connect  # noqa: E402
from geo_agent.r2_agent import R2Agent  # noqa: E402
from geo_agent.r2_schema import describe_lean_schema  # noqa: E402

OUT = ROOT / "eval" / "runs" / "ft"
rva.LOCAL_LEDGER = OUT / "ledger_ft.jsonl"
CALLS = OUT / "calls.jsonl"
SETS = {"heldout": HERE / "heldout" / "questions.json", "r2": ROOT / "eval" / "r2" / "questions.json"}
ALIAS = {"qwen3-8b": "base", "geo": "lora"}
_lock = threading.Lock()


class TracedLLM(rva.LocalLLM):
    """LocalLLM + 每次调用记录：是否返回结构化 tool_calls、文本长度。"""

    def chat(self, messages, **kw):
        msg, usage = super().chat(messages, **kw)
        rec = {"tag": self.tag, "qid": kw.get("qid", ""), "tool_calls": len(msg.tool_calls or []),
               "content_chars": len(msg.content or "")}  # fmt: skip
        with _lock, CALLS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return msg, usage


def run_job(
    base_url: str, model: str, set_name: str, run: int, workers: int, schema: str, ablation: bool
) -> str:
    qs = json.loads(SETS[set_name].read_text(encoding="utf-8"))
    arm = "ablation" if ablation else "agent"
    name = f"{ALIAS.get(model, model)}_run{run}"
    out = OUT / set_name / name / f"{arm}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines() if x.strip()}
    tag = f"ft:{set_name}:{name}:{arm}"
    agent = R2Agent(schema_text=schema, tag=tag, ablation=ablation,
                    llm=TracedLLM(base_url=base_url, model=model, tag=tag))  # fmt: skip

    def one(q: dict) -> dict:
        with connect() as conn:
            t0 = time.time()
            try:
                rec = agent.run(q["question"], qid=q["id"], conn=conn).to_dict()
            except Exception as err:  # 单题异常记为 error，不中断整批（与 run_eval.py 相同）
                rec = {"question": q["question"], "status": "error", "error": f"{type(err).__name__}: {err}"}
            rec.update(id=q["id"], tier=q["tier"], arm=arm, started=t0, rerun=False)
            return rec

    todo = [q for q in qs if q["id"] not in done]
    with ThreadPoolExecutor(workers) as pool, out.open("a", encoding="utf-8") as f:
        for fut in as_completed([pool.submit(one, q) for q in todo]):
            rec = fut.result()
            with _lock:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                f.flush()
    return f"{set_name}/{name}/{arm}: {len(todo)} run, {len(done)} already done"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--models", default="qwen3-8b,geo")
    ap.add_argument("--sets", default="heldout,r2")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2, help="threads per job; all jobs run concurrently")
    ap.add_argument("--ablation", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    schema = describe_lean_schema()
    jobs = [
        (m, s, r)
        for s in args.sets.split(",")
        for m in args.models.split(",")
        for r in range(1, args.runs + 1)
    ]
    with ThreadPoolExecutor(len(jobs)) as pool:
        futs = [
            pool.submit(run_job, args.base_url, m, s, r, args.workers, schema, args.ablation)
            for m, s, r in jobs
        ]
        for fut in as_completed(futs):
            print(fut.result(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
