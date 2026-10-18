"""运行第二轮评测臂：逐题调用，原始输出写入 eval/runs/<run>/<arm>.jsonl（只追加；断点续跑）。

用法：
  uv run python eval/r2/run_eval.py --arm baseline --set dev  --run r2dev1
  uv run python eval/r2/run_eval.py --arm agent    --set test --run r2run1
  uv run python eval/r2/run_eval.py --arm ablation --set test --run r2run1
  --only D01,E03   只补跑指定题（基础设施故障后），记录 rerun=true

arm: baseline = (a) 单次 text-to-SQL；agent = (b) 全工具 Agent；ablation = (c) 去掉路网/RAG/合规工具的 Agent。
--llm-url / --llm-model：改为任意 OpenAI 兼容端点（本地 vLLM 等，见 eval/local_llm/），不写 DeepSeek 台账。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 第二轮台账停止线 $20（用户确认费用不是约束）；必须在导入 geo_agent.llm 之前设置
os.environ.setdefault("GEO_AGENT_BUDGET_STOP_USD", "20")

from geo_agent.db import connect  # noqa: E402
from geo_agent.llm import BudgetExceeded, ledger_total  # noqa: E402
from geo_agent.r2_agent import R2Agent  # noqa: E402
from geo_agent.r2_baseline import R2Baseline  # noqa: E402
from geo_agent.r2_schema import describe_lean_schema  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build_arm(arm: str, schema: str, tag: str, llm=None):
    if arm == "baseline":
        return R2Baseline(schema_text=schema, tag=tag, llm=llm)
    return R2Agent(schema_text=schema, tag=tag, ablation=(arm == "ablation"), llm=llm)


def main(argv: list[str] | None = None, llm_factory=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "agent", "ablation"], required=True)
    ap.add_argument("--set", choices=["dev", "test"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-root", default=str(ROOT / "eval" / "runs"))
    args = ap.parse_args(argv)

    qfile = HERE / ("questions.json" if args.set == "test" else "dev_questions.json")
    questions = json.loads(qfile.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    out_dir = Path(args.out_root) / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.arm}.jsonl"
    done = set()
    if out.exists() and not args.only:
        done = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines() if x.strip()}
    todo = [q for q in questions if q["id"] not in done]
    print(f"{args.arm}/{args.set}: {len(todo)} to run, {len(done)} done; ledger ${ledger_total():.4f}")

    schema = describe_lean_schema()
    tag = f"{args.run}:{args.arm}"
    llm = llm_factory(tag) if llm_factory else None
    arm = build_arm(args.arm, schema, tag, llm)

    def one(q: dict) -> dict:
        with connect() as conn:
            t0 = time.time()
            try:
                rec = arm.run(q["question"], qid=q["id"], conn=conn).to_dict()
            except BudgetExceeded:
                raise
            except Exception as err:  # 单题异常记为 error，不中断整批
                rec = {"question": q["question"], "status": "error", "error": f"{type(err).__name__}: {err}"}
            rec.update(id=q["id"], tier=q["tier"], arm=args.arm, started=t0, rerun=bool(args.only))
            return rec

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out.open("a", encoding="utf-8") as f:
        futs = {pool.submit(one, q): q["id"] for q in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except BudgetExceeded as err:
                print(f"STOP: {err}", file=sys.stderr)
                pool.shutdown(cancel_futures=True)
                return 2
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            n += 1
            print(f"[{n}/{len(todo)}] {rec['id']} {rec['status']}")
    print(f"done; ledger ${ledger_total():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
