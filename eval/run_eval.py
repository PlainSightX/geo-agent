"""运行评测臂：对题集逐题调用基线或 Agent，把原始输出写入 eval/runs/<run>/<arm>.jsonl。

用法：
  uv run python eval/run_eval.py --arm baseline --set dev  --run dev1
  uv run python eval/run_eval.py --arm agent    --set test --run test1
  --only A01,B02   只跑指定题（用于基础设施故障后补跑，补跑会在输出中标记 rerun=true）

已完成的题会跳过（断点续跑）；输出只追加不覆盖。判分在 score.py 中离线完成，不需要 API。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from geo_agent.agent import GeoAgent
from geo_agent.baseline import SingleShotBaseline
from geo_agent.db import connect, describe_schema
from geo_agent.llm import BudgetExceeded, ledger_total

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "agent"], required=True)
    ap.add_argument("--set", choices=["dev", "test"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    qfile = ROOT / "eval" / ("questions.json" if args.set == "test" else "dev_questions.json")
    questions = json.loads(qfile.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    out_dir = ROOT / "eval" / "runs" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.arm}.jsonl"
    done = set()
    if out.exists() and not args.only:
        done = {
            json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
        }
    todo = [q for q in questions if q["id"] not in done]
    print(
        f"{args.arm}/{args.set}: {len(todo)} to run, {len(done)} already done; ledger ${ledger_total():.4f}"
    )

    schema = describe_schema()
    tag = f"{args.run}:{args.arm}"
    arm = (
        GeoAgent(schema_text=schema, tag=tag)
        if args.arm == "agent"
        else SingleShotBaseline(schema_text=schema, tag=tag)
    )

    def one(q: dict) -> dict:
        # 每个线程独立连接：psycopg 连接不能跨线程共享
        with connect() as conn:
            t0 = time.time()
            try:
                r = arm.run(q["question"], qid=q["id"], conn=conn)
                rec = r.to_dict()
            except BudgetExceeded:
                raise
            except Exception as err:  # 记录为 error，不让单题失败中断整批
                rec = {"question": q["question"], "status": "error", "error": f"{type(err).__name__}: {err}"}
            rec.update(id=q["id"], tier=q["tier"], arm=args.arm, started=t0, rerun=bool(args.only))
            return rec

    n_ok = 0
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
            n_ok += 1
            print(f"[{n_ok}/{len(todo)}] {rec['id']} {rec['status']}")
    print(f"done; ledger ${ledger_total():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
