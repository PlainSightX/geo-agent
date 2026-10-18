"""第二轮事后敏感性分析（非预注册，README 中单独标注；主结果仍以预注册口径为准）。

问题：Agent 在 T06/T07/T10 三题上 3 次全错，原因都是口径比金标更宽而不是写错：
- T06 外带：金标只数 takeaway=yes（164），Agent 把 takeaway=only 也算上（165）；
- T07 肯德基：金标只数 brand=肯德基（81），Agent 把 brand=KFC 也算上（83）；
- T10 蔚来：金标只数 brand=NIO Power（12），Agent 把 brand=蔚来换电站 也算上（13）。
若把这三个更宽的口径也视为正确，各臂、各次的正确率与 run1 配对检验如何变化？
"""

from __future__ import annotations

import json
from pathlib import Path

from geo_agent.stats import mcnemar_exact, wilson

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "eval" / "runs"
ALT = {"T06": 165, "T07": 83, "T10": 13}


def main() -> int:
    out: dict = {"alternative_counts": ALT, "runs": {}}
    run1: dict[str, dict[str, bool]] = {}
    for run in ("r2run1", "r2run2", "r2run3"):
        per = json.loads((RUNS / run / "r2_per_question.json").read_text(encoding="utf-8"))
        out["runs"][run] = {}
        for arm, rows in per.items():
            recs = {
                json.loads(x)["id"]: json.loads(x)
                for x in (RUNS / run / f"{arm}.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()
            }
            ok = {}
            for r in rows:
                c = r["correct"]
                rec = recs[r["id"]]
                if not c and r["id"] in ALT and rec.get("final_rows") and len(rec["final_rows"]) == 1:
                    c = ALT[r["id"]] in rec["final_rows"][0]
                ok[r["id"]] = c
            k = sum(ok.values())
            lo, hi = wilson(k, len(ok))
            out["runs"][run][arm] = {"k": k, "n": len(ok), "ci95": [round(lo, 4), round(hi, 4)]}
            if run == "r2run1":
                run1[arm] = ok
    b, a = run1["baseline"], run1["agent"]
    only_b = sum(b[i] and not a[i] for i in b)
    only_a = sum(a[i] and not b[i] for i in b)
    out["paired_run1_baseline_vs_agent"] = {
        "only_baseline": only_b,
        "only_agent": only_a,
        "mcnemar_exact_p": round(mcnemar_exact(only_b, only_a), 6),
    }
    (RUNS / "r2_sensitivity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
