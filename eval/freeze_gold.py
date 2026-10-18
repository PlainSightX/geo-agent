"""冻结金标：执行每道题的金标 SQL，做边界裕度检查，写出 eval/gold.json。

必须在任何模型调用之前运行并提交到 git；之后题目或金标若有改动，需在报告里披露。

边界裕度检查（防止“合理的不同算法”翻转金标）：
- radius_check：所有候选对象的距离/面积值都必须与阈值相差 ≥ 0.5%（geography 与 UTM/球面算法差异 < 0.5%）；
- topk_check：第 k 名与第 k+1 名的距离必须相差 ≥ 0.5%，否则 LIMIT k 的结果不唯一。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from geo_agent.db import connect, run_select
from geo_agent.matcher import canonical_hash

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 0.005


def check_radius(conn, spec: dict) -> list[str]:
    res = run_select(spec["sql"], conn=conn, row_limit=100000)
    thr = float(spec["threshold"])
    close = [r[0] for r in res.rows if r[0] is not None and abs(float(r[0]) - thr) < MARGIN * thr]
    return [f"value {v:.3f} within {MARGIN:.1%} of threshold {thr}" for v in close]


def check_topk(conn, spec: dict) -> list[str]:
    res = run_select(spec["sql"], conn=conn, row_limit=100000)
    k = int(spec["k"])
    vals = [float(r[0]) for r in res.rows]
    if len(vals) <= k:
        return []
    a, b = vals[k - 1], vals[k]
    if b - a < MARGIN * max(a, 1.0):
        return [f"rank {k} ({a:.3f}) and rank {k + 1} ({b:.3f}) differ by < {MARGIN:.1%}"]
    return []


def data_fingerprint(conn) -> dict:
    res = run_select("SELECT table_name, n FROM dataset_summary ORDER BY table_name", conn=conn, guard=False)
    ver = run_select("SELECT postgis_lib_version() AS v", conn=conn, guard=False)
    source = json.loads((ROOT / "data" / "SOURCE.json").read_text(encoding="utf-8"))
    return {
        "table_counts": {r[0]: r[1] for r in res.rows},
        "postgis_version": ver.rows[0][0],
        "osm_snapshot": source["file"],
        "osm_snapshot_sha256": source["sha256"],
    }


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_only = "--check" in sys.argv
    qpath = ROOT / "eval" / args[0] if args else ROOT / "eval" / "questions.json"
    out = qpath.with_name(qpath.stem.replace("questions", "gold") + ".json")
    questions = json.loads(qpath.read_text(encoding="utf-8"))
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    problems: dict[str, list[str]] = {}
    items = []
    with connect() as conn:
        for q in questions:
            issues: list[str] = []
            entry = {"id": q["id"], "tier": q["tier"], "answerable": q["gold_sql"] is not None}
            if q["gold_sql"] is not None:
                # 金标 SQL 也必须通过守卫：守卫若误杀合法查询，会在这里暴露
                res = run_select(q["gold_sql"], conn=conn, row_limit=1000)
                if res.truncated:
                    issues.append("gold result truncated")
                if not res.rows:
                    issues.append("gold result empty")
                elif len(res.rows) == 1 and len(res.columns) == 1 and res.rows[0][0] in (0, None):
                    issues.append("gold scalar is 0/NULL (weak signal)")
                for spec in q.get("radius_check") and (
                    q["radius_check"] if isinstance(q["radius_check"], list) else [q["radius_check"]]
                ) or []:
                    issues += check_radius(conn, spec)
                if q.get("topk_check"):
                    issues += check_topk(conn, q["topk_check"])
                entry.update(
                    columns=res.columns,
                    rows=res.rows,
                    order_matters=bool(q.get("order_matters")),
                    result_sha256=canonical_hash(res.columns, res.rows),
                )
            if issues:
                problems[q["id"]] = issues
            items.append(entry)
        fingerprint = data_fingerprint(conn)

    if problems:
        for qid, issues in problems.items():
            for i in issues:
                print(f"[{qid}] {i}")
        print(f"{len(problems)} questions failed checks; fix them before freezing")
        return 1

    if check_only:
        for e in items:
            if e["answerable"]:
                print(e["id"], e["columns"], e["rows"][:4], f"({len(e['rows'])} rows)")
        print("all checks passed (check only, nothing written)")
        return 0
    qbytes = qpath.read_bytes()
    gold = {
        "frozen_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "questions_file": qpath.name,
        "questions_sha256": hashlib.sha256(qbytes).hexdigest(),
        "data": fingerprint,
        "items": items,
    }
    gold["items_sha256"] = hashlib.sha256(
        json.dumps(items, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    out.write_text(json.dumps(gold, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    by_tier: dict[str, int] = {}
    for q in questions:
        by_tier[q["tier"]] = by_tier.get(q["tier"], 0) + 1
    print(f"frozen {len(items)} questions -> {out.name} {by_tier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
