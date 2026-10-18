"""冻结第二轮金标：执行 eval/r2/questions.json 的金标 SQL，做边界裕度检查，写出 eval/r2/gold.json。

与第一轮 eval/freeze_gold.py 相同的流程与哈希规则（questions 按 LF 规范化后 sha256；结果 canonical_hash），
区别：使用第二轮表白名单（r2=True）；边界裕度阈值按题型区分：
- 直线（geography）阈值题：候选值与阈值相差 ≥ 0.5%（与第一轮相同）；
- 路网阈值题（route/compliance）：候选设施网络距离与阈值相差 ≥ 1%
  （网络距离对吸附规则、拓扑细节更敏感，留更大裕度）；
- top-k：第 k 与 k+1 名相差 ≥ 0.5%。
必须在任何第二轮模型调用之前运行并提交。用法：
  uv run python eval/r2/freeze_gold.py           # 冻结
  uv run python eval/r2/freeze_gold.py --check   # 只检查（复现）
  --dev                                           # 开发集（dev_questions.json → dev_gold.json）
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from geo_agent.db import connect, run_select
from geo_agent.matcher import canonical_hash

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MARGIN_GEO = 0.005
MARGIN_NET = 0.01


def margin_issues(conn, q: dict) -> list[str]:
    spec = q.get("margin")
    if not spec:
        return []
    res = run_select(spec["sql"], conn=conn, row_limit=100000, r2=True)
    vals = [float(r[0]) for r in res.rows if r[0] is not None]
    if spec["kind"] == "radius":
        thr = float(spec["threshold"])
        m = MARGIN_NET if q["tier"] in ("route", "compliance") else MARGIN_GEO
        return [f"value {v:.2f} within {m:.1%} of threshold {thr}" for v in vals if abs(v - thr) < m * thr]
    if spec["kind"] == "topk":
        k = int(spec["k"])
        if len(vals) <= k:
            return []
        a, b = vals[k - 1], vals[k]
        if abs(a - b) < MARGIN_GEO * max(abs(a), 1.0):
            return [f"rank {k} ({a:.3f}) and rank {k + 1} ({b:.3f}) differ by < {MARGIN_GEO:.1%}"]
        return []
    return [f"unknown margin kind {spec['kind']}"]


def data_fingerprint(conn) -> dict:
    q = lambda s: run_select(s, conn=conn, guard=False, row_limit=1000).rows  # noqa: E731
    counts = {
        t: q(f"SELECT count(*) FROM {t}")[0][0]
        for t in (
            "districts",
            "pois",
            "metro_stations",
            "parks",
            "residential_areas",
            "bus_stops",
            "poi_tags",
            "walk_edges",
            "walk_vertices",
            "standard_clauses",
        )
    }
    stats = q(
        "SELECT edges, vertices, total_km, components, main_vertex_share, main_length_share FROM walk_network_stats"
    )[0]
    source = json.loads((ROOT / "data" / "SOURCE.json").read_text(encoding="utf-8"))
    clauses = (ROOT / "data" / "standards" / "clauses.json").read_bytes().replace(b"\r\n", b"\n")
    return {
        "table_counts": counts,
        "walk_network": dict(
            zip(
                ["edges", "vertices", "total_km", "components", "main_vertex_share", "main_length_share"],
                stats,
                strict=True,
            )
        ),
        "postgis_version": q("SELECT postgis_lib_version()")[0][0],
        "pgrouting_version": q("SELECT pgr_version()")[0][0],
        "pgvector_version": q("SELECT extversion FROM pg_extension WHERE extname = 'vector'")[0][0],
        "osm_snapshot": source["file"],
        "osm_snapshot_sha256": source["sha256"],
        "clauses_sha256": hashlib.sha256(clauses).hexdigest(),
    }


def main() -> int:
    check_only = "--check" in sys.argv
    dev = "--dev" in sys.argv
    qpath = HERE / ("dev_questions.json" if dev else "questions.json")
    out = HERE / ("dev_gold.json" if dev else "gold.json")
    questions = json.loads(qpath.read_text(encoding="utf-8"))
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    problems: dict[str, list[str]] = {}
    items = []
    with connect() as conn:
        for q in questions:
            issues: list[str] = []
            entry = {"id": q["id"], "tier": q["tier"], "answerable": q["gold_sql"] is not None}
            if q.get("compliance"):
                entry["compliance_clause"] = q["compliance"]
            if q["gold_sql"] is not None:
                res = run_select(q["gold_sql"], conn=conn, row_limit=1000, r2=True)
                if res.truncated:
                    issues.append("gold result truncated")
                if not res.rows:
                    issues.append("gold result empty")
                elif len(res.rows) == 1 and any(v is None for v in res.rows[0]):
                    issues.append("gold has NULL value")
                elif (
                    q["tier"] != "compliance"
                    and len(res.rows) == 1
                    and len(res.columns) == 1
                    and res.rows[0][0] in (0, None)
                ):
                    issues.append("gold scalar is 0/NULL (weak signal)")
                issues += margin_issues(conn, q)
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
        old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        mism = []
        if old:
            prev = {e["id"]: e for e in old["items"]}
            mism = [
                e["id"] for e in items if e.get("result_sha256") != prev.get(e["id"], {}).get("result_sha256")
            ]
        for e in items:
            if e["answerable"]:
                print(e["id"], e["columns"], str(e["rows"][:3])[:160], f"({len(e['rows'])} rows)")
        print(f"all checks passed; hash mismatches vs frozen gold: {mism if old else 'n/a (not frozen)'}")
        return 0 if not mism else 3
    qbytes = qpath.read_bytes().replace(b"\r\n", b"\n")
    gold = {
        "frozen_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "questions_file": "r2/" + qpath.name,
        "questions_sha256": hashlib.sha256(qbytes).hexdigest(),
        "data": fingerprint,
        "items": items,
    }
    gold["items_sha256"] = hashlib.sha256(
        json.dumps(items, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    out.write_text(json.dumps(gold, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    by_tier: dict[str, int] = {}
    for q in questions:
        by_tier[q["tier"]] = by_tier.get(q["tier"], 0) + 1
    print(f"frozen {len(items)} questions -> {out.name} {by_tier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
