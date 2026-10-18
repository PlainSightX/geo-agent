"""入口：生成 train 轨迹与 heldout 冻结集，写出产物并运行不相交检查（实现见 build_sets.py / disjoint.py）。

uv run python eval/ft/run_build.py [--limit N] [--out DIR]   # 生成 + 冻结 heldout 金标 + 不相交检查
uv run python eval/ft/run_build.py --check                   # 复算 heldout 金标并比对冻结哈希
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_sets as B  # noqa: E402
import disjoint  # noqa: E402

from geo_agent.db import connect  # noqa: E402

FIELDS = (
    "id",
    "tier",
    "kind",
    "question",
    "gold_sql",
    "margin",
    "order_matters",
    "max_rows",
    "compliance",
    "args",
    "ents",
)


def select(split: str, outs: list[dict]) -> list[dict]:
    """每层取前 quota 个通过的题；合规层按 (条文, 结论) 轮流取，使各条文“符合/不符合”尽量均衡。"""
    keep: list[dict] = []
    for tier, quota in B.QUOTA[split].items():
        ok = [o for o in outs if o["ok"] and o["item"]["tier"] == tier]
        if tier != "compliance":
            keep += ok[:quota]
            continue
        buckets = collections.defaultdict(list)
        for o in ok:
            buckets[(o["item"]["compliance"], o["item"]["label"])].append(o)
        order, i = sorted(buckets), 0
        while len([o for o in keep if o["item"]["tier"] == tier]) < quota and any(buckets.values()):
            b = buckets[order[i % len(order)]]
            if b:
                keep.append(b.pop(0))
            i += 1
    return keep


def write_gold(path: Path, qpath: Path, items: list[dict], conn) -> None:
    qbytes = qpath.read_bytes().replace(b"\r\n", b"\n")
    gold = {"frozen_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "questions_file": "ft/heldout/" + qpath.name,
            "questions_sha256": hashlib.sha256(qbytes).hexdigest(), "data": B.data_fingerprint(conn), "items": items}  # fmt: skip
    gold["items_sha256"] = hashlib.sha256(
        json.dumps(items, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    path.write_text(json.dumps(gold, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def check(out: Path) -> int:
    qs = json.loads((out / "heldout" / "questions.json").read_text("utf-8"))
    frozen = {e["id"]: e for e in json.loads((out / "heldout" / "gold.json").read_text("utf-8"))["items"]}
    bad = []
    with connect() as conn:
        for q in qs:
            e, issues = B.gold_entry(conn, q)
            if issues or e.get("result_sha256") != frozen[q["id"]].get("result_sha256"):
                bad.append(q["id"])
    print(f"heldout gold check: {len(qs)} items, mismatches {bad}")
    return 0 if not bad else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N candidates per tier")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out)
    if args.check:
        return check(out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "heldout").mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        pools = B.build_pools(conn)
        schema = B.describe_lean_schema(conn)
        cands = {s: B.sample(s, pools, conn) for s in ("train", "heldout")}
    summary: dict = {"pools": {s: {k: len(v) for k, v in pools[s].items()} for s in ("train", "heldout")},
                     "excluded": pools["excluded"], "entity_literals": len(pools["entity_literals"])}  # fmt: skip
    kept_all = {}
    for split, items in cands.items():
        if args.limit:  # 试跑：每层只取前 N 个候选
            items = [x for t in B.QUOTA[split] for x in [y for y in items if y["tier"] == t][: args.limit]]
        with ThreadPoolExecutor(args.workers) as pool:
            outs = list(pool.map(lambda it: B.evaluate(it, schema), items))
        for it, o in zip(items, outs, strict=True):
            o["item"] = it
        kept = select(split, outs) if not args.limit else [o for o in outs if o["ok"]]
        kept_all[split] = kept
        drops = collections.Counter(o["why"].split(":")[0][:60] for o in outs if not o["ok"])
        summary[split] = {
            "candidates": len(items), "passed": sum(o["ok"] for o in outs), "kept": len(kept),
            "by_tier": dict(collections.Counter(o["item"]["tier"] for o in kept)),
            "by_kind": dict(collections.Counter(o["item"]["kind"] for o in kept)),
            "drop_reasons": dict(drops.most_common(12)),
            "with_error_turn": sum(bool(o["err_turns"]) for o in kept),
            "error_types": dict(collections.Counter(o["err"] for o in kept if o["err_turns"])),
            "compliance_labels": dict(collections.Counter(f"{o['item']['compliance'].split(':')[-1]}:{o['item']['label']}"
                                                          for o in kept if o["item"].get("compliance"))),
            "model_calls_mean": round(sum(o["model_calls"] for o in kept) / max(len(kept), 1), 2),
        }  # fmt: skip
    tr = [{**{k: o["item"][k] for k in ("id", "split", "tier", "kind", "question")}, "messages": o["messages"],
           "loss_turns": o["loss_turns"], "err": o["err"] if o["err_turns"] else None} for o in kept_all["train"]]  # fmt: skip
    meta = {"system_prompt": B.R2Agent(schema_text=schema, llm=object()).system_prompt(), "tools": B.R2_SPECS}
    meta["system_prompt_sha256"] = hashlib.sha256(meta["system_prompt"].encode()).hexdigest()
    meta["tools_sha256"] = hashlib.sha256(json.dumps(B.R2_SPECS, ensure_ascii=False).encode()).hexdigest()
    (out / "data" / "render_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n", "utf-8"
    )
    with (
        open(out / "data" / "train_trajectories.jsonl.gz", "wb") as fh,
        gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz,
    ):
        for t in tr:
            gz.write((json.dumps(t, ensure_ascii=False) + "\n").encode("utf-8"))
    hq = [{k: o["item"][k] for k in FIELDS if k in o["item"]} for o in kept_all["heldout"]]
    qpath = out / "heldout" / "questions.json"
    qpath.write_text(json.dumps(hq, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    with connect() as conn:
        write_gold(out / "heldout" / "gold.json", qpath, [o["gold"] for o in kept_all["heldout"]], conn)
    rep = disjoint.report(pools, [o["item"] for o in kept_all["train"]], hq)
    (out / "disjointness.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (out / "data" / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", "utf-8"
    )
    print(json.dumps({s: {k: summary[s][k] for k in ("candidates", "passed", "kept", "by_tier")} for s in cands},
                     ensure_ascii=False))  # fmt: skip
    print("disjointness ok:", rep["ok"], {k: v for k, v in rep["checks"].items() if not v["ok"]})
    return 0 if rep["ok"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
