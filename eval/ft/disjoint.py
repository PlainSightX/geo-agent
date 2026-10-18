"""不相交检查（FINETUNE_PLAN.md 规则 1–5）：任一项冲突即 ok=false，run_build.py 以非 0 退出。

只输出计数、最大相似度与本项目自己生成的题号；不输出测试题面。
"""

from __future__ import annotations

import collections
import random

from common import (
    CLAUSES,
    MINUTES,
    RADII,
    SUBTYPES,
    SUBTYPES_TEST,
    TAGS,
    jaccard,
    ngrams,
    test_literals,
    test_numbers,
    test_texts,
)
from kindlib import KINDS
from pools import _hit

from geo_agent.db import connect


def _vals(items: list[dict], key: str) -> set:
    out = set()
    for it in items:
        a = it.get("args") or {}
        if a.get(key) is not None:
            out.add(a[key])
        spec = a.get("spec") or {}
        if key == "st":
            out |= set(spec.get("subtypes") or [])
    return out


def _chk(ok: bool, **kw) -> dict:
    return {"ok": bool(ok), **kw}


def report(pools: dict, train: list[dict], heldout: list[dict]) -> dict:
    checks: dict[str, dict] = {}
    ents = {
        s: {tuple(e) for it in its for e in it.get("ents") or []}
        for s, its in (("train", train), ("heldout", heldout))
    }
    lits = pools["entity_literals"]
    both = ents["train"] & ents["heldout"]
    hits = [e for s in ents for e in ents[s] if _hit(e[1], lits)]
    checks["entities"] = _chk(not both and not hits, train=len(ents["train"]), heldout=len(ents["heldout"]),
                              overlap=len(both), test_literal_hits=len(hits), entity_literals=len(lits))  # fmt: skip
    st = {s: _vals(its, "st") for s, its in (("train", train), ("heldout", heldout))}
    checks["subtypes"] = _chk(st["train"] <= set(SUBTYPES["train"]) and st["heldout"] <= set(SUBTYPES["heldout"])
                              and not (st["train"] & st["heldout"]) and not ((st["train"] | st["heldout"]) & SUBTYPES_TEST),
                              train=sorted(st["train"]), heldout=sorted(st["heldout"]))  # fmt: skip
    test_clauses = {x for x in test_literals() if x.startswith("GB50180")}
    cl = {
        s: {it["compliance"] for it in its if it.get("compliance")}
        for s, its in (("train", train), ("heldout", heldout))
    }
    checks["clauses"] = _chk(cl["train"] <= set(CLAUSES["train"]) and cl["heldout"] <= set(CLAUSES["heldout"])
                             and not (cl["train"] & cl["heldout"]) and not ((cl["train"] | cl["heldout"]) & test_clauses),
                             train=sorted(cl["train"]), heldout=sorted(cl["heldout"]), test_clauses=len(test_clauses))  # fmt: skip
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT key FROM poi_tags WHERE key = ANY(%s)", (list(test_literals()),))
        test_keys = {r[0] for r in cur.fetchall()}
    # 只看 tag 层（join 层别名题用 alt_name 等作名称查找，实体与取值另行不相交）
    keys = {
        s: _vals([i for i in its if i["tier"] == "tag"], "key")
        for s, its in (("train", train), ("heldout", heldout))
    }
    checks["tag_keys"] = _chk(keys["train"] <= {k for k, _ in TAGS["train"]} and keys["heldout"] <= {k for k, _ in TAGS["heldout"]}
                              and not (keys["train"] & keys["heldout"]) and not ((keys["train"] | keys["heldout"]) & test_keys),
                              train=sorted(keys["train"]), heldout=sorted(keys["heldout"]), test_keys=len(test_keys))  # fmt: skip
    nums = test_numbers()
    mins = {s: _vals(its, "t") for s, its in (("train", train), ("heldout", heldout))}
    rad = {s: _vals(its, "r") for s, its in (("train", train), ("heldout", heldout))}
    checks["numbers"] = _chk(not (mins["train"] & mins["heldout"]) and not (rad["train"] & rad["heldout"])
                             and not any(80.0 * m in nums for m in mins["train"] | mins["heldout"])
                             and not ((rad["train"] | rad["heldout"]) & nums)
                             and mins["train"] <= set(MINUTES["train"]) and rad["heldout"] <= set(RADII["heldout"]),
                             minutes={s: sorted(v) for s, v in mins.items()}, radii={s: sorted(v) for s, v in rad.items()})  # fmt: skip
    per_tier = collections.defaultdict(lambda: collections.Counter())
    same = []
    for k in KINDS.values():
        if set(k.phr["train"]) & set(k.phr["heldout"]):
            same.append(k.name)
        for s in ("train", "heldout"):
            per_tier[k.tier][s] += len(k.phr[s])
    checks["templates"] = _chk(not same and all(c["train"] >= 8 and c["heldout"] >= 8 for c in per_tier.values()),
                               shared_kinds=same, per_tier={t: dict(c) for t, c in per_tier.items()})  # fmt: skip
    tg = [ngrams(t) for t in test_texts()]
    worst = {}
    for s, its in (("train", train), ("heldout", heldout)):
        best = max(
            ((max(jaccard(ngrams(it["question"]), g) for g in tg), it["id"]) for it in its),
            default=(0.0, None),
        )
        worst[s] = {"max_jaccard_vs_test": round(best[0], 3), "closest_item": best[1]}
    hg = [ngrams(it["question"]) for it in heldout]
    worst["train_vs_heldout_max"] = round(
        max((max(jaccard(ngrams(it["question"]), g) for g in hg) for it in train), default=0), 3
    )
    checks["ngram5"] = _chk(
        worst["train"]["max_jaccard_vs_test"] < 0.5 and worst["heldout"]["max_jaccard_vs_test"] < 0.5, **worst
    )
    rng = random.Random(0)
    spot = rng.sample([(it["id"], it["question"]) for it in train], min(25, len(train)))
    spot += rng.sample([(it["id"], it["question"]) for it in heldout], min(25, len(heldout)))
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks, "spot_check_sample": spot}
