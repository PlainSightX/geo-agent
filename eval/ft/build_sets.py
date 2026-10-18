"""生成 train 轨迹与 heldout 冻结集（FINETUNE_PLAN.md）。不读取测试题面与测试金标（只用 common.test_literals 排除）。

用法（数据库已按主 README 构建）：
  uv run python eval/ft/build_sets.py              # 生成 + 过滤 + 冻结 heldout 金标 + 不相交检查
  uv run python eval/ft/build_sets.py --check      # 只复算 heldout 金标并比对冻结哈希
输出：eval/ft/data/train_trajectories.jsonl.gz、eval/ft/heldout/questions.json、eval/ft/heldout/gold.json、
     eval/ft/data/build_summary.json、eval/ft/disjointness.json
流程：顺序、确定性地采样候选题（seed 0）→ 并行执行作者金标 SQL 与边界裕度检查（freeze_gold.py 同一阈值）→
     对每题运行求解脚本（真实工具）并用 matcher_r2 对本题金标判分，只保留判对且没有意外校验问题的轨迹。
heldout 不注入错误；求解脚本也必须判对（保证存在参考解法），否则丢弃——丢弃发生在冻结之前、不涉及任何模型输出。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import psycopg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "eval" / "r2"))

import kinds_comp  # noqa: E402,F401  注册题型
import kinds_crs  # noqa: E402,F401
import kinds_join  # noqa: E402,F401
import kinds_route  # noqa: E402,F401
import kinds_tag  # noqa: E402,F401
from freeze_gold import MARGIN_GEO, MARGIN_NET, data_fingerprint  # noqa: E402,F401  data_fingerprint 供 run_build 使用
from kindlib import KINDS, Abort, Ctx  # noqa: E402
from policy import trajectory  # noqa: E402
from pools import build_pools  # noqa: E402,F401

from geo_agent.db import QueryError, connect, run_select  # noqa: E402
from geo_agent.matcher import canonical_hash  # noqa: E402
from geo_agent.r2_agent import R2_SPECS, R2Agent  # noqa: E402,F401
from geo_agent.r2_schema import describe_lean_schema  # noqa: E402,F401

QUOTA = {
    "train": {"tag": 250, "join": 260, "crs": 230, "route": 320, "compliance": 240, "unanswerable": 200},
    "heldout": {t: 25 for t in ("tag", "join", "crs", "route", "compliance", "unanswerable")},
}
# 候选/配额倍数（试跑通过率：train 约 96%；heldout 路网约 45%，稀有设施的可达题常为 0 或空）
OVERSAMPLE = {"train": {"route": 1.4, "compliance": 1.5}, "heldout": {"route": 3.2, "compliance": 2.2}}
ERR_RATE = 0.25
PREFIX = {"tag": "T", "join": "J", "crs": "C", "route": "D", "compliance": "E", "unanswerable": "U"}


def margin_issues(conn, item: dict) -> list[str]:
    specs = item.get("margin") or []
    out: list[str] = []
    for spec in specs if isinstance(specs, list) else [specs]:
        vals = [
            float(r[0])
            for r in run_select(spec["sql"], conn=conn, row_limit=100000, r2=True).rows
            if r[0] is not None
        ]
        if spec["kind"] == "radius":
            thr, m = (
                float(spec["threshold"]),
                MARGIN_NET if item["tier"] in ("route", "compliance") else MARGIN_GEO,
            )
            out += [f"value {v:.2f} within {m:.1%} of {thr}" for v in vals if abs(v - thr) < m * thr]
        elif spec["kind"] == "ordered_topk":
            rel = spec.get("rel", MARGIN_GEO)
            out += [
                f"rank gap < {rel:.1%}"
                for a, b in zip(vals, vals[1:], strict=False)
                if abs(b - a) < rel * max(abs(a), 1.0)
            ]
        elif spec["kind"] == "gap_count":
            if len(vals) < 2 or vals[0] <= vals[1]:
                out.append("top value not unique")
        else:
            out.append(f"unknown margin {spec['kind']}")
    return out


def gold_entry(conn, item: dict) -> tuple[dict, list[str]]:
    e = {"id": item["id"], "tier": item["tier"], "answerable": item["gold_sql"] is not None}
    if item.get("compliance"):
        e["compliance_clause"] = item["compliance"]
    if item["gold_sql"] is None:
        return e, []
    res = run_select(item["gold_sql"], conn=conn, row_limit=1000, r2=True)
    issues = []
    if res.truncated or not res.rows:
        issues.append("gold empty or truncated")
    elif len(res.rows) > item.get("max_rows", 1000):
        issues.append("too many rows")
    elif any(v is None for r in res.rows for v in r):
        issues.append("gold has NULL")
    elif (
        item["tier"] != "compliance"
        and len(res.rows) == 1
        and len(res.columns) == 1
        and res.rows[0][0] in (0, None)
    ):
        issues.append("gold scalar is 0")
    issues += margin_issues(conn, item)
    e.update(columns=res.columns, rows=res.rows, order_matters=bool(item.get("order_matters")),
             result_sha256=canonical_hash(res.columns, res.rows))  # fmt: skip
    return e, issues


def sample(split: str, pools: dict, conn) -> list[dict]:
    """顺序、确定性地生成候选题（含 id）；同一题面不重复。"""
    rng = random.Random(f"ft-{split}-0")
    out, seen = [], set()
    for tier, quota in QUOTA[split].items():
        kinds = [k for k in KINDS.values() if k.tier == tier]
        target = int(quota * OVERSAMPLE[split].get(tier, 1.3)) + 2
        ctx = Ctx(split, rng, pools, conn)
        ctx.counts["comp_quota"] = -(-target // (2 * len(kinds_comp.CLAUSES[split])))
        n, tries = 0, 0
        while n < target and tries < 40 * target:
            tries += 1
            k = rng.choices(kinds, [k.weight for k in kinds])[0]
            try:
                it = k.make(ctx)
            except (IndexError, QueryError, psycopg.Error):
                conn.rollback()
                it = None
            if not it:
                continue
            q = it.get("question") or rng.choice(k.phr[split]).format(**it["slots"])
            key = it["gold_sql"] or q  # 同一金标 SQL 只出一题（避免同一实体换句式重复出现）
            if key in seen or q in seen:
                continue
            seen.update((key, q))
            if it.get("compliance"):
                ctx.counts[(it["compliance"], it["label"])] = (
                    ctx.counts.get((it["compliance"], it["label"]), 0) + 1
                )
            n += 1
            pre = ("FT-" if split == "train" else "HO-") + PREFIX[tier]
            out.append(
                {**it, "id": f"{pre}{n:04d}", "split": split, "tier": tier, "kind": k.name, "question": q}
            )
    return out


def evaluate(item: dict, schema: str) -> dict:
    with connect() as conn:
        try:
            gold, issues = gold_entry(conn, item)
        except Exception as err:  # 守卫拒绝、超时等：该题丢弃
            return {"id": item["id"], "ok": False, "why": f"gold error: {type(err).__name__}: {err}"[:200]}
        if issues:
            return {"id": item["id"], "ok": False, "why": "; ".join(issues)[:200]}
        rng = random.Random(item["id"])
        k = KINDS[item["kind"]]
        err = rng.choice(k.errs) if item["split"] == "train" and rng.random() < ERR_RATE else None
        agent = R2Agent(schema_text=schema, llm=object(), tag="ft")
        try:
            t = trajectory(agent, k.solve(item["args"], err), item["question"], gold, conn)
        except Abort as e:
            return {"id": item["id"], "ok": False, "why": f"abort: {e}"}
        except Exception as e:  # 求解脚本提前结束等：丢弃
            return {"id": item["id"], "ok": False, "why": f"solver error: {type(e).__name__}: {e}"[:200]}
        steps_ok = _steps_clean(t)
        return {"id": item["id"], "gold": gold, "err": err, **t, "ok": t["ok"] and steps_ok,
                "why": t["why"] if steps_ok else "unexpected validator problem"}  # fmt: skip


def _steps_clean(t: dict) -> bool:
    """非错误轮次的工具调用不得出现校验问题（否则轨迹会教模型忽略校验器）。"""
    turn, bad = 0, set(t["err_turns"])
    for m in t["messages"]:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            content = next((x["content"] for x in t["messages"] if x.get("tool_call_id") == tc["id"]), "")
            if turn not in bad and tc["function"]["name"] != "final_answer" and "Validator:" in str(content):
                return False
        turn += 1
    return True
