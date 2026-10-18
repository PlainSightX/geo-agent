"""题型框架：每个题型 = 采样槽位 + 题面句式（train/heldout 各写一套）+ 作者金标 SQL + 边界裕度 + 求解脚本。

求解脚本是生成器：yield (工具名, 参数, 是否为注入的错误轮次)，收到 {"json": 工具结果预览, "text": 原文}；
它通过真实 R2Agent 循环调用真实工具（policy.py），所以轨迹中的结果、编号与校验器提示都是真实的。
错误轮次（先带后缀查名、选错图层、单位错误、纯文本调用等）在训练时不计损失，只学习其后的改正。
"""

from __future__ import annotations

import random
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any

KINDS: dict[str, Kind] = {}


class Abort(Exception):
    """求解脚本无法得到合规轨迹（例如精确名称不在 find_entity 结果中）：丢弃该题。"""


@dataclass
class Kind:
    name: str
    tier: str
    phr: dict[str, list[str]]
    make: Callable[[Ctx], dict | None]
    solve: Callable[..., Generator] | None = None
    weight: float = 1.0
    errs: tuple[str, ...] = ("text",)


def kind(name: str, tier: str, train: list[str], heldout: list[str], weight: float = 1.0, errs=("text",)):
    def deco(make):
        KINDS[name] = Kind(name, tier, {"train": train, "heldout": heldout}, make, None, weight, tuple(errs))
        return make

    return deco


def solver(name: str):
    def deco(fn):
        KINDS[name].solve = fn
        return fn

    return deco


@dataclass
class Ctx:
    split: str
    rng: random.Random
    pools: dict
    conn: Any
    used: set = field(default_factory=set)
    cache: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)

    def q(self, sql: str, params: Any = None) -> list[tuple]:
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        self.conn.rollback()
        return rows

    def pick(self, pool: str) -> Any:
        return self.rng.choice(self.pools[self.split][pool])


def lit(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def res_mention(name: str) -> str:
    return name if name.endswith("小区") else name + "小区"


def metro_mention(name: str, rng: random.Random) -> str:
    base = name if name.endswith("站") else name + "站"
    return rng.choice([f"地铁{base}", f"{name}地铁站"]) if not name.endswith("站") else f"地铁{base}"


def strip_mention(m: str) -> str:
    for pre in ("地铁",):
        if m.startswith(pre) and len(m) > len(pre) + 1:
            m = m[len(pre) :]
    for suf in ("地铁站", "小区", "站"):
        if m.endswith(suf) and len(m) - len(suf) >= 2:
            return m[: -len(suf)]
    return m


# ---------------------------------------------------------------- 求解脚本的小工具


def call(name: str, args: dict, err: bool = False):
    obs = yield (name, args, err)
    return obs


def rows(obs: dict) -> list:
    return (obs.get("json") or {}).get("rows") or []


def rid(obs: dict) -> str:
    r = (obs.get("json") or {}).get("id")
    if not r:
        raise Abort(f"tool returned no result id: {str(obs.get('text'))[:120]}")
    return r


def col_index(obs: dict, col: str, value: Any) -> int | None:
    j = obs.get("json") or {}
    cs = j.get("columns") or []
    if col not in cs:
        return None
    i = cs.index(col)
    return next((k for k, r in enumerate(j.get("rows") or []) if r[i] == value), None)


def place(layer: str, name: str) -> dict:
    return {"layer": layer, "name": name}


def resolve(layer: str, exact: str, mention: str, err: str | None):
    """find_entity 解析精确名称（题面名称去掉“小区/地铁站”等后缀，指定图层）。"""
    q = strip_mention(mention)
    if err == "suffix" and mention != q:
        o = yield from call("find_entity", {"name": mention, "layer": layer}, True)
        if col_index(o, "name", exact) is not None:
            return o
    elif err == "layer":
        yield from call("find_entity", {"name": q, "layer": "roads"}, True)
    o = yield from call("find_entity", {"name": q, "layer": layer})
    if col_index(o, "name", exact) is None:
        raise Abort("exact name not in find_entity result")
    return o


def finish(ref: str, answer: str, err: str | None, **extra):
    """final_answer（参数按工具定义顺序）；err='text' 先输出一次纯文本调用（不计损失），收到提醒后再正确调用。"""
    args = {"result_ref": ref, **extra, "answer": answer}
    if err == "text":
        import json

        yield from call("__text__", "final_answer\n" + json.dumps(args, ensure_ascii=False), True)
    yield from call("final_answer", args)
