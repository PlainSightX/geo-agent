"""route 层：最短步行距离/时间、等时圈面积、N 分钟可达设施（计数、最近、列表）。口径同 gold_sql.py / routing.py。"""

from __future__ import annotations

import sys

from common import MINUTES, ROOT, SUBTYPES
from kindlib import Ctx, call, finish, kind, lit, metro_mention, res_mention, resolve, rid, solver

sys.path.insert(0, str(ROOT / "eval" / "r2"))
import gold_sql as G  # noqa: E402

PT = {"residential_areas": "ST_PointOnSurface(geom)", "parks": "ST_PointOnSurface(geom)"}


def origin(ctx: Ctx, weights=(0.55, 0.3, 0.15)) -> tuple[str, str, str]:
    lay = ctx.rng.choices(["residential_areas", "metro_stations", "parks"], weights)[0]
    n = ctx.pick(lay)
    m = (
        res_mention(n)
        if lay == "residential_areas"
        else metro_mention(n, ctx.rng)
        if lay == "metro_stations"
        else n
    )
    return lay, n, m


def dest_near(ctx: Ctx, o: tuple[str, str, str]) -> tuple[str, str, str] | None:
    """离起点直线 300–2500 m 的终点（本 split 的地铁站或地标 POI）。"""
    lay = "metro_stations" if ctx.rng.random() < 0.6 else "pois"
    pool = (
        ctx.pools[ctx.split][lay]
        if lay == "metro_stations"
        else [a[0] for a in ctx.pools[ctx.split]["poi_anchors"]]
    )
    r = ctx.q(f"SELECT d.name FROM {lay} d, (SELECT {PT.get(o[0], 'geom')} AS p FROM {o[0]} WHERE name = %s) o "
              "WHERE d.name = ANY(%s) AND ST_DWithin(d.geom::geography, o.p::geography, 2500) "
              "AND NOT ST_DWithin(d.geom::geography, o.p::geography, 300) ORDER BY d.name", (o[1], pool))  # fmt: skip
    if not r:
        return None
    n = ctx.rng.choice(r)[0]
    return lay, n, metro_mention(n, ctx.rng) if lay == "metro_stations" else n


def _res(lay, n, m, err):
    if lay in ("residential_areas", "metro_stations"):
        yield from resolve(lay, n, m, err if err in ("suffix", "layer") else None)


def _route_kind(name, col, unit_cn, train, heldout):
    @kind(name, "route", train, heldout, 1.0, ("text", "suffix", "layer", "unresolved"))
    def make(ctx: Ctx):
        for _ in range(4):  # heldout 地标少：换几个起点找 300–2500 m 内的终点
            o = origin(ctx)
            d = dest_near(ctx, o)
            if d is not None:
                break
        else:
            return None
        gold = f"SELECT {col} FROM ({G.route_sql(o[:2], d[:2])}) q"
        return {"slots": {"a": o[2], "b": d[2]}, "args": {"o": list(o), "d": list(d)}, "gold_sql": gold,
                "ents": [o[:2], d[:2]]}  # fmt: skip

    @solver(name)
    def solve(a, err):
        o, d = a["o"], a["d"]
        if err == "unresolved" and o[2] != o[1]:
            yield from call(
                "walk_route", {"a": {"layer": o[0], "name": o[2]}, "b": {"layer": d[0], "name": d[1]}}, True
            )
        yield from _res(*o, err)
        yield from _res(*d, None)
        r = rid(
            (
                yield from call(
                    "walk_route", {"a": {"layer": o[0], "name": o[1]}, "b": {"layer": d[0], "name": d[1]}}
                )
            )
        )
        yield from finish(r, f"沿步行路网约 {{{r}.rows[0].{col}}} {unit_cn}。", err)


_route_kind("route_m", "walk_m", "米",
            ["按 4.8 km/h 步速沿步行路网，从{a}走到{b}最短要走多少米？", "{a}到{b}沿步行路网的最短距离是多少米？"],
            ["沿步行网络从{a}到{b}，最短步行路程（米）是？", "步行路网上{a}与{b}之间最短距离几米？"])  # fmt: skip
_route_kind("route_min", "walk_min", "分钟",
            ["步速 4.8 km/h，沿步行路网从{a}走到{b}要几分钟？", "{a}步行到{b}（沿路网，4.8 km/h）需要多少分钟？"],
            ["以 4.8 km/h 沿步行网络，{a}到{b}大约走几分钟？", "从{a}出发沿步行路网到{b}，按 4.8 km/h 需几分钟？"])  # fmt: skip


@kind("iso_area", "route", ["{o}步行 {t} 分钟（4.8 km/h，沿路网）的等时圈面积是多少平方公里？按可达路段 50 m 缓冲计。",
                            "沿步行路网 {t} 分钟内从{o}可达的范围（可达路段各缓冲 50 m 后合并）有几平方公里？步速 4.8 km/h。"],
      ["以 4.8 km/h 计，{o}的 {t} 分钟步行等时圈（路段 50 m 缓冲口径）面积为多少平方公里？",
       "从{o}出发步行 {t} 分钟、沿路网可达路段做 50 m 缓冲，合并后的面积（平方公里）是？速度 4.8 km/h。"], 0.7,
      ("text", "suffix", "unresolved"))  # fmt: skip
def make_iso(ctx: Ctx):
    o = origin(ctx)
    t = ctx.rng.choice(MINUTES[ctx.split])
    return {"slots": {"o": o[2], "t": t}, "args": {"o": list(o), "t": t}, "gold_sql": G.iso_area_sql(o[:2], 80.0 * t),
            "ents": [o[:2]]}  # fmt: skip


@solver("iso_area")
def solve_iso(a, err):
    o = a["o"]
    if err == "unresolved" and o[2] != o[1]:
        yield from call("isochrone", {"origin": {"layer": o[0], "name": o[2]}, "minutes": a["t"]}, True)
    yield from _res(*o, err)
    r = rid((yield from call("isochrone", {"origin": {"layer": o[0], "name": o[1]}, "minutes": a["t"]})))
    yield from finish(r, f"等时圈面积约 {{{r}.rows[0].area_km2}} 平方公里。", err)


def near_origins(ctx: Ctx, lay: str, cond_layer: str, cond: str, meters: float) -> list[str]:
    """本 split 中直线距离 meters 内至少有一个目标要素的起点名称（缓存）。"""
    ck = ("near", ctx.split, lay, cond_layer, cond, meters)
    if ck not in ctx.cache:
        esc = cond.replace("%", "%%")  # 条件里的 LIKE 通配符不能被当作参数占位符
        pt = PT.get(lay, "geom").replace("geom", "o.geom")
        tp = "ST_PointOnSurface(t.geom)" if cond_layer == "parks" else "t.geom"
        ctx.cache[ck] = [r[0] for r in ctx.q(
            f"SELECT o.name FROM {lay} o WHERE o.name = ANY(%s) AND EXISTS (SELECT 1 FROM {cond_layer} t WHERE {esc} "
            f"AND ST_DWithin({tp}::geography, {pt}::geography, %s)) ORDER BY o.name", (ctx.pools[ctx.split][lay], meters))]  # fmt: skip
    return ctx.cache[ck]


def _mention(ctx: Ctx, lay: str, n: str) -> str:
    return (
        res_mention(n)
        if lay == "residential_areas"
        else metro_mention(n, ctx.rng)
        if lay == "metro_stations"
        else n
    )


def _reach_parts(ctx: Ctx, named: bool):
    """可达题：75% 从“直线 0.7·80·t 米内有目标设施”的起点中抽（否则稀有设施的可达数几乎都是 0）。"""
    st = ctx.rng.choice(list(SUBTYPES[ctx.split]))
    t = ctx.rng.choice(MINUTES[ctx.split])
    cond = f"t.subtype = {lit(st)}" + (" AND t.name IS NOT NULL" if named else "")
    o = None
    if ctx.rng.random() < 0.75:
        lay = ctx.rng.choices(["residential_areas", "metro_stations", "parks"], (0.55, 0.3, 0.15))[0]
        cands = near_origins(ctx, lay, "pois", cond, 0.7 * 80.0 * t)
        if cands:
            n = ctx.rng.choice(cands)
            o = (lay, n, _mention(ctx, lay, n))
    return o or origin(ctx), st, t, ("pois", cond)


@kind("reach_count", "route", ["{o}步行 {t} 分钟内（沿路网，4.8 km/h）能到达几家{sub}？", "从{o}出发沿步行路网 {t} 分钟可达的{sub}有多少个？步速 4.8 km/h。"],
      ["按 4.8 km/h，{o}沿路网步行 {t} 分钟以内可到的{sub}数量？", "{o}的 {t} 分钟步行圈（路网，4.8 km/h）里有几个{sub}？"], 1.0,
      ("text", "suffix", "layer"))  # fmt: skip
def make_reach_count(ctx: Ctx):
    o, st, t, fac = _reach_parts(ctx, False)
    m = 80.0 * t
    return {"slots": {"o": o[2], "t": t, "sub": SUBTYPES[ctx.split][st]},
            "args": {"o": list(o), "st": st, "t": t, "list": ctx.rng.random() < 0.3},
            "gold_sql": G.reach_sql(o[:2], fac, m, "count"),
            "margin": {"kind": "radius", "threshold": m, "sql": G.reach_margin_sql(o[:2], fac, m)}, "ents": [o[:2]]}  # fmt: skip


@solver("reach_count")
def solve_reach_count(a, err):
    o = a["o"]
    yield from _res(*o, err)
    args = {
        "origin": {"layer": o[0], "name": o[1]},
        "target": {"layer": "pois", "subtypes": [a["st"]]},
        "minutes": a["t"],
    }
    if not a["list"]:
        args["count_only"] = True
    r = rid((yield from call("reachable", args)))
    if a["list"]:
        yield from finish(r, f"共有 {{{r}.count}} 个。", err, result_as="count")
    else:
        yield from finish(r, f"共有 {{{r}.n}} 个。", err)


@kind("reach_nearest", "route", ["{o}步行 {t} 分钟内（路网，4.8 km/h）最近的有名称的{sub}是哪家？走多少米？",
                                 "沿步行路网，从{o}出发 {t} 分钟内可达、路程最短的有名称{sub}是？距离几米？步速 4.8 km/h。"],
      ["{o}的 {t} 分钟步行范围（4.8 km/h，路网）内，网络距离最近的有名称{sub}及距离（米）？",
       "按 4.8 km/h 沿路网，{o}在 {t} 分钟内能走到的最近的有名称{sub}是哪个，路程多少米？"], 0.9, ("text", "suffix"))  # fmt: skip
def make_reach_nearest(ctx: Ctx):
    o, st, t, fac = _reach_parts(ctx, True)
    m = 80.0 * t
    lst = G.reach_sql(o[:2], fac, m, "list")
    return {"slots": {"o": o[2], "t": t, "sub": SUBTYPES[ctx.split][st]}, "args": {"o": list(o), "st": st, "t": t},
            "gold_sql": G.reach_sql(o[:2], fac, m, "nearest"),
            "margin": [{"kind": "ordered_topk", "k": 1, "sql": f"SELECT walk_m FROM ({lst}) q ORDER BY walk_m LIMIT 2",
                        "rel": 0.01}, {"kind": "radius", "threshold": m, "sql": G.reach_margin_sql(o[:2], fac, m)}],
            "ents": [o[:2]]}  # fmt: skip


@solver("reach_nearest")
def solve_reach_nearest(a, err):
    o = a["o"]
    yield from _res(*o, err)
    tgt = {"layer": "pois", "subtypes": [a["st"]], "named_only": True}
    r = rid(
        (
            yield from call(
                "reachable", {"origin": {"layer": o[0], "name": o[1]}, "target": tgt, "minutes": a["t"]}
            )
        )
    )
    yield from finish(
        r, f"最近的是{{{r}.rows[0].name}}，步行 {{{r}.rows[0].walk_m}} 米。", err, result_rows=[0]
    )


@kind("reach_list", "route", ["列出{o}沿步行路网 {t} 分钟内（4.8 km/h）可达的有名称{sub}及各自路程（米）。",
                              "{o}步行 {t} 分钟（路网，4.8 km/h）能到的有名称{sub}有哪些？各走多少米？"],
      ["按 4.8 km/h 沿路网，{o}在 {t} 分钟以内能到达哪些有名称的{sub}？给出网络距离（米）。",
       "{o}的 {t} 分钟步行可达范围（路网，4.8 km/h）中，所有有名称{sub}的名称与步行距离（米）？"], 0.6, ("text", "suffix"))  # fmt: skip
def make_reach_list(ctx: Ctx):
    o, st, t, fac = _reach_parts(ctx, True)
    m = 80.0 * t
    return {"slots": {"o": o[2], "t": t, "sub": SUBTYPES[ctx.split][st]}, "args": {"o": list(o), "st": st, "t": t},
            "gold_sql": G.reach_sql(o[:2], fac, m, "list"), "max_rows": 10,
            "margin": {"kind": "radius", "threshold": m, "sql": G.reach_margin_sql(o[:2], fac, m)}, "ents": [o[:2]]}  # fmt: skip


@solver("reach_list")
def solve_reach_list(a, err):
    o = a["o"]
    yield from _res(*o, err)
    tgt = {"layer": "pois", "subtypes": [a["st"]], "named_only": True}
    r = rid(
        (
            yield from call(
                "reachable", {"origin": {"layer": o[0], "name": o[1]}, "target": tgt, "minutes": a["t"]}
            )
        )
    )
    yield from finish(r, f"可达的有：{{{r}.rows}}。", err)
