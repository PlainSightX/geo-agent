"""join 层：别名/英文名、点在面内、跨表计数与最近邻。实体全部来自本 split 的实体池。"""

from __future__ import annotations

from common import RADII, SUBTYPES, TOPK
from kindlib import (
    Abort,
    Ctx,
    call,
    col_index,
    finish,
    kind,
    lit,
    metro_mention,
    res_mention,
    resolve,
    rid,
    solver,
)


def anchor_near(ctx: Ctx, st: str) -> tuple[str, str, str]:
    """在某类设施附近取一个锚点（本 split 的地铁站或地标 POI），返回 (layer, 精确名, 题面说法)。"""
    ck = ("ids", st)
    if ck not in ctx.cache:
        ctx.cache[ck] = [r[0] for r in ctx.q("SELECT id FROM pois WHERE subtype = %s ORDER BY id", (st,))]
    fid = ctx.rng.choice(ctx.cache[ck])
    if ctx.rng.random() < 0.5:
        pool = ctx.pools[ctx.split]["metro_stations"]
        r = ctx.q("SELECT m.name FROM metro_stations m, (SELECT geom FROM pois WHERE id = %s) f "
                  "WHERE m.name = ANY(%s) ORDER BY m.geom <-> f.geom, m.name LIMIT 1", (fid, pool))  # fmt: skip
        return "metro_stations", r[0][0], metro_mention(r[0][0], ctx.rng)
    pool = [a[0] for a in ctx.pools[ctx.split]["poi_anchors"]]
    r = ctx.q("SELECT p.name FROM pois p, (SELECT geom FROM pois WHERE id = %s) f "
              "WHERE p.name = ANY(%s) ORDER BY p.geom <-> f.geom, p.name LIMIT 1", (fid, pool))  # fmt: skip
    return "pois", r[0][0], r[0][0]


def _resolve_anchor(a, err):
    if a["layer"] == "metro_stations":
        yield from resolve("metro_stations", a["name"], a["mention"], err)
    elif err in ("suffix", "layer"):
        yield from resolve("pois", a["name"], a["mention"], err)


@kind("alias_district", "join", ["“{alias}”这个叫法对应的是哪个地点？它在哪个区？", "别名为“{alias}”的地方正式名称是什么，属于哪个区？"],
      ["人们说的“{alias}”指哪里？位于哪个行政区？", "“{alias}”是哪个兴趣点的别称？所在区是？"])  # fmt: skip
def make_alias(ctx: Ctx):
    name, key, value = ctx.pick("aliases")
    return _alias_item(name, key, value)


@kind("english_name", "join", ["英文名为“{alias}”的地点中文叫什么？在哪个区？", "“{alias}”对应的中文名称与所在区是什么？"],
      ["哪个地方的英文名是“{alias}”？请给出中文名和行政区。", "把英文名“{alias}”换成中文名称，并说明它在哪个区。"])  # fmt: skip
def make_english(ctx: Ctx):
    name, key, value = ctx.pick("english")
    return _alias_item(name, key, value)


def _alias_item(name, key, value):
    gold = (f"SELECT p.name, (SELECT d.name FROM districts d WHERE ST_Within(p.geom, d.geom) LIMIT 1) AS district "
            f"FROM pois p WHERE p.name = {lit(name)}")  # fmt: skip
    return {"slots": {"alias": value}, "args": {"name": name, "key": key, "value": value}, "gold_sql": gold,
            "ents": [["pois", name]]}  # fmt: skip


def solve_alias(a, err):
    o = yield from call("find_entity", {"name": a["value"]})
    j = o.get("json") or {}
    cs = j.get("columns") or []
    want = f"tag {a['key']}={a['value']}"
    i = next((k for k, r in enumerate(j.get("rows") or []) if r[cs.index("name")] == a["name"]
              and r[cs.index("matched_by")] == want), None) if "matched_by" in cs else None  # fmt: skip
    if i is None:
        raise Abort("alias row not found")
    r = rid(o)
    yield from finish(
        r, f"指的是{{{r}.rows[{i}].name}}，位于{{{r}.rows[{i}].district}}。", err, result_rows=[i]
    )


solver("alias_district")(solve_alias)
solver("english_name")(solve_alias)


@kind("place_district", "join", ["{place}属于哪个区？", "{place}位于南京的哪个行政区？"],
      ["{place}在哪个区的范围内？", "请问{place}所在的区是哪个？"], 1.0, ("text", "suffix", "layer"))  # fmt: skip
def make_place_district(ctx: Ctx):
    if ctx.rng.random() < 0.6:
        layer, name = "residential_areas", ctx.pick("residential_areas")
        mention, pt = res_mention(name), "ST_PointOnSurface(t.geom)"
    else:
        layer, name = "metro_stations", ctx.pick("metro_stations")
        mention, pt = metro_mention(name, ctx.rng), "t.geom"
    gold = f"SELECT d.name AS district FROM {layer} t JOIN districts d ON ST_Within({pt}, d.geom) WHERE t.name = {lit(name)}"
    return {"slots": {"place": mention}, "args": {"layer": layer, "name": name, "mention": mention},
            "gold_sql": gold, "ents": [[layer, name]]}  # fmt: skip


@solver("place_district")
def solve_place_district(a, err):
    o = yield from resolve(a["layer"], a["name"], a["mention"], err)
    i = col_index(o, "name", a["name"])
    r = rid(o)
    yield from finish(r, f"{a['mention']}位于{{{r}.rows[{i}].district}}。", err, result_rows=[i])


@kind("count_in_park", "join", ["{park}的范围内有多少个兴趣点？", "落在{park}边界以内的兴趣点有几个？"],
      ["{park}里面一共有几处兴趣点？", "统计位于{park}内部的兴趣点个数。"])  # fmt: skip
def make_count_in_park(ctx: Ctx):
    ck = ("parks_n", ctx.split)
    if ck not in ctx.cache:
        ctx.cache[ck] = ctx.q("SELECT k.name, count(*) FROM parks k JOIN pois t ON ST_Within(t.geom, k.geom) "
                              "WHERE k.name = ANY(%s) GROUP BY k.name HAVING count(*) BETWEEN 2 AND 150 ORDER BY 1",
                              (ctx.pools[ctx.split]["parks"],))  # fmt: skip
    name, _ = ctx.rng.choice(ctx.cache[ck])
    gold = f"SELECT count(*) AS n FROM pois t JOIN parks k ON ST_Within(t.geom, k.geom) WHERE k.name = {lit(name)}"
    return {"slots": {"park": name}, "args": {"name": name, "list": ctx.rng.random() < 0.5}, "gold_sql": gold,
            "ents": [["parks", name]]}  # fmt: skip


@solver("count_in_park")
def solve_count_in_park(a, err):
    if a["list"]:
        o = yield from call("spatial_relation", {"left_layer": "pois", "relation": "within",
                                                 "right": {"layer": "parks", "name": a["name"]}})  # fmt: skip
        r = rid(o)
        yield from finish(r, f"共有 {{{r}.count}} 个兴趣点。", err, result_as="count")
    else:
        sql = ("SELECT count(*) AS n FROM pois p JOIN parks k ON ST_Within(p.geom, k.geom) "
               f"WHERE k.name = {lit(a['name'])}")  # fmt: skip
        o = yield from call("run_sql", {"sql": sql})
        r = rid(o)
        yield from finish(r, f"共有 {{{r}.n}} 个兴趣点。", err)


@kind("sub_in_district", "join", ["{district}一共有多少家{sub}？", "{district}境内的{sub}有几个？"],
      ["{sub}在{district}有多少处？", "位于{district}的{sub}数量是多少？"])  # fmt: skip
def make_sub_in_district(ctx: Ctx):
    st = ctx.rng.choice(list(SUBTYPES[ctx.split]))
    d = ctx.rng.choice(ctx.pools[ctx.split]["districts"])
    gold = (f"SELECT count(*) AS n FROM pois t JOIN districts d ON ST_Within(t.geom, d.geom) "
            f"WHERE d.name = {lit(d)} AND t.subtype = {lit(st)}")  # fmt: skip
    return {"slots": {"district": d, "sub": SUBTYPES[ctx.split][st]}, "args": {"st": st, "d": d,
            "probe": ctx.rng.random() < 0.3}, "gold_sql": gold, "ents": [["districts", d]]}  # fmt: skip


@solver("sub_in_district")
def solve_sub_in_district(a, err):
    if a["probe"]:
        yield from call("distinct_values", {"layer": "pois", "prefix": a["st"][:3]})
    o = yield from call(
        "count_by_district", {"target": {"layer": "pois", "subtypes": [a["st"]], "district": a["d"]}}
    )
    r = rid(o)
    yield from finish(r, f"{a['d']}共有 {{{r}.rows[0].n}} 个。", err)


@kind("within_count", "join", ["距{anchor}直线{r}米以内有几家{sub}？", "以{anchor}为中心、半径{r}米（直线）范围内的{sub}有多少？"],
      ["{anchor}周边直线距离{r}米内共有多少个{sub}？", "统计离{anchor}直线不超过{r}米的{sub}数量。"], 1.0,
      ("text", "suffix", "layer"))  # fmt: skip
def make_within_count(ctx: Ctx):
    st = ctx.rng.choice(list(SUBTYPES[ctx.split]))
    layer, name, mention = anchor_near(ctx, st)
    r = ctx.rng.choice(RADII[ctx.split])
    excl = f" AND t.name IS DISTINCT FROM {lit(name)}" if layer == "pois" else ""
    a_cte = f"WITH a AS (SELECT ST_Union(geom) AS g FROM {layer} WHERE name = {lit(name)}) "
    cond = f"t.subtype = {lit(st)}{excl}"
    gold = f"{a_cte}SELECT count(*) AS n FROM pois t, a WHERE {cond} AND ST_DWithin(t.geom::geography, a.g::geography, {r})"
    msql = (f"{a_cte}SELECT ST_Distance(t.geom::geography, a.g::geography) FROM pois t, a WHERE {cond} "
            f"AND ST_DWithin(t.geom::geography, a.g::geography, {2 * r + 500})")  # fmt: skip
    return {"slots": {"anchor": mention, "r": r, "sub": SUBTYPES[ctx.split][st]},
            "args": {"layer": layer, "name": name, "mention": mention, "st": st, "r": r, "list": ctx.rng.random() < 0.3},
            "gold_sql": gold, "margin": {"kind": "radius", "threshold": r, "sql": msql}, "ents": [[layer, name]]}  # fmt: skip


@solver("within_count")
def solve_within_count(a, err):
    yield from _resolve_anchor(a, err)
    args = {"anchor": {"layer": a["layer"], "name": a["name"]}, "target": {"layer": "pois", "subtypes": [a["st"]]},
            "radius_m": a["r"], "count_only": not a["list"]}  # fmt: skip
    o = yield from call("within_distance", args)
    r = rid(o)
    if a["list"]:
        yield from finish(r, f"共有 {{{r}.count}} 个。", err, result_as="count")
    else:
        yield from finish(r, f"共有 {{{r}.n}} 个。", err)


@kind("nearest_k", "join", ["离{anchor}最近的{k}个有名称的{sub}依次是哪些？", "按直线距离，{anchor}附近最近的{k}家{sub}（有名称）是？"],
      ["列出距{anchor}直线最近的{k}个有名字的{sub}，由近到远。", "与{anchor}直线距离最小的前{k}个{sub}（须有名称）分别是？"], 1.0,
      ("text", "suffix", "layer"))  # fmt: skip
def make_nearest_k(ctx: Ctx):
    st = ctx.rng.choice(list(SUBTYPES[ctx.split]))
    layer, name, mention = anchor_near(ctx, st)
    k = ctx.rng.choice(TOPK[ctx.split])
    excl = f" AND t.name IS DISTINCT FROM {lit(name)}" if layer == "pois" else ""
    a_cte = f"WITH a AS (SELECT ST_Union(geom) AS g FROM {layer} WHERE name = {lit(name)}) "
    base = f"FROM pois t, a WHERE t.subtype = {lit(st)} AND t.name IS NOT NULL{excl} ORDER BY d"
    sel = "ST_Distance(t.geom::geography, a.g::geography) AS d"
    gold = f"{a_cte}SELECT t.name, {sel} {base} LIMIT {k}"
    return {"slots": {"anchor": mention, "k": k, "sub": SUBTYPES[ctx.split][st]},
            "args": {"layer": layer, "name": name, "mention": mention, "st": st, "k": k},
            "gold_sql": f"SELECT name FROM ({gold}) q ORDER BY d", "order_matters": True,
            "margin": {"kind": "ordered_topk", "k": k, "sql": f"{a_cte}SELECT {sel} {base} LIMIT {k + 1}"},
            "ents": [[layer, name]]}  # fmt: skip


@solver("nearest_k")
def solve_nearest_k(a, err):
    yield from _resolve_anchor(a, err)
    args = {"anchor": {"layer": a["layer"], "name": a["name"]},
            "target": {"layer": "pois", "subtypes": [a["st"]], "named_only": True}, "k": a["k"]}  # fmt: skip
    o = yield from call("nearest_k", args)
    r = rid(o)
    yield from finish(r, f"由近到远依次是：{{{r}.rows}}。", err)


@kind("nearest_metro", "join", ["从{res}（取小区面内代表点）出发，直线最近的地铁站是哪个？距离多少米？",
                                "{res}直线距离最近的地铁站及其距离（米，按小区面内代表点计算）？"],
      ["哪个地铁站离{res}最近（直线，以小区面内代表点为准）？相距几米？", "以{res}面内代表点计，最近地铁站的名称和直线距离（米）是？"],
      1.0, ("text", "suffix", "anchor"))  # fmt: skip
def make_nearest_metro(ctx: Ctx):
    name = ctx.pick("residential_areas")
    o = f"WITH o AS (SELECT ST_PointOnSurface(geom) AS p FROM residential_areas WHERE name = {lit(name)}) "
    sel = "ST_Distance(m.geom::geography, o.p::geography)"
    gold = f"{o}SELECT m.name, {sel} AS dist_m FROM metro_stations m, o ORDER BY dist_m LIMIT 1"
    margin = {
        "kind": "ordered_topk",
        "k": 1,
        "sql": f"{o}SELECT {sel} AS d FROM metro_stations m, o ORDER BY d LIMIT 2",
    }
    mention = res_mention(name)
    return {"slots": {"res": mention}, "args": {"name": name, "mention": mention}, "gold_sql": gold,
            "margin": margin, "ents": [["residential_areas", name]]}  # fmt: skip


@solver("nearest_metro")
def solve_nearest_metro(a, err):
    yield from resolve("residential_areas", a["name"], a["mention"], err if err == "suffix" else None)
    if err == "anchor":
        yield from call("nearest_k", {"anchor": {"layer": "residential_areas", "name": a["name"]},
                                      "target": {"layer": "metro_stations"}, "k": 1}, True)  # fmt: skip
    sql = ("SELECT m.name, ST_Distance(m.geom::geography, ST_PointOnSurface(r.geom)::geography) AS dist_m "
           f"FROM metro_stations m, residential_areas r WHERE r.name = {lit(a['name'])} ORDER BY dist_m LIMIT 1")  # fmt: skip
    o = yield from call("run_sql", {"sql": sql})
    r = rid(o)
    yield from finish(r, f"最近的是{{{r}.rows[0].name}}，直线距离 {{{r}.rows[0].dist_m}} 米。", err)
