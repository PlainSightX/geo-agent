"""tag 层：原始 OSM 标签题。train 与 heldout 用不相交的标签键（common.TAGS），都不含测试集的键值。"""

from __future__ import annotations

from common import SUBTYPES, TAGS
from kindlib import Abort, Ctx, call, col_index, finish, kind, lit, rid, solver

KEY_CN = {"wheelchair": "轮椅通行情况", "internet_access": "上网方式", "outdoor_seating": "户外座位情况",
          "cuisine": "菜系", "operator": "运营方", "smoking": "吸烟规定", "level": "所在楼层",
          "drive_through": "免下车服务情况", "changing_table": "婴儿护理台情况"}  # fmt: skip
TOP_KEYS = {"train": ["operator", "wheelchair", "internet_access", "outdoor_seating", "cuisine"],
            "heldout": ["smoking", "level", "drive_through", "changing_table"]}  # fmt: skip
WRONG_KEY = {"wheelchair": "accessibility", "internet_access": "wifi", "outdoor_seating": "seating",
             "delivery": "home_delivery", "air_conditioning": "aircon", "operator": "company", "cuisine": "food",
             "payment:wechat": "wechat", "payment:unionpay": "unionpay", "payment:cash": "cash",
             "smoking": "no_smoking", "drive_through": "drive", "changing_table": "baby", "level": "floor",
             "diet:vegetarian": "vegetarian"}  # fmt: skip


def combos(ctx: Ctx, by: str) -> list[tuple]:
    """(by 维度取值, key, value, 计数)，by ∈ {subtype, district}；按 split 的取值集合预先算好并缓存。"""
    ck = (ctx.split, by)
    if ck not in ctx.cache:
        tags = TAGS[ctx.split]
        vals = list(SUBTYPES[ctx.split]) if by == "subtype" else ctx.pools[ctx.split]["districts"]
        if by == "subtype":
            sql = ("SELECT p.subtype, g.key, g.value, count(DISTINCT p.id) FROM pois p JOIN poi_tags g ON g.poi_id = p.id "
                   "WHERE p.subtype = ANY(%s) GROUP BY 1, 2, 3")  # fmt: skip
        else:
            sql = ("SELECT d.name, g.key, g.value, count(DISTINCT p.id) FROM pois p JOIN poi_tags g ON g.poi_id = p.id "
                   "JOIN districts d ON ST_Within(p.geom, d.geom) WHERE d.name = ANY(%s) GROUP BY 1, 2, 3")  # fmt: skip
        ctx.cache[ck] = [r for r in ctx.q(sql, (vals,)) if (r[1], r[2]) in tags]
    return ctx.cache[ck]


def _probe_key(key: str, err: str | None):
    if err == "wrongkey":
        yield from call("tag_values", {"key": WRONG_KEY.get(key, key + "_tag")}, True)
        yield from call("tag_values", {})
    o = yield from call("tag_values", {"key": key})
    return o


@kind("tag_count", "tag", ["数据里{desc}的地点一共有多少个？", "有多少处兴趣点{desc}？"],
      ["统计{desc}的兴趣点数量。", "库中标注为{desc}的点位有几处？"], 1.0, ("text", "wrongkey"))  # fmt: skip
def make_tag_count(ctx: Ctx):
    (key, value), desc = ctx.rng.choice(list(TAGS[ctx.split].items()))
    gold = ("SELECT count(DISTINCT p.id) AS n FROM pois p JOIN poi_tags g ON g.poi_id = p.id "
            f"WHERE g.key = {lit(key)} AND g.value = {lit(value)}")  # fmt: skip
    return {"slots": {"desc": desc}, "args": {"key": key, "value": value}, "gold_sql": gold, "ents": []}


@solver("tag_count")
def solve_tag_count(a, err):
    o = yield from _probe_key(a["key"], err)
    i = col_index(o, "value", a["value"])
    if i is None:
        raise Abort("value not in tag_values")
    r = rid(o)
    yield from finish(
        r, f"共有 {{{r}.rows[0].n}} 个兴趣点。", err if err == "text" else None, result_rows=[i]
    )


@kind("tag_sub_count", "tag", ["{sub}当中{desc}的有几家？", "在所有{sub}里，{desc}的数量是多少？"],
      ["{desc}的{sub}共有多少个？", "请问{sub}中{desc}的有多少？"], 1.0, ("text", "wrongkey"))  # fmt: skip
def make_tag_sub_count(ctx: Ctx):
    cs = [c for c in combos(ctx, "subtype") if c[3] >= 1]
    st, key, value, _ = ctx.rng.choice(cs)
    gold = ("SELECT count(DISTINCT p.id) AS n FROM pois p JOIN poi_tags g ON g.poi_id = p.id "
            f"WHERE p.subtype = {lit(st)} AND g.key = {lit(key)} AND g.value = {lit(value)}")  # fmt: skip
    slots = {"sub": SUBTYPES[ctx.split][st], "desc": TAGS[ctx.split][(key, value)]}
    return {"slots": slots, "args": {"st": st, "key": key, "value": value}, "gold_sql": gold, "ents": []}


@solver("tag_sub_count")
def solve_tag_sub_count(a, err):
    yield from _probe_key(a["key"], err)
    sql = ("SELECT count(*) AS n FROM pois p JOIN poi_tags g ON g.poi_id = p.id "
           f"WHERE p.subtype = {lit(a['st'])} AND g.key = {lit(a['key'])} AND g.value = {lit(a['value'])}")  # fmt: skip
    o = yield from call("run_sql", {"sql": sql})
    r = rid(o)
    yield from finish(r, f"共 {{{r}.n}} 个。", err if err == "text" else None)


@kind("tag_sub_list", "tag", ["列出{desc}的{sub}的名称。", "哪些有名称的{sub}{desc}？"],
      ["{desc}的{sub}都叫什么名字？", "给出所有{desc}且有名称的{sub}。"], 0.8)  # fmt: skip
def make_tag_sub_list(ctx: Ctx):
    cs = [c for c in combos(ctx, "subtype") if 1 <= c[3] <= 12]
    st, key, value, _ = ctx.rng.choice(cs)
    gold = ("SELECT DISTINCT p.name FROM pois p JOIN poi_tags g ON g.poi_id = p.id "
            f"WHERE p.subtype = {lit(st)} AND g.key = {lit(key)} AND g.value = {lit(value)} AND p.name IS NOT NULL")  # fmt: skip
    slots = {"sub": SUBTYPES[ctx.split][st], "desc": TAGS[ctx.split][(key, value)]}
    return {"slots": slots, "args": {"st": st, "key": key, "value": value}, "gold_sql": gold, "ents": []}


@solver("tag_sub_list")
def solve_tag_sub_list(a, err):
    yield from _probe_key(a["key"], err)
    sql = ("SELECT DISTINCT p.name FROM pois p JOIN poi_tags g ON g.poi_id = p.id "
           f"WHERE p.subtype = {lit(a['st'])} AND g.key = {lit(a['key'])} AND g.value = {lit(a['value'])} "
           "AND p.name IS NOT NULL ORDER BY p.name")  # fmt: skip
    o = yield from call("run_sql", {"sql": sql})
    r = rid(o)
    yield from finish(r, f"符合条件的有：{{{r}.rows}}。", err if err == "text" else None)


@kind("tag_top", "tag", ["{scope}的{keycn}标签里，出现最多的取值是什么？有多少个？", "{scope}按{keycn}统计，最常见的是哪一种，几个？"],
      ["就{keycn}而言，{scope}中最常见的标注值与数量是？", "{scope}的{keycn}，哪个取值最多？给出数量。"], 0.6)  # fmt: skip
def make_tag_top(ctx: Ctx):
    key = ctx.rng.choice(TOP_KEYS[ctx.split])
    st = ctx.rng.choice([None] + list(SUBTYPES[ctx.split])) if ctx.rng.random() < 0.6 else None
    where = f"g.key = {lit(key)}" + (f" AND p.subtype = {lit(st)}" if st else "")
    base = f"FROM pois p JOIN poi_tags g ON g.poi_id = p.id WHERE {where} GROUP BY g.value"
    gold = f"SELECT g.value, count(DISTINCT p.id) AS n {base} ORDER BY n DESC LIMIT 1"
    margin = {"kind": "gap_count", "sql": f"SELECT count(DISTINCT p.id) AS n {base} ORDER BY n DESC LIMIT 2"}
    scope = SUBTYPES[ctx.split][st] if st else "所有兴趣点"
    slots = {"scope": scope, "keycn": KEY_CN[key]}
    return {"slots": slots, "args": {"key": key, "st": st}, "gold_sql": gold, "margin": margin, "ents": []}


@solver("tag_top")
def solve_tag_top(a, err):
    if not a["st"]:
        o = yield from call("tag_values", {"key": a["key"]})
    else:
        yield from call("tag_values", {"key": a["key"]})
        sql = ("SELECT g.value, count(*) AS n FROM pois p JOIN poi_tags g ON g.poi_id = p.id "
               f"WHERE p.subtype = {lit(a['st'])} AND g.key = {lit(a['key'])} GROUP BY g.value ORDER BY n DESC")  # fmt: skip
        o = yield from call("run_sql", {"sql": sql})
    r = rid(o)
    yield from finish(r, f"最常见的是 {{{r}.rows[0].value}}，共 {{{r}.rows[0].n}} 个。", err, result_rows=[0])


@kind("tag_district", "tag", ["{district}有多少个{desc}的兴趣点？", "{district}内{desc}的地点有几处？"],
      ["{district}范围内{desc}的点有多少？", "位于{district}且{desc}的兴趣点数量是？"], 0.8)  # fmt: skip
def make_tag_district(ctx: Ctx):
    d, key, value, _ = ctx.rng.choice([c for c in combos(ctx, "district") if c[3] >= 1])
    gold = ("SELECT count(DISTINCT p.id) AS n FROM pois p JOIN poi_tags g ON g.poi_id = p.id "
            f"JOIN districts d ON ST_Within(p.geom, d.geom) WHERE d.name = {lit(d)} "
            f"AND g.key = {lit(key)} AND g.value = {lit(value)}")  # fmt: skip
    slots = {"district": d, "desc": TAGS[ctx.split][(key, value)]}
    return {
        "slots": slots,
        "args": {"d": d, "key": key, "value": value},
        "gold_sql": gold,
        "ents": [["districts", d]],
    }


@solver("tag_district")
def solve_tag_district(a, err):
    yield from _probe_key(a["key"], err)
    sql = ("SELECT count(*) AS n FROM pois p JOIN poi_tags g ON g.poi_id = p.id "
           f"JOIN districts d ON ST_Within(p.geom, d.geom) WHERE d.name = {lit(a['d'])} "
           f"AND g.key = {lit(a['key'])} AND g.value = {lit(a['value'])}")  # fmt: skip
    o = yield from call("run_sql", {"sql": sql})
    r = rid(o)
    yield from finish(r, f"{a['d']}共有 {{{r}.n}} 个。", err if err == "text" else None)
