"""crs 层：面积、长度、周长、按区界裁剪、直线距离（单位换算由题面指定）。"""

from __future__ import annotations

from kindlib import Ctx, call, finish, kind, lit, metro_mention, res_mention, resolve, rid, solver


def _measure_kind(name, layer, metric, unit_cn, train, heldout, wrong):
    sqlf = (
        "sum(ST_Area(geom::geography)) / 1e6"
        if metric == "area_km2"
        else "sum(ST_Length(geom::geography)) / 1000"
    )

    @kind(name, "crs", train, heldout, 1.0, ("text", "metric"))
    def make(ctx: Ctx):
        n = ctx.pick(layer)
        gold = f"SELECT {sqlf} AS {metric} FROM {layer} WHERE name = {lit(n)}"
        return {"slots": {"x": n}, "args": {"name": n, "probe": ctx.rng.random() < 0.3}, "gold_sql": gold,
                "ents": [[layer, n]]}  # fmt: skip

    @solver(name)
    def solve(a, err):
        if a["probe"]:
            yield from call("find_entity", {"name": a["name"], "layer": layer})
        if err == "metric":
            yield from call("measure", {"layer": layer, "name": a["name"], "metric": wrong}, True)
        o = yield from call("measure", {"layer": layer, "name": a["name"], "metric": metric})
        r = rid(o)
        yield from finish(r, f"{a['name']}约为 {{{r}.rows[0].{metric}}} {unit_cn}。", err)


_measure_kind("park_area_km2", "parks", "area_km2", "平方公里",
              ["{x}的面积有多少平方公里？", "{x}占地多少平方公里？"],
              ["用平方公里表示，{x}有多大？", "{x}的总面积（平方公里）是多少？"], "length_km")  # fmt: skip
_measure_kind("water_area_km2", "water_areas", "area_km2", "平方公里",
              ["{x}这片水面的面积是多少平方公里？", "水体{x}有几平方公里？"],
              ["{x}的水域面积按平方公里计是多少？", "以平方公里为单位，{x}水面多大？"], "length_km")  # fmt: skip
_measure_kind("waterway_len_km", "waterways", "length_km", "公里",
              ["{x}的总长度是多少公里？", "河道{x}一共有多长（公里）？"],
              ["以公里计，{x}全长多少？", "{x}这条水系线的长度有几公里？"], "area_km2")  # fmt: skip


@kind("park_area_ha", "crs", ["{park}的面积是多少公顷？", "{park}占地多少公顷？"],
      ["以公顷计，{park}有多大？", "{park}的占地面积（公顷）是多少？"], 1.0, ("text", "unit"))  # fmt: skip
def make_park_ha(ctx: Ctx):
    n = ctx.pick("parks")
    gold = f"SELECT sum(ST_Area(geom::geography)) / 1e4 AS area_ha FROM parks WHERE name = {lit(n)}"
    return {"slots": {"park": n}, "args": {"name": n}, "gold_sql": gold, "ents": [["parks", n]]}


@solver("park_area_ha")
def solve_park_ha(a, err):
    if err == "unit":
        yield from call(
            "run_sql",
            {"sql": f"SELECT sum(ST_Area(geom)) AS area FROM parks WHERE name = {lit(a['name'])}"},
            True,
        )
    sql = f"SELECT sum(ST_Area(geom::geography)) / 10000 AS area_ha FROM parks WHERE name = {lit(a['name'])}"
    o = yield from call("run_sql", {"sql": sql})
    r = rid(o)
    yield from finish(r, f"{a['name']}约 {{{r}.rows[0].area_ha}} 公顷。", err)


def _res_kind(name, expr, col, unit_cn, train, heldout):
    @kind(name, "crs", train, heldout, 1.0, ("text", "suffix", "metric", "unit"))
    def make(ctx: Ctx):
        n = ctx.pick("residential_areas")
        gold = (
            f"SELECT {expr.format(g='geom::geography')} AS {col} FROM residential_areas WHERE name = {lit(n)}"
        )
        m = res_mention(n)
        return {"slots": {"res": m}, "args": {"name": n, "mention": m}, "gold_sql": gold,
                "ents": [["residential_areas", n]]}  # fmt: skip

    @solver(name)
    def solve(a, err):
        yield from resolve("residential_areas", a["name"], a["mention"], err if err == "suffix" else None)
        if err == "metric":
            yield from call(
                "measure", {"layer": "residential_areas", "name": a["name"], "metric": "area_km2"}, True
            )
        if err == "unit":
            bad = f"SELECT {expr.format(g='geom')} AS {col} FROM residential_areas WHERE name = {lit(a['name'])}"
            yield from call("run_sql", {"sql": bad}, True)
        sql = f"SELECT {expr.format(g='geom::geography')} AS {col} FROM residential_areas WHERE name = {lit(a['name'])}"
        o = yield from call("run_sql", {"sql": sql})
        r = rid(o)
        yield from finish(r, f"约 {{{r}.rows[0].{col}}} {unit_cn}。", err)


_res_kind("res_perimeter_m", "ST_Perimeter({g})", "perimeter_m", "米",
          ["{res}边界的周长是多少米？", "{res}这个面的周长有几米？"],
          ["沿{res}的边界走一圈是多少米（周长）？", "{res}轮廓线的周长（米）为多少？"])  # fmt: skip
_res_kind("res_area_m2", "ST_Area({g})", "area_m2", "平方米",
          ["{res}的占地面积是多少平方米？", "{res}有多少平方米？"],
          ["按平方米计，{res}的面积多大？", "{res}这块用地的面积（平方米）是？"])  # fmt: skip


@kind("road_len_in_district", "crs", ["{road}在{district}境内的长度是多少公里？", "{road}位于{district}范围内的那一段有多长（公里）？"],
      ["按{district}行政边界裁剪后，{road}的长度为多少公里？", "{road}落在{district}内的部分总长几公里？"])  # fmt: skip
def make_road_len(ctx: Ctx):
    d = ctx.rng.choice(ctx.pools[ctx.split]["districts"])
    ck = ("cross", ctx.split, d)
    if ck not in ctx.cache:
        ctx.cache[ck] = [r[0] for r in ctx.q("SELECT DISTINCT t.name FROM roads t, districts d WHERE d.name = %s AND "
                         "ST_Crosses(t.geom, d.geom) AND t.name = ANY(%s) ORDER BY 1", (d, ctx.pools[ctx.split]["roads"]))]  # fmt: skip
    if not ctx.cache[ck]:
        return None
    road = ctx.rng.choice(ctx.cache[ck])
    sql = ("SELECT sum(ST_Length(ST_Intersection(t.geom, d.geom)::geography)) / 1000 AS len_km FROM roads t "
           f"JOIN districts d ON ST_Intersects(t.geom, d.geom) WHERE t.name = {lit(road)} AND d.name = {lit(d)}")  # fmt: skip
    return {"slots": {"road": road, "district": d}, "args": {"sql": sql, "road": road}, "gold_sql": sql,
            "ents": [["roads", road], ["districts", d]]}  # fmt: skip


@solver("road_len_in_district")
def solve_road_len(a, err):
    o = yield from call("run_sql", {"sql": a["sql"]})
    r = rid(o)
    yield from finish(r, f"{a['road']}在该区内约 {{{r}.rows[0].len_km}} 公里。", err)


@kind("dist_between_km", "crs", ["{a}和{b}之间的直线距离是多少公里？", "{a}到{b}直线相距几公里？"],
      ["以公里计，{a}与{b}的直线距离？", "{a}离{b}有多少公里（直线）？"], 1.0, ("text", "unresolved"))  # fmt: skip
def make_dist(ctx: Ctx):
    def one(layers):
        lay = ctx.rng.choice(layers)
        if lay == "metro_stations":
            n = ctx.pick("metro_stations")
            return lay, n, metro_mention(n, ctx.rng)
        if lay == "pois":
            n = ctx.pick("poi_anchors")[0]
            return lay, n, n
        n = ctx.pick("parks")
        return lay, n, n

    a, b = one(["metro_stations", "pois"]), one(["parks", "pois", "metro_stations"])
    if a[1] == b[1]:
        return None
    gold = (f"WITH a AS (SELECT ST_Union(geom) g FROM {a[0]} WHERE name = {lit(a[1])}), "
            f"b AS (SELECT ST_Union(geom) g FROM {b[0]} WHERE name = {lit(b[1])}) "
            "SELECT ST_Distance(a.g::geography, b.g::geography) / 1000 AS dist_km FROM a, b")  # fmt: skip
    return {"slots": {"a": a[2], "b": b[2]}, "args": {"a": list(a), "b": list(b)}, "gold_sql": gold,
            "ents": [[a[0], a[1]], [b[0], b[1]]]}  # fmt: skip


@solver("dist_between_km")
def solve_dist(a, err):
    (la, na, ma), (lb, nb, mb) = a["a"], a["b"]
    if err == "unresolved" and ma != na:
        yield from call(
            "distance_between", {"a": {"layer": la, "name": ma}, "b": {"layer": lb, "name": nb}}, True
        )
    if la == "metro_stations":
        yield from resolve(la, na, ma, None)
    if lb == "metro_stations":
        yield from resolve(lb, nb, mb, None)
    o = yield from call("distance_between", {"a": {"layer": la, "name": na}, "b": {"layer": lb, "name": nb}})
    r = rid(o)
    yield from finish(r, f"直线距离约 {{{r}.rows[0].dist_km}} 公里。", err)
