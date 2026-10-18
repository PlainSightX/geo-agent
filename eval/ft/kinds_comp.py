"""compliance 层（检索条文 → check_compliance → 引用条文 id）与 unanswerable 层（识别数据缺失后拒答）。

train 只用 common.CLAUSES['train'] 的 6 条，heldout 只用另外 3 条；两者都不含测试集的 7 条。
每条条文“符合/不符合”尽量各半：一半样本先选一个目标设施、再取离它最近的本 split 小区（多为符合），另一半随机取小区。
"""

from __future__ import annotations

import sys

from common import CLAUSES, ROOT, facility_cond
from kindlib import Abort, Ctx, call, finish, kind, metro_mention, res_mention, resolve, rid, rows, solver

sys.path.insert(0, str(ROOT / "eval" / "r2"))
import gold_sql as G  # noqa: E402
from kinds_route import near_origins  # noqa: E402

COMP_TRAIN = [
    "按照 GB 50180-2018 中{fac}服务半径的规定，{res}沿步行路网是否满足？请注明依据条文。",
    "{res}是否符合《城市居住区规划设计标准》对{fac}服务半径的要求（按步行路网距离）？给出条文编号。",
    "依据居住区规划标准，{res}步行范围内的{fac}配置达标吗？引用所用条文。",
    "以 GB 50180-2018 的{fac}服务半径为步行网络距离上限，{res}是否合规？说明条文。",
    "{res}附近的{fac}，按标准规定的服务半径走路网能否覆盖？请给出条文 id。",
    "用规划标准里{fac}的服务半径核查{res}（步行路网距离），合规吗？附上条文。",
    "{res}居民步行到{fac}的距离是否在 GB 50180-2018 规定的服务半径内？注明条文。",
    "请判断{res}在{fac}服务半径方面是否达标（沿步行路网），并说明依据的条文编号。",
]
COMP_HELDOUT = [
    "对照 GB 50180-2018 关于{fac}服务半径的条文，{res}沿路网步行是否达到要求？请写出条文号。",
    "{res}的{fac}步行可达性是否满足居住区规划设计标准规定的服务半径？注明依据。",
    "检查{res}：按标准中{fac}的服务半径（步行路网距离），结论是符合还是不符合？附条文。",
    "以《城市居住区规划设计标准》{fac}服务半径衡量，{res}是否满足？需给出引用条文。",
    "{res}能否在标准给定的{fac}服务半径内（路网步行）找到设施？请列明条文号。",
    "根据 GB 50180-2018 对{fac}的服务半径要求，评估{res}是否达标，并写明依据条文。",
    "{res}一带：{fac}的步行网络距离是否落在规划标准的服务半径之内？给出条文。",
    "就{fac}的服务半径而言，{res}按步行路网计算是否合规？注明所引条文。",
]


@kind("compliance", "compliance", COMP_TRAIN, COMP_HELDOUT, 1.0, ("text", "suffix", "unresolved", "clause"))
def make_compliance(ctx: Ctx):
    clause = ctx.rng.choice(sorted(CLAUSES[ctx.split]))
    fac_cn, spec = CLAUSES[ctx.split][clause]
    pool = ctx.pools[ctx.split]["residential_areas"]
    if ctx.rng.random() < 0.6:  # 一半以上从“直线半径内有目标设施”的小区中抽，使“符合”样本足够
        radius = ctx.q("SELECT max_radius_m FROM standard_clauses WHERE clause_id = %s", (clause,))[0][0]
        cands = near_origins(ctx, "residential_areas", spec["layer"], facility_cond(spec), float(radius))
        res = ctx.rng.choice(cands) if cands else ctx.rng.choice(pool)
    else:
        res = ctx.rng.choice(pool)
    fac = (spec["layer"], facility_cond(spec))
    gold = G.compliance_sql(("residential_areas", res), fac, clause)
    got = ctx.q(gold)
    if not got:
        return None
    label = bool(got[0][0])
    quota = ctx.counts.get("comp_quota", 10**9)
    if ctx.counts.get((clause, label), 0) >= quota:
        return None
    radius = float(got[0][2])
    mention = res_mention(res)
    return {"slots": {"res": mention, "fac": fac_cn}, "compliance": clause, "label": label,
            "args": {"res": res, "mention": mention, "clause": clause, "fac": fac_cn.split("（")[0], "spec": spec,
                     "qv": ctx.rng.randrange(3), "k": ctx.rng.choice([3, 5])},
            "gold_sql": gold, "margin": {"kind": "radius", "threshold": radius,
                                         "sql": G.reach_margin_sql(("residential_areas", res), fac, radius)},
            "ents": [["residential_areas", res]]}  # fmt: skip


@solver("compliance")
def solve_compliance(a, err):
    if err == "unresolved":
        yield from call(
            "check_compliance",
            {"residential": a["mention"], "clause_id": a["clause"], "target": a["spec"]},
            True,
        )
    yield from resolve("residential_areas", a["res"], a["mention"], "suffix" if err == "suffix" else None)
    q = [f"{a['fac']}服务半径", f"{a['fac']}的服务半径规定", f"居住区{a['fac']}服务半径"][a["qv"]]
    o = yield from call("retrieve_standard", {"query": q, "k": a["k"]})
    if not rows(o) or rows(o)[0][0] != a["clause"]:
        raise Abort("retrieval top-1 is not the target clause")
    if err == "clause":
        yield from call("check_compliance", {"residential": a["res"], "clause_id": a["clause"].split(":", 1)[1],
                                             "target": a["spec"]}, True)  # fmt: skip
    o = yield from call(
        "check_compliance", {"residential": a["res"], "clause_id": a["clause"], "target": a["spec"]}
    )
    r = rid(o)
    if rows(o) and rows(o)[0][0]:
        ans = (f"符合：{{{r}.rows[0].nearest_name}}沿步行路网 {{{r}.rows[0].nearest_walk_m}} 米，"
               f"在条文 {{{r}.rows[0].clause_id}} 规定的 {{{r}.rows[0].radius_m}} 米以内。")  # fmt: skip
    else:
        ans = f"不符合：条文 {{{r}.rows[0].clause_id}} 规定服务半径 {{{r}.rows[0].radius_m}} 米，沿步行路网在此范围内没有该类设施。"
    yield from finish(r, ans, err)


# ---------------------------------------------------------------- 不可回答（话题与测试集不重叠：不问人口、客流、驾车、房价、评分、换乘、招生、绿化率）
UNANS = {
    "train": [("{res}的物业费是每平方米多少钱？", "物业费"), ("{metro}早高峰的发车间隔是几分钟？", "列车时刻"),
              ("{poi}去年的营业额是多少？", "经营数据"), ("{res}是哪一年建成的？", "建成年份"),
              ("{park}夜间的平均噪声是多少分贝？", "噪声监测"), ("{poi}有多少名员工？", "员工人数"),
              ("{res}现在还有多少个空余车位？", "实时车位"), ("{park}里种了多少棵树？", "树木数量"),
              ("{district}去年空气质量优良的天数是多少？", "空气质量")],
    "heldout": [("{poi}周边的商铺空置率是多少？", "商铺空置"), ("{metro}附近道路晚高峰的拥堵指数是多少？", "交通拥堵"),
                ("{res}冬至日的日照时长有几小时？", "日照"), ("{district}每天的快递量有多少件？", "快递量"),
                ("{res}每栋楼有几部电梯？", "电梯数量"), ("{park}的土壤是什么类型？", "土壤"),
                ("{district}去年的碳排放量是多少吨？", "碳排放"), ("{poi}的营业执照编号是什么？", "营业执照")],
}  # fmt: skip


@kind(
    "unanswerable",
    "unanswerable",
    [t for t, _ in UNANS["train"]],
    [t for t, _ in UNANS["heldout"]],
    1.0,
    ("text",),
)
def make_unans(ctx: Ctx):
    tpl, noun = ctx.rng.choice(UNANS[ctx.split])
    slot = tpl[tpl.index("{") + 1 : tpl.index("}")]
    layer = {
        "res": "residential_areas",
        "metro": "metro_stations",
        "park": "parks",
        "poi": "pois",
        "district": "districts",
    }[slot]
    if slot == "res":
        n = ctx.pick("residential_areas")
        m = res_mention(n)
    elif slot == "metro":
        n = ctx.pick("metro_stations")
        m = metro_mention(n, ctx.rng)
    elif slot == "poi":
        n = m = ctx.pick("poi_anchors")[0]
    elif slot == "district":
        n = m = ctx.rng.choice(ctx.pools[ctx.split]["districts"])
    else:
        n = m = ctx.pick("parks")
    return {"question": tpl.format(**{slot: m}), "gold_sql": None, "ents": [[layer, n]],
            "args": {"layer": layer, "name": n, "noun": noun, "mode": ctx.rng.choices([0, 1, 2], [0.5, 0.35, 0.15])[0]}}  # fmt: skip


@solver("unanswerable")
def solve_unans(a, err):
    if a["mode"] == 1 and a["layer"] != "districts":
        yield from call("find_entity", {"name": a["name"], "layer": a["layer"]})
    elif a["mode"] == 2:
        yield from call("tag_values", {})
    if err == "text":
        yield from call("__text__", "抱歉，数据库中没有这类信息，无法回答。", True)
    reason = f"数据库只有 OSM 的位置、类别与标签，没有{a['noun']}数据"
    yield from call(
        "final_answer", {"answer": "无法用当前数据回答：" + reason, "refuse": True, "reason": reason}
    )
