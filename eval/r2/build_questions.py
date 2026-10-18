"""生成第二轮预注册题集 eval/r2/questions.json（只在冻结前运行一次；冻结后此文件不再修改）。

题目用中文写给模型；gold_sql 由作者编写（路网题用 gold_sql.py 的模板，口径与 routing.py 一致）。
每题字段：
  id, tier, question, gold_sql (None=不可回答), order_matters,
  margin: 边界裕度检查规格（见 freeze_gold.py），
  compliance: 合规题的条文 id（判分要求答案引用该条文）
层级 tier：
  tag        原始 OSM 标签语义（精简 schema 下需要自己发现 poi_tags 与标签值）
  join       多表连接、名称消歧、别名
  crs        投影/面积/长度陷阱（单位换算、Web 墨卡托、跨表度量）
  route      路网：最短路径、等时圈面积、步行可达设施
  compliance 标准合规：检索条文得到阈值 + 路网可达性计算，答案须给出结论与条文 id
  unanswerable 数据无法回答
"""

from __future__ import annotations

import json
from pathlib import Path

from gold_sql import compliance_sql, fac, iso_area_sql, place, reach_margin_sql, reach_sql, route_sql

ROOT = Path(__file__).resolve().parent
Q: list[dict] = []


def add(qid, tier, question, gold_sql, **extra):
    Q.append({"id": qid, "tier": tier, "question": question, "gold_sql": gold_sql, **extra})


def tagq(key: str, value: str) -> str:
    return (
        f"EXISTS (SELECT 1 FROM poi_tags t WHERE t.poi_id = p.id AND t.key = '{key}' AND t.value = '{value}')"
    )


def in_district(name: str, alias: str = "p") -> str:
    return f"EXISTS (SELECT 1 FROM districts d WHERE d.name = '{name}' AND ST_Within({alias}.geom, d.geom))"


R = lambda n: place("residential_areas", n)  # noqa: E731
M = lambda n: place("metro_stations", n)  # noqa: E731
P = lambda n: place("pois", n)  # noqa: E731
K = lambda n: place("parks", n)  # noqa: E731

# ---------------------------------------------------------------- (a) 原始标签语义 12
add(
    "T01",
    "tag",
    "数据里品牌标注为中国石化的加油站有多少座？",
    f"SELECT count(*) AS n FROM pois p WHERE p.subtype = 'fuel' AND {tagq('brand', '中国石化')}",
)
add(
    "T02",
    "tag",
    "标注为佛教的宗教场所有多少处？",
    f"SELECT count(*) AS n FROM pois p WHERE p.subtype = 'place_of_worship' AND {tagq('religion', 'buddhist')}",
)
add(
    "T03",
    "tag",
    "列出所有清真寺（伊斯兰教礼拜场所）的名称。",
    f"SELECT p.name FROM pois p WHERE p.subtype = 'place_of_worship' AND {tagq('religion', 'muslim')}",
)
add(
    "T04",
    "tag",
    "营业时间标注为 24 小时的便利店有几家？",
    f"SELECT count(*) AS n FROM pois p WHERE p.subtype = 'convenience' AND {tagq('opening_hours', '24/7')}",
)
add(
    "T05",
    "tag",
    "明确标注免费的公共厕所有多少个？",
    f"SELECT count(*) AS n FROM pois p WHERE p.subtype = 'toilets' AND {tagq('fee', 'no')}",
)
add(
    "T06",
    "tag",
    "支持外带（takeaway）的快餐店有多少家？",
    f"SELECT count(*) AS n FROM pois p WHERE p.subtype = 'fast_food' AND {tagq('takeaway', 'yes')}",
)
add(
    "T07",
    "tag",
    "数据里有多少家肯德基（按品牌标签）？",
    f"SELECT count(*) AS n FROM pois p WHERE {tagq('brand', '肯德基')}",
)
add(
    "T08",
    "tag",
    "鼓楼区有多少家肯德基（按品牌标签）？",
    f"SELECT count(*) AS n FROM pois p WHERE {tagq('brand', '肯德基')} AND {in_district('鼓楼区')}",
)
add(
    "T09",
    "tag",
    "秦淮区有几家中国工商银行的银行网点（按品牌标签）？",
    f"SELECT count(*) AS n FROM pois p WHERE p.subtype = 'bank' AND {tagq('brand', '中国工商银行')} AND {in_district('秦淮区')}",
)
add(
    "T10",
    "tag",
    "蔚来（NIO Power）品牌的充电/换电站有几座？",
    f"SELECT count(*) AS n FROM pois p WHERE p.subtype = 'charging_station' AND {tagq('brand', 'NIO Power')}",
)
add(
    "T11",
    "tag",
    "支持支付宝付款的兴趣点一共有多少个？",
    "SELECT count(DISTINCT poi_id) AS n FROM poi_tags WHERE key = 'payment:alipay' AND value = 'yes'",
)
add(
    "T12",
    "tag",
    "英文名为 Presidential Palace 的地点叫什么中文名？",
    "SELECT p.name FROM pois p JOIN poi_tags t ON t.poi_id = p.id WHERE t.key = 'name:en' AND t.value = 'Presidential Palace'",
)

# ---------------------------------------------------------------- (b) 多表连接 / 消歧 / 别名 12
add(
    "J01",
    "join",
    "简称“南京炮院”的学校，正式名称是什么？",
    "SELECT p.name FROM pois p JOIN poi_tags t ON t.poi_id = p.id WHERE t.key IN ('alt_name', 'old_name', 'short_name') AND t.value = '南京炮院'",
)
add(
    "J02",
    "join",
    "“瞻园”在数据里对应的博物馆叫什么？",
    "SELECT p.name FROM pois p JOIN poi_tags t ON t.poi_id = p.id WHERE p.subtype = 'museum' AND t.key IN ('alt_name', 'old_name', 'short_name') AND t.value = '瞻园'",
)
add(
    "J03",
    "join",
    "东南大学在数据里有哪几个校区（大学类兴趣点，名称含“东南大学”但不含“成贤学院”）？",
    "SELECT name FROM pois WHERE subtype = 'university' AND name LIKE '%东南大学%' AND name NOT LIKE '%成贤%'",
)
add(
    "J04",
    "join",
    "哪个区的有名称小区（居住用地）数量最多？返回区名和数量。",
    "SELECT d.name AS district, count(*) AS n FROM residential_areas r JOIN districts d ON ST_Within(ST_PointOnSurface(r.geom), d.geom) GROUP BY d.name ORDER BY n DESC LIMIT 1",
    margin={
        "kind": "topk",
        "k": 1,
        "sql": "SELECT count(*)::float AS v FROM residential_areas r JOIN districts d ON ST_Within(ST_PointOnSurface(r.geom), d.geom) GROUP BY d.name ORDER BY v DESC",
    },
)
add(
    "J05",
    "join",
    "有多少个地铁站在 120 米内有公交站点？",
    "SELECT count(*) AS n FROM metro_stations m WHERE EXISTS (SELECT 1 FROM bus_stops b WHERE ST_DWithin(b.geom::geography, m.geom::geography, 120))",
    margin={
        "kind": "radius",
        "threshold": 120,
        "sql": "SELECT min(ST_Distance(b.geom::geography, m.geom::geography)) AS d FROM metro_stations m CROSS JOIN LATERAL (SELECT geom FROM bus_stops b ORDER BY b.geom <-> m.geom LIMIT 5) b GROUP BY m.id",
    },
)
add(
    "J06",
    "join",
    "建邺区有多少个有名称的小区（居住用地，按面内点判断所在区）？",
    "SELECT count(*) AS n FROM residential_areas r JOIN districts d ON d.name = '建邺区' AND ST_Within(ST_PointOnSurface(r.geom), d.geom)",
)
add(
    "J07",
    "join",
    "玄武区内有多少家星巴克（按品牌标签）？",
    f"SELECT count(*) AS n FROM pois p WHERE {tagq('brand', '星巴克')} AND {in_district('玄武区')}",
)
add(
    "J08",
    "join",
    "距离新街口地铁站最近的 3 家麦当劳快餐店（按品牌标签）各有多远？按距离从近到远返回直线距离（米）。",
    f"SELECT ST_Distance(p.geom::geography, m.geom::geography) AS dist_m FROM pois p JOIN metro_stations m ON m.name = '新街口' WHERE p.subtype = 'fast_food' AND {tagq('brand', '麦当劳')} ORDER BY dist_m LIMIT 3",
    order_matters=True,
    margin={
        "kind": "topk",
        "k": 3,
        "sql": f"SELECT ST_Distance(p.geom::geography, m.geom::geography) AS v FROM pois p JOIN metro_stations m ON m.name = '新街口' WHERE p.subtype = 'fast_food' AND {tagq('brand', '麦当劳')} ORDER BY v",
    },
)
add(
    "J09",
    "join",
    "哪些宗教场所标注为天主教（denomination=catholic）？返回名称。",
    "SELECT p.name FROM pois p JOIN poi_tags t ON t.poi_id = p.id WHERE t.key = 'denomination' AND t.value = 'catholic'",
)
add(
    "J10",
    "join",
    "百水芊城这个小区位于哪个区？",
    "SELECT d.name FROM residential_areas r JOIN districts d ON ST_Within(ST_PointOnSurface(r.geom), d.geom) WHERE r.name = '百水芊城'",
)
add(
    "J11",
    "join",
    "南京站地铁站 300 米内有多少个公交站点（站点记录，不合并同名）？",
    "SELECT count(*) AS n FROM bus_stops b JOIN metro_stations m ON m.name = '南京站' WHERE ST_DWithin(b.geom::geography, m.geom::geography, 300)",
    margin={
        "kind": "radius",
        "threshold": 300,
        "sql": "SELECT ST_Distance(b.geom::geography, m.geom::geography) AS d FROM bus_stops b JOIN metro_stations m ON m.name = '南京站'",
    },
)
add(
    "J12",
    "join",
    "牙科诊所（OSM 中 amenity=dentist 或 healthcare=dentist）一共有多少个？",
    "SELECT count(*) AS n FROM pois p WHERE p.subtype = 'dentist' OR EXISTS (SELECT 1 FROM poi_tags t WHERE t.poi_id = p.id AND t.key = 'healthcare' AND t.value = 'dentist')",
)

# ---------------------------------------------------------------- (c) 投影 / 面积 / 长度陷阱 10
add(
    "C01",
    "crs",
    "小区“东方城”的占地面积是多少公顷？",
    "SELECT ST_Area(geom::geography) / 1e4 AS area_ha FROM residential_areas WHERE name = '东方城'",
)
add(
    "C02",
    "crs",
    "小区“百水芊城”的边界周长是多少米？",
    "SELECT ST_Perimeter(geom::geography) AS perimeter_m FROM residential_areas WHERE name = '百水芊城'",
)
add(
    "C03",
    "crs",
    "用 Web 墨卡托（EPSG:3857）直接量算，玄武区面积是多少平方公里？与椭球面积相比放大了多少倍？",
    "SELECT ST_Area(ST_Transform(geom, 3857)) / 1e6 AS area_3857_km2, ST_Area(ST_Transform(geom, 3857)) / ST_Area(geom::geography) AS ratio FROM districts WHERE name = '玄武区'",
)
add(
    "C04",
    "crs",
    "步行路网中，人行道（highway=footway）的总长度是多少公里？",
    "SELECT sum(cost_m) / 1000 AS km FROM walk_edges WHERE highway = 'footway'",
)
add(
    "C05",
    "crs",
    "玄武区范围内的人行道（步行路网 highway=footway，按区界裁剪）总长多少公里？",
    "SELECT sum(ST_Length(ST_Intersection(e.geom, d.geom)::geography)) / 1000 AS km FROM walk_edges e JOIN districts d ON d.name = '玄武区' AND ST_Intersects(e.geom, d.geom) WHERE e.highway = 'footway'",
)
add(
    "C06",
    "crs",
    "面积最大的 3 个有名称小区是哪些？返回名称和面积（公顷）。",
    "SELECT name, ST_Area(geom::geography) / 1e4 AS area_ha FROM residential_areas ORDER BY area_ha DESC LIMIT 3",
    order_matters=True,
    margin={
        "kind": "topk",
        "k": 3,
        "sql": "SELECT ST_Area(geom::geography) / 1e4 AS v FROM residential_areas ORDER BY v DESC",
    },
)
add(
    "C07",
    "crs",
    "莫愁湖公园的周长是多少公里？",
    "SELECT ST_Perimeter(geom::geography) / 1000 AS km FROM parks WHERE name = '莫愁湖公园'",
)
add(
    "C08",
    "crs",
    "建邺区所有有名称小区的总占地面积是多少平方公里？（按面内点判断所在区）",
    "SELECT sum(ST_Area(r.geom::geography)) / 1e6 AS km2 FROM residential_areas r JOIN districts d ON d.name = '建邺区' AND ST_Within(ST_PointOnSurface(r.geom), d.geom)",
)
add(
    "C09",
    "crs",
    "小区“锁金四村”的面积是多少平方米？",
    "SELECT ST_Area(geom::geography) AS area_m2 FROM residential_areas WHERE name = '锁金四村'",
)
add(
    "C10",
    "crs",
    "南京站地铁站到南京南站地铁站的直线距离是多少公里？",
    "SELECT ST_Distance(a.geom::geography, b.geom::geography) / 1000 AS km FROM metro_stations a, metro_stations b WHERE a.name = '南京站' AND b.name = '南京南站'",
)

# ---------------------------------------------------------------- (d) 路网 / 等时圈 / 步行可达 16
M = lambda n: place("metro_stations", n)  # noqa: E731
RT = [
    ("D01", M("珠江路"), M("鼓楼"), "从珠江路地铁站步行到鼓楼地铁站，最短步行距离是多少米？"),
    ("D02", M("夫子庙"), P("江南贡院"), "从夫子庙地铁站步行到江南贡院要走多远（米）？"),
    ("D03", P("六朝博物馆"), P("总统府"), "六朝博物馆步行到总统府的最短路径有多长（米）？"),
    ("D04", M("玄武门"), M("鼓楼"), "按每小时 4.8 公里的步速，从玄武门地铁站走到鼓楼地铁站需要多少分钟？"),
    ("D05", R("百水芊城"), M("马群"), "从百水芊城小区步行到马群地铁站，最短步行距离是多少米？"),
    ("D06", M("学则路"), M("仙林中心"), "学则路地铁站到仙林中心地铁站，步行大约要多少分钟（4.8 km/h）？"),
    (
        "D07",
        P("侵华日军南京大屠杀遇难同胞纪念馆"),
        M("云锦路"),
        "侵华日军南京大屠杀遇难同胞纪念馆步行到云锦路地铁站有多远（米）？",
    ),
    ("D08", M("新街口"), M("大行宫"), "新街口地铁站步行到大行宫地铁站的最短步行距离（米）是多少？"),
]
# 题面只问一个量：问“米/多远”的金标只保留 walk_m，问“分钟”的只保留 walk_min（不要求多答）
ASK_MIN = {"D04", "D06"}
for qid, a, b, text in RT:
    col = "walk_min" if qid in ASK_MIN else "walk_m"
    add(qid, "route", text, f"SELECT {col} FROM ({route_sql(a, b)}) r")
add(
    "D09",
    "route",
    "新街口地铁站步行 15 分钟（4.8 km/h）可达范围的面积是多少平方公里？（可达道路两侧各 50 米缓冲）",
    iso_area_sql(M("新街口"), 1200),
)
add(
    "D10",
    "route",
    "鼓楼地铁站步行 10 分钟可达范围有多大（平方公里，可达道路两侧 50 米缓冲，4.8 km/h）？",
    iso_area_sql(M("鼓楼"), 800),
)


def reach_q(qid, origin, kind, meters, mode, text):
    """路网可达题：金标 + 网络距离边界裕度检查（所有候选设施与阈值相差 ≥ 1%）。"""
    sql = reach_sql(origin, fac(kind), meters, mode)
    if mode == "list":
        sql = sql.replace("SELECT name, walk_m FROM hit", "SELECT name FROM hit")
    add(
        qid,
        "route",
        text,
        sql,
        margin={"kind": "radius", "threshold": meters, "sql": reach_margin_sql(origin, fac(kind), meters)},
    )


reach_q(
    "D11",
    R("龙岸花园幸福苑"),
    "kindergarten",
    400,
    "list",
    "从龙岸花园幸福苑小区出发，步行 5 分钟（4.8 km/h）内能到达哪些幼儿园？",
)
reach_q(
    "D12",
    M("新街口"),
    "bank",
    800,
    "count",
    "新街口地铁站步行 10 分钟（4.8 km/h，沿步行路网）内有多少家银行？",
)
reach_q(
    "D13",
    R("银城东苑"),
    "bus_stop",
    400,
    "count",
    "银城东苑小区步行 5 分钟（4.8 km/h）内有几个公交站点（站点记录）？",
)
reach_q(
    "D14", M("鼓楼"), "restaurant", 800, "count", "鼓楼地铁站步行 10 分钟（4.8 km/h，沿路网）内有多少家餐厅？"
)
reach_q(
    "D15",
    R("明都雅苑"),
    "metro_station",
    1200,
    "nearest",
    "明都雅苑小区步行 15 分钟（4.8 km/h）内最近的地铁站是哪个？步行距离多少米？",
)
reach_q(
    "D16", P("江南贡院"), "cafe", 800, "count", "江南贡院步行 10 分钟（4.8 km/h，沿路网）内有几家咖啡馆？"
)

# ---------------------------------------------------------------- (e) 标准合规（RAG + 路网）12
# 判定口径（预注册）：取条文 max_radius_m 作为步行网络距离阈值；阈值内存在至少 1 个该类设施即“符合”。
CP = [
    (
        "E01",
        "南湾营融康苑",
        "middle_school",
        "GB50180-2018:C.0.1:初中",
        "按《城市居住区规划设计标准》中初中服务半径的要求，南湾营融康苑小区沿步行路网在服务半径内是否有中学？",
    ),
    (
        "E02",
        "百水芊城",
        "middle_school",
        "GB50180-2018:C.0.1:初中",
        "按国标 GB 50180-2018 对初中服务半径的规定，百水芊城小区步行可达范围内是否有中学？",
    ),
    (
        "E03",
        "龙岸花园幸福苑",
        "primary_school",
        "GB50180-2018:C.0.1:小学",
        "按 GB 50180-2018 关于小学服务半径的规定，龙岸花园幸福苑小区是否满足（沿步行路网）？",
    ),
    (
        "E04",
        "锁金四村",
        "primary_school",
        "GB50180-2018:C.0.1:小学",
        "依据居住区规划设计标准中小学的服务半径，锁金四村小区沿路网步行范围内有没有小学？",
    ),
    (
        "E05",
        "大发·燕澜湾西区",
        "kindergarten",
        "GB50180-2018:C.0.2:幼儿园",
        "按国标对幼儿园服务半径的要求，大发·燕澜湾西区小区沿步行路网是否有幼儿园覆盖？",
    ),
    (
        "E06",
        "香格里拉花园",
        "kindergarten",
        "GB50180-2018:C.0.2:幼儿园",
        "按 GB 50180-2018 幼儿园服务半径，香格里拉花园小区是否被幼儿园覆盖（沿步行路网）？",
    ),
    (
        "E07",
        "邓府新村",
        "metro_station",
        "GB50180-2018:C.0.1:轨道交通站点",
        "按居住区规划设计标准中轨道交通站点的服务半径，邓府新村小区沿路网步行范围内是否有地铁站？",
    ),
    (
        "E08",
        "银城东苑",
        "metro_station",
        "GB50180-2018:C.0.1:轨道交通站点",
        "根据 GB 50180-2018 对轨道交通站点服务半径的规定，银城东苑小区是否在地铁站服务范围内（沿步行路网）？",
    ),
    (
        "E09",
        "东方城",
        "bus_stop",
        "GB50180-2018:C.0.1:公交车站",
        "按国标公交车站服务半径的要求，东方城小区沿步行路网是否有公交站覆盖？",
    ),
    (
        "E10",
        "湖景花园",
        "bus_stop",
        "GB50180-2018:C.0.1:公交车站",
        "依据 GB 50180-2018 公交车站服务半径，湖景花园小区是否满足（沿步行路网）？",
    ),
    (
        "E11",
        "青秀城",
        "hospital",
        "GB50180-2018:C.0.1:卫生服务中心",
        "按标准中卫生服务中心（社区医院）的服务半径，以数据中的医院（hospital）计，青秀城小区沿路网步行范围内是否有医院？",
    ),
    (
        "E12",
        "莲花新城南苑",
        "police",
        "GB50180-2018:C.0.1:派出所",
        "按 GB 50180-2018 中派出所的服务半径，莲花新城南苑小区沿步行路网是否有派出所（数据中的 police）覆盖？",
    ),
]
RADIUS = {
    "GB50180-2018:C.0.1:初中": 1000,
    "GB50180-2018:C.0.1:小学": 500,
    "GB50180-2018:C.0.2:幼儿园": 300,
    "GB50180-2018:C.0.1:轨道交通站点": 800,
    "GB50180-2018:C.0.1:公交车站": 500,
    "GB50180-2018:C.0.1:卫生服务中心": 1000,
    "GB50180-2018:C.0.1:派出所": 800,
}
for qid, resid, kind, clause, text in CP:
    add(
        qid,
        "compliance",
        text,
        compliance_sql(R(resid), fac(kind), clause),
        compliance=clause,
        margin={
            "kind": "radius",
            "threshold": RADIUS[clause],
            "sql": reach_margin_sql(R(resid), fac(kind), RADIUS[clause]),
        },
    )

# ---------------------------------------------------------------- (f) 不可回答 10
U = [
    "南湾营融康苑小区有多少户居民？",
    "新街口地铁站早高峰的客流量是多少？",
    "从鼓楼地铁站开车到南京南站，晚高峰需要多长时间？",
    "百水芊城的二手房均价是多少？",
    "按 GB 50180-2018，南京市所有十五分钟生活圈的居住人口分别是多少？",
    "哪家肯德基的顾客评分最高？",
    "从新街口坐公交到夫子庙要换乘几次？",
    "龙岸花园幸福苑附近的初中今年招生多少个班？",
    "南京各区的绿化覆盖率是多少？",
    "按《城市居住区规划设计标准》，十五分钟生活圈内中学的最低建筑面积是多少平方米？",
]
for i, text in enumerate(U, 1):
    add(f"U{i:02d}", "unanswerable", text, None)


if __name__ == "__main__":
    ids = [q["id"] for q in Q]
    assert len(ids) == len(set(ids))
    lines = ",\n".join(json.dumps(q, ensure_ascii=False) for q in Q)
    (ROOT / "questions.json").write_text("[\n" + lines + "\n]\n", encoding="utf-8", newline="\n")
    tiers: dict[str, int] = {}
    for q in Q:
        tiers[q["tier"]] = tiers.get(q["tier"], 0) + 1
    print(len(Q), tiers)
