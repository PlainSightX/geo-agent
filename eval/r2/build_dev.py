"""第二轮开发集（12 题）：只用于调试两个评测臂的提示词与工具；实体与测试集不重叠。

写在测试集冻结之后、任何模型调用之前。开发集结果会在 README 中报告，但不进入主结论。
"""

from __future__ import annotations

import json
from pathlib import Path

from build_questions import in_district, tagq
from gold_sql import compliance_sql, fac, iso_area_sql, place, reach_sql, route_sql

HERE = Path(__file__).resolve().parent
R = lambda n: place("residential_areas", n)  # noqa: E731
M = lambda n: place("metro_stations", n)  # noqa: E731

DEV = [
    {
        "id": "V01",
        "tier": "tag",
        "question": "品牌为中国邮政的邮局有多少个？",
        "gold_sql": f"SELECT count(*) AS n FROM pois p WHERE p.subtype = 'post_office' AND {tagq('brand', '中国邮政')}",
    },
    {
        "id": "V02",
        "tier": "tag",
        "question": "标注为基督教的宗教场所有多少处？",
        "gold_sql": f"SELECT count(*) AS n FROM pois p WHERE p.subtype = 'place_of_worship' AND {tagq('religion', 'christian')}",
    },
    {
        "id": "V03",
        "tier": "join",
        "question": "简称“陆指”的学校正式名称是什么？",
        "gold_sql": "SELECT p.name FROM pois p JOIN poi_tags t ON t.poi_id = p.id WHERE t.key IN ('alt_name', 'old_name', 'short_name') AND t.value = '陆指'",
    },
    {
        "id": "V04",
        "tier": "join",
        "question": "栖霞区有多少家星巴克（按品牌标签）？",
        "gold_sql": f"SELECT count(*) AS n FROM pois p WHERE {tagq('brand', '星巴克')} AND {in_district('栖霞区')}",
    },
    {
        "id": "V05",
        "tier": "crs",
        "question": "小区“银城东苑”的占地面积是多少公顷？",
        "gold_sql": "SELECT ST_Area(geom::geography) / 1e4 AS area_ha FROM residential_areas WHERE name = '银城东苑'",
    },
    {
        "id": "V06",
        "tier": "crs",
        "question": "白鹭洲公园的周长是多少米？",
        "gold_sql": "SELECT ST_Perimeter(geom::geography) AS perimeter_m FROM parks WHERE name = '白鹭洲公园'",
    },
    {
        "id": "V07",
        "tier": "route",
        "question": "从云南路地铁站步行到鼓楼地铁站，最短步行距离是多少米？",
        "gold_sql": f"SELECT walk_m FROM ({route_sql(M('云南路'), M('鼓楼'))}) r",
    },
    {
        "id": "V08",
        "tier": "route",
        "question": "奥体中心地铁站步行 15 分钟（4.8 km/h，可达道路两侧 50 米缓冲）可达范围有多少平方公里？",
        "gold_sql": iso_area_sql(M("奥体中心"), 1200),
    },
    {
        "id": "V09",
        "tier": "route",
        "question": "南湾营泰康苑小区步行 5 分钟（4.8 km/h）内有几个公交站点（站点记录）？",
        "gold_sql": reach_sql(R("南湾营泰康苑"), fac("bus_stop"), 400, "count"),
    },
    {
        "id": "V10",
        "tier": "compliance",
        "question": "按 GB 50180-2018 对初中服务半径的要求，诚品城小区沿步行路网是否有中学？",
        "gold_sql": compliance_sql(R("诚品城"), fac("middle_school"), "GB50180-2018:C.0.1:初中"),
        "compliance": "GB50180-2018:C.0.1:初中",
    },
    {
        "id": "V11",
        "tier": "compliance",
        "question": "按国标公交车站服务半径，青秀城小区是否被公交站覆盖（沿步行路网）？",
        "gold_sql": compliance_sql(R("青秀城"), fac("bus_stop"), "GB50180-2018:C.0.1:公交车站"),
        "compliance": "GB50180-2018:C.0.1:公交车站",
    },
    {"id": "V12", "tier": "unanswerable", "question": "诚品城小区的物业费是多少？", "gold_sql": None},
]

if __name__ == "__main__":
    lines = ",\n".join(json.dumps(q, ensure_ascii=False) for q in DEV)
    (HERE / "dev_questions.json").write_text("[\n" + lines + "\n]\n", encoding="utf-8", newline="\n")
    print(len(DEV))
