"""微调数据的公共定义：测试集排除字面量、确定性哈希切分、train/heldout 互不相交的取值集合。

规则来自 eval/local_llm/FINETUNE_PLAN.md“数据：三个互不相交的集合”：
- 测试集（eval/r2/questions.json）只程序化读取 gold_sql 中的字符串字面量与 compliance 条文 id，用于排除；
  题面文本只在内存中做字符 5-gram Jaccard 检查（test_texts），任何脚本都不打印、不写入题面；
- 实体按 sha256(layer:name) 二分（约 80% train / 20% heldout）；
- 设施类型、条文、标签键、分钟数/半径在 train 与 heldout 之间各自取不相交的集合，且都不含测试集用到的取值。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FT = ROOT / "eval" / "ft"
TEST_Q = ROOT / "eval" / "r2" / "questions.json"
HELDOUT_FRAC = 0.2
CLASSIFIERS = {"中学", "小学"}  # gold_sql 中的 LIKE 片段，是学段口径而不是实体名


def _test_items() -> list[dict]:
    return json.loads(TEST_Q.read_text(encoding="utf-8"))


def test_literals() -> set[str]:
    """测试集 gold_sql 的全部字符串字面量（LIKE 去掉 %）与合规条文 id；不读取题面。"""
    out: set[str] = set()
    for q in _test_items():
        for m in re.findall(r"'((?:[^']|'')*)'", q.get("gold_sql") or ""):
            out.add(m.replace("''", "'").strip("%"))
        if q.get("compliance"):
            out.add(q["compliance"])
    return {x for x in out if x}


def test_numbers() -> set[float]:
    """测试集 gold_sql 中字符串字面量之外出现的数字（半径、米数、分钟换算值等）。"""
    nums: set[float] = set()
    for q in _test_items():
        s = re.sub(r"'(?:[^']|'')*'", "", q.get("gold_sql") or "")
        nums |= {float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", s)}
    return nums


def test_texts() -> list[str]:
    """测试题面，只供 5-gram 相似度在内存中比较；调用方不得打印或落盘。"""
    return [q["question"] for q in _test_items()]


def split_of(layer: str, name: str) -> str:
    h = int(hashlib.sha256(f"{layer}:{name}".encode()).hexdigest()[:8], 16) / 2**32
    return "heldout" if h < HELDOUT_FRAC else "train"


def ngrams(s: str, n: int = 5) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i : i + n] for i in range(max(len(s) - n + 1, 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# ---------------------------------------------------------------- 设施类型（规则 2）
# 测试集用到的 subtype（计划列出的 + gold_sql 中出现的），train/heldout 一律不用作过滤条件
SUBTYPES_TEST = {
    "bank", "cafe", "fuel", "school", "kindergarten", "hospital", "police", "toilets", "museum", "restaurant",
    "fast_food", "charging_station", "dentist", "convenience", "university", "place_of_worship",
}  # fmt: skip
SUBTYPES = {
    "train": {
        "supermarket": "超市", "post_office": "邮局", "library": "图书馆", "pharmacy": "药店",
        "marketplace": "农贸市场", "mall": "商场", "community_centre": "社区活动中心", "clinic": "诊所",
        "hotel": "酒店", "bus_station": "公交枢纽站", "bakery": "面包店", "bicycle_rental": "公共自行车租赁点",
        "clothes": "服装店", "car_repair": "汽车维修店", "telecommunication": "通信营业厅", "books": "书店",
    },
    "heldout": {
        "department_store": "百货商店", "greengrocer": "果蔬店", "arts_centre": "艺术中心", "theatre": "剧院",
        "jewelry": "珠宝店", "mobile_phone": "手机店", "fire_station": "消防站", "cinema": "电影院",
        "hostel": "青年旅舍", "driver_training": "驾校训练场", "hairdresser": "理发店", "courthouse": "法院",
    },
}  # fmt: skip

# ---------------------------------------------------------------- 条文（规则 3）：clause → (题面设施说法, 目标过滤)
CLAUSES = {
    "train": {
        "GB50180-2018:C.0.1:商场": ("商场", {"layer": "pois", "subtypes": ["mall"]}),
        "GB50180-2018:C.0.1:菜市场": ("菜市场", {"layer": "pois", "subtypes": ["marketplace"]}),
        "GB50180-2018:C.0.1:邮政营业场所": ("邮政营业场所（邮局）", {"layer": "pois", "subtypes": ["post_office"]}),
        "GB50180-2018:C.0.1:门诊部": ("门诊部（诊所）", {"layer": "pois", "subtypes": ["clinic"]}),
        "GB50180-2018:C.0.1:文化活动中心": ("文化活动中心（社区活动中心）", {"layer": "pois", "subtypes": ["community_centre"]}),
        "GB50180-2018:C.0.2:小超市": ("小超市（超市）", {"layer": "pois", "subtypes": ["supermarket"]}),
    },
    "heldout": {
        "GB50180-2018:C.0.1:体育场馆": ("体育场馆（名称含“体育”的公园）", {"layer": "parks", "name_like": "体育"}),
        "GB50180-2018:C.0.2:社区卫生服务站": ("社区卫生服务站（名称含“社区卫生服务”的设施）", {"layer": "pois", "name_like": "社区卫生服务"}),
        "GB50180-2018:C.0.2:室外综合健身场地": ("室外综合健身场地（公园）", {"layer": "parks"}),
    },
}  # fmt: skip

# ---------------------------------------------------------------- 标签键（规则 4）：(key, value) → 中文说法
TAGS = {
    "train": {
        ("wheelchair", "yes"): "可以无障碍通行轮椅", ("wheelchair", "no"): "轮椅无法进入",
        ("wheelchair", "limited"): "轮椅只能部分通行", ("internet_access", "wlan"): "提供无线局域网",
        ("internet_access", "no"): "不提供上网服务", ("outdoor_seating", "yes"): "设有户外座位",
        ("outdoor_seating", "no"): "没有户外座位", ("delivery", "yes"): "提供送货上门",
        ("air_conditioning", "yes"): "装有空调", ("operator", "中国邮政"): "由中国邮政运营",
        ("operator", "苏果超市有限公司"): "由苏果超市有限公司运营", ("cuisine", "chinese"): "菜系标注为中餐",
        ("operator", "南京公共自行车有限公司"): "由南京公共自行车有限公司运营",
        ("access", "private"): "仅限内部使用、不对公众开放", ("access", "customers"): "只对顾客开放",
        ("access", "permissive"): "经业主许可可以进入", ("indoor", "yes"): "位于室内", ("indoor", "room"): "设在室内房间里",
        ("covered", "yes"): "有顶棚遮盖", ("atm", "yes"): "设有自动取款机", ("information", "guidepost"): "属于指路牌",
        ("information", "office"): "属于游客咨询处", ("operator", "罗森（中国）投资有限公司"): "由罗森（中国）投资有限公司运营",
    },
    "heldout": {
        ("payment:wechat", "yes"): "支持微信付款", ("payment:unionpay", "yes"): "可以刷银联卡",
        ("payment:cash", "yes"): "收现金", ("smoking", "no"): "全面禁烟", ("smoking", "isolated"): "设有隔离吸烟区",
        ("drive_through", "no"): "没有免下车取餐窗口", ("changing_table", "yes"): "配有婴儿护理台",
        ("level", "-1"): "位于地下一层", ("diet:vegetarian", "yes"): "供应素食",
    },
}  # fmt: skip

# ---------------------------------------------------------------- 分钟数与直线半径（规则 5）
MINUTES = {"train": [6, 8, 12, 14, 18, 20], "heldout": [7, 9, 11, 13, 16, 17]}
RADII = {"train": [150, 250, 350, 450, 600, 700, 900], "heldout": [200, 280, 380, 550, 650, 750]}
TOPK = {"train": [2, 4, 6], "heldout": [3, 5]}


def facility_cond(spec: dict, alias: str = "t") -> str:
    """目标过滤 → 金标 SQL 条件（与 tools._target_where 同一语义）。"""
    parts = []
    if spec.get("subtypes"):
        parts.append(f"{alias}.subtype IN (" + ", ".join(f"'{s}'" for s in spec["subtypes"]) + ")")
    if spec.get("name_like"):
        parts.append(f"{alias}.name LIKE '%{spec['name_like']}%'")
    return " AND ".join(parts) or "TRUE"
