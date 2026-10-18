"""实体池：从数据库取各图层的名称，排除测试集字面量涉及的实体，再按 sha256(layer:name) 切成 train / heldout。

排除规则（规则 1）：实体字面量 = 测试集 gold_sql 中等于某图层名称或某别名标签值的字面量，以及 LIKE 片段（学段口径除外）。
任一图层中名称包含某个实体字面量、或被某个实体字面量包含（长度 ≥ 2）的要素，train 与 heldout 一律不用；
别名或英文名标签值命中的 POI 同样排除。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from common import CLASSIFIERS, split_of, test_literals

NAME_LAYERS = ("residential_areas", "metro_stations", "parks", "water_areas", "waterways", "roads", "pois",
               "districts", "bus_stops")  # fmt: skip
ALIAS_KEYS = ("alt_name", "old_name", "short_name", "official_name")
ANCHOR_SUBTYPES = ("attraction", "hotel", "mall", "library", "theatre", "townhall", "exhibition_centre",
                   "theme_park", "supermarket", "department_store", "bus_station", "college", "hostel")  # fmt: skip


def _rows(conn, sql: str, params: Any = None) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def entity_literals(conn) -> list[str]:
    lits = test_literals()
    found: set[str] = set()
    for layer in NAME_LAYERS:
        found |= {
            r[0]
            for r in _rows(conn, f"SELECT DISTINCT name FROM {layer} WHERE name = ANY(%s)", (list(lits),))
        }
    found |= {
        r[0]
        for r in _rows(
            conn,
            "SELECT DISTINCT value FROM poi_tags WHERE value = ANY(%s) AND (key = ANY(%s) OR key LIKE 'name:%%')",
            (list(lits), list(ALIAS_KEYS)),
        )
    }
    # LIKE 片段（如学校名的一部分）也是实体字面量；学段口径词除外
    like_frags = {x for x in lits if x not in CLASSIFIERS and any("一" <= ch <= "鿿" for ch in x)}
    found |= {x for x in like_frags if not x.startswith("GB50180")}
    return sorted(x for x in found if len(x) >= 2)


def _hit(name: str, ents: list[str]) -> bool:
    return any(e in name or (len(name) >= 2 and name in e) for e in ents)


def build_pools(conn) -> dict[str, Any]:
    """返回 {'excluded': {...}, 'train': {pool: [...]}, 'heldout': {...}}。"""
    ents = entity_literals(conn)
    excluded: dict[str, list[str]] = {}
    pools: dict[str, dict[str, list]] = {"train": defaultdict(list), "heldout": defaultdict(list)}

    def add(pool: str, layer: str, name: str, item: Any = None) -> None:
        pools[split_of(layer, name)][pool].append(name if item is None else item)

    bad_poi = {
        r[0]
        for r in _rows(
            conn,
            "SELECT DISTINCT p.name FROM pois p JOIN poi_tags g ON g.poi_id = p.id "
            "WHERE p.name IS NOT NULL AND (g.key = ANY(%s) OR g.key LIKE 'name:%%') AND g.value = ANY(%s)",
            (list(ALIAS_KEYS), ents),
        )
    }
    for layer in ("residential_areas", "metro_stations", "parks", "water_areas", "waterways", "roads"):
        names = _rows(
            conn, f"SELECT name, count(*) FROM {layer} WHERE name IS NOT NULL GROUP BY name ORDER BY name"
        )
        ex = [n for n, _ in names if _hit(n, ents)]
        excluded[layer] = ex
        exs = set(ex)
        for n, c in names:
            if n in exs or len(n) < 2:
                continue
            if layer in ("residential_areas", "parks", "metro_stations") and c != 1:
                continue  # 起终点要求同名唯一，避免“同名取 id 最小”成为题目难点
            add(layer, layer, n)
    dists = [r[0] for r in _rows(conn, "SELECT name FROM districts ORDER BY name")]
    excluded["districts"] = [d for d in dists if _hit(d, ents)]
    rest = sorted(
        (d for d in dists if d not in excluded["districts"]), key=lambda d: split_of("districts", d)
    )
    pools["heldout"]["districts"], pools["train"]["districts"] = rest[:2], rest[2:]
    anchors = _rows(
        conn,
        "SELECT name, min(subtype) FROM pois WHERE name IS NOT NULL AND subtype = ANY(%s) "
        "GROUP BY name HAVING count(*) = 1 ORDER BY name",
        (list(ANCHOR_SUBTYPES),),
    )
    excluded["pois"] = sorted({n for n, _ in anchors if _hit(n, ents) or n in bad_poi})
    for n, st in anchors:
        if n not in excluded["pois"] and len(n) >= 2:
            add("poi_anchors", "pois", n, [n, st])
    for key_like, pool in (("alias", "aliases"), ("name:en", "english")):
        cond = "g.key = ANY(%s)" if key_like == "alias" else "g.key = 'name:en'"
        rows = _rows(
            conn,
            "SELECT p.name, g.key, g.value FROM pois p JOIN poi_tags g ON g.poi_id = p.id "
            f"WHERE p.name IS NOT NULL AND {cond} AND g.value <> p.name "
            "AND (SELECT count(*) FROM poi_tags h WHERE h.value = g.value) = 1 "
            "AND (SELECT count(*) FROM pois q WHERE q.name = p.name) = 1 ORDER BY 1, 2",
            (list(ALIAS_KEYS),) if key_like == "alias" else None,
        )
        for n, k, v in rows:
            if _hit(n, ents) or n in bad_poi or _hit(v, ents):
                continue
            if pool == "english" and not (v.isascii() and 4 <= len(v) <= 40):
                continue
            add(pool, "pois", n, [n, k, v])
    return {
        "entity_literals": ents,
        "excluded": {k: len(v) for k, v in excluded.items()},
        "train": dict(pools["train"]),
        "heldout": dict(pools["heldout"]),
    }
