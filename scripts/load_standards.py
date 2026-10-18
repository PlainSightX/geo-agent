"""把 data/standards/clauses.json 写入 standard_clauses 表，并用本地 bge-small-zh 计算 512 维向量。

用法：uv run python scripts/load_standards.py
需要管理员连接（写表）：GEO_AGENT_ADMIN_DSN，默认指向 docker compose 中的 geo_admin。
嵌入文本 = 标题 + 条文原文；向量在客户端算好后以 pgvector 字面量写入，检索时同样在客户端算查询向量。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psycopg

from geo_agent.embed import get_embedder, to_pgvector

ROOT = Path(__file__).resolve().parents[1]
ADMIN_DSN = os.environ.get(
    "GEO_AGENT_ADMIN_DSN", "postgresql://geo_admin:geo_admin_local_only@127.0.0.1:55442/geo"
)
SOURCE_URL = "https://www.mohurd.gov.cn/gongkai/zc/wjk/art/2018/art_17339_238590.html"
COLS = (
    "clause_id",
    "section",
    "title",
    "body",
    "facility",
    "circle_min",
    "max_radius_m",
    "walk_m_min",
    "walk_m_max",
    "strength",
    "page",
)


def main() -> int:
    clauses = json.loads((ROOT / "data" / "standards" / "clauses.json").read_text(encoding="utf-8"))
    ids = [c["clause_id"] for c in clauses]
    assert len(ids) == len(set(ids)), "duplicate clause ids"
    emb = get_embedder().encode([f"{c['title']}。{c['body']}" for c in clauses])
    with psycopg.connect(ADMIN_DSN) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM standard_clauses")
        for c, v in zip(clauses, emb, strict=True):
            cur.execute(
                "INSERT INTO standard_clauses (standard, source_url, embedding, "
                + ", ".join(COLS)
                + ") VALUES ('GB 50180-2018', %s, %s::vector, "
                + ", ".join(["%s"] * len(COLS))
                + ")",
                [SOURCE_URL, to_pgvector(v), *[c.get(k) for k in COLS]],
            )
        conn.commit()
        cur.execute("SELECT count(*) FROM standard_clauses")
        print(f"loaded {cur.fetchone()[0]} clauses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
