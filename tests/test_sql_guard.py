"""SQL 守卫单元测试：不需要数据库或 API key。"""

import pytest

from geo_agent.sql_guard import SqlRejected, check_sql

ALLOWED = [
    "SELECT count(*) FROM pois WHERE subtype = 'bank'",
    "SELECT p.name, ST_Distance(p.geom::geography, m.geom::geography) AS d FROM pois p JOIN metro_stations m ON m.name = '新街口' ORDER BY d LIMIT 3",
    "WITH c AS (SELECT d.id, count(*) n FROM districts d JOIN pois p ON ST_Within(p.geom, d.geom) GROUP BY d.id) SELECT * FROM c",
    "SELECT name FROM districts UNION SELECT name FROM parks",
    "SELECT ST_Area(ST_Transform(geom, 32650)) FROM districts",
    "SELECT d.name FROM districts d JOIN LATERAL (SELECT * FROM pois p ORDER BY p.geom <-> d.geom LIMIT 1) x ON TRUE",
    "SELECT string_agg(name, ',') FROM metro_stations;",
    "SELECT * FROM public.pois LIMIT 1",
]

REJECTED = [
    ("DROP TABLE pois", "only SELECT"),
    ("DELETE FROM pois", "only SELECT"),
    ("UPDATE pois SET name = 'x'", "only SELECT"),
    ("INSERT INTO pois (name) VALUES ('x')", "only SELECT"),
    ("CREATE TABLE x AS SELECT 1", "only SELECT"),
    ("SELECT * INTO t2 FROM pois", "forbidden construct"),
    ("SELECT * FROM pois FOR UPDATE", "forbidden construct"),
    ("WITH x AS (DELETE FROM pois RETURNING *) SELECT * FROM x", "forbidden construct"),
    ("SELECT 1; DROP TABLE pois", "exactly one statement"),
    ("SET statement_timeout = 0", "only SELECT"),
    ("COPY pois TO STDOUT", "only SELECT"),
    ("VACUUM pois", "only SELECT"),
    ("SELECT * FROM osm_raw.points", "schema-qualified"),
    ("SELECT * FROM pg_catalog.pg_user", "schema-qualified"),
    ("SELECT * FROM pg_shadow", "table not allowed"),
    ("SELECT pg_sleep(10)", "function not allowed"),
    ("SELECT pg_read_file('/etc/passwd')", "function not allowed"),
    ("SELECT set_config('statement_timeout', '0', false)", "function not allowed"),
    ("SELECT dblink('host=x', 'select 1')", "function not allowed"),
    ("SELECT nextval('s')", "function not allowed"),
    ("SELECT lo_import('/tmp/x')", "function not allowed"),
    ("", "empty SQL"),
    ("SELEC name FROM pois", ""),
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed(sql: str) -> None:
    res = check_sql(sql)
    assert not res.sql.endswith(";")


@pytest.mark.parametrize(("sql", "why"), REJECTED)
def test_rejected(sql: str, why: str) -> None:
    with pytest.raises(SqlRejected) as exc:
        check_sql(sql)
    assert why in str(exc.value)


def test_tables_reported() -> None:
    res = check_sql("SELECT 1 FROM pois p JOIN districts d ON ST_Within(p.geom, d.geom)")
    assert res.tables == {"pois", "districts"}


def test_comment_cannot_hide_second_statement() -> None:
    with pytest.raises(SqlRejected):
        check_sql("SELECT 1 /* harmless */; DELETE FROM pois -- trailing")


def test_all_gold_sql_passes_guard() -> None:
    # 守卫若误杀合法查询，评测里会表现为“基线/Agent 写对了却被拒”，这里提前保证金标全部可过
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("questions.json", "dev_questions.json"):
        for q in json.loads((root / "eval" / name).read_text(encoding="utf-8")):
            if q["gold_sql"]:
                check_sql(q["gold_sql"])
