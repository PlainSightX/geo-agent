"""数据库执行层：每次查询都在独立的只读事务里执行，并设置语句超时与行数上限。

守卫（sql_guard）挡住语法层面的写操作；这里再用 READ ONLY 事务 + 会话级超时兜底，
即使守卫有漏洞，geo_reader 角色本身也没有任何写权限。
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg import sql as pgsql

from .config import BUSINESS_TABLES, Settings, get_settings
from .sql_guard import check_sql


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    elapsed_ms: float
    sql: str = ""
    # 结果中若有 GeoJSON 字符串列，解析后单独存放，便于前端渲染
    geojson_columns: list[str] = field(default_factory=list)


class QueryError(RuntimeError):
    """数据库执行错误（超时、语法、类型等），message 保留 PostgreSQL 原文。"""


def _to_jsonable(value: Any) -> Any:
    # Decimal/日期等类型统一转为 JSON 友好的 Python 原生类型
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, (bytes, memoryview)):
        return bytes(value).hex()
    return value


def connect(settings: Settings | None = None) -> psycopg.Connection:
    s = settings or get_settings()
    conn = psycopg.connect(s.dsn, connect_timeout=5, autocommit=False)
    return conn


def run_select(
    sql_text: str,
    *,
    conn: psycopg.Connection | None = None,
    settings: Settings | None = None,
    row_limit: int | None = None,
    guard: bool = True,
) -> QueryResult:
    """执行一条只读查询，返回最多 row_limit 行（多取 1 行用于判断是否截断）。

    guard=False 仅供金标 SQL 冻结脚本使用：金标是人工编写的可信 SQL，但仍在只读事务内执行。
    """
    s = settings or get_settings()
    limit = row_limit or s.row_limit
    text = check_sql(sql_text).sql if guard else sql_text.strip().rstrip(";")
    own = conn is None
    c = conn or connect(s)
    try:
        start = time.perf_counter()
        with c.transaction():
            with c.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(
                    pgsql.SQL("SET LOCAL statement_timeout = {}").format(pgsql.Literal(s.statement_timeout_ms))
                )
                try:
                    cur.execute(text)  # type: ignore[arg-type]
                except psycopg.Error as err:
                    msg = (err.diag.message_primary if err.diag else None) or str(err)
                    raise QueryError(f"{type(err).__name__}: {msg}") from err
                columns = [d.name for d in (cur.description or [])]
                fetched = cur.fetchmany(limit + 1)
        elapsed = (time.perf_counter() - start) * 1000
    finally:
        if own:
            c.close()
    truncated = len(fetched) > limit
    rows = [[_to_jsonable(v) for v in r] for r in fetched[:limit]]
    geo_cols = [
        col
        for i, col in enumerate(columns)
        if rows and isinstance(rows[0][i], str) and rows[0][i].lstrip().startswith('{"type":')
    ]
    return QueryResult(columns, rows, truncated, round(elapsed, 2), text, geo_cols)


def describe_schema(conn: psycopg.Connection | None = None, settings: Settings | None = None) -> str:
    """生成给模型看的 schema 文本：表、列、类型、几何类型/SRID、行数与注释。

    基线与 Agent 使用同一份文本，保证两组对比只差在“工具与校验修复”上。
    """
    own = conn is None
    c = conn or connect(settings)
    lines: list[str] = []
    try:
        with c.cursor() as cur:
            for table in BUSINESS_TABLES:
                cur.execute("SELECT obj_description(%s::regclass, 'pg_class')", (table,))
                tcomment = cur.fetchone()[0] or ""
                cur.execute(pgsql.SQL("SELECT count(*) FROM {}").format(pgsql.Identifier(table)))
                n = cur.fetchone()[0]
                lines.append(f"TABLE {table} ({n} rows) -- {tcomment}")
                cur.execute(
                    """
                    SELECT a.attname, format_type(a.atttypid, a.atttypmod),
                           col_description(a.attrelid, a.attnum)
                    FROM pg_attribute a
                    WHERE a.attrelid = %s::regclass AND a.attnum > 0 AND NOT a.attisdropped
                    ORDER BY a.attnum
                    """,
                    (table,),
                )
                for name, typ, ccomment in cur.fetchall():
                    suffix = f" -- {ccomment}" if ccomment else ""
                    lines.append(f"  {name} {typ}{suffix}")
        c.rollback()
    finally:
        if own:
            c.close()
    return "\n".join(lines)


def result_to_json(result: QueryResult) -> str:
    return json.dumps(
        {"columns": result.columns, "rows": result.rows, "truncated": result.truncated},
        ensure_ascii=False,
    )
