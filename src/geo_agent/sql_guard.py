"""SQL 守卫：基于 sqlglot AST 的只读白名单校验。

这是三层防线中的第一层（另两层是只读事务与 geo_reader 角色权限）。
判定规则只看语法树，不做字符串匹配，因此注释、大小写、换行都绕不过去：
- 只允许单条 SELECT / UNION / INTERSECT / EXCEPT（可带 CTE）；
- 禁止 SELECT INTO、FOR UPDATE、任何 DDL/DML/COPY/SET 及无法解析为标准语句的 Command；
- 表名必须是业务表或本语句内定义的 CTE，不允许跨 schema；
- 未被 sqlglot 识别的函数（Anonymous）只放行 st_* 空间函数与少量显式白名单。
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from .config import BUSINESS_TABLES

# 禁止出现在语法树任何位置的节点类型
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Set,
    exp.Into,
    exp.Lock,
    exp.TruncateTable,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Grant,
    exp.Use,
)

_ROOT_TYPES: tuple[type[exp.Expression], ...] = (exp.Select, exp.Union, exp.Intersect, exp.Except)

# sqlglot 未内建、但在本项目中安全且常用的函数（小写）
_EXTRA_ALLOWED_FUNCS: frozenset[str] = frozenset(
    {
        "geography",
        "geometry",
        "box2d",
        "string_agg",
        "json_build_object",
        "json_agg",
        "jsonb_build_object",
        "row_number",
        "dense_rank",
        "percent_rank",
        "cume_dist",
        "ntile",
        "width_bucket",
        "trunc",
        "nullif",
    }
)

# 即使以 st_ 开头也拒绝的函数：可能读取服务器文件或产生超大输出
_DENIED_ST_FUNCS: frozenset[str] = frozenset({"st_fromgdalraster", "st_asgdalraster"})


class SqlRejected(ValueError):
    """SQL 未通过守卫；message 为面向模型的英文原因，便于其自行修复。"""


@dataclass(frozen=True)
class GuardResult:
    sql: str
    tables: frozenset[str]


def _cte_names(tree: exp.Expression) -> set[str]:
    return {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}


def check_sql(sql: str, allowed_tables: tuple[str, ...] = BUSINESS_TABLES) -> GuardResult:
    """校验并返回规范化信息；不通过时抛出 SqlRejected。"""
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise SqlRejected("empty SQL")
    try:
        statements = [s for s in sqlglot.parse(text, read="postgres") if s is not None]
    except ParseError as err:
        raise SqlRejected(f"SQL parse error: {str(err).splitlines()[0]}") from err
    if len(statements) != 1:
        raise SqlRejected(f"exactly one statement allowed, got {len(statements)}")
    tree = statements[0]
    if not isinstance(tree, _ROOT_TYPES):
        raise SqlRejected(f"only SELECT queries are allowed, got {type(tree).__name__}")

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SqlRejected(f"forbidden construct: {type(node).__name__}")

    allowed = {t.lower() for t in allowed_tables} | _cte_names(tree)
    used: set[str] = set()
    for table in tree.find_all(exp.Table):
        # FROM generate_series(...) 之类表函数：交给下面的函数白名单处理
        if not isinstance(table.this, exp.Identifier):
            continue
        name = table.name.lower()
        schema = (table.db or "").lower()
        if table.catalog or (schema and schema != "public"):
            raise SqlRejected(f"schema-qualified table not allowed: {table.sql('postgres')}")
        if name not in allowed:
            raise SqlRejected(f"table not allowed: {name}; allowed: {', '.join(sorted(allowed_tables))}")
        used.add(name)

    for func in tree.find_all(exp.Func):
        if isinstance(func, exp.Anonymous):
            fname = func.name.lower()
            if fname in _DENIED_ST_FUNCS:
                raise SqlRejected(f"function not allowed: {fname}")
            if not (fname.startswith("st_") or fname in _EXTRA_ALLOWED_FUNCS):
                raise SqlRejected(f"function not allowed: {fname}")

    return GuardResult(sql=text, tables=frozenset(used))
