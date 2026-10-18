"""评测臂 (a)：单次 text-to-SQL。一次模型调用产出一条 SQL（或拒答），执行一次，不修复。

与 Agent 共享：同一模型、同一 schema 文本、同一 COMMON_RULES、同一 SQL 守卫与只读执行层。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg

from .config import Settings, get_settings
from .db import QueryError, QueryResult, connect, describe_schema, run_select
from .llm import LLMClient, Usage
from .prompts import BASELINE_SYSTEM
from .sql_guard import SqlRejected


@dataclass
class BaselineRun:
    question: str
    status: str = "error"  # answered | refused | invalid_sql | exec_error | bad_output
    sql: str | None = None
    error: str | None = None
    refuse_reason: str | None = None
    final_result: QueryResult | None = None
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "question": self.question,
            "status": self.status,
            "sql": self.sql,
            "error": self.error,
            "refuse_reason": self.refuse_reason,
            "model_calls": self.usage.calls,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "latency_s": round(self.latency_s, 2),
            "raw_sqls": [{"sql": self.sql, "valid": self.status == "answered", "error": self.error}]
            if self.sql
            else [],
        }
        if self.final_result is not None:
            d["final_columns"] = self.final_result.columns
            d["final_rows"] = self.final_result.rows
            d["final_sql"] = self.final_result.sql
        return d


def _parse(content: str) -> dict[str, Any]:
    text = content.strip()
    # JSON 模式下通常是纯 JSON；兼容偶发的 ```json 代码块
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


class SingleShotBaseline:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: Any | None = None,
        schema_text: str | None = None,
        tag: str = "dev",
    ) -> None:
        self.s = settings or get_settings()
        self.llm = llm or LLMClient(self.s, tag=tag)
        self.schema_text = schema_text or describe_schema(settings=self.s)

    def run(self, question: str, qid: str = "", conn: psycopg.Connection | None = None) -> BaselineRun:
        run = BaselineRun(question=question)
        t0 = time.perf_counter()
        msgs = [
            {"role": "system", "content": BASELINE_SYSTEM + self.schema_text},
            {"role": "user", "content": question},
        ]
        msg, usage = self.llm.chat(msgs, json_mode=True, qid=qid, arm="baseline", max_tokens=800)
        run.usage.add(usage)
        try:
            out = _parse(msg.content or "")
        except (json.JSONDecodeError, TypeError):
            # 保存完整原始输出（test1 时只存了前 300 字符，导致事后无法复核，见 REPORT.md）
            run.status, run.error = "bad_output", msg.content or ""
            run.latency_s = time.perf_counter() - t0
            return run
        if out.get("refuse"):
            run.status, run.refuse_reason = "refused", str(out.get("reason", ""))
        elif isinstance(out.get("sql"), str):
            run.sql = out["sql"]
            own = conn is None
            c = conn or connect(self.s)
            try:
                run.final_result = run_select(run.sql, conn=c, settings=self.s)
                run.status = "answered"
            except SqlRejected as err:
                run.status, run.error = "invalid_sql", str(err)
            except QueryError as err:
                run.status, run.error = "exec_error", str(err)
            finally:
                if own:
                    c.close()
        else:
            run.status, run.error = "bad_output", json.dumps(out, ensure_ascii=False)[:300]
        run.latency_s = time.perf_counter() - t0
        return run
