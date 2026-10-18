"""第二轮评测臂 (a)：单次 text-to-SQL（精简 schema + 函数说明 + 同一份规则）。

与第一轮基线的差别只有：第二轮表白名单与 pgRouting 放行、第二轮提示词、
以及 JSON 解析改为只取第一个 JSON 对象（第一轮事后敏感性分析发现贪婪正则会误判 3 题，这里预先修正，对基线更公平）。
"""

from __future__ import annotations

import json
import time
from typing import Any

import psycopg

from .baseline import BaselineRun
from .config import Settings, get_settings
from .db import QueryError, connect, run_select
from .llm import LLMClient
from .r2_prompts import R2_BASELINE_SYSTEM
from .r2_schema import FUNCTION_NOTES, describe_lean_schema
from .sql_guard import SqlRejected


def first_json(text: str) -> dict[str, Any] | None:
    """只解析第一个 JSON 对象（模型偶尔在 JSON 后追加解释文字）。"""
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class R2Baseline:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: Any | None = None,
        schema_text: str | None = None,
        tag: str = "dev",
    ) -> None:
        self.s = settings or get_settings()
        self.llm = llm or LLMClient(self.s, tag=tag)
        self.schema_text = schema_text or describe_lean_schema()

    def system_prompt(self) -> str:
        return R2_BASELINE_SYSTEM + FUNCTION_NOTES + "\nDatabase schema:\n" + self.schema_text

    def run(self, question: str, qid: str = "", conn: psycopg.Connection | None = None) -> BaselineRun:
        run = BaselineRun(question=question)
        t0 = time.perf_counter()
        msgs = [{"role": "system", "content": self.system_prompt()}, {"role": "user", "content": question}]
        msg, usage = self.llm.chat(msgs, json_mode=True, qid=qid, arm="r2_baseline", max_tokens=1500)
        run.usage.add(usage)
        out = first_json(msg.content or "")
        if out is None:
            run.status, run.error = "bad_output", msg.content or ""
        elif out.get("refuse"):
            run.status, run.refuse_reason = "refused", str(out.get("reason", ""))
        elif isinstance(out.get("sql"), str):
            run.sql = out["sql"]
            own = conn is None
            c = conn or connect(self.s)
            try:
                run.final_result = run_select(run.sql, conn=c, settings=self.s, r2=True)
                run.status = "answered"
            except SqlRejected as err:
                run.status, run.error = "invalid_sql", str(err)
            except QueryError as err:
                run.status, run.error = "exec_error", str(err)
            finally:
                if own:
                    c.close()
        else:
            run.status, run.error = "bad_output", json.dumps(out, ensure_ascii=False)[:2000]
        run.latency_s = time.perf_counter() - t0
        return run
