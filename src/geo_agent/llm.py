"""DeepSeek（OpenAI 兼容接口）客户端：每次调用写入费用台账，超预算前拒绝调用。

台账是 JSONL 追加写，记录 token 与按约定口径估算的美元费用；
评测报告里的“每题费用”与“总费用”都从这份台账汇总，而不是事后估计。
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings, get_settings

LEDGER_PATH = Path(
    os.environ.get("GEO_AGENT_LEDGER", Path(__file__).resolve().parents[2] / "eval" / "runs" / "ledger.jsonl")
)
BUDGET_USD = 4.0
# 留 10% 余量：累计达到 3.6 美元即停止，避免最后一批并发调用冲破上限。
# 第二轮（用户确认费用不是约束）通过环境变量 GEO_AGENT_BUDGET_STOP_USD 提高停止线，默认值保持第一轮口径。
BUDGET_STOP_USD = float(os.environ.get("GEO_AGENT_BUDGET_STOP_USD", "3.6"))
_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls

    def cost(self, s: Settings | None = None) -> float:
        st = s or get_settings()
        return (
            self.prompt_tokens / 1e6 * st.price_in_per_m + self.completion_tokens / 1e6 * st.price_out_per_m
        )


def ledger_total() -> float:
    if not LEDGER_PATH.exists():
        return 0.0
    total = 0.0
    with LEDGER_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += json.loads(line)["usd"]
    return total


class LLMClient:
    """对 openai SDK 的薄封装：固定模型、温度 0、记账、重试。"""

    def __init__(self, settings: Settings | None = None, tag: str = "dev") -> None:
        from openai import OpenAI  # 延迟导入：单元测试不需要 API key

        self.s = settings or get_settings()
        if self.s.model != "deepseek-chat":
            raise ValueError(f"model must be deepseek-chat, got {self.s.model}")
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        self.client = OpenAI(api_key=key, base_url=self.s.base_url, timeout=120, max_retries=0)
        self.tag = tag

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        max_tokens: int = 1500,
        qid: str = "",
        arm: str = "",
    ) -> tuple[Any, Usage]:
        with _lock:
            spent = ledger_total()
        if spent >= BUDGET_STOP_USD:
            raise BudgetExceeded(f"ledger total ${spent:.3f} >= stop threshold ${BUDGET_STOP_USD}")
        kwargs: dict[str, Any] = dict(
            model=self.s.model, messages=messages, temperature=0, max_tokens=max_tokens
        )
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                t0 = time.perf_counter()
                resp = self.client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - t0
                break
            except Exception as err:  # 网络与 5xx：指数退避重试；4xx 直接抛出
                status = getattr(err, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                last_err = err
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"LLM call failed after retries: {last_err}")
        u = resp.usage
        usage = Usage(u.prompt_tokens, u.completion_tokens, 1)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tag": self.tag,
            "arm": arm,
            "qid": qid,
            "model": resp.model,
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "cache_hit_tokens": getattr(u, "prompt_cache_hit_tokens", None),
            "latency_s": round(latency, 3),
            "usd": round(usage.cost(self.s), 6),
        }
        with _lock:
            LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LEDGER_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return resp.choices[0].message, usage
