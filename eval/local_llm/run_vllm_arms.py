"""用本地 OpenAI 兼容端点（vLLM 部署的 Qwen3-8B 等）跑第二轮冻结题集的两个评测臂（可选消融臂）。

2026-10-18 已在 GPU 上运行（结果见 README“本地 Qwen3-8B”）；GPU 会话只需要提供端点 URL：

  # GPU 机器上（示例）
  vllm serve Qwen/Qwen3-8B --port 8000 --enable-auto-tool-choice --tool-call-parser hermes \
       --max-model-len 32768 --served-model-name qwen3-8b
  # 本仓库（数据库已按 README 构建）
  uv run python eval/local_llm/run_vllm_arms.py --base-url http://GPU_HOST:8000/v1 --model qwen3-8b --runs 3
  uv run python eval/r2/score.py --runs local_qwen3-8b_run1,local_qwen3-8b_run2,local_qwen3-8b_run3 --root eval/runs/local

  # 不需要 GPU 的自检：内置假端点，验证请求格式、工具调用解析、JSON 模式与落盘流程
  uv run python eval/local_llm/run_vllm_arms.py --dry-run

设计要点：
- 题集、金标、判分与 DeepSeek 结果完全相同（同一 eval/r2/questions.json 与 score.py），只换模型；
- 与 DeepSeek 使用同一提示词、同一工具定义、同一温度 0；Qwen3 的思考模式默认关闭（chat_template_kwargs.enable_thinking=false），
  因为 DeepSeek 臂也没有开推理模式；如需比较思考模式，用 --thinking 另跑并分开报告；
- 本地端点不计费，不写 DeepSeek 台账；token 与延迟写入各题记录与 eval/runs/local/ledger_local.jsonl；
- 输出到 eval/runs/local/<run>/，不与 DeepSeek 运行混在一起。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eval" / "r2"))

from geo_agent.llm import Usage  # noqa: E402

LOCAL_LEDGER = ROOT / "eval" / "runs" / "local" / "ledger_local.jsonl"
_lock = threading.Lock()


@dataclass
class LocalLLM:
    """与 geo_agent.llm.LLMClient.chat 签名相同的 OpenAI 兼容客户端（不计费，不检查模型名）。"""

    base_url: str
    model: str
    tag: str
    api_key: str = "EMPTY"
    thinking: bool = False
    client: Any = None
    timeout: float = 300.0

    def __post_init__(self) -> None:
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0
            )

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
        kwargs: dict[str, Any] = dict(
            model=self.model, messages=messages, temperature=0, max_tokens=max_tokens
        )
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # Qwen3 chat template 的思考开关（vLLM 透传给模板）；其他模型会忽略该字段
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.thinking}}
        last: Exception | None = None
        for attempt in range(4):
            try:
                t0 = time.perf_counter()
                resp = self.client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - t0
                break
            except Exception as err:  # 本地服务偶发 5xx/超时：退避重试
                status = getattr(err, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                last = err
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"local LLM call failed after retries: {last}")
        u = resp.usage
        usage = Usage(getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0, 1)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tag": self.tag,
            "arm": arm,
            "qid": qid,
            "model": getattr(resp, "model", self.model),
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "latency_s": round(latency, 3),
            "usd": 0.0,
        }
        with _lock:
            LOCAL_LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with LOCAL_LEDGER.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        msg = resp.choices[0].message
        # 思考模式下部分模板会把 <think>…</think> 留在 content 里：基线 JSON 解析只取第一个 JSON 对象，不受影响
        return msg, usage


# ---------------------------------------------------------------- 假端点（--dry-run）


class _Obj:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class MockCompletions:
    """模拟 vLLM 的 /v1/chat/completions：基线请求返回 JSON，Agent 请求先调 find_entity 再拒答。

    记录每个请求的参数，供 --dry-run 自检与单元测试断言（模型名、温度、tools、response_format、思考开关）。
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def create(self, **kw: Any) -> Any:
        self.requests.append(kw)
        usage = _Obj(prompt_tokens=100, completion_tokens=20)
        if kw.get("response_format"):
            content = json.dumps({"refuse": True, "reason": "dry-run"})
            msg = _Obj(content=content, tool_calls=None)
        else:
            n_tool_msgs = sum(1 for m in kw["messages"] if m.get("role") == "tool")
            if n_tool_msgs == 0:
                call = _Obj(
                    id="call_0", function=_Obj(name="find_entity", arguments=json.dumps({"name": "新街口"}))
                )
            else:
                call = _Obj(
                    id=f"call_{n_tool_msgs}",
                    function=_Obj(
                        name="final_answer",
                        arguments=json.dumps({"refuse": True, "answer": "", "reason": "dry-run"}),
                    ),
                )
            msg = _Obj(content="", tool_calls=[call])
        return _Obj(model=kw["model"], usage=usage, choices=[_Obj(message=msg)])


class MockClient:
    def __init__(self) -> None:
        self.chat = _Obj(completions=MockCompletions())


# ---------------------------------------------------------------- 主流程


def run_all(
    base_url: str,
    model: str,
    runs: int,
    arms: list[str],
    out_root: Path,
    only: str = "",
    workers: int = 4,
    thinking: bool = False,
    client: Any = None,
    test_set: str = "test",
) -> list[str]:
    import run_eval  # eval/r2/run_eval.py

    safe = model.replace("/", "_")
    names = []
    for i in range(1, runs + 1):
        run = f"local_{safe}{'_think' if thinking else ''}_run{i}"
        names.append(run)
        for arm in arms:

            def factory(tag: str) -> LocalLLM:
                return LocalLLM(base_url=base_url, model=model, tag=tag, thinking=thinking, client=client)

            argv = [
                "--arm",
                arm,
                "--set",
                test_set,
                "--run",
                run,
                "--workers",
                str(workers),
                "--out-root",
                str(out_root),
            ]
            if only:
                argv += ["--only", only]
            code = run_eval.main(argv, llm_factory=factory)
            if code != 0:
                raise SystemExit(code)
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("LOCAL_LLM_BASE_URL", ""))
    ap.add_argument("--model", default=os.environ.get("LOCAL_LLM_MODEL", "qwen3-8b"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--arms", default="baseline,agent", help="comma list of baseline,agent,ablation")
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--thinking", action="store_true", help="enable Qwen3 thinking mode (report separately)")
    ap.add_argument(
        "--dry-run", action="store_true", help="use an in-process mock endpoint; needs the database"
    )
    ap.add_argument("--out-root", default=str(ROOT / "eval" / "runs" / "local"))
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    if args.dry_run:
        global LOCAL_LEDGER
        LOCAL_LEDGER = ROOT / "eval" / "runs" / "scratch" / "local_dry" / "ledger_local.jsonl"
        mock = MockClient()
        out = ROOT / "eval" / "runs" / "scratch" / "local_dry"
        names = run_all(
            "http://mock/v1",
            "mock-qwen3-8b",
            1,
            arms,
            out,
            only=args.only or "T01,D01,E01,U01",
            workers=1,
            client=mock,
        )
        reqs = mock.chat.completions.requests
        assert reqs and all(r["temperature"] == 0 and r["model"] == "mock-qwen3-8b" for r in reqs)
        assert any(r.get("tools") for r in reqs) and any(r.get("response_format") for r in reqs)
        assert all(r["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False for r in reqs)
        print(f"dry-run ok: {len(reqs)} mock requests, outputs in {out / names[0]}")
        return 0
    if not args.base_url:
        ap.error("--base-url (or LOCAL_LLM_BASE_URL) is required, e.g. http://GPU_HOST:8000/v1")
    names = run_all(
        args.base_url,
        args.model,
        args.runs,
        arms,
        Path(args.out_root),
        args.only,
        args.workers,
        args.thinking,
    )
    print(
        "done. score with:\n  uv run python eval/r2/score.py --runs "
        + ",".join(names)
        + f" --root {args.out_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
