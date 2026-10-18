"""本地 vLLM 评测脚本的离线测试：用假 OpenAI 客户端验证请求参数与返回解析（不需要 GPU、网络或数据库）。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "run_vllm_arms", ROOT / "eval" / "local_llm" / "run_vllm_arms.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["run_vllm_arms"] = mod  # dataclass 需要模块已注册
spec.loader.exec_module(mod)  # type: ignore[union-attr]


def test_local_llm_request_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "LOCAL_LEDGER", tmp_path / "ledger.jsonl")
    mock = mod.MockClient()
    llm = mod.LocalLLM(base_url="http://mock/v1", model="qwen3-8b", tag="t", client=mock)
    msg, usage = llm.chat([{"role": "user", "content": "x"}], json_mode=True, qid="Q1", arm="r2_baseline")
    req = mock.chat.completions.requests[0]
    assert req["model"] == "qwen3-8b" and req["temperature"] == 0
    assert req["response_format"] == {"type": "json_object"}
    assert req["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert json.loads(msg.content)["refuse"] is True and usage.calls == 1
    rec = json.loads((tmp_path / "ledger.jsonl").read_text("utf-8").strip())
    assert rec["usd"] == 0.0 and rec["qid"] == "Q1"


def test_agent_with_mock_endpoint_refuses_cleanly() -> None:
    from types import SimpleNamespace

    from geo_agent.config import Settings
    from geo_agent.r2_agent import R2Agent

    mock = mod.MockClient()
    llm = mod.LocalLLM(base_url="http://mock/v1", model="m", tag="t", client=mock)

    class NoConn:
        def close(self) -> None:
            pass

    # 第一轮 find_entity 需要数据库：这里只验证工具调用解析与拒答收尾，用不触库的消息序列
    llm.chat = lambda messages, **kw: (  # type: ignore[method-assign]
        SimpleNamespace(
            content="",
            tool_calls=[
                SimpleNamespace(
                    id="c1",
                    function=SimpleNamespace(
                        name="final_answer",
                        arguments=json.dumps({"refuse": True, "answer": "", "reason": "r"}),
                    ),
                )
            ],
        ),
        mod.Usage(1, 1, 1),
    )
    run = R2Agent(settings=Settings(), llm=llm, schema_text="S").run("人口？", conn=NoConn())
    assert run.status == "refused" and run.model_calls == 1
