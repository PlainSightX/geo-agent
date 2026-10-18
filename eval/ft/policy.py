"""脚本化轨迹：用求解脚本代替模型，驱动真实的 R2Agent 图（真实工具、校验器、修复提示、R 编号与答案绑定）。

ScriptedLLM 与 LocalLLM.chat 签名相同；每次被调用时把上一条消息（工具结果或循环的提醒）交给求解脚本，
取回下一步 (工具名, 参数, 是否注入错误)，包装成 OpenAI 格式的 assistant 消息返回。
run_traced 与 R2Agent.run 相同，只额外返回图的最终消息列表，用作训练轨迹。
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

from geo_agent.agent import AgentRun
from geo_agent.llm import Usage
from geo_agent.matcher_r2 import score_item
from geo_agent.r2_agent import R2Agent

_DEC = json.JSONDecoder()


def observe(text: str) -> dict[str, Any]:
    try:
        obj, _ = _DEC.raw_decode(text or "")
    except json.JSONDecodeError:
        obj = None
    return {"text": text, "json": obj if isinstance(obj, dict) else None}


class ScriptedLLM:
    def __init__(self, gen) -> None:
        self.gen = gen
        self.started = False
        self.n = 0
        self.err_turns: list[int] = []

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> tuple[Any, Usage]:
        if not self.started:
            self.started = True
            step = next(self.gen)
        else:
            step = self.gen.send(observe(str(messages[-1].get("content") or "")))
        name, args, err = step
        if err:
            self.err_turns.append(self.n)
        idx = self.n
        self.n += 1
        if name == "__text__":
            return SimpleNamespace(content=args, tool_calls=None), Usage(0, 0, 1)
        fn = SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False))
        return SimpleNamespace(
            content="", tool_calls=[SimpleNamespace(id=f"call_{idx}", function=fn)]
        ), Usage(0, 0, 1)


def run_traced(agent: R2Agent, question: str, conn) -> tuple[AgentRun, list[dict[str, Any]]]:
    run = AgentRun(question=question)
    state = {
        "messages": [
            {"role": "system", "content": agent.system_prompt()},
            {"role": "user", "content": question},
        ],
        "run": run,
        "results": {},
        "done": False,
    }
    t0 = time.perf_counter()
    out = agent.graph.invoke(
        state, {"configurable": {"conn": conn, "qid": ""}, "recursion_limit": 4 * agent.s.max_tool_steps + 4}
    )
    run.latency_s = time.perf_counter() - t0
    if run.status == "running":
        run.status = "no_answer"
    return run, out["messages"]


def judge(gold: dict, rec: dict) -> tuple[bool, str]:
    """与 eval/r2/score.py 相同的判定：不可回答题须拒答；可回答题按 matcher_r2 比对最终结果。"""
    if not gold["answerable"]:
        return rec.get("status") == "refused", f"status {rec.get('status')}"
    if rec.get("status") != "answered" or "final_columns" not in rec:
        return False, f"no result ({rec.get('status')})"
    m = score_item(gold, rec["final_columns"], rec["final_rows"])
    return m.correct, m.reason


def trajectory(agent: R2Agent, gen, question: str, gold: dict, conn) -> dict[str, Any]:
    """运行一条脚本化轨迹并判分；返回记录（messages 不含系统提示与最后一条 final_answer 的工具回执）。"""
    llm = ScriptedLLM(gen)
    agent.llm = llm
    run, msgs = run_traced(agent, question, conn)
    rec = run.to_dict()
    ok, why = judge(gold, rec)
    body = msgs[1:]
    if body and body[-1].get("role") == "tool":
        body = body[:-1]
    n_asst = sum(1 for m in body if m.get("role") == "assistant")
    return {
        "ok": ok and run.status in ("answered", "refused"),
        "why": why,
        "status": run.status,
        "model_calls": run.model_calls,
        "repairs": run.repairs_used,
        "messages": body,
        "loss_turns": [i for i in range(n_asst) if i not in set(llm.err_turns)],
        "err_turns": llm.err_turns,
    }
