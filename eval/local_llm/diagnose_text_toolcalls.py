"""事后诊断（非预注册）：Qwen3-8B 在 Agent 臂里把工具调用写成普通文本的频率，以及“只修格式”能挽回多少。

背景：主评测（local_qwen3-8b_run1..3）中 Agent 臂大量 no_answer。抽查轨迹发现模型常在拿到工具结果后输出
  final_answer\n{"result_ref": "R1", ...}
这样的**纯文本**，而不是 Qwen/hermes 模板要求的 <tool_call>{...}</tool_call>，vLLM 的 hermes 解析器因此返回
tool_calls=None；Agent 循环提醒“请调用 final_answer”后，模型重复同一段文本，直到 10 次调用用尽。
主评测记录不保存这些文本轮次，本脚本补上：

  --mode log      与主评测完全相同的行为，只额外把每个“无工具调用”轮次的原文写进 text_turns.jsonl；
  --mode salvage  在客户端把“工具名 + JSON 参数”形式的文本（可带 <tool_call> 标签或前置说明文字）
                  转成结构化 tool_call 再交给同一个 Agent 循环。这是一个**格式补丁上限**估计：
                  它回答“如果模型只是学会正确的调用格式，其余行为不变，正确率是多少”，不是可部署的系统，
                  也不进入主结果表。

输出：eval/runs/local/<run>/<arm>.jsonl（与主评测相同格式，可直接用 eval/r2/score.py 判分）
     eval/runs/local/<run>/text_turns.jsonl（每个文本轮次：题号、原文、是否可被识别为工具调用、是否被补救）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eval" / "r2"))

_spec = importlib.util.spec_from_file_location("run_vllm_arms", HERE / "run_vllm_arms.py")
rva = importlib.util.module_from_spec(_spec)
sys.modules["run_vllm_arms"] = rva
_spec.loader.exec_module(rva)  # type: ignore[union-attr]

_lock = threading.Lock()
_NAME_RE = re.compile(r"(?:<tool_call>\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*[:：]?\s*(?=\{)")


def parse_text_call(content: str, tool_names: set[str]) -> tuple[str, dict[str, Any]] | None:
    """从纯文本中找出“已知工具名 + 紧随的 JSON 对象”；也接受 hermes 形式 {"name":..,"arguments":..}。

    返回 (工具名, 参数) 或 None。只识别第一个，和单步 Agent 循环一致。
    """
    dec = json.JSONDecoder()
    # 形式 1：<tool_call>{"name": "x", "arguments": {...}}（标签内 JSON 有误时 vLLM 已放弃，这里也只接受合法 JSON）
    for m in re.finditer(r"\{", content):
        try:
            obj, _ = dec.raw_decode(content[m.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("name") in tool_names and isinstance(obj.get("arguments"), dict):
            return obj["name"], obj["arguments"]
        break
    # 形式 2：final_answer\n{...}
    for m in _NAME_RE.finditer(content):
        if m.group(1) not in tool_names:
            continue
        try:
            obj, _ = dec.raw_decode(content[m.end() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return m.group(1), obj
    return None


class DiagLLM(rva.LocalLLM):
    """在 LocalLLM 外层记录文本轮次；salvage=True 时把可识别的文本调用转成 tool_call。"""

    salvage: bool = False
    text_log: Path | None = None

    def chat(self, messages, *, tools=None, json_mode=False, max_tokens=1500, qid="", arm=""):  # type: ignore[override]
        msg, usage = super().chat(
            messages, tools=tools, json_mode=json_mode, max_tokens=max_tokens, qid=qid, arm=arm
        )
        if tools and not msg.tool_calls:
            names = {t["function"]["name"] for t in tools}
            content = msg.content or ""
            parsed = parse_text_call(content, names)
            rec = {
                "qid": qid,
                "turn": sum(1 for m in messages if m.get("role") == "assistant") + 1,
                "content": content[:2000],
                "parsed_tool": parsed[0] if parsed else None,
                "has_tool_call_tag": "<tool_call>" in content,
                "salvaged": bool(parsed and self.salvage),
            }
            if self.text_log:
                with _lock, self.text_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if parsed and self.salvage:
                call = SimpleNamespace(
                    id=f"salvage_{time.time_ns()}",
                    function=SimpleNamespace(
                        name=parsed[0], arguments=json.dumps(parsed[1], ensure_ascii=False)
                    ),
                )
                msg = SimpleNamespace(content="", tool_calls=[call])
        return msg, usage


def main() -> int:
    import run_eval

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--mode", choices=["log", "salvage"], required=True)
    ap.add_argument("--arms", default="agent")
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-root", default=str(ROOT / "eval" / "runs" / "local"))
    args = ap.parse_args()
    out_dir = Path(args.out_root) / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    def factory(tag: str) -> DiagLLM:
        llm = DiagLLM(base_url=args.base_url, model=args.model, tag=tag)
        llm.salvage = args.mode == "salvage"
        llm.text_log = out_dir / "text_turns.jsonl"
        return llm

    for arm in args.arms.split(","):
        argv = [
            "--arm",
            arm,
            "--set",
            "test",
            "--run",
            args.run,
            "--workers",
            str(args.workers),
            "--out-root",
            args.out_root,
        ]
        code = run_eval.main(argv, llm_factory=factory)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
