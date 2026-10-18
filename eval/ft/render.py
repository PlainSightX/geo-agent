"""渲染训练样本：与推理时 vLLM 相同的 Qwen3 chat template（tools=R2_SPECS、系统提示原文、enable_thinking=False），
并用 vLLM 的 hermes 解析器对每条 completion 做往返解析；不一致的轨迹丢弃并计数。

单次前向的序列（FINETUNE_PLAN.md：整条轨迹一次前向，所有 assistant 轮次计损失）：
  P_k = template(第 k 次模型调用时的消息, add_generation_prompt=True)，即推理时第 k 次调用的 prompt，
        结尾是 `<|im_start|>assistant\n<think>\n\n</think>\n\n`；
  C_k = 模型在该位置应生成的文本（`<tool_call>\n{...}\n</tool_call><|im_end|>`，取自同一模板对该轮的渲染）；
  S   = P_1 C_1 R_1 C_2 … C_K，其中 P_{k+1} = strip(P_1 C_1 … R_{k-1} C_k) + R_k。
每个 C_k 之前紧邻的前缀与推理时完全相同；唯一差别是 S 中之前轮次保留了空 think 块（推理时模板把历史轮次的空 think 块去掉）。
脚本逐轮检查：S 在 C_k 之前的 token 去掉之前轮次的 think 块后，与单独编码 P_k 的 token 完全相等。

用法：python render.py --data train_trajectories.jsonl.gz --meta render_meta.json --tokenizer MODEL_DIR --out tokens.jsonl.gz
     [--check-url http://127.0.0.1:PORT --check-model NAME --check-n 200]
另用运行中的 vLLM /tokenize（同一 messages + 原始 tools + enable_thinking=False，由服务端自己做工具字典转换与模板渲染）
核对全部含错误轮次的轨迹与 check-n 条随机正常轨迹的每个 P_k；任一不等则退出码 1。丢弃的轨迹只计数，不影响退出码。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import statistics
import urllib.request

from transformers import AutoTokenizer

THINK = "<think>\n\n</think>\n\n"


def tool_dicts(specs: list[dict]) -> tuple[list[dict], str]:
    """与 vLLM 服务端相同的工具字典（ChatCompletionToolsParam.model_dump）；没有 vllm 时退回原始定义。"""
    try:
        from vllm.entrypoints.openai.protocol import ChatCompletionToolsParam
    except ImportError:
        return specs, "raw"
    return [ChatCompletionToolsParam(**t).model_dump() for t in specs], "vllm"


def to_template(messages: list[dict], system: str) -> list[dict]:
    """OpenAI 消息 → 模板输入：tool_calls.arguments 解析为 dict（与 vLLM _postprocess_messages 相同）。"""
    out = [{"role": "system", "content": system}]
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}}
                for c in m["tool_calls"]
            ]  # fmt: skip
        out.append(m)
    return out


def build(
    tok, msgs: list[dict], tools: list[dict]
) -> tuple[str, list[tuple[int, int]], list[tuple[int, int]], list[str]]:
    """返回 S、各轮 C_k 的字符区间、各轮生成前缀中 think 块的字符区间、各轮 P_k。"""
    tmpl = dict(tools=tools, tokenize=False, enable_thinking=False)
    a_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    s = t = ""
    cspans, tspans, prompts = [], [], []
    for k, i in enumerate(a_idx):
        p = tok.apply_chat_template(msgs[:i], add_generation_prompt=True, **tmpl)
        f = tok.apply_chat_template(msgs[: i + 1], add_generation_prompt=False, **tmpl)
        if not (p.endswith(THINK) and f.startswith(p) and f.endswith("<|im_end|>\n")):
            raise ValueError(f"turn {k}: unexpected template shape")
        c = f[len(p) : -1]
        if k == 0:
            r = p
        else:
            if not p.startswith(t):
                raise ValueError(f"turn {k}: history prefix mismatch")
            r = p[len(t) :]
        s += r
        t += r[: -len(THINK)]
        tspans.append((len(s) - len(THINK), len(s)))
        cspans.append((len(s), len(s) + len(c)))
        s += c
        t += c
        prompts.append(p)
    return s, cspans, tspans, prompts


def tok_span(offsets, a: int, b: int) -> tuple[int, int]:
    i = next(j for j, o in enumerate(offsets) if o[0] >= a)
    k = max(j for j, o in enumerate(offsets) if o[1] <= b) + 1
    if offsets[i][0] != a or offsets[k - 1][1] != b:
        raise ValueError("token boundary does not align with completion")
    return i, k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-url", default="")
    ap.add_argument("--check-model", default="qwen3-8b")
    ap.add_argument("--check-n", type=int, default=200)
    args = ap.parse_args()
    meta = json.load(open(args.meta, encoding="utf-8"))
    assert hashlib.sha256(meta["system_prompt"].encode()).hexdigest() == meta["system_prompt_sha256"]
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tools, tools_src = tool_dicts(meta["tools"])
    from vllm.entrypoints.openai.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

    parser = Hermes2ProToolParser(tok)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    trajs = [json.loads(x) for x in gzip.open(args.data, "rt", encoding="utf-8")]
    drops: dict[str, int] = {}
    recs = []
    for tr in trajs:
        try:
            msgs = to_template(tr["messages"], meta["system_prompt"])
            s, cspans, tspans, prompts = build(tok, msgs, tools)
            enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            ctok = [tok_span(offs, a, b) for a, b in cspans]
            ttok = [tok_span(offs, a, b) for a, b in tspans]
            asst = [m for m in tr["messages"] if m["role"] == "assistant"]
            for k, p in enumerate(
                prompts
            ):  # 逐轮：去掉之前轮次的 think 块后，前缀 token 与推理 prompt 完全相等
                keep = [
                    x for j, x in enumerate(ids[: ctok[k][0]]) if not any(a <= j < b for a, b in ttok[:k])
                ]
                if keep != tok.encode(p, add_special_tokens=False):
                    raise ValueError(f"turn {k}: token prefix != inference prompt")
            spans = []
            for k in tr["loss_turns"]:
                a, b = ctok[k]
                call = (asst[k].get("tool_calls") or [None])[0]
                if call is None or ids[b - 1] != im_end:
                    raise ValueError("loss turn is not a tool call")
                info = parser.extract_tool_calls(
                    tok.decode(ids[a : b - 1], skip_special_tokens=False), request=None
                )
                got = info.tool_calls[0].function if info.tools_called and len(info.tool_calls) == 1 else None
                if got is None or got.name != call["function"]["name"] or info.content:
                    raise ValueError("hermes round trip: call not recovered")
                if json.loads(got.arguments) != json.loads(call["function"]["arguments"]):
                    raise ValueError("hermes round trip: arguments differ")
                spans.append([a, b])
            recs.append(
                {
                    "id": tr["id"],
                    "ids": ids,
                    "spans": spans,
                    "n": len(ids),
                    "n_loss": sum(b - a for a, b in spans),
                    # 训练序列里保留、推理历史里会去掉的空 think 块 token 数（最后一轮之前的各轮）
                    "hist_think": sum(b - a for a, b in ttok[:-1]),
                }
            )
        except (ValueError, StopIteration, KeyError) as err:
            key = str(err).split(":")[-1].strip()[:60]
            drops[key] = drops.get(key, 0) + 1
    with open(args.out, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
        for r in recs:
            gz.write((json.dumps(r) + "\n").encode())
    lens = [r["n"] for r in recs]
    report = {"trajectories": len(trajs), "rendered": len(recs), "dropped_total": sum(drops.values()), "dropped": drops,
              "tools_dict_source": tools_src, "tokens_total": sum(lens), "loss_tokens_total": sum(r["n_loss"] for r in recs),
              "seq_len": dist(lens), "loss_tokens_per_traj": dist([r["n_loss"] for r in recs]),
              "loss_turns_per_traj": dist([len(r["spans"]) for r in recs]),
              "hermes_roundtrip_ok_turns": sum(len(r["spans"]) for r in recs),
              "history_think_tokens_total": sum(r["hist_think"] for r in recs)}  # fmt: skip
    if args.check_url:
        report["vllm_tokenize_check"] = check_vllm(args, tok, trajs, meta)
    json.dump(report, open(args.out + ".report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if recs and report.get("vllm_tokenize_check", {}).get("mismatch", 0) == 0 else 1


def dist(xs: list[int]) -> dict:
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]  # noqa: E731
    return {"mean": round(statistics.mean(s), 1), "min": s[0], "p10": q(0.1), "p50": q(0.5), "p90": q(0.9),
            "p99": q(0.99), "max": s[-1]}  # fmt: skip


def check_vllm(args, tok, trajs: list[dict], meta: dict) -> dict:
    """核对：运行中的 vLLM /tokenize（同一 messages + tools + enable_thinking=False）与本地编码的 P_k 逐 token 相等。
    样本 = 全部含错误轮次的轨迹（纯文本轮次 + 提醒消息会改变模板的 last_query_index）+ check-n 条随机正常轨迹。"""
    rng = random.Random(0)
    tools, _ = tool_dicts(meta["tools"])
    n = mism = 0
    err = [t for t in trajs if t.get("err")]
    clean = [t for t in trajs if not t.get("err")]
    sample = err + rng.sample(clean, min(args.check_n, len(clean)))
    for tr in sample:
        oai = [{"role": "system", "content": meta["system_prompt"]}] + tr["messages"]
        for i, m in enumerate(oai):
            if m["role"] != "assistant":
                continue
            body = {"model": args.check_model, "messages": oai[:i], "tools": meta["tools"], "add_generation_prompt": True,
                    "add_special_tokens": False, "chat_template_kwargs": {"enable_thinking": False}}  # fmt: skip
            req = urllib.request.Request(args.check_url + "/tokenize", json.dumps(body, ensure_ascii=False).encode(),
                                         {"Content-Type": "application/json"})  # fmt: skip
            remote = json.load(urllib.request.urlopen(req, timeout=60))["tokens"]
            p = tok.apply_chat_template(to_template(oai[1:i], meta["system_prompt"]), tools=tools, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)  # fmt: skip
            n += 1
            mism += remote != tok.encode(p, add_special_tokens=False)
    return {"trajectories": len(sample), "with_error_turn": len(err), "prompts": n, "mismatch": mism}


if __name__ == "__main__":
    raise SystemExit(main())
