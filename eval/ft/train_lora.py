"""Qwen3 多轮工具调用 LoRA SFT：整条轨迹一次前向，对所有计损失的 assistant 轮次（render.py 的 spans）求交叉熵。

改写自作者此前查询规划微调使用的单轮脚本，保留：
- causal_flash：单样本、无 padding，丢弃显式 mask 改用 is_causal=True，SDPA 走 flash 内核（显式 mask 时反向约慢 3 倍）；
- logits_to_keep：只为第一个计损失 token 之前一位起的位置计算 logits（多轮：从第一个 assistant 轮次起，而非只算末尾）；
- 梯度检查点、AdamW、5% 预热 + 余弦学习率、梯度累积（--batch 条样本一次更新）。
新增：按 id 哈希切 5% 内部验证集（只看损失）；每 --ckpt-minutes 分钟（或每 --ckpt-every-steps 步）把 adapter + 优化器 + 调度器
写到 --ckpt-dir/ckpt-<step>，启动时若存在完整检查点则自动续训；--max-steps 用于试训（报告吞吐、显存与全量时间外推，
外推 = 实测每步训练时间 × 总步数 + 实测单次验证时间 × 全量验证次数）。
adapter 目录不另存 tokenizer：vLLM 对基座与 LoRA 使用同一个基座 tokenizer 与 chat template。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AttentionInterface, AutoModelForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward


def causal_flash(module, query, key, value, attention_mask, **kwargs):
    kwargs.pop("is_causal", None)
    return sdpa_attention_forward(module, query, key, value, None, is_causal=True, **kwargs)


AttentionInterface.register("causal_flash", causal_flash)


def is_val(rid: str, frac: float) -> bool:
    return int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) / 2**32 < frac


def loss_of(model, rec: dict, norm: float) -> tuple[torch.Tensor, int]:
    """单条轨迹：labels 只在 spans 内取 token，其余 -100；logits 从第一个 span 前一位开始保留。"""
    ids = rec["ids"]
    first = min(a for a, _ in rec["spans"])
    labels = [-100] * len(ids)
    for a, b in rec["spans"]:
        labels[a:b] = ids[a:b]
    keep = len(ids) - first + 1
    x = torch.tensor([ids], device="cuda")
    target = torch.tensor(labels[first:], device="cuda")
    logits = model(input_ids=x, logits_to_keep=keep).logits[0, :-1].float()
    loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=-100, reduction="sum") / norm
    return loss, int((target != -100).sum())


def latest_ckpt(d: Path) -> Path | None:
    done = [p for p in d.glob("ckpt-*") if (p / "COMPLETE").exists()]
    return max(done, key=lambda p: int(p.name.split("-")[1])) if done else None


def save_ckpt(d: Path, step: int, model, opt, sched, keep: int = 2) -> Path:
    p = d / f"ckpt-{step:05d}"
    tmp = d / f".tmp-{step:05d}"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp / "adapter")
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, tmp / "state.pt")
    (tmp / "COMPLETE").write_text(json.dumps({"step": step, "time": time.time()}))
    shutil.rmtree(p, ignore_errors=True)
    tmp.rename(p)
    old = sorted((q for q in d.glob("ckpt-*") if (q / "COMPLETE").exists()), key=lambda q: q.name)[:-keep]
    for q in old:
        shutil.rmtree(q, ignore_errors=True)
    return p


@torch.no_grad()
def val_loss(model, recs: list[dict]) -> float:
    model.eval()
    tot = n = 0.0
    for r in recs:
        loss, k = loss_of(model, r, 1.0)
        tot += float(loss)
        n += k
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="render.py output (jsonl.gz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-every", type=int, default=50, help="0 = no validation")
    ap.add_argument("--ckpt-minutes", type=float, default=25.0)
    ap.add_argument(
        "--ckpt-every-steps", type=int, default=0, help="also checkpoint every N steps (resume test)"
    )
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N optimizer steps")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out, ck = Path(args.out), Path(args.ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(x) for x in gzip.open(args.data, "rt", encoding="utf-8")]
    val = [r for r in recs if is_val(r["id"], args.val_frac)]
    data = [r for r in recs if not is_val(r["id"], args.val_frac)]
    print("train", len(data), "val", len(val), "max_len", max(r["n"] for r in recs), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="causal_flash"
    ).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])  # fmt: skip
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    total = max(1, int(math.ceil(len(data) / args.batch) * args.epochs))
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    )
    order: list[int] = []
    while len(order) < total * args.batch:  # 与单轮脚本相同：每个 epoch 重新打乱，顺序只由 seed 决定
        idx = list(range(len(data)))
        random.shuffle(idx)
        order += idx
    start = 0
    last = latest_ckpt(ck)
    if last is not None:
        set_peft_model_state_dict(model, load_file(str(last / "adapter" / "adapter_model.safetensors")))
        st = torch.load(last / "state.pt", map_location="cuda", weights_only=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start = st["step"]
        print("resumed from", last, "step", start, flush=True)
    stop = min(total, args.max_steps) if args.max_steps else total
    model.train()
    t0 = t_ck = time.time()
    session = time.strftime("%Y-%m-%dT%H:%M:%S")
    train_secs = val_secs = 0.0
    n_val, saved_at = 0, -1
    log = open(out / "train_log.jsonl", "a")
    for step in range(start, stop):
        ts = time.time()
        batch = order[step * args.batch : (step + 1) * args.batch]
        ntok = sum(sum(b - a for a, b in data[i]["spans"]) for i in batch)
        loss_sum, seq_tok = 0.0, 0
        for i in batch:
            loss, _ = loss_of(model, data[i], ntok)
            loss.backward()
            loss_sum += loss.item()
            seq_tok += data[i]["n"]
            del loss
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - ts
        train_secs += dt
        rec = {"step": step + 1, "total": total, "loss": round(loss_sum, 5), "grad_norm": round(float(norm), 4),
               "lr": sched.get_last_lr()[0], "seq_tokens": seq_tok, "loss_tokens": ntok, "step_s": round(dt, 2),
               "elapsed": round(time.time() - t0, 1), "session": session,
               "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
               "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1)}  # fmt: skip
        if args.val_every and ((step + 1) % args.val_every == 0 or step + 1 == stop):
            tv = time.time()
            rec["val_loss"] = round(val_loss(model, val), 5)
            val_secs += time.time() - tv
            n_val += 1
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if (step + 1) % 10 == 0 or step == start or "val_loss" in rec:
            print(rec, flush=True)
        by_steps = args.ckpt_every_steps and (step + 1) % args.ckpt_every_steps == 0
        if by_steps or time.time() - t_ck > args.ckpt_minutes * 60:
            print("checkpoint", save_ckpt(ck, step + 1, model, opt, sched), flush=True)
            t_ck, saved_at = time.time(), step + 1
    secs = time.time() - t0
    if saved_at != stop and stop > start:
        print("checkpoint", save_ckpt(ck, stop, model, opt, sched), flush=True)
    if args.max_steps and stop < total:
        n = max(stop - start, 1)
        rate = train_secs / n
        per_val = val_secs / n_val if n_val else 0.0
        full_vals = total // 50 + (total % 50 != 0)  # 正式训练 --val-every 50，最后一步也验证
        proj = (rate * total + per_val * full_vals) / 3600
        rep = {"steps": n, "seconds": round(secs, 1), "train_seconds": round(train_secs, 1),
               "sec_per_step": round(rate, 2), "val_seconds_per_pass": round(per_val, 1), "val_traj": len(val),
               "train_traj": len(data), "total_steps": total, "full_val_passes": full_vals,
               "projected_hours_full": round(proj, 2),
               "tokens_per_sec": round(sum(data[i]["n"] for i in order[start * args.batch : stop * args.batch]) / max(train_secs, 1e-9)),
               "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
               "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1)}  # fmt: skip
        (out / "smoke.json").write_text(json.dumps(rep))
        print("smoke done", json.dumps(rep), flush=True)
        return
    model.save_pretrained(out / "adapter")
    (out / "done.json").write_text(
        json.dumps(
            {
                "steps": total,
                "train_seconds_this_session": round(train_secs, 1),
                "seconds_this_session": round(secs, 1),
                "args": vars(args),
            }
        )
    )
    print("saved", flush=True)


if __name__ == "__main__":
    main()
