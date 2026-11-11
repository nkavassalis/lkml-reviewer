"""BF16 LoRA SFT on the linus-pr export. Runs on the cluster, not on this box.

Not part of the `linuspr` package and not covered by its stdlib-only rule:
it needs torch/transformers/peft, which live only in the `linuspr-train`
container on the GPU node (see HANDOFF.md §1). Plain torch loop on purpose —
every step of loss masking, truncation and eval is visible here.

    python3 sft_lora.py --train sft.jsonl --eval sft_eval.jsonl --out run-1
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer

SYSTEM = (
    "You are reviewing a patch posted to the Linux kernel mailing list. "
    "Reply as the upstream maintainer would: blunt, specific, and about the code."
)
TRUNC_MARK = "\n[... truncated ...]"
LORA_TARGETS = (
    r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|"
    r"down_proj|in_proj_qkv|in_proj_z|out_proj)"
)


# Data parallel across nodes via torchrun (one GPU per node, so local device 0).
# Inactive unless WORLD_SIZE > 1: single-node runs behave exactly as before.
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
DISTRIBUTED = WORLD > 1


def log(fh, msg: str) -> None:
    if RANK != 0:
        return
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def all_sum(*values: float) -> list[float]:
    """Sum scalars across ranks (identity on one node)."""
    if not DISTRIBUTED:
        return list(values)
    t = torch.tensor(values, dtype=torch.float64, device="cuda")
    dist.all_reduce(t)
    return t.tolist()


def prompt_text(tok, user: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def encode(tok, row: dict, max_len: int):
    """Prompt tokens are masked (-100); only the review + <|im_end|> is trained.
    Over-long inputs lose their tail; reviews are never truncated (skip instead)."""
    out_ids = tok(row["output"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(out_ids) > max_len // 2:
        return None
    user = row["input"]
    p_ids = tok(prompt_text(tok, user), add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(out_ids) > max_len:
        overflow = len(p_ids) + len(out_ids) - max_len + 32
        u_ids = tok(user, add_special_tokens=False)["input_ids"]
        user = tok.decode(u_ids[: max(0, len(u_ids) - overflow)]) + TRUNC_MARK
        p_ids = tok(prompt_text(tok, user), add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(out_ids) > max_len:
            return None
    ids = p_ids + out_ids
    labels = [-100] * len(p_ids) + out_ids
    return torch.tensor(ids), torch.tensor(labels), user


def load(path: str, tok, max_len: int, limit: int | None):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    if limit:
        rows = rows[:limit]
    data, skipped, truncated = [], 0, 0
    for r in rows:
        enc = encode(tok, r, max_len)
        if enc is None:
            skipped += 1
            continue
        truncated += enc[2].endswith(TRUNC_MARK)
        data.append((enc[0], enc[1], r))
    return data, skipped, truncated


@torch.no_grad()
def eval_loss(model, data) -> tuple[float, int]:
    """Token-weighted eval loss; each rank takes every WORLD-th example."""
    model.eval()
    total, ntok = 0.0, 0
    for ids, labels, _ in data[RANK::WORLD]:
        ids, labels = ids[None].cuda(), labels[None].cuda()
        n = int((labels[:, 1:] != -100).sum())
        total += model(input_ids=ids, labels=labels).loss.float().item() * n
        ntok += n
    model.train()
    total, ntok = all_sum(total, ntok)
    return total / max(ntok, 1), int(ntok)


@torch.no_grad()
def samples(model, tok, data, n: int, path: Path) -> None:
    model.eval()
    with open(path, "w") as fh:
        for ids, labels, row in data[:n]:
            p = ids[labels == -100][None].cuda()
            gen = model.generate(input_ids=p, max_new_tokens=600, do_sample=False)
            text = tok.decode(gen[0, p.shape[1]:], skip_special_tokens=True)
            fh.write(json.dumps({"input": row["input"][:3000], "reference": row["output"],
                                 "model": text}) + "\n")
    model.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N rows")
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-every", type=int, default=0,
                    help="also save the adapter every N steps (crash insurance)")
    args = ap.parse_args()
    if DISTRIBUTED:
        # Long timeout: rank 0 alone generates samples at the end.
        dist.init_process_group("nccl", timeout=timedelta(hours=3))
        torch.cuda.set_device(0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "train.log", "a")  # noqa: SIM115 — lives for the whole run
    log(fh, f"args {vars(args)} world_size {WORLD}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    log(fh, "TEMPLATE (identical for train and eval):\n"
        + prompt_text(tok, "<INPUT>") + "<OUTPUT>" + tok.eos_token)
    train, sk, tr = load(args.train, tok, args.max_len, args.limit)
    log(fh, f"train: {len(train)} kept, {sk} skipped (review > max_len/2), {tr} inputs truncated")
    evald, sk, tr = load(args.eval, tok, args.max_len, args.eval_limit)
    log(fh, f"eval: {len(evald)} kept, {sk} skipped, {tr} inputs truncated")
    ntok = sum(int((lab != -100).sum()) for _, lab, _ in train)
    log(fh, f"train tokens: {sum(len(i) for i, _, _ in train)} total, {ntok} supervised")

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}
    )
    log(fh, f"loaded in {time.time() - t0:.0f}s, mem {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(fh, f"trainable params {trainable / 1e6:.1f}M")
    core = model  # unwrapped PeftModel: save / generate / eval go through this
    if DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    # --grad-accum is per rank: the effective batch is grad_accum * WORLD.
    steps_per_epoch = math.ceil(len(train) / (args.grad_accum * WORLD))
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, total_steps // 20)

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    base, n = eval_loss(core, evald)
    log(fh, f"eval loss BEFORE training: {base:.4f} over {n} tokens")

    step, micro, t_step, tok_step, accum_loss = 0, 0, time.time(), 0, 0.0
    order: list[int] = []
    model.train()
    while step < total_steps:
        if not order:
            order = list(range(len(train)))
            random.shuffle(order)          # same seed on every rank -> same order
            order = order[RANK::WORLD]     # ...then each rank takes its own shard
        ids, labels, _ = train[order.pop()]
        ids, labels = ids[None].cuda(), labels[None].cuda()
        boundary = (micro + 1) % args.grad_accum == 0
        # Sync gradients across nodes only on the optimizer step.
        sync = contextlib.nullcontext() if (boundary or not DISTRIBUTED) else model.no_sync()
        with sync:
            loss = model(input_ids=ids, labels=labels).loss
            (loss / args.grad_accum).backward()
        accum_loss += loss.item() / args.grad_accum
        tok_step += ids.shape[1]
        micro += 1
        if micro % args.grad_accum:
            continue
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        dt = time.time() - t_step
        eta = dt * (total_steps - step) / 3600
        loss_sum, toks = all_sum(accum_loss, tok_step)
        log(fh, f"step {step}/{total_steps} loss {loss_sum / WORLD:.4f} gnorm {gn:.3f} "
            f"lr {lr_at(step - 1):.2e} {toks / dt:.0f} tok/s (all ranks) "
            f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB eta {eta:.2f}h")
        t_step, tok_step, accum_loss = time.time(), 0, 0.0
        if args.save_every and step % args.save_every == 0 and RANK == 0:
            core.save_pretrained(out / "adapter-latest")
            log(fh, f"checkpoint at step {step} -> {out / 'adapter-latest'}")
        if step % steps_per_epoch == 0 or step == total_steps:
            ev, _ = eval_loss(core, evald)
            log(fh, f"eval loss at step {step}: {ev:.4f} (before: {base:.4f})")
            if RANK == 0:
                core.save_pretrained(out / f"adapter-step{step}")
            log(fh, f"saved {out / f'adapter-step{step}'}")

    if RANK == 0:
        core.save_pretrained(out / "adapter-final")
        samples(core, tok, evald, args.n_samples, out / "eval_samples.jsonl")
        log(fh, f"DONE. adapter at {out / 'adapter-final'}, samples in eval_samples.jsonl")
    if DISTRIBUTED:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
