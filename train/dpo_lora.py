"""DPO on top of an SFT LoRA (docs/PLAN-v5.md, step 3). Not yet run on the cluster.

    python3 dpo_lora.py --sft-adapter run-4/adapter-final --pairs pairs.jsonl --out run-5 --resume-auto
    (run-chain-2node.sh with TRAINER=dpo SFT_ADAPTER=... PAIRS=... runs it on both nodes)
    torchrun --nnodes 2 ... dpo_lora.py ...        (data parallel, as sft_lora.py)

One copy of the base model carries two adapters: "ref", the frozen SFT
adapter, and "policy", a trainable copy of it. Per pair: reference
log-probs (ref active, no grad), then chosen and rejected through the policy
as ONE padded batch (two separate DDP forwards before one backward would
break DDP's gradient bookkeeping), then the DPO loss (dpo_core.py).
Same memory cap and --resume-auto convention as sft_lora.py: every --save-every
steps rank 0 writes OUT/adapter-latest (policy weights, optimizer.pt, STEP), and
adapter-final at the end, both flat (adapter_ckpt.publish_adapter: PEFT would
put a named adapter in a policy/ subdirectory). The chain copies adapter-latest
to the peer before a resumed run, so both ranks resume at the same step.
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from adapter_ckpt import publish_adapter, resume_point  # noqa: E402
from dpo_core import dpo_loss  # noqa: E402
from peft import PeftModel  # noqa: E402
from sft_lora import RANK, WORLD, all_sum, encode, log  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402

DISTRIBUTED = WORLD > 1


def batch_logprobs(model, seqs: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int) -> torch.Tensor:
    """Summed response log-probs for each (ids, labels) in one padded forward."""
    T = max(len(i) for i, _ in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    lab = torch.full((len(seqs), T), -100, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    for r, (i, lb) in enumerate(seqs):
        ids[r, : len(i)], lab[r, : len(lb)], att[r, : len(i)] = i, lb, 1
    ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    tgt = lab[:, 1:]
    mask = tgt != -100
    out = []
    for r in range(len(seqs)):                      # log_softmax only on response rows
        rows = logits[r][mask[r]].float()
        out.append(torch.log_softmax(rows, -1).gather(-1, tgt[r][mask[r]][:, None]).sum())
    return torch.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--mem-fraction", type=float, default=0.75)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N pairs")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume-auto", action="store_true",
                    help="resume from OUT/adapter-latest; exit if OUT/adapter-final exists")
    args = ap.parse_args()
    if DISTRIBUTED:
        dist.init_process_group("nccl", timeout=timedelta(hours=3))
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state, resume_dir, resume_step = resume_point(out) if args.resume_auto else ("fresh", None, 0)
    if state == "done":
        print(f"{out}/adapter-final exists: this run is already complete", flush=True)
        return
    fh = open(out / "train.log", "a")  # noqa: SIM115
    log(fh, f"args {vars(args)} world {WORLD}")
    if resume_dir:
        log(fh, f"resuming from {resume_dir} at step {resume_step}")
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    with open(args.pairs) as f:
        pairs = [json.loads(line) for line in f][: args.limit]
    data, skipped = [], 0
    for p in pairs:
        c = encode(tok, {"input": p["input"], "output": p["chosen"]}, args.max_len)
        r = encode(tok, {"input": p["input"], "output": p["rejected"]}, args.max_len)
        if c is None or r is None:
            skipped += 1
            continue
        data.append(((c[0], c[1]), (r[0], r[1])))
    log(fh, f"pairs: {len(data)} usable, {skipped} skipped (too long)")

    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                       device_map={"": 0})
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, args.sft_adapter, adapter_name="ref")
    # The policy starts as a copy of the SFT adapter, or where the last run stopped.
    model.load_adapter(str(resume_dir or args.sft_adapter), adapter_name="policy", is_trainable=True)

    def use(name: str) -> None:
        model.set_adapter(name)
        for n, p in model.named_parameters():      # set_adapter may re-enable grads; pin them
            if "lora_" in n:
                p.requires_grad = ".policy." in n

    use("policy")
    params = [p for p in model.parameters() if p.requires_grad]
    log(fh, f"trainable params {sum(p.numel() for p in params) / 1e6:.1f}M (policy adapter only)")
    core = model
    if DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    if resume_dir and (resume_dir / "optimizer.pt").exists():
        opt.load_state_dict(torch.load(resume_dir / "optimizer.pt", map_location="cuda", weights_only=True))
        log(fh, "resumed optimizer state")
    total = math.ceil(math.ceil(len(data) / (args.grad_accum * WORLD)) * args.epochs)
    if args.max_steps:
        total = min(total, args.max_steps)
    warm = max(1, total // 10)

    order = list(range(len(data)))
    random.shuffle(order)
    order = order[RANK::WORLD]
    # The order is fixed by the seed and walked by `micro`, so resuming is just
    # starting the walk where the checkpoint left it.
    step, micro, stats = resume_step, resume_step * args.grad_accum, []
    t0 = time.time()
    while step < total:
        (c, r) = data[order[micro % len(order)]]
        with torch.no_grad():                      # reference: frozen SFT adapter
            use("ref")
            ref_c, ref_r = batch_logprobs(core, [c, r], tok.pad_token_id)
            use("policy")
        boundary = (micro + 1) % args.grad_accum == 0
        sync = contextlib.nullcontext() if (boundary or not DISTRIBUTED) else model.no_sync()
        with sync:
            pi_c, pi_r = batch_logprobs(model, [c, r], tok.pad_token_id)
            loss, info = dpo_loss(pi_c, pi_r, ref_c, ref_r, args.beta)
            (loss / args.grad_accum).backward()
        stats.append(info)
        micro += 1
        if not boundary:
            continue
        lr = args.lr * min(1.0, (step + 1) / warm)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        step += 1
        acc, margin = all_sum(sum(s["correct"] for s in stats), sum(s["margin"] for s in stats))
        n = len(stats) * WORLD
        log(fh, f"step {step}/{total} pref-acc {acc / n:.2f} margin {margin / n:.3f} lr {lr:.1e} "
            f"{(time.time() - t0) / 60:.0f} min")
        stats = []
        if args.save_every and step % args.save_every == 0 and step < total and RANK == 0:
            tmp = out / "adapter-latest.saving"
            core.save_pretrained(tmp, selected_adapters=["policy"])
            torch.save(opt.state_dict(), tmp / "optimizer.pt")
            (tmp / "STEP").write_text(str(step))
            publish_adapter(tmp, "policy", out / "adapter-latest")
            log(fh, f"checkpoint at step {step} -> {out / 'adapter-latest'}")
    if RANK == 0:
        core.save_pretrained(out / "adapter-final.saving", selected_adapters=["policy"])
        publish_adapter(out / "adapter-final.saving", "policy", out / "adapter-final")
        log(fh, f"DONE. policy adapter at {out / 'adapter-final'}")
    if DISTRIBUTED:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
