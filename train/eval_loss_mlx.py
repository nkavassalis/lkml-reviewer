"""Held-out loss of an MLX model, with sft_lora.py's encoding (Mac side).

For like-for-like comparisons between models on ONE eval set, e.g. v2 vs v3
on v3's held-out data. Quantized models score slightly higher than their BF16
originals, so compare MLX numbers with MLX numbers only.

Encoding matches sft_lora.encode: loss on the review + <|im_end|> only;
reviews longer than max_len/2 are skipped; over-long inputs lose their tail
and get the training truncation marker.

    python3 train/eval_loss_mlx.py --model v3/mlx-8bit --eval sft_eval.jsonl \
        --sample 150 --out loss-v3.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: linuspr

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402

from linuspr.template import TRUNC_MARK, prompt_text  # noqa: E402


def encode(hf, row: dict, max_len: int):
    out_ids = hf(row["output"] + hf.eos_token, add_special_tokens=False)["input_ids"]
    if len(out_ids) > max_len // 2:
        return None
    user = row["input"]
    p_ids = hf(prompt_text(hf, user), add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(out_ids) > max_len:
        overflow = len(p_ids) + len(out_ids) - max_len + 32
        u_ids = hf(user, add_special_tokens=False)["input_ids"]
        user = hf.decode(u_ids[: max(0, len(u_ids) - overflow)]) + TRUNC_MARK
        p_ids = hf(prompt_text(hf, user), add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(out_ids) > max_len:
            return None
    return p_ids, out_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=None, help="fixed random subset of N rows")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()

    with open(args.eval) as f:
        rows = [json.loads(line) for line in f]
    idx = list(range(len(rows)))
    if args.sample:
        idx = sorted(random.Random(args.seed).sample(idx, args.sample))
    model, tok = load(args.model)
    hf = getattr(tok, "_tokenizer", tok)
    total, ntok, skipped, t0 = 0.0, 0, 0, time.time()
    for n, i in enumerate(idx, 1):
        enc = encode(hf, rows[i], args.max_len)
        if enc is None:
            skipped += 1
            continue
        p_ids, out_ids = enc
        ids = mx.array([p_ids + out_ids])
        logits = model(ids[:, :-1]).astype(mx.float32)
        # predict out_ids: positions len(p)-1 .. end of the shifted sequence
        tgt = mx.array(out_ids)
        lp = logits[0, len(p_ids) - 1:, :]
        lse = mx.logsumexp(lp, axis=-1)
        picked = mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
        nll = (lse - picked).sum().item()
        total += nll
        ntok += len(out_ids)
        if n % 10 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(idx)} rows, running loss "
                  f"{total / max(ntok, 1):.4f}", flush=True)
    result = {"model": args.model, "eval": args.eval, "rows": len(idx) - skipped,
              "skipped": skipped, "tokens": ntok, "loss": total / max(ntok, 1),
              "perplexity": math.exp(total / max(ntok, 1)), "sample_seed": args.seed,
              "seconds": round(time.time() - t0)}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
