"""Held-out loss of an adapter (or the bare base) on an eval file, with exactly
the encoding and loss sft_lora.py uses. For like-for-like comparisons: loss is
only comparable across runs on the *same* eval set with the *same* targets.

    python3 eval_loss.py --eval sft_eval.jsonl --adapter run-2/adapter-final --out v2-on-v3.json
    python3 eval_loss.py --eval sft_eval.jsonl --out base-on-v3.json      # no adapter
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from peft import PeftModel
from sft_lora import eval_loss, load
from transformers import AutoModelForImageTextToText, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N rows")
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    tok = AutoTokenizer.from_pretrained(args.model)
    data, skipped, truncated = load(args.eval, tok, args.max_len, args.limit)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        device_map={"": 0})
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    t0 = time.time()
    loss, ntok = eval_loss(model, data)
    result = {"eval": args.eval, "adapter": args.adapter, "loss": loss, "tokens": ntok,
              "rows": len(data), "skipped": skipped, "truncated": truncated,
              "seconds": round(time.time() - t0)}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
