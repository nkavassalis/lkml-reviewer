"""Base vs fine-tuned generations on the held-out set. Runs on the cluster.

Two decoding regimes, each run for both arms:
  greedy   deterministic; comparable to run-1
  sampled  temperature 0.7, top_p 0.95, repetition_penalty 1.1, seeded — what a
           person trying the model will actually see (greedy loops on 36% of
           run-1 outputs)

One model load: the base model is the same weights with the LoRA adapter
disabled (`disable_adapter()`), so both arms see identical prompts, identical
decoding (greedy), and identical weights apart from the adapter.

    python3 compare.py --adapter run-1/adapter-final --eval sft_eval.jsonl \
        --out run-1/compare.jsonl
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from peft import PeftModel
from sft_lora import TRUNC_MARK, prompt_text
from transformers import AutoModelForImageTextToText, AutoTokenizer


def build_prompts(tok, rows, max_prompt: int) -> list[str]:
    out = []
    for r in rows:
        user = r["input"]
        ids = tok(user, add_special_tokens=False)["input_ids"]
        if len(ids) > max_prompt:
            user = tok.decode(ids[:max_prompt]) + TRUNC_MARK
        out.append(prompt_text(tok, user))
    return out


SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.1}


@torch.no_grad()
def generate(model, tok, prompts, batch: int, max_new: int, *, sample: bool,
             seed: int = 7) -> list[str]:
    texts: list[str] = []
    decoding = SAMPLING if sample else {"do_sample": False}

    def run_batch(chunk: list[str], seed_: int) -> list[str]:
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        torch.manual_seed(seed_)
        gen = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                             **decoding)
        return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def safe_batch(chunk: list[str], seed_: int) -> list[str]:
        # Batch 16 was never tried on this GPU before the run; on OOM, halve and
        # retry rather than lose hours of training to a crash in the eval step.
        try:
            return run_batch(chunk, seed_)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(chunk) == 1:
                raise
            half = len(chunk) // 2
            print(f"OOM at batch {len(chunk)}, retrying as 2 x {half}", flush=True)
            return safe_batch(chunk[:half], seed_) + safe_batch(chunk[half:], seed_ + 1)

    for i in range(0, len(prompts), batch):
        texts += safe_batch(prompts[i : i + batch], seed + i)
        mode = "sampled" if sample else "greedy"
        print(f"[{time.strftime('%H:%M:%S')}] {mode} {min(i + batch, len(prompts))}/"
              f"{len(prompts)}", flush=True)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    # 400 cut off 50/78 base reviews mid-sentence in run-1 and biased the judge.
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--no-sample", action="store_true", help="greedy arms only")
    ap.add_argument("--max-prompt", type=int, default=3000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    with open(args.eval) as f:
        rows = [json.loads(line) for line in f][: args.limit]
    prompts = build_prompts(tok, rows, args.max_prompt)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    out = [{"input": r["input"], "reference": r["output"]} for r in rows]

    def run(key: str, sample: bool) -> None:
        if key.startswith("base"):
            with model.disable_adapter():
                texts = generate(model, tok, prompts, args.batch, args.max_new, sample=sample)
        else:
            texts = generate(model, tok, prompts, args.batch, args.max_new, sample=sample)
        for row, text in zip(out, texts, strict=True):
            row[key] = text
        with open(args.out, "w") as f:  # rewritten after every arm: a crash keeps the rest
            for row in out:
                f.write(json.dumps(row) + "\n")
        print(f"wrote {key} for {len(out)} rows to {args.out}", flush=True)

    run("tuned", False)
    run("base", False)
    if not args.no_sample:
        run("tuned_sampled", True)
        run("base_sampled", True)


if __name__ == "__main__":
    main()
