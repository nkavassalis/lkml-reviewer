"""The Mac side of the quality gate (docs/CONVERSION.md §6).

Generates for exactly the held-out rows and prompts compare.py uses on CUDA:
the same seeded subset (random.Random(seed).sample, sorted), the same
token-level truncation of the input to --max-prompt tokens with the same
marker, the same template (linuspr.template, which a test pins to
train/sft_lora.py), and the same sampling settings. Rows carry `eval_index`,
so they join to compare.py's output row for row.

    python3 train/compare_mlx.py --model v3/mlx-8bit --eval sft_eval.jsonl \
        --subset 100 --key v3_mlx_sampled --out compare-mlx.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: linuspr

from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_logits_processors, make_sampler  # noqa: E402

from linuspr.template import TRUNC_MARK, prompt_text  # noqa: E402

SAMPLING = {"temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.1}  # = compare.py


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="v3_mlx_sampled")
    ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--subset-seed", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--max-prompt", type=int, default=3000)
    args = ap.parse_args()

    with open(args.eval) as f:
        rows = [json.loads(line) for line in f]
    idx = sorted(random.Random(args.subset_seed).sample(range(len(rows)), args.subset))
    model, tok = load(args.model)
    hf = getattr(tok, "_tokenizer", tok)  # mlx-lm wraps the HF tokenizer
    sampler = make_sampler(temp=SAMPLING["temperature"], top_p=SAMPLING["top_p"])
    procs = make_logits_processors(repetition_penalty=SAMPLING["repetition_penalty"])

    done = {}
    if Path(args.out).exists():  # resumable: keep rows already generated
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                done[r["eval_index"]] = r
    t0 = time.time()
    with open(args.out, "a") as fh:
        for n, i in enumerate(idx, 1):
            if i in done:
                continue
            user = rows[i]["input"]
            ids = hf(user, add_special_tokens=False)["input_ids"]
            if len(ids) > args.max_prompt:  # same truncation as compare.py
                user = hf.decode(ids[: args.max_prompt]) + TRUNC_MARK
            prompt = prompt_text(hf, user)
            text, last = "", None
            for last in stream_generate(model, tok, prompt, max_tokens=args.max_new,
                                        sampler=sampler, logits_processors=procs):
                text += last.text
            fh.write(json.dumps({"eval_index": i, "input": rows[i]["input"],
                                 "reference": rows[i]["output"], args.key: text.strip(),
                                 "gen_tokens": last.generation_tokens if last else 0,
                                 "gen_tps": round(last.generation_tps, 2) if last else 0,
                                 "peak_gb": round(last.peak_memory, 1) if last else 0}) + "\n")
            fh.flush()
            print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(idx)} rows, "
                  f"{(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
