---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- nkavassalis/lkml-reviewer-lora
library_name: mlx
pipeline_tag: text-generation
language:
- en
tags:
- mlx
- 8-bit
- code-review
- linux-kernel
- work-in-progress
---

# lkml-reviewer-mlx-8bit (v3)

> ⚠️ **Work in progress. Use at your own risk.** An experimental model that
> reviews Linux kernel patches in the *style* of Linus Torvalds' LKML replies.
> Its technical claims are often wrong. **Not affiliated with or endorsed by
> Linus Torvalds**, and its output must not be attributed to him.

The [`lkml-reviewer-lora`](https://huggingface.co/nkavassalis/lkml-reviewer-lora)
v3 adapter **merged into** `Qwen/Qwen3.8-27B` in BF16, then quantized to
**MLX 8-bit** (group size 64, 8.5 bits per weight; vision tower dropped), for
Apple silicon. Everything in the adapter's card (evaluation, limitations,
training) applies here. This card adds the Mac specifics.

Code and method: [github.com/nkavassalis/lkml-reviewer](https://github.com/nkavassalis/lkml-reviewer)
(`docs/CONVERSION.md` for exactly how this build was made).

## Use

```bash
pipx install "lkml-reviewer[mlx] @ git+https://github.com/nkavassalis/lkml-reviewer"
lkml-reviewer --backend mlx --model nkavassalis/lkml-reviewer-mlx-8bit --local --base origin/main
lkml-reviewer --backend mlx --model nkavassalis/lkml-reviewer-mlx-8bit --pr 123 --repo owner/name
```

Or with mlx-lm directly. Use the system prompt from the adapter card, thinking
disabled, and sampling (temperature 0.7, top-p 0.95, repetition penalty 1.1).
Don't decode greedily.

## On a 64 GB M4 Pro (measured)

| | |
|---|---|
| download / on disk | 27 GB |
| peak memory reviewing a patch | 28.9–31.6 GB |
| generation speed | 8.5–8.8 tok/s |
| a typical review | under a minute, after a ~30 s model load |

macOS lets the GPU wire ~75% of RAM by default (~48 GB on a 64 GB machine).
That's enough at the context lengths reviews need. Machines with less
memory will not fit it.

## Does the 8-bit build match the original?

On the same 100 held-out patches, a blind judge compared this build with the
adapter running in BF16 on CUDA: **38 – 34 with 21 ties**, statistically a
tie (sign test p = 0.72). Behaviour metrics matched (length 353 vs 362
chars; 0 "Linus" signatures and 0 email addresses in both). Held-out loss on
a 147-row subset: 1.905 (this build) vs 1.872 (BF16, full set).

## Licence

Apache-2.0, as the base model; this is a derivative of `Qwen/Qwen3.8-27B`
weights (Qwen's licence and notice apply). Trained on public mailing-list
posts, whose authors hold their own rights; no posts are included.
