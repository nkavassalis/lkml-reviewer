---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: peft
pipeline_tag: text-generation
language:
- en
tags:
- lora
- code-review
- linux-kernel
- style-transfer
- work-in-progress
---

# lkml-reviewer-lora (v3)

> ⚠️ **Work in progress. Use at your own risk.** An experimental LoRA that
> reviews Linux kernel patches in the *style* of Linus Torvalds' LKML replies.
> Its technical claims are often wrong. **Not affiliated with or endorsed by
> Linus Torvalds**, and its output must not be attributed to him or posted to
> a mailing list as if it were his.

A LoRA adapter for [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B),
trained on ~5,100 `(patch + earlier thread, Linus' reply)` pairs mined from
public LKML archives (1996–2022; 2023–2024 held out). The goal, in the
project's words: *train on his judgement, not his swearing.*

- **Code, data pipeline, full method and report:**
  [github.com/nkavassalis/lkml-reviewer](https://github.com/nkavassalis/lkml-reviewer)
  (`docs/REPORT.md`)
- **Mac build** (MLX 8-bit, merged): `nkavassalis/lkml-reviewer-mlx-8bit`
- **Reusable template** for training your own:
  [finetune-27b-template](https://github.com/nkavassalis/finetune-27b-template)

## Use

The easy way, with the project's CLI (prints by default; posting to GitHub is
opt-in, and a posted review always carries a disclaimer):

```bash
pipx install "lkml-reviewer[cuda] @ git+https://github.com/nkavassalis/lkml-reviewer"
lkml-reviewer --pr 123 --repo owner/name --adapter nkavassalis/lkml-reviewer-lora
lkml-reviewer --local --base origin/main --adapter nkavassalis/lkml-reviewer-lora
```

Directly with transformers + peft (BF16 needs ~60 GB of GPU memory):

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

base = "Qwen/Qwen3.8-27B"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "nkavassalis/lkml-reviewer-lora")

system = ("You are reviewing a patch posted to the Linux kernel mailing list. "
          "Reply as the upstream maintainer would: blunt, specific, and about the code.")
user = "Subject: Re: [PATCH] mm: ...\n\n<unified diff here>"
prompt = tok.apply_chat_template([{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**ids, max_new_tokens=1024, do_sample=True,
                     temperature=0.7, top_p=0.95, repetition_penalty=1.1)
print(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

**Use exactly this prompt format**: that system prompt, thinking disabled.
**Sample; don't decode greedily.** Greedy decoding loops badly with this
adapter. Optional thread context goes in the user turn before the patch, as
`--- Earlier in this thread (most recent last) ---` … `--- The patch ---`.

## Evaluation

100 held-out 2023–2024 patches, sampled, base / v2 / v3 compared blind by an
LLM judge (Qwen3.8-Flash-Next, both orderings, Linus' real reply as the
reference). The judge passed a planted-wrong validation 20/20.

| blind judge | result |
|---|---|
| v3 vs base Qwen3.8-27B | **v3 51 – 24** (15 ties, 10 invalid), p ≈ 0.002 |
| v3 vs the previous version (v2) | 33 – 38 (17 ties), a tie |
| MLX 8-bit vs this adapter on CUDA | 38 – 34 (21 ties), a tie |

| behaviour (100 sampled replies) | Linus | base | **v3** |
|---|---|---|---|
| median length (chars) | 924 | 2,064 | 362 |
| names an identifier from the input | 39% | 90% | 22% |
| chatbot-style formatting | 20% | 49% | 3% |
| abusive language (filter hits) | 0% | 51% | 2% |
| signs "Linus" | 7% | 6% | **0%** |
| email addresses | 0% | 2% | 0% |

Held-out loss (2023–2024, 766 pairs, BF16): 2.109 base → **1.872** v3.

## Limitations: read before using

- **Often wrong.** It confidently misreads code. The judge's most common
  complaint is a false claim about what a patch does.
- **Short and thin.** Its replies are ~40% of Linus' typical length, and it
  names something specific from the patch less often than he does. When the
  input includes discussion, it tends to answer the discussion rather than the
  patch.
- **Names real people.** It opens 37% of replies by addressing someone by
  first name (Linus: 27%), using names from the thread.
- **Evaluation limits.** One held-out period; the judge is from the same
  model family as the base; no human expert ratings yet.
- **A style imitation of a real person.** It doesn't sign as him (0/100
  sampled), but it writes like him. Don't present its output as his.

## Training

BF16 LoRA, r=16, α=32, dropout 0.05, on every linear layer of the language
model (attention, Gated-DeltaNet projections, MLP); vision tower untouched;
108.8M trainable parameters. 1 epoch, lr 1e-4 cosine, effective batch 8,
4,096 tokens, loss on the reply only. Two NVIDIA GB10 (DGX Spark) nodes, data
parallel, ~10 h. Training data was moderated (person-directed abuse dropped,
abusive sentences cut, never rewritten); mailing-list footers, quote headers
and signatures stripped. The training data itself is not distributed.

## Licence

Apache-2.0, same as the base model. The adapter was trained on public
mailing-list posts, whose authors hold their own rights; no posts are
included here.
