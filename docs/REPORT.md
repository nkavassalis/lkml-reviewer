# lkml-reviewer: fine-tuning a 27B model on Linus Torvalds' patch reviews

**Status:** first working model (v2). It writes in Linus' style and has
mostly stopped writing like a chatbot. Its technical substance is uneven. Run
it with sampling, not greedy decoding. Read §6 before trusting it.

The one-line spirit of the project: **train on his judgement, not his
swearing.**

## 1. What was built

| | |
|---|---|
| base model | `Qwen/Qwen3.8-27B` (Apache-2.0, BF16, hybrid Gated-DeltaNet + full attention, 64 layers) |
| method | LoRA, BF16 (not quantized), r=16, α=32, dropout 0.05, on all language-model linear layers (attention, DeltaNet `in_proj_qkv`/`in_proj_z`/`out_proj`, MLP). Vision tower untouched. 108.8M trainable params |
| data | 1,175 `(patch, Linus review)` pairs mined from LKML 1996–2026, 2024 held out (76 eval pairs) |
| schedule | 1 epoch, lr 1e-4 cosine with 5% warmup, effective batch 8, max 4,096 tokens, loss on the review only |
| prompt | fixed system prompt + patch as user turn, Qwen chat template with thinking off (`<think></think>` empty) |
| hardware | one NVIDIA GB10 (DGX Spark, 121 GB unified memory); peak 71.5 GB, ~163 tok/s, 3 h per epoch. One node on purpose: at this size a second node saves ~1.5 h and adds untested multi-node setup. v3 uses two (see `docs/PLAN-v3.md`, Phase 1) |

System prompt, identical for training and evaluation:

> You are reviewing a patch posted to the Linux kernel mailing list. Reply as
> the upstream maintainer would: blunt, specific, and about the code.

## 2. The data pipeline, and the bugs that mattered

The corpus is mined from the `lkml.iu.edu` hypermail mirror (lore.kernel.org
blocks stateless clients). A pair is *what Linus quoted* plus *what Linus wrote*
in the same message, upgraded to the real posted patch where the thread lets
us find it. Full design: `AGENTS.md`.

| stage | count |
| --- | --- |
| Linus messages harvested | 8,802 |
| gated review pairs | 1,377 (15.6%) |
| pairs with the real posted patch | 1,048 (76.1%) |
| after abuse moderation | 1,026 keep / 239 scrub / 112 drop |
| final train / eval (2024 held out) | 1,187 / 78 (1,175 / 76 fit in 4,096 tokens) |

**Every one of these bugs passed the test suite and produced plausible counts.
Each was found by reading output by hand.**

1. **Export flattened all code indentation.** A whitespace cleanup meant for
   email addresses collapsed every run of spaces and tabs, so every diff in the
   training set lost its indentation.
2. **52% of training targets ended with the mailing list's unsubscribe
   footer.** The model would have learned to sign every review with
   `To unsubscribe from this list...`.
3. **The abuse filter cut his approvals.** Bare `sane` was on the abuse list,
   so "Looks sane to me" was scrubbed from 86 reviews.
4. **Patches were attached to the wrong reviews.** Patch texts were keyed by
   the per-month message number, which repeats in every one of the 1,472
   months. A 2024 eventfs review carried a 2010 block-layer patch. Found because
   the *smoke-test model* replied "this is a block layer patch, not an eventfs
   patch". Real patch coverage was 76.1%, not the 88.2% previously reported.
5. **The model invented who it was replying to.** Training targets kept the
   `On <date>, <name> wrote:` header of a quote that had already been moved into
   the input. Run-1 opened 94% of reviews with an invented date and the wrong
   person (44 of 55 named someone else). v2 strips these headers at export:
   0%.

The lesson, repeated five times: **a percentage you haven't sampled by eye
isn't a metric.**

## 3. Runs

| run | data | epochs | held-out eval loss (before → after) |
| --- | --- | --- | --- |
| run-1 | with bug 5 (headers in targets) | 2 | 2.18 → 1.92 (epoch 1) → 1.93 (epoch 2, slight overfit) |
| **v2** | headers stripped | 1 | **2.24 → 2.03** |

The two runs' losses aren't directly comparable. Run-1's targets contained
the easy-to-predict header lines; v2 scores only the review prose.

## 4. Evaluation method

All 78 held-out 2024 patches. In one process, with one model load, the same
weights generate with the adapter **on** ("fine-tuned") and **off** ("base"),
so the LoRA is the only difference. Two decoding regimes:

- **greedy**: deterministic, max 1,024 new tokens;
- **sampled**: temperature 0.7, top_p 0.95, repetition_penalty 1.1, seeded
  identically for both arms. This is what a person using the model will see.

Three kinds of evidence:

1. **Eval loss** (above).
2. **Proxy metrics** (`train/score.py`): cheap and deterministic, and they
   measure behaviour, not correctness.
3. **A blind pairwise judge** (`train/judge.py`). Qwen3.8-Flash-Next sees the
   patch, Linus' actual reply as the reference, and the two candidates
   unlabelled, and is asked which is the better review *of this patch*:
   technical accuracy first, overlap with the reference's points second,
   specificity third, ignoring tone. Each patch is judged in both orders; a
   model wins only if it wins both, and disagreement counts as a tie.
   **Validated** by a planted-wrong test (`train/judge_check.py`): a real
   review of the patch against a fluent review of a different patch.

## 5. Results

### Proxy metrics, v2 (n = 78)

| metric | Linus (reference) | base, greedy | v2, greedy | base, sampled | v2, sampled |
|---|---|---|---|---|---|
| median length (chars) | 1063 | 2252 | 492 | 2078 | 373 |
| names an identifier from the patch | 60% | 99% | 56% | 92% | 36% |
| assistant-style formatting/phrases | 26% | 41% | 1% | 45% | 5% |
| flagged by abuse filter (scrub\|drop) | 4% | 33% | 1% | 46% | 3% |
| starts with `On ... wrote:` | 0% | 0% | 0% | 0% | 0% |
| signs off "Linus" | 91% | 0% | 59% | 0% | 100% |
| repeats a line 3+ times (loop) | 1% | 0% | 44% | 1% | 0% |

(The reference column is measured after export stripping, so its `On ...
wrote:` rate is 0% by construction. In the raw corpus it is 72%.)

### Judge

**Judge validation passed: 20/20** on the planted-wrong test (0 invalid). In
every case, in both orderings, the judge preferred a real review of the patch
over a fluent review of a different patch.

**v2 vs base, both sampled, 78 held-out patches × 2 orderings:**

| outcome | patches |
|---|---|
| v2 wins (both orderings) | **44** |
| base wins (both orderings) | 16 |
| tie (judge disagreed with itself, or said tie) | 4 |
| invalid (judge returned no verdict) | 14 |

Of the 60 decided patches, v2 wins 73%. The judge was order-consistent on 60
of 64 patches where both orderings returned a verdict (94%), and picked slot A
53% of the time, so there was no meaningful position bias.

**Read the reasons, not just the tally.** v2 often wins by *saying less*. A
typical verdict: *"Candidate A misreads the diff and makes false ordering
claims, while Candidate B is generic but does not assert incorrect technical
facts."* The judge puts accuracy first, and the base model's confident
fabrications lose more often than v2's thin replies. That's a real win on
not being wrong, not evidence that v2 has more substance than the base. Where
the base wins, it's usually because v2 gave a bare ack where the patch needed
a review.

## 5b. v3 (2026-11-13)

**What changed from v2:** 4.4× the data (5,110 trained pairs; the gate now
accepts any reply whose real patch was recovered), the earlier thread messages
in the input (78–85% of rows), bookkeeping replies dropped, "Linus" sign-offs
stripped, 2023–2024 held out (766 pairs, 10× v2's), 2 nodes, 1 epoch.

**Evaluation:** 100 fixed held-out patches, sampled (T 0.7, top-p 0.95,
repetition penalty 1.1), base / v2 / v3 from BF16 on CUDA, v3 also as the MLX
8-bit Mac build. Blind pairwise judge (both orders); it passed the
planted-wrong check 20/20.

| | Linus | base | v2 | v3 |
|---|---|---|---|---|
| median length (chars) | 924 | 2,064 | 437 | 362 |
| names an identifier from the input | 39% | 90% | 33% | 22% |
| chatbot-style phrasing | 20% | 49% | 6% | 3% |
| abuse-filter flags | 0% | 51% | 8% | 2% |
| signs "Linus" (anywhere) | 7% | 6% | 100% | **0%** |
| email addresses | 0% | 2% | 0% | 0% |
| loops | 0% | 1% | 0% | 0% |

| blind judge | result | reading |
|---|---|---|
| v3 vs base | **51 – 24** (15 tie, 10 invalid) | v3 better (sign test p ≈ 0.002) |
| v3 vs v2 | 33 – 38 (17 tie, 12 invalid) | a tie (p ≈ 0.6) |
| Mac 8-bit vs CUDA v3 | 38 – 34 (21 tie, 7 invalid) | a tie (p = 0.72): the conversion loses nothing measurable |

**Held-out loss, like for like** (same 147 rows of v3's eval set, both 8-bit
on the Mac): v2 1.983 (perplexity 7.27), **v3 1.905** (6.72). On CUDA BF16,
v3 went 2.109 → 1.872 on the full 766-row set.

**Reading:** v3 is a real improvement in hygiene (it never signs as Linus,
and is the least abusive and least chatbot-like of the three) and in how well
it predicts his replies. It is **not** a better reviewer than v2. The
judge's reasons say why: v2 wins when it "directly addresses the patch's…"
change, v3 loses when it makes a false claim about the code. Two diagnostics
point at the cause:

1. v3's training targets have a median of 620 chars against 906 held out.
   Not a selection bug, as first thought: after cleaning, Linus' replies run
   ~580–590 chars through 2005–2019 and 778 in 2020–2024. The held-out years
   are a wordier era; v3 learned the 25-year average.
2. Thread context pulled it off the patch: on inputs with context, v3 names
   something from the *patch* 8% of the time (v2 24%, Linus 26%); without,
   31%.

Both are the v4 levers (`docs/PLAN-v4.md`).

**Release checks** (100 sampled replies, CUDA and Mac alike): 0 "Linus"
signatures, 0 email addresses. It does open by addressing a real developer by
first name in 37% of replies (Linus: 27%).

## 6. What we learned

- **The style transferred, from 1,175 examples and one epoch.** The fine-tune
  writes plain-text, terse, first-person reviews ("Ack.", "I'm not seeing this
  in my tree?", "This looks odd."). Assistant-style formatting dropped from
  ~45% to 1–5%.
- **The base model, told to be "blunt", becomes a caricature.** It opens with
  "This is a mess" / "This is nonsense", trips the abuse filter on 33–46% of
  reviews, and states false things confidently (for example, that `asm_inline`
  doesn't exist; it's a real kernel macro). The fine-tune, trained on
  moderated data, trips the filter on 1–3%, below Linus' own 4%. *Train on his
  judgement, not his swearing* worked in the direction we wanted.
- **The fine-tune is less specific than he is.** It names an identifier from
  the patch 36–56% of the time, against his 60%. Its reviews are shorter (373–492
  chars vs his 1,063), and some are confidently wrong, e.g. claiming the
  kernel's `min()` macro's advantage is a *lack* of type checking. It learned
  how he sounds faster than what he knows.
- **Greedy decoding loops (44%); sampling with a repetition penalty doesn't
  (0%).** Never evaluate or ship this with greedy decoding.
- **It signs every sampled review "Linus".** That's in the style, and also
  impersonation. See §8.
- **Most quality problems were in the data, not the training.** Five pipeline
  bugs (§2) would each have shaped the model. Two were found only because a
  model's output looked wrong.

## 7. Limits of this evaluation

- 76–78 held-out pairs from one year (2024): enough to see large effects, too
  few to rank close variants.
- The judge is a sibling of the base model (same Qwen3.8 family) and may share
  its taste; it is a small-active-parameter MoE served in FP4. The planted-wrong
  test shows whether it can tell on-topic from off-topic, not whether its
  technical calls are right.
- No human expert has scored these reviews.
- Linus' reply is a single reference; a different, correct review can disagree
  with it.

## 8. Using it responsibly

This is an **unofficial style imitation**, not endorsed by or affiliated with
Linus Torvalds, and its technical claims are frequently wrong. Don't post its
output to a mailing list, and don't attribute it to him. It signs reviews
"Linus" because he does; strip that if you show output to anyone.

## 9. Next steps

1. **Grow the data.** Loosen the gate (`code_evidence >= 3`, `--patch-only`);
   1,175 pairs taught style, not substance.
2. **Strip the "Linus" sign-off** at export, as was done with the headers.
3. **Hold out 2023–2024** for a bigger eval set, and report n next to every score.
4. **A judge from a different model family**, and a few dozen human-scored
   reviews to calibrate it.
5. **A cross-month ThreadMap** to recover the 24% of pairs without the real patch.

## Reproduce

Pipeline: `AGENTS.md` (crawl → thread → pairs → resolve → export). Training and
evaluation: `train/sft_lora.py`, `train/compare.py`, `train/score.py`,
`train/judge.py`, `train/judge_check.py`, orchestrated by `train/run-chain.sh`.
