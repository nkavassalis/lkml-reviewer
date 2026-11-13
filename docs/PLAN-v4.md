# Plan: v4 (draft, pending v3's judge results)

**Status:** criteria and candidate levers. The levers get picked once the v3
blind-judge results are in (v3 vs v2, v3 vs base, Mac vs CUDA). Each row of
the lever table below names the result that would trigger it.

## What v3 showed so far (100 sampled held-out patches, 2026-11-13)

| | Linus | base | v2 | v3 |
|---|---|---|---|---|
| median length (chars) | 924 | 2,064 | 437 | 362 |
| names an identifier from the input | 39% | 90% | 33% | 22% |
| chatbot-style phrasing | 20% | 49% | 6% | 3% |
| abuse-filter flags | 0% | 51% | 8% | 2% |
| signs "Linus" | 7% | 6% | 100% | **0%** |
| loops (sampled) | 0% | 1% | 0% | 0% |
| held-out loss (v3 eval set, CUDA BF16) | | 2.109 | (Mac run pending) | **1.872** |

v3's 8-bit Mac build matched CUDA on every proxy except identifier-naming
(34% vs 22%, borderline on n = 100), and scored 1.875 held-out loss on a
150-row subset against CUDA's 1.872 on the full set.

## Two diagnostics that already point at levers

1. **v3's training replies are shorter than the held-out ones, because Linus
   writes longer now.** Training targets: median **620** chars; held-out
   (2023–24): **906**. *Correction (2026-11-13):* I first blamed the data
   selection, claiming his length was "stable across eras". That came from raw
   replies padded with footers and quote headers. Measured after cleaning, his
   median is ~580–590 chars through 2005–2019 and **778 for 2020–2024**, and
   within each gate the held-out years run 35–40% longer than the training
   years (recovered-patch rows 581 → 808; quoted-code rows 737 → 958). A
   length filter would also drop real terse reviews: in a 15-row sample about
   half the short replies it caught were technical ("Just test bit #2…",
   "…rather than when you're in a COW section."). So the lever is **recency
   weighting**, not filtering.
2. **Thread context pulls v3 off the patch.** On the 87 patches with thread
   context, v3 names something from the *patch* 8% of the time (v2 24%,
   Linus 26%). On the 13 without, 31%. v3 appears to answer the discussion and
   forget the code.

Both are one sample per patch on a crude proxy: suggestive, not proven.

## v4 criteria (set before training, like v3's)

**Gates, no regression allowed:** 0% sign-offs and email addresses;
abuse-filter rate ≤ Linus' own; 0% loops when sampled; low chatbot phrasing;
the Mac 8-bit build ties CUDA in the blind judge.

**Targets, measured against v3:**
- blind judge win rate vs v3 and vs base, with the reasons tallied by type
  (false claim / generic / missed the reference's point / answered the
  thread instead of the patch);
- naming something from the **patch** at a rate closer to Linus' 26–39%;
- median length closer to his ~900 chars, with substance rather than padding;
- held-out loss on the same eval set as v3 (like for like);
- if possible, 30 blind human ratings (base / v3 / v4).

## Candidate levers

| if v3's results show… | v4 change |
|---|---|
| short replies because the held-out era writes longer (diagnostic 1, corrected) | **recency weighting**: up-weight recent years in training until train and held-out lengths roughly match; no length filter (it drops real terse reviews) |
| it answers the thread, not the patch (diagnostic 2) | patch first, thread second, each with its own token budget; trim the thread, never the patch |
| truncated inputs score worse | keep every file's hunks instead of cutting the tail, or 6k context if memory allows |
| more false claims than v2 (judge reasons) | preference training: sample v3 several times per patch, rank against Linus' reply with the judge, DPO on chosen vs rejected |
| the Mac build loses to CUDA | change quantization (group size, mixed precision), not the training |
| only cosmetic issues remain | v4 is just the cleaned data (sign-off lines now stripped everywhere) |

## Process

Same as v3: write the chosen levers and criteria here before training;
smoke-test the **whole** chain (training, copies, evaluation) at tiny sizes;
watcher plus hourly status; one epoch; same LoRA config, so differences come
from the data.
