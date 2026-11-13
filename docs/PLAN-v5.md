# Plan: v5, preference training for accuracy

**Status:** queued (2026-11-13, Nick: "queue those work items to happen after
v4"). Candidate generation, scoring and the reading pack run from `QUEUE.toml`
after v4; DPO training waits for the human check of the pairs (HANDOFF §4e).

## Why

Every version so far learned how Linus *sounds*. None learned to be *right*
more often. The blind judge's most common complaint about v3 is a false claim
about what the patch does, and v3 vs v2 was a tie. Supervised fine-tuning
(SFT) only imitates; it has no signal for "this review is wrong". Preference
training gives it one: show the model two of *its own* reviews of the same
patch, one judged closer to the truth, and push it toward that one.

## Method: DPO on the model's own samples, starting from v4 (by default; see below)

1. **Candidates.** For N training patches (not held out), sample K reviews
   from v4 (sampling settings as in evaluation).
2. **Rank them.** The judge scores each candidate against Linus' real reply on
   the same criteria as evaluation: accuracy first, then overlap with his
   substantive points, then specificity. Keep a pair (best, worst) only when the
   judge separates them clearly; ties and judge failures are dropped, never
   guessed.
3. **Train.** DPO with β ≈ 0.1, continuing from v4: the trainable "policy"
   adapter starts as a copy of v4's adapter, and a frozen copy of v4 is the
   reference model. Both are the same base weights with different adapters, so
   memory stays at one 27B copy. v5 ships as one adapter (v4's, moved by DPO),
   not a second adapter stacked on v4. This is the first version that starts
   from an earlier one: v1–v4 were each a fresh LoRA on the base model.
4. **Evaluate** with the stronger evaluation (several samples per patch,
   paired CIs, noise floor, `train/paired_stats.py`), against v4 and v3.

DPO, not PPO/RL: no reward model to train, no online rollouts; a single
offline pass over preference pairs. That suits nights on two GB10s.

## Which model v5 starts from (rule set 2026-11-13, before the data)

v5 starts from **v4 by default**. It starts from **v3 with patch-first input**
instead only if the multi-sample comparison (200 held-out patches × 3
samples, byte-identical patch-first inputs for both) shows v4 clearly worse:

- its patch grounding is lower, with the whole 95% CI below zero, or
- it loops more, with the whole 95% CI above zero.

Why: DPO sharpens the model it starts from, so it should start from the better
one; and the input-order probe showed v3 gets much of v4's gain just from the
reordered input. A difference inside the noise keeps the default. v4's
single-sample judge results don't enter the rule: they compare v4 against v3
in its *thread-first* layout, which isn't the alternative here.

Mechanics: the queue's `multisample-report` runs `scripts/q/choose_base.py`
(tested) and writes `data/v5/BASE`; `v5-candidates` waits for it and samples
from that model; the decision and its numbers are in the report email. The
DPO step must use the same model for `--sft-adapter` (both the policy's
starting point and the frozen reference).

## Costs (estimates; measured numbers in brackets)

| step | work | time |
|---|---|---|
| candidates | N = 800 patches × K = 4 = 3,200 reviews, max 600 tokens | ~17 h on 2 nodes [CUDA generation ≈ 10 min per batch of 16] → **2 nights** |
| ranking | 3,200 judge calls | ~17 h with 8 parallel calls inside the window [156 calls ≈ 90–110 min at 4 parallel] → **2 nights**, vLLM up but under load |
| DPO | ~700 pairs × 4 forward passes (policy + reference, chosen + rejected) | ~6–7 h → **1 night** |
| evaluation | 200 patches × 3 samples, v5 vs v4 vs v3 | Mac (no cluster) ≈ 5–10 h, then judging |

**About five nights in total.** Cheaper variants: N = 400 (halves steps 1–2,
weaker signal), or K = 2 (one pair per patch, no "worst of 4" selection).

## Success criteria (set now, before any of it runs)

- **Headline:** v5 beats v4 in the blind judge with a paired CI excluding 0,
  and the judge's reasons tally fewer "false claim" verdicts against v5.
- **Gates:** no regression in style or hygiene. Sign-offs and emails at 0;
  abuse at or below Linus' rate; no loops; chatbot phrasing at or below v4's;
  the Mac 8-bit build ties CUDA.
- **Watch for:** reward hacking toward the judge's taste (vaguer, hedged
  reviews that avoid being wrong by saying nothing). Length and specificity
  must not fall below v4's; read 20 side-by-sides before believing a win.

## Risks

- **The judge is the teacher.** Whatever it gets wrong, v5 learns. It passed
  the planted-wrong check 20/20, but nobody has measured whether its accuracy
  calls are right. Before step 3, check ~30 of its rankings against a human
  (Nick's friend?).
- **Same-family judge** (Qwen3.8-Flash-Next ranking a Qwen3.8-27B): it may
  prefer its family's habits. A judge from another family would be safer,
  but means an external API and sending the data out; Nick's call.
- Five nights of window plus judge load on the serving stack.

## Code (drafted, not run)

- `train/make_preferences.py`: candidates → judge scores → (chosen, rejected)
  pairs, resumable, with the drop rules above.
- `train/dpo_lora.py`: DPO with policy and reference as two adapters on one
  base, same DDP / memory cap / resume / checkpointing as `sft_lora.py`.

Both get the smoke-test treatment (the whole chain at tiny sizes, inside a
window) before any real run.
