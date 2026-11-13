# Plan: v5, preference training for accuracy (draft, for review)

**Status:** proposal. Nothing is scheduled. Needs Nick's go-ahead, because it
costs several nights of the maintenance window plus sustained judge load on
vLLM.

## Why

Every version so far learned how Linus *sounds*. None learned to be *right*
more often. The blind judge's most common complaint about v3 is a false claim
about what the patch does, and v3 vs v2 was a tie. Supervised fine-tuning
(SFT) only imitates; it has no signal for "this review is wrong". Preference
training gives it one: show the model two of *its own* reviews of the same
patch, one judged closer to the truth, and push it toward that one.

## Method: DPO on the model's own samples, starting from v4

1. **Candidates.** For N training patches (not held out), sample K reviews
   from v4 (sampling settings as in evaluation).
2. **Rank them.** The judge scores each candidate against Linus' real reply on
   the same criteria as evaluation: accuracy first, then overlap with his
   substantive points, then specificity. Keep a pair (best, worst) only when the
   judge separates them clearly; ties and judge failures are dropped, never
   guessed.
3. **Train.** DPO with β ≈ 0.1: a new LoRA on top of v4, with v4 itself
   (frozen) as the reference model. Both are the same base weights with
   different adapters, so memory stays at one 27B copy.
4. **Evaluate** with the stronger evaluation (several samples per patch,
   paired CIs, noise floor, `train/paired_stats.py`), against v4 and v3.

DPO, not PPO/RL: no reward model to train, no online rollouts; a single
offline pass over preference pairs. That suits nights on two GB10s.

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
