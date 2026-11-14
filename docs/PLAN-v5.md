# Plan: v5, preference training for accuracy

**Status:** queued (2026-11-13, Nick: "queue those work items to happen after
v4"). Candidate generation, scoring and the reading pack run from `QUEUE.toml`
after v4; DPO training waits for the human check of the pairs (HANDOFF §4e).
2026-11-14, Nick: "start everything now, I am ok with some day time downtime":
the queue may use the cluster outside the window until 07:40 ET on 11-15.

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

## Decision: which model v5 starts from (rule set 2026-11-13, before the data)

**Optimizes:** v5 (preference training) is for correctness: fewer false claims
about the patch.
**Measured by:** as first written, nothing in the rule did (it watched patch
grounding and looping). Since 2026-11-14, v5's own rubric (`train/rubric.py`
reusing `make_preferences.ask`) scores v3, v4 and base on the 100 held-out
patches (queue item `rubric-score`), and that score is `train/decide.py`'s
target: if it leans away from the rule's choice, a person decides. The
pairwise judge is a non-target signal: it has no "saying little is not a
virtue" clause, so it rewards caution that v5's scoring won't.
**Pre-mortem (added 2026-11-14, after the judge result, so not blind):** the
rule's choice is wrong if v4 makes more false claims than v3 (judge, reasons),
if v4's grounding gain is inside the noise floor, or if v4 loops more.

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

### 2026-11-14: a person confirms the starting model (Nick: "is v4 worse than v3? why would we build v5 on v4 if thats the case?")

v4's blind judge against v3 (each in its own layout, 100 patches): **v3 45,
v4 28**, 18 ties, 9 invalid. The reasons: v3 mostly wins because v4 makes a
false claim about the patch; v4 wins when v3 is vague. v4 engages with the
code more, and is wrong more often when it does. The rule above can't see
this: its two metrics (grounding, loops) don't measure correctness, which is
the thing v5 exists to improve, and grounding probably favours v4. So the rule
still runs and writes `data/v5/BASE`, but the queue now waits at a human gate
(`v5-base-decision`) before candidate generation; the decision, with both the
multi-sample report and the judge results, gets dated here. The rule was
honest but incomplete; changing the default after seeing a result is exactly
what it was meant to prevent, so this is a person's call, written down, not a
quiet edit to `choose_base.py`.

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

## Code

- `train/make_preferences.py`: candidates → judge scores → (chosen, rejected)
  pairs, resumable, with the drop rules above. Tested end to end against a fake
  judge.
- `train/dpo_lora.py`: DPO with policy and reference as two adapters on one
  base, DDP and the memory cap as `sft_lora.py`. Reading it before its first
  run (2026-11-14) found no resume and adapter saves that PEFT would put in a
  `policy/` subdirectory; both fixed (`train/adapter_ckpt.py`, tests).
- `train/run-chain-2node.sh` runs it with `TRAINER=dpo SFT_ADAPTER=...
  PAIRS=...` and copies those inputs, and any checkpoint, to the peer.

Not run on the cluster yet. The `dpo-smoke` queue item (16 synthetic pairs, 2
steps, a checkpoint, then evaluation loads the result) runs before
candidate generation, so mechanics are proven before the real pairs exist.
