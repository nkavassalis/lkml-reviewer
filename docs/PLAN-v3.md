# Plan: v3

**Goal:** close the gap v2 exposed. It learned how Linus *sounds*, not what he
*knows* (shorter than him, names patch identifiers 36–56% of the time vs his
60%, sometimes confidently wrong). v3 changes the data, not the training code.

## Measured before planning (2026-11-11, all local, no network)

| lever | measurement |
|---|---|
| gate on "has the real patch", not only on Linus' quote | **6,352** candidate pairs vs 1,377 today (4.6×), no crawl needed |
| what the new 4,975 look like | 5 of 6 sampled are real reviews; 5% bookkeeping-ish; 42% have no code token in the reply (vs 20% today), so more discussion |
| thread context (earlier messages in the same thread) | ~9.8 per pair; **71% already cached**; the rest is a polite crawl |
| Linus replies outside `[PATCH]` threads | 12,378 never crawled. Out of scope for v3 (no patch to review) |
| held-out size if 2023–2024 is held out | **840** pairs (vs 78) |

## The constraint: training time

At the measured 163 tok/s on one GB10:

| v3 input | tokens/epoch | 1 node | 2 nodes (est., unmeasured) |
|---|---|---|---|
| patch only, 4k cap | 10.1M | 17 h | ~9 h |
| patch + ~1k tokens of thread | 16.7M | 28 h | ~15 h |
| patch + ~2k thread, 8k cap | 23.8M | 41 h | ~22 h |

vLLM is down for all of it. The 2-node speedup is an estimate: LoRA gradients
are small (109M params), so data-parallel over the 200 Gb link should scale well,
but it's untested on this cluster.

## Phase 0: data (no cluster, no downtime)

1. **New gate:** accept a review when we have its real patch, or when the
   quote has code evidence ≥ 3 (today's rule).
2. **Drop bookkeeping targets:** ack / applied / "which tree?" replies with no
   technical claim. They can't be predicted from the input and teach vague replies.
3. **Thread context in the input:** patch first, then the most recent earlier
   messages in the thread up to a token budget, each marked with its author.
   Crawl the missing ~29% of those pages at the mirror's politeness floor.
4. **Strip the "Linus" sign-off** at export, as was done with the headers.
5. **Hold out 2023–2024** (840 pairs). Evaluate generation on a fixed random
   100 of them.
6. **Moderation at 4.6× volume:** read a sample of `scrub` and `drop`
   decisions (`audit`). The vocabulary has never seen this much text.
7. **Read 20 pairs by eye before training.** Five bugs in v1/v2 were found this
   way and by no test.

## Phase 1: measure the 2-node speedup (~30 min of vLLM downtime)

A short `torchrun` data-parallel smoke test across both Sparks. If it scales,
v3 trains on both nodes; if not, on one.

## Phase 2: train v3

**Decided (2026-11-11):** patch + ~1k tokens of thread context, all 6,352
candidates, on both nodes if Phase 1 shows it scales (~15 h of vLLM downtime;
~28 h if it has to run on one node, which needs re-confirming first).

1 epoch (run-1's second epoch overfit), with the budget chosen from the table
above. Same LoRA config as v2, so differences come from the data.

## Phase 3: evaluate

- Eval loss on all 840 held-out pairs: base, v2, v3.
- Generation on the fixed 100: **sampled only** (what people will use), three
  arms, base / v2 / v3, in one model load (~2–3 h).
- Judge v3 vs v2 and v3 vs base, after the planted-wrong check passes.
- **Human ratings:** a friend who knows the kernel rates 30 blind triples
  (base / v2 / v3, unlabelled). This is worth more than the automatic judge.

## Success criteria, set in advance

v3 is better than v2 if, on the fixed 100 sampled: the judge prefers v3 over v2
on more patches than the reverse, it names patch identifiers closer to Linus'
60%, its median length moves toward his ~1,060 chars, abuse-filter hits stay
≤ 4%, loops stay ≈ 0%, and the friend's blind ratings agree.
