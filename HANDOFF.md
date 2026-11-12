# HANDOFF: session state for the next agent

Read this **and** `AGENTS.md` before touching anything. AGENTS.md is the durable
design doc; this file is "where the work actually stopped". Results are in
`docs/REPORT.md`, and the next iteration is planned in `docs/PLAN-v3.md`.

---

## 0. Status (2026-11-11)

- The corpus pipeline is built and has run over the whole archive (1996 → Nov 2026).
- **Two LoRA fine-tunes of `Qwen/Qwen3.8-27B` exist: run-1 and v2.**
  - v2 is the current model: 1,175 train pairs, 1 epoch, held-out eval loss 2.24 → 2.03.
  - It learned Linus' *style* (plain text, terse, abuse below his own rate) but not
    his *substance*: shorter than him, less specific, sometimes confidently wrong.
  - Greedy decoding loops on 44% of outputs; sampling with a repetition penalty
    doesn't loop at all.
- **v3 is in progress (Phase 0, data).** See `docs/PLAN-v3.md`.
- Adapters and data are **not** in git and not published. Hugging Face upload
  waits until v3 shows where it lands.

## 1. What lives where

| artefact | where | notes |
| --- | --- | --- |
| cached hypermail HTML | `data/raw/` (1.7 GB) | **never delete**: hours of polite crawling |
| bare torvalds/linux | `data/git/linux.git` (4 GB) | **never delete** |
| msgid / commit-subject indexes | `data/msgid_index.json`, `data/commit_subjects.json` | each costs a full `git log` pass |
| Linus messages | `data/linus_messages.jsonl` | 8,802, `[PATCH]`-thread replies only |
| recovered patches | `data/patch_texts.json` | keyed `month/mid` (see trap 8) |
| gated pairs, unmoderated | `data/review_pairs.jsonl` | 1,377; the audit source |
| SFT export | `data/sft.jsonl`, `data/sft_eval.jsonl` | 1,187 / 78 (2024 held out) |
| run outputs | `data/train/run-1/`, `data/train/run-2/` | adapters, train.log, compare/judge JSONL; run-2 also has the exact data it trained on |

Training ran on a 2-node DGX Spark cluster (GB10, ~121 GB unified memory each)
that also serves a vLLM model. Its addresses aren't in the repo. The operator
keeps them, and the stop/restore procedure for the vLLM service, outside git.
The rule: **stop the serving container for a training run, and restore it when
the run ends**. `train/run-chain.sh` does this with an `EXIT` trap, so a crash
can't leave the service down.

## 2. Measured numbers (beat them, don't re-derive them)

| stage | count |
| --- | --- |
| buckets crawled | 1,472 (9603.0 = Mar 1996 → 2611.1 = Nov 2026) |
| Linus messages harvested | 8,802 |
| gated pairs | 1,377 (15.6%) |
| pairs with the real posted patch | 1,048 (76.1%) |
| pairs matched to a merged commit | 564 (41.0%) |
| moderation | 1,026 keep / 239 scrub / 112 drop |
| final train / eval | 1,187 / 78 |
| v3 candidates (new gate, measured) | 6,352; 840 if 2023–2024 is held out |

Training: one GB10 runs BF16 LoRA on the 27B at **~163 tok/s, peak 71.5 GB at
4,096 tokens**. 1.7M tokens ≈ 3 h per epoch. Model load ≈ 6 min. Two nodes
(`train/run-chain-2node.sh`, torchrun data parallel): **258–276 tok/s
(1.6–1.7×)**, 71.9 GB per node. Use one node for small runs (v1/v2: not worth
the setup) and two when a run would exceed ~half a day on one (v3: ~28 h →
~14 h). Keep the effective batch fixed (`--grad-accum` is per rank) so runs
stay comparable.

## 3. Traps already paid for: do not re-pay them

1. **Bucket ids are `YYMM.N`** (2019-01 = `1901`). `bucket_year()` pivots at
   `PRE_2000_THRESHOLD = 50`. `yy >= 93` reads `1901` as 1919.
2. **`git log` output is not valid UTF-8.** Decode with `errors="replace"`
   (`GitRepo.run_bytes`).
3. **hypermail's "Previous message" is document order, not parentage.** Use
   `ThreadMap` (group by normalised subject); keep the subject gate.
4. **Abuse vocabulary needs whole-word matching** (*Hell*wig, un*sane*), and bare
   `sane` is approval, not abuse.
5. **Stage order:** `crawl` → `thread` → `pairs` → `resolve` → `export`.
6. **lore.kernel.org, patchwork and git.kernel.org's web UI are behind Anubis.**
   Only git and the `lkml.iu.edu` mirror work stateless.
7. **Never commit `data/`.**
8. **`mid` is a per-bucket counter.** It repeats in all 1,472 buckets; key
   anything per-message by `month/mid` (`patch_key()`). Keying on `mid` attached
   2010 patches to 2024 reviews.
9. **Export transforms must not touch code.** A whitespace cleanup flattened every
   diff's indentation; a sentence re-join did the same in scrubbed reviews.
10. **Strip what isn't his words to the patch:** list footers (52% of targets had
    one) and `On <date>, <name> wrote:` headers (run-1 learned to invent them).
11. **Never evaluate or ship with greedy decoding.** It loops on 44% of outputs.
12. **A reasoning-model judge needs a big token budget.** At 6,000 it returned
    empty content on 60 of 156 calls. Empty is not a verdict.
13. **Training data mismatch is visible in generations, not in loss.** The
    wrong-patch bug was found because a 3-step smoke-test model said "this is a
    block layer patch, not an eventfs patch". Always read the smoke-test sample.

## 4. Known weaknesses (disclose them, don't hide them)

- 76–78 held-out pairs from a single year is too few to rank close variants
  (v3 fixes this with 2023–2024).
- The judge is a sibling of the base model (same Qwen3.8 family). It passed a
  planted-wrong check (20/20), but nobody has checked whether its technical
  calls are right.
- The model signs reviews "Linus". It is a style imitation, and the output must
  not be attributed to him.
- Commit matching is subject+year based and hand-verified on a sample only.
- Moderation is a vocabulary heuristic; `stupid` always drops a pair, even when
  it's aimed at code.
- Mirror completeness is unverified against lore.

## 4b. Running long jobs

Follow `docs/OPERATIONS.md`: chain script with a restore trap, allocator cap,
checkpoints with optimizer state, the off-cluster watcher, and an hourly status
run (`scripts/status.sh`, hosts in a gitignored `.status.env`, scheduled with
a session cron job). The run-3 freeze is written up there.

## 4c. After v3 finishes (agreed with Nick, 2026-11-12)

1. Confirm vLLM serves again on both nodes.
2. Score + judge v3 vs v2 vs base on the fixed 100 held-out (sampled); score
   both adapters' eval loss on v3's eval set (the only like-for-like loss).
3. Release checks on v3 generations: "Linus" sign-offs, email addresses,
   named people. Report the rates on the card.
4. Merge check on a GPU node; convert v3 to MLX 8-bit; Mac-vs-CUDA quality gate.
5. Publish **publicly**, Apache-2.0: `nkavassalis/lkml-reviewer-lora` (adapter)
   and `nkavassalis/lkml-reviewer-mlx-8bit` (only if it passes step 4). The card
   leads with "work in progress, use at your own risk" and "not affiliated
   with or endorsed by Linus Torvalds", and links this repo and the template.
   Show Nick the final card first.

## 5. Next actions

Follow `docs/PLAN-v3.md`: Phase 0 (data), then a 2-node data-parallel smoke test,
then train v3, then evaluate against v2 with blind human ratings. Keep AGENTS.md,
this file and the report updated in the same commit as any change.

## 6. Not verified (do not present as done)

- Whether 2-node training scales on this cluster (v3 Phase 1 measures it).
- Whether the judge's technical verdicts are right.
- `match_commit` precision at scale; `patch_quality` on Kconfig/DTS/Rust/asm;
  `DomainIndex` msgid recovery (0 of 564 matches).
- Crawler behaviour under real rate-limiting or an outage.
- The moderation vocabulary at v3's volume (6,352 candidates).
