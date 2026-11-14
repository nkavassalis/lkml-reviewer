# HANDOFF: session state for the next agent

Read this **and** `AGENTS.md` before touching anything. AGENTS.md is the durable
design doc; this file is "where the work actually stopped". Results are in
`docs/REPORT.md`, lessons in `docs/LEARNINGS.md`, and the next iterations in
`docs/PLAN-v4.md` and `docs/PLAN-v5.md`.

---

## 0. Status (2026-11-14)

Rewrite this section whenever the state changes; history goes in
`docs/LEARNINGS.md` and `docs/REPORT.md`, not here.

- **Published:** v3 (adapter + MLX 8-bit) as `nkavassalis/lkml-reviewer-lora`.
  Never signs as Linus, beats base 51–24 blind, ties v2. README's journey
  table has v1–v5 in two lines each.
- **Running:** v4 (patch-first layout). Training finished 2026-11-14 06:49
  (eval loss 2.121 → 1.885 at step 639); its evaluation needs ~100 min and
  runs at the start of the next window (20:00 ET 2026-11-14). The reporter
  emails on problems, each morning and when `RESULTS-v4.md` lands (§4d); it
  missed the first morning's report (fixed, `docs/LEARNINGS.md`).
- **Queued after v4:** fetch, Mac build, multi-sample evaluation, v5 DPO data
  (candidates, judge scores, reading pack), then a human check; after v5, a
  greedy-decoding loop check (§4e). Runner: `scripts/queue-runner.sh status`.
- **On the Mac:** v3 multi-sample done (200 patches × 3, both layouts); idle
  until the queue builds v4.
- **Not built yet:** chain support for `train/dpo_lora.py` (DPO smoke and
  training items get added at the human gate).

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

## 2. Measured numbers (v1/v2 gate; v3 onward in docs/REPORT.md)

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

## 4d. RUNNING UNATTENDED from 2026-11-13 (Nick away a couple of days)

**On the master node** (`~/linuspr-train`): a loop runs `night-run.sh` with
`night-plan-v4.sh` for up to three nights (stops early once
`RESULTS-v4.md` exists). Each night: smoke test of the whole v4 chain (once;
failure stops the night), v4 training (`--resume-auto`, both nodes), v4
evaluation on the same 100 held-out patches, then the blind judge (v4 vs v3,
v4 vs base) and `RESULTS-v4.md` via `make_results_v4.py`, all inside
20:00–08:00 ET. Hard stop 07:40 ET; vLLM restored by the chains, double-checked
by the launcher. The plan and results script are cluster-local copies of
`data/cluster/night-plan-v4.sh` and `data/cluster/make_results_v4.py` (they hold
cluster addresses, so they're not in git).

- Logs: `~/linuspr-train/night-YYYY-MM-DD.log` (`latest-run.log` points at the
  current one), `chain-<run>.log`, `run-4/train.log`, `run-4/judges.log`.
- Stop it safely: `pkill -f "for n in 1 2 3; do \[ -f RESULTS-v4.md \]"` and
  `pkill -f night-run.sh` *outside* the window. Inside the window, stop the
  chain instead (`pkill -TERM -f run-chain-2node.sh`): its trap restores vLLM.

**On this box:** `scripts/night-reporter.sh` (supervised loop; pid in
`data/reporter/reporter.pid`; `scripts/night-reporter.sh status|stop`) emails
the owner via the monitoring box's Gmail sender (configured in the gitignored `.status.env`):
PROBLEM at once, a Report after each night, FINISHED with `RESULTS-v4.md`;
then it stops itself. A test email was sent and confirmed in the Sent folder.

**Also pending:** the Mac input-order probe (v3 with patch-first inputs, same
100 patches): `~/lkml-mac/compare-v3-patchfirst.jsonl` on the Mac. Compare with
`data/train/run-3b/compare-v3-mlx.jsonl` using `slice.py` (template) to see how
much of the layout effect needs no retraining.

**Also running on the Mac (no cluster):** `~/lkml-mac/multi-sample-queue.sh`:
3 samples per patch on 200 held-out patches for v3, thread-first
(`ms-v3-threadfirst.jsonl`) and patch-first (`ms-v3-patchfirst.jsonl`), ~10 h.
Analyse with `train/paired_stats.py A:v3_tf B:v3_pf --primary-marker '--- The
patch ---' --context-marker '--- Earlier in this thread'` (noise floor included).

**Queued (Nick, 2026-11-13: "queue those work items to happen after v4"):**
v5 = DPO for accuracy (`docs/PLAN-v5.md`; `train/dpo_core.py` tested,
`make_preferences.py` tested offline, `dpo_lora.py` drafted). See §4e.

v4 data: `data/v4/sft*.jsonl` (patch-first layout, `export --layout
patch-first`). v3's as-trained data: `data/train/run-3b/`.

## 4e. The improvement queue (after v4), from 2026-11-13

`QUEUE.toml` + `scripts/queue-runner.sh` on this box (supervised loop, every
5 min; `scripts/queue-runner.sh status|stop|start`; state and logs in
`data/queue/`). Items, in order, each emailing Started / Done / FAILED:

1. `v4-fetch`: when `RESULTS-v4.md` exists and the cluster is idle, copy run-4's
   adapter, logs and judge files to `data/train/run-4/`.
2. `v4-mac-convert`: when the Mac is idle, merge + MLX 8-bit into `~/lkml-mac/v4`
   (frees disk by removing the stock mlx-community 8-bit model if < 90 GB free).
3. `v4-multisample`: starts v4 × 200 patches × 3 samples on the Mac, detached.
4. `v5-candidates` (window, after `multisample-report`): 800 training patches
   (seeded, `data/v5/in/`), 4 samples each from v5's starting model
   (`data/v5/BASE`: v4 unless the report shows v4 clearly worse than v3
   patch-first; PLAN-v5 "Which model v5 starts from"), two chunks of 100 per round (one per node), via a GO_NOW
   `night-run.sh` with `night-plan-cands.sh` (cluster-local copy in
   `data/cluster/`). ~4.5 h per round, two rounds a night: ~2 nights. Exits 75
   between windows ("Paused" email).
5. `multisample-report`: when the Mac is idle: `paired_stats.py` v3-tf / v3-pf / v4
   with noise floors → `data/train/multisample/MULTISAMPLE.md`, emailed, plus
   the v5 starting-model decision (`scripts/q/choose_base.py` → `data/v5/BASE`).
   v3 patch-first and v4 use byte-identical inputs (`data/v4/sft_eval.jsonl`).
6. `v5-score` (window): the judge (cluster vLLM) scores 3,200 candidates,
   resumable, cut at the window's end; then best-vs-worst pairs (gap ≥ 3) →
   `data/v5/pairs.jsonl`. Throughput unmeasured: expect 1–3 nights.
7. `v5-check-pack`: `data/v5/CHECK.md`, 20 pairs to read, emailed.
8. `v5-human-check` (**human gate**, emails "Needs you"): read the pack; approve
   with `python3 train/workqueue.py approve v5-human-check`. Then add the DPO
   smoke, DPO training and v5 eval items (they need chain support for
   `dpo_lora.py`, not written yet).

9. `greedy-loops` + `greedy-loops-report` (Nick, 2026-11-13: "measure after
   v5"): greedy decoding (`compare_mlx.py --greedy`, no repetition penalty) for
   v3, v4 and v5 on 100 held-out patches; loop rate emailed. v2 looped on 44%
   greedy vs base 0%; DPO with looping candidates as "rejected" may reduce it.
   Waits for `~/lkml-mac/v5/mlx-8bit`: **the v5 convert item must build there.**

When v4's and v5's results land, update their rows in README's "The journey so far" table (Nick wants the learning documented front and centre).

A failed item halts the queue (`workqueue.py reset ID` to retry, `skip ID`).
Tested: the v5 data path end to end against a fake judge (chunking, merge,
scoring, pairs, check pack); the runner's email sequence (tests/test_workqueue.py).
Not tested against the real cluster: `v5-candidates`' plan and the Mac items.

## 5. Next actions

1. When v4's results email arrives: REPORT section, README journey row,
   PLAN-v4 status and outcome, LEARNINGS entry (AGENTS.md "Documenting as we
   go"). The queue carries on by itself.
2. When the multi-sample report arrives: is v4 better than v3 patch-first
   beyond the noise floor? That answers whether v4's training bought anything
   over reordering at inference.
3. Before the v5 human gate opens: write chain support for `dpo_lora.py` and a
   DPO smoke test, so approval can go straight to training.

## 6. Not verified (do not present as done)

- Whether 2-node training scales on this cluster (v3 Phase 1 measures it).
- Whether the judge's technical verdicts are right.
- `match_commit` precision at scale; `patch_quality` on Kconfig/DTS/Rust/asm;
  `DomainIndex` msgid recovery (0 of 564 matches).
- Crawler behaviour under real rate-limiting or an outage.
- The moderation vocabulary at v3's volume (6,352 candidates).
