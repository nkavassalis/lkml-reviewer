# Learnings

Append-only. One entry per thing we believed and found to be wrong (or found
out the hard way), in date order, newest at the bottom. Never edit an old
entry's facts; if a later finding overturns one, add a new entry that says so.
`scripts/docs_check.py` (in the gate) checks the format.

Each entry: **Believed** (what we thought), **Evidence** (how we found out, with
numbers), **Changed** (code, test, doc: where the fix lives), **Generalizes**
(does it belong in the template? where it went, or why not).

The README's journey table is the two-line summary per version; `docs/REPORT.md`
has the full numbers. This file is the reasoning between them.

## 2026-11-10: Bucket ids are not what they look like

**Believed:** mirror bucket `YYMM` with a `yy >= 93` pivot for the 1900s.
**Evidence:** `--year 2019` read `1901` as January 1919, harvested 48 wrong
months and reported a plausible "355,530 messages, 614 from Linus". A second
attempt read the id as `YYYM` and turned 2024 into 2140.
**Changed:** `bucket_year()` with `PRE_2000_THRESHOLD = 50`, calibrated to the
mirror's real range; `test_bucket_year_*`; AGENTS.md "Do not touch".
**Generalizes:** yes: plausible totals prove nothing; check one unit by hand
(template METHOD.md, "a percentage you haven't sampled by eye").

## 2026-11-10: A coverage number we hadn't sampled was wrong a third of the time

**Believed:** walking hypermail's "Previous message" link recovers the patch a
reply answers (62% coverage).
**Evidence:** reading samples: "previous" is document order, not parentage; a
third of recovered patches came from an unrelated thread (a KEYS review wearing
an `n_hdlc.c` diff).
**Changed:** `ThreadMap` groups by subject (76% of pairs, hand-checked); the
walk survives only as a subject-gated fallback.
**Generalizes:** yes: METHOD.md, sample every coverage metric by eye.

## 2026-11-10: Export bugs that no count showed

**Believed:** the export was clean because the counts looked right.
**Evidence:** reading `sft.jsonl`: whitespace collapsed in every diff; 52% of
outputs ended in the vger unsubscribe trailer; bare `sane` scrubbed 86 approvals
("Looks sane to me").
**Changed:** `scrub_mangled` keeps whitespace, `strip_list_footers()`, the
abuse vocabulary matches "no sane person" only; each with a regression test.
**Generalizes:** yes: read the final training file, not just the stage counts
(METHOD.md).

## 2026-11-11: The smoke-test model found a data bug

**Believed:** patch coverage was 88.2%.
**Evidence:** a smoke-test generation answered "this is a block layer patch, not
an eventfs patch". `patch_texts.json` was keyed by the per-bucket message
number, which repeats in all 1,472 buckets; patches overwrote each other across
years. Real coverage 76.1%.
**Changed:** keys are `month/mid` (`patch_key()`); patches dated more than 2
years from their review went from 17 to 0.
**Generalizes:** yes: read a small model's outputs before the big run;
wrong-looking output is often a data bug (METHOD.md).

## 2026-11-11: v1 learned our quoting headers

**Believed:** moving the quoted text into the input was enough.
**Evidence:** run-1 opened 94% of reviews with an invented "On <date>, <name>
wrote:"; 44 of 55 named someone else. The targets still carried the header.
**Changed:** v2 strips attribution headers at export: 0%.
**Generalizes:** yes: the model learns every regularity in the targets,
including the ones you didn't mean to teach.

## 2026-11-11: v2 won the judge by saying less

**Believed:** a 44–16 blind-judge win meant v2 reviews better than the base.
**Evidence:** reading the judge's reasons: v2 usually won because the base made
confident false claims and v2's thin reply didn't. Where the base won, v2 had
given a bare ack.
**Changed:** REPORT §5 reads the reasons, not just the tally; proxy metrics for
specificity (names an identifier from the patch).
**Generalizes:** yes: METHOD.md, read the judge's reasons.

## 2026-11-11: Greedy decoding loops

**Believed:** greedy is the neutral, deterministic way to evaluate.
**Evidence:** v2 greedy repeated a line 3+ times in 44% of reviews (base 0%);
sampled with repetition penalty 1.1: 0%.
**Changed:** every evaluation and the reviewer sample (T 0.7, top-p 0.95,
penalty 1.1); model cards say never greedy. A greedy loop check for v3/v4/v5 is
queued after v5.
**Generalizes:** yes: evaluate with the decoding you'll ship.

## 2026-11-11: An unwatched run froze a node

**Believed:** a run that started cleanly would finish or fail visibly.
**Evidence:** run-3 ran out of unified memory and hung the master node; nothing noticed
until Nick asked.
**Changed:** allocator cap (`--mem-fraction 0.75`), checkpoints with optimizer
state and `--resume-auto`, an off-cluster watcher, hourly `status.sh` runs.
**Generalizes:** yes: template OPERATIONS.md, "The four rules" and monitoring.

## 2026-11-12: Leaks get as far as the commit message

**Believed:** the secret scan on files was enough.
**Evidence:** a hostname in a code comment, host names and an email in HANDOFF,
and a commit message naming a host were caught only by the scan or by review
before pushing.
**Changed:** commit-msg hook, a local `.secret-watchlist` of hostnames,
addresses only in the gitignored `.status.env`.
**Generalizes:** yes: template hooks and scan.

## 2026-11-12: Local config overrode the test environment

**Believed:** `.status.env` was only read by the status script.
**Evidence:** it was sourced unconditionally, so a test could have used the
real email sender and the real work directory.
**Changed:** `load_env` sets defaults only; anything in the environment wins.
**Generalizes:** yes: template `notify.sh` and `status.sh`.

## 2026-11-13: v3's short replies were drift over time, not a selection bug

**Believed:** v3's training targets (median 620 chars vs 906 held out) were
shorter because the new gate selected short replies.
**Evidence:** per-year medians after cleaning: ~580–590 chars through
2005–2019, 778 in 2020–2024. The held-out years are a wordier era.
**Changed:** REPORT §5b corrected; length filters and recency weighting were
measured and rejected.
**Generalizes:** yes: with a time-based split, check for drift before blaming
the pipeline.

## 2026-11-13: Much of a training change was available at inference

**Believed:** fixing "thread context pulls v3 off the patch" needed retraining
with the patch first (v4).
**Evidence:** v3 unchanged, patch-first input, 87 patches with context: names
something from the patch 43% vs 21% with its own layout (about 1.2× vs 0.8×
Linus' rate on the same input).
**Changed:** PLAN-v4 update; v4 trained anyway as the base for v5.
**Generalizes:** yes: METHOD.md §4, probe at inference before training.

## 2026-11-13: One sample per patch is too noisy for small effects

**Believed:** 100 patches × 1 sample could compare two models.
**Evidence:** on 13 patches with byte-identical inputs in both runs, the same
model scored 46% vs 23%.
**Changed:** `compare.py`/`compare_mlx.py --samples K`, `train/paired_stats.py`
(paired bootstrap CIs and each baseline's noise floor).
**Generalizes:** yes: template `paired_stats.py`, METHOD.md noise floor.

## 2026-11-13: Docs drifted while the code moved

**Believed:** updating docs "in the same commit" was a habit we had.
**Evidence:** the README still called v2 current after v3 shipped; HANDOFF's
status was two days and two versions old; PLAN-v5 said "nothing is scheduled"
after it was queued, and "a new LoRA on top of v4" while the code continues
v4's adapter; AGENTS.md quoted 170 tests (217).
**Changed:** AGENTS.md "Documenting as we go" (where each fact lives, what to
update on each event); this file; `scripts/docs_check.py` in the gate.
**Generalizes:** yes: template has the same rules, log and check.

## 2026-11-13: Two shell traps in unattended tooling

**Believed:** `ssh host "pgrep -f compare.py"` tells whether a job is running;
copying a fixed script over the old one is safe.
**Evidence:** pgrep matched the remote shell running it (always "busy"); a
watcher timed out for the same reason. bash reads a running script as it goes,
so overwriting `night-run.sh` under the waiting v4 loop could run garbage.
**Changed:** bracketed patterns (`'[c]ompare.py'`); updates copy to a temporary
name and `mv`.
**Generalizes:** yes: template OPERATIONS.md, the improvement queue.

## 2026-11-13: Two different files were both called "patch-first"

**Believed:** v3's patch-first multi-sample run and v4's would see the same
inputs, so v3 patch-first vs v4 would isolate what v4's training added.
**Evidence:** checksums differed. The Mac's file was an earlier rendering made
for the input-order probe: 672 of 774 inputs differed (patch length budget, a
blank line after the marker) and 58 references (older sign-off stripping).
Found while wiring v5's starting-model decision to that comparison.
**Changed:** the v3 patch-first run now uses v4's exact eval file
(`data/v4/sft_eval.jsonl`; the probe's copy kept as
`sft_eval_patchfirst.probe.jsonl` on the Mac). The probe's numbers stand, but
they used the earlier rendering.
**Generalizes:** yes: before a paired comparison, checksum the inputs; "the same
layout" by name is not the same bytes.

## 2026-11-14: The morning report raced the next night

**Believed:** the reporter, which follows `latest-run.log`, would see every
night's "night done" line and send that night's report.
**Evidence:** v4 training finished at 06:49 on 2026-11-14 and no email came.
The multi-night loop started the next night's run 10 seconds after "night
done", which repointed `latest-run.log` to a fresh log; the reporter checks every
5 minutes, so it never saw the old night end. It also could have missed a plan
failure the same way.
**Changed:** `scripts/night-reporter.sh` checks the newest two run logs (per-log
`check_log`). Dry run against the real logs with a fake sender: one report for
night-2026-11-13, none repeated on the next check.
**Generalizes:** yes: template reporter fixed the same way, and template
OPERATIONS.md says why. A watcher that follows a "current" pointer misses events
when the pointer moves faster than it polls.

## 2026-11-14: A "go now" launch measured against the wrong deadline

**Believed:** `GO_NOW=1 MAX_MINUTES=480 night-run.sh` gives the plan 8 hours.
**Evidence:** launched at 07:59 ET, the plan logged "skipping v4 evaluation:
needs ~100 min, -19 left" and exited. `need_minutes` in the plan only knew the
regular window, whose hard stop (07:40) had passed; night-run's own deadline
never reached it. The queue's cluster steps launch the same way.
**Changed:** `night-run.sh` exports `PLAN_DEADLINE`; `window_deadline` uses it;
`test_a_go_now_deadline_overrides_the_window`.
**Generalizes:** yes: template `night-run.sh`, `window-lib.sh`, test and
OPERATIONS.md. Pass a deadline down to whatever checks it, rather than letting
each layer recompute it.

## 2026-11-14: Code written ahead of time, read before its first run

**Believed:** the drafted DPO trainer worked like `sft_lora.py` ("same
checkpointing and --resume-auto conventions"), and the cluster ran the code we
had tested.
**Evidence:** read before wiring it in: (1) `dpo_lora.py` had no resume; (2)
PEFT 0.21 saves an adapter not named "default" to `adapter-final/policy/`, so
the chain's success check and the Mac conversion would have found nothing;
(3) on two nodes only rank 0 writes checkpoints, so a resumed `sft_lora.py`
run would restart rank 1 from step 0 (never hit: v4 trained in one night);
(4) the cluster's `compare.py` predated `--samples`, which v5's candidate step
needs. Separately, `pkill -f` over ssh killed the ssh session itself, a
variant of the 2026-11-13 trap.
**Changed:** `train/adapter_ckpt.py` (flat, atomic saves; resume point) with
`tests/test_adapter_ckpt.py`; `dpo_lora.py --resume-auto`; `TRAINER=dpo` and a
checkpoint copy to the peer in `run-chain-2node.sh`, `tests/test_chain.py`;
`deploy_train` in the queue steps; a `dpo-smoke` queue item before any real
DPO run.
**Generalizes:** yes: template chain copies checkpoints to the peer (test
added); OPERATIONS.md says to deploy in the step and checksum the nodes, and
to test a resume on purpose.

## 2026-11-14: Deleting a cached model freed nothing

**Believed:** `rm -rf ~/.cache/huggingface/hub/models--<model>` frees that
model's disk space (the v4 Mac step did this to make room).
**Evidence:** free space stayed at 63 GB and the v4 build failed (needs ~88).
This `huggingface_hub` (2.4) keeps file data in a shared, deduplicated store
(`hub/blobs`); model folders hold small pointers. `hf cache ls` then reported
"6 unreferenced shared blob(s) totalling 29.5G".
**Changed:** `scripts/q/v4-mac-convert.sh` uses `hf cache rm` + `hf cache
prune` and re-checks free space before converting; CONVERSION.md.
**Generalizes:** yes: template CONVERSION.md. Free cache space with the
cache's own tool, and re-measure after.

## 2026-11-14: The starting-model rule couldn't see what the next step is for

**Believed:** a rule set before the data (v4 unless clearly worse on patch
grounding or loops) was enough to pick v5's starting model automatically.
**Evidence:** v4's blind judge against v3 came in at v3 45, v4 28; the reasons
say v4 makes more false claims about the patch. Neither of the rule's metrics
measures correctness, and v5 is preference training *for correctness*. Nick
asked why we'd build on v4; the rule would have picked it without looking.
**Changed:** a human gate (`v5-base-decision`) before candidate generation;
PLAN-v5 dated note. `choose_base.py` unchanged (its output is one input).
**Generalizes:** yes: template METHOD.md item 5: a pre-set rule must include
the metric the next step optimizes, or it gets a human gate.
