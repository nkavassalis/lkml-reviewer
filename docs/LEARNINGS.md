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
Then, to catch it without a person noticing: `train/decide.py` (exit 76 when
the evidence disagrees; a `--target` signal that leans the other way at all,
since 45–28 is p = 0.06 and a significance test alone passes it), run by
`multisample-report`; queue exit 76 = a gate the step raises itself;
`docs_check.py` requires **Optimizes:**/**Measured by:** in PLAN `## Decision:`
sections; AGENTS.md "Automatic decisions".
**Generalizes:** yes: template METHOD.md item 5, `decide.py`, the queue's exit
76, the docs check, PLAN-template's Decision section, AGENTS.md.

## 2026-11-14: Two judges, two different questions

**Believed:** the pairwise judge (`judge.py`) and v5's preference rubric
(`make_preferences.py`) measure the same thing: review quality, accuracy first.
**Evidence:** both rank accuracy first, but only the rubric says "saying little
is not a virtue". The pairwise judge's v4-vs-v3 reasons are often "A makes no
false claims, whereas B…": a cautious, vague review beats a specific one with
a slip. That's the v2 effect, and v5's scoring deliberately won't reward it.
**Changed:** `train/rubric.py` scores models with the next step's own scorer
(`make_preferences.ask`); queue item `rubric-score`; its v3-vs-v4 score is
the `--target` of v5's starting-model check; the judge is a non-target
signal. Tests in `tests/test_rubric.py`.
**Generalizes:** yes: template `rubric.py`, METHOD.md and AGENTS.md: measure a
decision with the scoring function the next step will optimize, not a proxy
with a different rubric.

## 2026-11-14: The layout probe didn't replicate, and the data was the ceiling

**Believed:** thread context pulls the model off the patch; putting the patch
first fixes much of it (probe: v3 patch-first named the patch 43% vs 21%), so
v4 retrained with that layout.
**Evidence:** 200 held-out patches x 3 samples on identical bytes: v3 24%
thread-first vs 25% patch-first, v4 26%, all inside the noise floor; v4 = v3
on every measure. Only 33% of the training replies name anything from the
patch (held-out 37%); the models match the data's rate. The probe was 1 sample
per patch on an earlier rendering of the inputs.
**Changed:** REPORT §5c, PLAN-v4 done; the review-only test (PLAN-v5) trains
on the 1,718 replies that engage the patch.
**Generalizes:** yes: template METHOD.md: measure what fraction of the
training targets do the task you want before blaming the model, and never
build a version on a single-sample probe.

## 2026-11-14: We measured the models, never the data

**Believed:** the metrics we tracked (names the patch, length, loops) were
model diagnostics, and Linus' rate on held-out patches was the bar to clear.
**Evidence:** the same metric on the training targets, never run until
today, was 33%; the models were at 25%. v3 had loosened the data gate (any
reply with the real patch, 4.4x the data) without re-measuring what the new
pairs were about, and v4's target ("name the patch >= 20%, Linus 24%") treated
the data's ceiling as a benchmark. Nick: "How did we miss this?"
**Changed:** `train/score.py --targets` and `--train` (the training targets'
column in every results table; the review-only report uses it); `docs_check.py`
requires a **Data rate:** line in every active plan; AGENTS.md events "the data
selection changes" and "a model falls short on a metric".
**Generalizes:** yes: the same in the template (score.py, docs_check,
PLAN-template, AGENTS.md, METHOD.md section 4 "Diagnose the data before the
model").

## 2026-11-15: The DPO trainer only broke on two nodes

**Believed:** the DPO trainer was ready once its loss math was tested and the
code read before its first run.
**Evidence:** the 2-node smoke test (2 steps, synthetic pairs) failed after
loading the model: `'DistributedDataParallel' object has no attribute
'set_adapter'`. The adapter switch called PEFT's method on `model`, which had
just been wrapped in DDP. A single-GPU run never wraps, so it would have passed.
**Changed:** the switch and every PEFT call use the unwrapped `core`;
`tests/test_dpo_ddp.py` fails on any PEFT call on the wrapped model.
**Generalizes:** yes: template OPERATIONS.md launch checklist: smoke test on as
many nodes as the real run.

## 2026-11-15: "Serving confirmed" meant "container started"

**Believed:** the night runner's safety net confirmed serving was back.
**Evidence:** after the DPO smoke crashed at 03:19, the chain restarted vLLM
6 s later. The peer's GPU memory wasn't free yet ("53.95/121.69 GiB free ...
less than desired 102.22 GiB"), so vLLM exited 2.5 min later on both nodes. The
safety net had already logged "serving confirmed" after checking only that the
containers were running, 10 s in. vLLM was down for 7 hours. Nobody was
told. The next consumer, the thinking test's judge, failed all 60 calls in
2 s, and `rubric.py` exited 0, so its report went out as "done" with nothing
scored.
**Changed:** the chain's restore waits until no trainer is left on either node;
`ensure_serving` waits past a model load, requires every node still up,
retries once, and logs FAILED (the reporter emails it); `rubric.py` exits
non-zero when verdicts fail and retries them on a rerun. Tests:
`test_serving.py`, `test_chain.py`, `test_rubric.py`.
**Generalizes:** yes: template OPERATIONS.md rule 1, with the same code and
tests. A health check must outlast the thing's startup time, and a step built
on zero results has failed, whatever its exit code.

## 2026-11-15: DPO's memory isn't SFT's memory

**Believed:** a trainer that fit for SFT at 4,096 tokens would fit for DPO.
**Evidence:** with the DDP bug fixed, the smoke test ran out of memory in the
policy forward pass (92 GB in use, 91 GB allowed). DPO runs chosen and rejected
together, and `batch_logprobs` built full-vocabulary logits for every
position, prompt included, then used only the response rows.
**Changed:** `logits_to_keep` computes logits only from the earliest
response onwards. Checked on the cluster against the full computation with a
toy causal model (exact match, including a response starting at position 1).
**Generalizes:** partly: the template has no DPO trainer yet. When it gets one,
size memory from DPO's own smoke test (two sequences per step), not from SFT's.

## 2026-11-15: The DPO out-of-memory wasn't the logits

**Believed:** (entry above) full-vocabulary logits ran the DPO smoke test out
of memory.
**Evidence:** with logits only for response positions, it failed at the same
place with identical numbers (90.57 GB allocated). The trainer never called
`model.train()`. `from_pretrained` returns eval mode, and HF's gradient
checkpointing does nothing in eval mode, so every layer's activations were
kept for both sequences. `sft_lora.py` calls `train()`; the DPO trainer,
written from it, didn't.
**Changed:** `core.train()` before the loop, and dropout off (TRL's DPO
default: the reference pass must be deterministic). `test_dpo_ddp.py` checks
that `train()` is called. The logits change stays (it's correct and smaller).
**Generalizes:** yes: template OPERATIONS.md launch checklist: a trainer must
call `train()`; checkpointing is silently off without it. And a fix for a
resource failure isn't confirmed until the number moves: identical memory
figures meant the first diagnosis was wrong.

## 2026-11-15: The smoke test passed and the queue said it failed

**Believed:** the DPO smoke step checked the cluster's result correctly.
**Evidence:** the plan passed (two DPO steps, checkpoint, final adapter, eval
on both nodes) and wrote its marker with `touch`. The queue step checked it with
`test -s`, which also requires the file to be non-empty, so it reported "did not
pass" and the queue halted: nothing ran on the night of 2026-11-15. Separately, an
approval made while that step was running was lost. `workqueue.py run` kept the
state it loaded at the start and wrote all of it back hours later.
**Changed:** `test -f`; `tests/test_markers.py` rejects `-s` on any `*.ok`;
`workqueue.py run` re-reads the state before writing its own item, with a test
for an approval made mid-step.
**Generalizes:** yes: template. A check that has only ever seen failures is
untested: run the after-check against a real pass before trusting it. And a
long-running writer must re-read shared state before saving (workqueue.py, in
the template too).
