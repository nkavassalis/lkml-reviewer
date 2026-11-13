# AGENTS.md — linus-pr

## Mode

**PUBLIC** (since 2026-11-11): published on GitHub under the MIT licence. The
history starts from a single clean commit. `.githooks/pre-push` runs
`scripts/gate.sh` (tests, lint, secret scan) before any network push and refuses on
failure. It also refuses to push the local-only `local-history` branch, which
holds the pre-publication commits and contains private infrastructure details.

**Never committed:** `data/` (corpus, indexes, adapters, run outputs), model
weights, cluster addresses or hostnames, home-directory paths. The secret scan
catches private IP ranges and `/home/<user>/` paths; hostnames are on you.

## Purpose

Mine Linus Torvalds' LKML replies into `(patch context, Linus review)` pairs so a
reasoning model can be fine-tuned to review code the way he does — blunt, specific,
about the code.

The one-line spirit: **we are training on his judgement, not his swearing.** Every
curation decision below should be traceable back to that sentence.

## Layout

```
linuspr/
  parse.py      pure HTML -> data for lkml.iu.edu hypermail pages (no network)
  fetch.py      stdlib-only polite crawler for that mirror (delay, retries, cache)
  pipeline.py   month listing, crawl orchestration, corpus JSONL (merge-on-write)
  pairs.py      Message -> ReviewPair + the quality gates (the interesting file)
  moderate.py   abuse filter: keep | scrub | drop, applied at export
  split.py      year-based train/eval partition (random splits leak near-duplicates)
  gitrepo.py    the bare torvalds/linux clone: msgid + subject -> commit
  resolve.py    attaching the real diff: thread walk, msgid recovery, subject match
  __main__.py   CLI: months | crawl | thread | pairs | resolve | export | sample |
                stats | index | progress
data/           gitignored. raw/ = cached HTML, linus_messages.jsonl,
                patch_texts.json, review_pairs.jsonl, sft.jsonl,
                msgid_index.json, commit_subjects.json, git/linux.git, crawl-*.log
tests/          hermetic tier only; fixtures/ holds REAL saved pages, not toy HTML
scripts/        gate.sh, secret-scan.sh, crawl-all.sh, status.sh (read-only status
                of a long run; hosts in gitignored .status.env), install-reviewer.sh,
                convert-mlx.sh
train/          fine-tune + evaluation; runs in a container on the GPU node, not
                in .venv, and is outside the stdlib-only rule:
                sft_lora.py (BF16 LoRA SFT), compare.py (base vs tuned generations),
                score.py (proxy metrics, stdlib), judge.py + judge_check.py (blind
                pairwise judge and its planted-wrong validation, stdlib),
                run-chain.sh (stop serving -> train -> compare -> restore),
                merge_lora.py (streaming LoRA merge, a few GB of RAM)
reviewer/       review a GitHub PR / local diff with the LoRA: gh.py (stdlib, every
                gh failure mapped to a remedy), prompt.py (PR -> training format),
                backends.py (cuda now; mlx planned), __main__.py (CLI). See its README
linuspr/template.py  the prompt format, shared by export and reviewer; a test fails
                if train/sft_lora.py's copy drifts
docs/           OPERATIONS.md (long jobs on shared machines: restore traps,
                memory caps, resume, watcher + hourly status runs; read it before
                any long run), REPORT.md (results + learnings), PLAN-v3.md, PLAN-v4.md (draft),
                CONVERSION.md (adapter -> merged BF16 -> MLX 8-bit; the standard
                last stage for every model, targeting a 64 GB M4 Pro)
HANDOFF.md      where the last session actually stopped — read it alongside this file
```

## How to run

```bash
cd ~/linus-pr
.venv/bin/python -m linuspr months                      # what the mirror publishes
.venv/bin/python -m linuspr crawl --year 2024 --patch-only
.venv/bin/python -m linuspr crawl --month 2406.1        # one bucket
.venv/bin/python -m linuspr thread                      # recover the posted patch
.venv/bin/python -m linuspr pairs                       # gate -> review_pairs.jsonl
.venv/bin/python -m linuspr resolve                     # attach the merged commit
.venv/bin/python -m linuspr sample -n 3 --seed 7        # eyeball it
.venv/bin/python -m linuspr audit -n 5 --seed 7         # eyeball moderation calls
.venv/bin/python -m linuspr export                      # {"input","output"} SFT file
.venv/bin/python -m linuspr export --eval-year 2024     # sft.jsonl + sft_eval.jsonl
.venv/bin/python -m linuspr stats                       # funnel numbers
.venv/bin/python -m linuspr progress                    # how far we have got
scripts/crawl-all.sh                                    # everything, end to end
```

**Stage order matters:** `crawl` -> `thread` -> `pairs` -> `resolve` -> `export`.
`pairs` reads `data/patch_texts.json`, so if you re-run `thread` you must re-run
`pairs` before `resolve`, or the pairs file carries stale (empty) patch text.

The linux clone lives at `data/git/linux.git` (bare, ~4 GB, one-time):

```bash
git clone --bare https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git \
    data/git/linux.git
```

Without it `crawl`/`thread`/`pairs` still work; only `resolve`'s commit matching
needs it.

Long crawls: `nohup scripts/crawl-all.sh > data/crawl-all.log 2>&1 &` (or the same
`linuspr crawl` invocation directly). This box has no systemd and an orphaned
process gets killed — keep the parent loop alive, or launch from a login shell that
stays up. Crawls are **resumable**: every fetched page is cached under `data/raw/`,
and `write_jsonl` merges per-month instead of clobbering, so re-running is safe and
cheap. `--delay 0.2` is the floor of politeness on that volunteer mirror.

## How to test

```bash
.venv/bin/python -m pytest        # 170 hermetic tests, no network, no sleeping
.venv/bin/python -m ruff check .
scripts/gate.sh                   # pytest + lint + compileall + secret scan
```

There is no `ui` tier — this is a data pipeline, not a web app. `gate.sh` reports
`SKIP — no tests marked ui` (pytest exit 5) instead of failing for that tier; a real
ui failure still fails the gate. The hermetic tier is the whole suite and must be
green before any change is committed.

## Measured numbers (so a regression is visible)

**Full archive, 1996 → Nov 2026 (1,472 buckets), measured end to end:**

| stage | count | rate |
| ----- | ----- | ----- |
| Linus messages harvested | 8,802 | — |
| patch text recovered by `thread` | 6,120 / 8,802 | 69.5% |
| gated pairs (`pairs`) | **1,377** | 15.6% of his messages |
| pairs with the real posted patch | **1,048** | 76.1% |
| pairs matched to a merged commit | **564** | 41.0% |
| pairs with no context at all | 249 | 18.1% |
| moderation (`export`) | 1,026 keep / 239 scrub / 112 drop | 8.1% dropped |
| **final train / eval (2024 held out)** | **1,187 / 78** | — |

2019 alone, for a per-year sanity check: 48 buckets, 355,530 messages seen, 614
Linus messages, 93 pairs, 81 with the real posted patch.

**1,187 training pairs is small** — enough to prove the loop and the style
transfer, not enough to judge the approach. The gate is deliberately conservative
(`code_evidence >= 3`, `--patch-only`); loosening it is the first lever. Review
length median 1,001 chars (p90 2,483); quoted-context median 389 chars.

## Design decisions and why

**The corpus source is `lkml.iu.edu/hypermail/linux/kernel/`, not lore.kernel.org.**
lore sits behind Techaro Anubis ("Making sure you're not a bot!") and refuses
stateless HTTP; the hypermail mirror is open, complete from 1993 to current, and
`Fetcher` raises `FetchError` rather than silently ingesting a challenge page. The
mirror mangles email addresses to `xxxxx` but **keeps real `Message-ID`, `Subject`
and `Date` in `<!--X-...-->` comments**, which is what makes threading and later
lore/git reconciliation possible. If the mirror dies, the fallback ladder is
`bot-check-ladder` (rung 2, persistent Chromium profile) against lore — not written
yet.

**Month buckets are plain `YYMM.N`** (two-digit year, two-digit month, part counter
0..3): `2406` = April 2024, `1901` = January 2019, `9603` = March 1996. The mirror
publishes `9603` (Mar 1996) .. `2611` (Nov 2026), so the first two digits are only
ever 96..99 or 00..26 and `bucket_year()` pivots at `PRE_2000_THRESHOLD = 50`.
This looks trivial and cost two wrong attempts: the obvious `yy >= 93` rule read
`1901` as January **1919**, so `--year 2019` silently harvested 48 pre-kernel months
and reported "355,530 messages seen, 614 from Linus" — plausible-looking and
completely wrong; a second attempt misread the id as `YYYM` and turned 2024 into
2140. Any new date filtering must go through `bucket_year`, and any change to it
must keep `test_bucket_year_*` green.

**All thread topology comes from the month `index.html`.** hypermail leaves `<li>`
and `<ul>` unclosed, so depth is tracked on `<ul>` alone with a spine of enclosing
anchors. `Message.prev_mid` (from the "Previous message" nav link) is the reliable
single-parent pointer. Do not trust `IndexEntry.depth` for anything subtle.

**A pair is `(what Linus quoted, what Linus wrote)` from the *same* message.** No
cross-message joining, no inference. That makes a pair verifiable by eye and immune
to threading bugs — at the cost that `context` is only as complete as Linus' quoting
habit (median ~385 chars, only ~11% contain a real diff hunk). This is the single
biggest quality limit on the dataset; see Known issues.

**`pairs.py` gates hard, because most of LKML is noise.** A message becomes a pair
only if: it is `Re:` in a `[PATCH]` thread, not a release announcement, not
off-topic, not an "applied"/"pulled" bookkeeping reply, its own text is ≥120 chars,
its quoted context is ≥80 chars, and `code_evidence(context) >= 3` (diff headers,
`@@` hunks, code punctuation, snake_case symbols). Net yield measured on real data:
~15% of Linus messages.

**Author names come from the `On <date>, <name> wrote:` lines Linus leaves behind**
(`attributed_authors`). That is free signal about who he is talking to; the emails
are mangled by the mirror so we keep names only.

**Attaching the real diff went through four designs; the fourth is the one that
works.**

1. *Resolve via lore msgid.* Dead end: only 8 of 614 Linus messages contain any
   msgid at all — he quotes the code, never the `Link: lore.kernel.org/...` line.
2. *Recover the mangled msgid.* hypermail eats the domain (`brauner@xxxxxxxxx`)
   but leaves the local part, so `DomainIndex` can ask "which real domain has a
   msgid with exactly this local part?". It works and is tested, but there were
   still only 6 candidate tokens in 93 pairs. Kept as the highest-trust route.
3. *Walk the nav link* (`walk_thread_for_patch`). Following `Message.prev_mid`
   looked great — **62% coverage** — and was wrong: hypermail's "Previous
   message" is *document order*, not parentage, so a third of those "recovered"
   patches were the tail of an unrelated thread (a KEYS review wearing an
   `n_hdlc.c` diff). It survives only as a fallback, behind a subject gate.
   **The lesson: a coverage number you have not sampled by eye is not a metric.**
4. *Group by subject* (`ThreadMap`, the current route). hypermail's own thread
   view groups by subject line, so the patch a reply answers is the earliest
   message sharing its subject. Coverage **76% of pairs** (an earlier 87–88% figure was inflated by
   the mid-collision bug below), and hand-checked
   samples show the right files (access_ok() -> `arch/powerpc/.../uaccess.h`,
   mincore -> `Documentation/sysctl/vm.txt`).

Both routes require the parent's subject (taken from the month **index**, not the
nav link — the link renders `Author: "Subject"` and subjects contain colons the
nav regex cannot delimit) and a `patch_quality` score, so a wrong-thread blob
cannot slip through on shape alone.

5. *Match the merged commit by subject* (`resolve`). The commit index gave 0 hits
   (same reason as 1), so `SubjectIndex` inverts every commit subject in the tree
   and `match_commit` requires **both** word-overlap ≥ 0.6 **and** the commit's
   author year within 2 years of the message — kernel subjects repeat across
   decades, and the year gate is what stops a 2019 patch binding to its 2015
   namesake. Verified by hand: matched commits resolve to the right subject and
   date (e.g. `arch/openrisc: Fix issues with access_ok()` -> 2019-01-08).

**`git log` output is decoded with `errors="replace"`.** Commit messages are not
guaranteed valid UTF-8 (Latin-1 author names), and `subprocess(text=True)` raised
`UnicodeDecodeError` at byte 0xf6 after 29 minutes of scanning. All git I/O goes
through `GitRepo.run_bytes`.

**Mangled emails are scrubbed at export only.** `scrub_mangled()` deletes
`name@xxxxxxxxx` addresses and the dead lore links that contain them, and it is
applied in `ReviewPair.as_sft()` — never to the corpus on disk, so the transform
stays reversible. The regex requires the domain to start *and* end with `x`, which
is what keeps `a@xxxxxx.com` (a real-looking placeholder) from being eaten.

**`stupid` is code-level, and always cut (2026-11-11, Nick's call).** It used to
be a person attack that dropped the whole pair: 159 v3 pairs for that word
alone, mostly aimed at code or tools. Now its sentence is cut and the review
kept; "you're stupid" / "stupid people" still drop. Unlike other code-level
words, a code-directed `stupid` sentence is never kept (`ALWAYS_CUT`).

**Abuse is dropped, never rewritten.** `moderate.py` is a vocabulary heuristic with
three outcomes: `keep`, `scrub` (cut abusive sentences/clauses — the surviving
prose is byte-identical to his, there is a test for exactly that), and `drop`
(person-directed insults, or abuse with no technical content). When unsure it
drops: a lost pair costs nothing, a pair that teaches "this is braindead code" as a
*style* costs the project. Measured on the 2019 pairs: **62 keep / 20 scrub /
11 drop** — 12% of the corpus removed before training. It runs at `export` by
default (`--no-moderate` to skip); `review_pairs.jsonl` stays unmoderated so you
can audit what it did.

**Crawler etiquette is code, not a comment.** `Fetcher` sends a self-identifying UA,
sleeps ≥0.2s between requests, does exponential backoff on transport errors, caps at
3 tries, and never caches a bot-check page. It is volunteer-run infrastructure we
do not own.

**Stdlib only.** No `requests`, no `b4`, no `pip install` of anything new — urllib
plus a cookie-free GET is enough for this mirror and keeps the pipeline auditable.

## Invariants

- `linuspr/parse.py` never touches the network; `fetch.py` owns all I/O. Tests rely
  on this split.
- Corpus JSONL is merge-on-write per month. A crawl of one month must never delete
  another month's rows.
- `data/` is gitignored. Never commit the corpus — it is tens of MB of other
  people's words, and diffs through it are noise.
- No secrets, hostnames, paths, or credentials in anything committed; the corpus is
  public mailing-list text but still large — keep it out of git.
- Only `main` is pushed. `local-history` never leaves this machine.

## Conventions

- One CLI subcommand per pipeline stage, each idempotent and re-runnable.
- Stats are printed as a funnel (`messages -> replies -> patch replies -> pairs`), so
  a regression in a gate shows up as a changed percentage, not a silent shrink.
- Fixtures are real pages copied out of `data/raw/`, with the filename carrying
  `month_mid`. Add a fixture rather than hand-writing HTML.

## Known issues / TODOs

1. **5% of pairs still have only Linus' thin quote as context**, and the subject
   group only ever contains messages from the *same* month bucket — a reply to a
   patch posted last month falls back to the nav-link walk. A cross-bucket
   `ThreadMap` (built from all crawled indexes) is the obvious next step; the
   indexes are already cached on disk.
2. The abuse filter is a **vocabulary heuristic, not a classifier**. Its word lists
   will miss things (sarcasm, novel insults) and occasionally cut a sentence that
   was fine. Read a sample of `scrub`/`drop` decisions before trusting a training
   run; the vocabulary is the thing to extend, not the logic.
3. The eval split exists (`split.py`, `export --eval-year`); **nothing has been
   trained on it yet.** Default held-out year is 2024. Splitting by year rather
   than randomly is deliberate — same-subsystem, same-maintainer threads cluster
   in time and a random split leaks them into train.
4. **The mirror mangles every email address to `xxxxxxxx`.** `scrub_mangled()`
   removes them at export time (see the design decision above); the on-disk corpus
   keeps them so the decision stays reversible. If you train on `review_pairs.jsonl`
   directly instead of `sft.jsonl`, you are training on `peterz@xxxxxxxxxxxxx`.
5. `stats` still reports a couple of legacy funnel rows (`own has tech vocab`) that
   are diagnostic only and are not used by the gate.
6. Duplicate detection is `(subject, review[:200])`; near-duplicates (quotation
   fixes, reposts) survive.

## Testing limitations / unverified

- **Verified by me, this session:** 126 hermetic tests green; `ruff check .` clean;
  real crawl of 2024-06 (4 buckets, 37,883 messages seen, 119 Linus messages) and of
  full 2019 (48 buckets, 355,530 messages seen, 614 Linus messages); `thread`
  recovered patch text for 375/614; `pairs` produced 93 gated pairs, 58 with the
  real posted patch; `resolve` matched 52 pairs to merged commits, spot-checked by
  hand against `git log` (correct subject **and** date); the bare linux clone built
  and scanned (1,506,282 commits, 453,946 msgids); the pre-push guard refuses a
  network remote; `bucket_year` regression tests for the 1996/2019 boundary;
  `gate.sh` PASS (pytest, ruff, compileall, secret scan).
- **Not verified:** whether the hypermail index is *complete* — I compared it to no
  independent source, so a systematic gap would be invisible to me. Check one month
  against lore by hand before trusting absolute counts.
- **Not verified:** `match_commit` at scale. I hand-checked a handful of matches and
  the year gate is doing its job, but 0.6 word-overlap on short subjects ("fix a
  typo") is exactly where a wrong patch will bind to a wrong commit. Audit
  `commit` assignments before treating them as ground truth.
- **Not verified:** `patch_quality` on non-C content (Kconfig, DTS, asm, Rust) and
  on patches posted inline-but-truncated by the mailing list.
- **Not verified:** `Message.prev_mid` correctness on deep threads — the walk is
  tested against a synthetic chain, not against a real 30-message thread.
- **Not verified:** behaviour when the mirror rate-limits or goes down mid-crawl;
  retries are unit-tested against a fake, never against the real host.
- **Not verified:** the msgid-recovery route (`DomainIndex`/`recover_msgid`) on real
  data — it is unit-tested but contributed 0 of the 52 matches, so it is unproven
  in anger.
- **Not verified:** the moderation vocabulary against real corpus volume — the
  62/20/11 split is from 93 pairs. Sample `scrub`/`drop` decisions by hand before
  a training run (`python -m linuspr audit -n 5 --seed 7`).
- **Found by hand, and now regression-tested:** substring matching in the abuse
  vocabulary hit *Christoph **Hell**wig* and *un**sane*** in `filldir64()/unsafe`,
  dropping a perfectly good review. Matching is whole-word now; the lesson is
  that the vocabulary needs reading, not just extending.
- **Found by hand (2026-11-10, second session), now regression-tested:**
  (1) `scrub_mangled` collapsed *every* run of spaces/tabs in the export, so
  every diff and code block in `sft.jsonl` had its indentation flattened;
  (2) the moderation scrub re-joined sentences after `.strip()`, same damage
  to code inside scrubbed reviews; (3) 616/1,184 training outputs (52%) ended
  in the vger "To unsubscribe from this list" trailer — `strip_list_footers()`
  now removes vger and Google Groups trailers at export; (4) bare `sane`
  scrubbed 86 reviews, mostly verdicts like "Looks sane to me" — now only the
  phrase "no sane person" matches. None of these showed up in any count.
- **Found by a smoke-test generation, now regression-tested:** `patch_texts.json`
  was keyed by `mid`, a per-bucket counter that repeats in every bucket, so
  patches overwrote each other across years and `thread` skipped messages whose
  `mid` was already present. A 2024 eventfs review carried a 2010 block-layer
  patch. Keys are now `month/mid` (`patch_key()`). Coverage fell from a false
  88.2% to a real 76.1%; patches dated >2 years from their review went from 17
  to 0; a hand-check of 6 pairs still flagged by a subject-word heuristic found
  all 6 correct. The old file is kept as `data/patch_texts.collided-mid-keys.json`.
- **Should be checked by hand:** read 20 sampled pairs (`sample -n 20 --seed 7`)
  before believing the gate is selecting reviews. Percentages do not prove quality.

## Do not touch

- `data/raw/` — re-fetching costs hours of polite crawling.
- `data/git/linux.git` — a ~4 GB, one-time clone; `git -C data/git/linux.git gc`
  rather than deleting it.
- `data/msgid_index.json`, `data/commit_subjects.json` — each costs a full
  `git log` pass over 1.5M commits to rebuild.
- `.githooks/pre-push` — the gate + secret scan before every push.
- `bucket_year()`'s `PRE_2000_THRESHOLD = 50` — it is calibrated to the mirror's
  actual bucket range (9603..2611). Do not "tidy" it to 93 or 70, and do not
  re-derive the id format from the id alone: `1901` is 2019, not 1919.
- The fixtures under `tests/fixtures/` — they are captured truth about the mirror's
  markup; regenerating them silently launders parser regressions into passing tests.

## Guardrails in effect

- Pre-push hook: gate must pass, secret scan must be clean, `local-history` refused.
- `scripts/secret-scan.sh` runs in `scripts/gate.sh`; run it before any commit.
- Crawler: self-identifying UA, ≥0.2s delay, 3 retries max, bot-check pages raise
  instead of being cached.
- No new dependencies without asking Nick.
