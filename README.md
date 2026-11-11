# lkml-reviewer

Teaching a language model to review Linux kernel patches the way Linus Torvalds
does: blunt, specific, about the code. **We are training on his judgement, not
his swearing.**

This repo has the whole method: a polite crawler and data pipeline that mines
`(patch, Linus review)` pairs from 30 years of LKML, the LoRA fine-tune of
`Qwen/Qwen3.8-27B`, the evaluation harness, and a candid write-up of what
worked, what didn't, and the bugs that nearly poisoned the data.

- **Results and learnings:** [`docs/REPORT.md`](docs/REPORT.md)
- **Next iteration:** [`docs/PLAN-v3.md`](docs/PLAN-v3.md)
- **Design decisions, invariants, measured numbers:** [`AGENTS.md`](AGENTS.md)
- **Where the work stopped:** [`HANDOFF.md`](HANDOFF.md)

Not included: the corpus, the training data, or the model weights. They're
other people's words and a derived model, and everything here rebuilds them.

## Where it stands

v2, the current model, learned the *style* from 1,175 examples in one epoch:
plain-text, terse reviews, with abusive language below Linus' own rate. It did
not learn the *substance*: its reviews are shorter and less specific than his,
and sometimes confidently wrong. v3 attacks that with 4.6× more data and thread
context. Details and numbers are in the report.

**This is an unofficial style imitation, not affiliated with or endorsed by
Linus Torvalds.** Don't post its output to a mailing list or attribute it to him.

## Quick start: the data pipeline

Stdlib-only Python. No network access in tests.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest            # hermetic suite
scripts/gate.sh                       # tests + lint + compileall + secret scan

# one-time, for commit matching (~4 GB):
git clone --bare https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git \
    data/git/linux.git

.venv/bin/python -m linuspr crawl --year 2024 --patch-only   # polite: >= 0.2 s/request
.venv/bin/python -m linuspr thread      # recover the patch each reply answers
.venv/bin/python -m linuspr pairs       # quality gates -> review_pairs.jsonl
.venv/bin/python -m linuspr resolve     # attach the merged commit
.venv/bin/python -m linuspr sample -n 5 --seed 7    # read some. Really.
.venv/bin/python -m linuspr audit -n 5 --seed 7     # read the moderation calls
.venv/bin/python -m linuspr export --eval-year 2024 # sft.jsonl + sft_eval.jsonl
```

Stage order matters: `crawl` → `thread` → `pairs` → `resolve` → `export`.
Crawls are resumable (every page is cached under `data/raw/`). The source is the
`lkml.iu.edu` hypermail mirror, volunteer-run infrastructure; the crawler
identifies itself, rate-limits, and backs off. Please keep it that way.

## Training and evaluation

`train/` runs inside a container on a GPU node (it needs torch, transformers,
peft and flash-linear-attention, which the stdlib pipeline doesn't):

| script | does |
| --- | --- |
| `sft_lora.py` | BF16 LoRA SFT, loss on the review only, fixed prompt template printed to the log |
| `compare.py` | base vs fine-tuned on the held-out set, greedy and sampled, one model load |
| `score.py` | proxy metrics (specificity, chatbot-isms, abuse, loops) — stdlib |
| `judge.py` | blind pairwise judge on any OpenAI-compatible endpoint (`JUDGE_URL`) — stdlib |
| `judge_check.py` | planted-wrong test: can the judge tell on-topic from off-topic? |
| `run-chain.sh` | stop a serving container, train, compare, restore it in an `EXIT` trap |

One GB10 (DGX Spark) trains the 27B at ~163 tok/s with a 71.5 GB peak at 4,096
tokens.

## The lesson, if you read nothing else

Five data bugs passed the test suite and produced plausible-looking counts:
flattened code indentation, mailing-list footers in half the targets, an abuse
filter that cut "looks sane to me", patches attached to reviews from a different
decade, and invented `On <date>, <name> wrote:` headers. Every one was found by
reading output by hand, two of them by reading what a model generated. **A
percentage you haven't sampled by eye isn't a metric.**

## Licence

MIT, see [`LICENSE`](LICENSE). The base model is Apache-2.0. LKML posts belong to
their authors; this repo ships three of them as test fixtures and none as
training data.
