# lkml-reviewer

Teaching a language model to review Linux kernel patches the way Linus Torvalds
does: blunt, specific, about the code. **We are training on his judgement, not
his swearing.**

This repo has the whole method: a polite crawler and data pipeline that mines
`(patch, Linus review)` pairs from 30 years of LKML, the LoRA fine-tune of
`Qwen/Qwen3.8-27B`, the evaluation harness, and a candid write-up of what
worked, what didn't, and the bugs that nearly poisoned the data.

- **Results and learnings:** [`docs/REPORT.md`](docs/REPORT.md)
- **Next iteration:** [`docs/PLAN-v5.md`](docs/PLAN-v5.md)
- **The model (v3):** [nkavassalis/lkml-reviewer-lora](https://huggingface.co/nkavassalis/lkml-reviewer-lora) on Hugging Face
- **Design decisions, invariants, measured numbers:** [`AGENTS.md`](AGENTS.md)
- **Where the work stopped:** [`HANDOFF.md`](HANDOFF.md)
- **Reusable version for other datasets:** [finetune-27b-template](https://github.com/nkavassalis/finetune-27b-template),
  the generalised pipeline (config-driven training, evaluation, Mac conversion, operations)

Not included: the corpus, the training data, or the model weights. They're
other people's words and a derived model, and everything here rebuilds them.

## The journey so far

Every version so far learned how Linus *sounds*; v5 is the first aimed at
being *right*.

| version | what changed | what we learned |
| --- | --- | --- |
| **v1** | First LoRA on ~1,200 review pairs, 2 epochs, one GPU. | Opened 94% of reviews with an invented "On <date>, <name> wrote:": the training targets still carried quote headers. The model learns your data bugs; read its output. |
| **v2** | Headers stripped, 1 epoch. | Learned the style (terse, plain text, less abusive than Linus) and beat the base model 44–16 in a blind judge, mostly by saying less rather than saying wrong things. Signed every review "Linus"; greedy decoding looped 44% of the time. |
| **v3** | 4.4× the data (5,110 pairs), earlier thread messages in the input, sign-offs stripped, 2 nodes. | Never signs as Linus and has the best held-out loss; beats the base 51–24 but ties v2, because the thread context pulled it off the patch. [Published](https://huggingface.co/nkavassalis/lkml-reviewer-lora), with an MLX 8-bit build for a 64 GB Mac that loses nothing measurable. |
| **v4** | Same data, patch first and thread after. | Before training, a probe showed much of the gain comes free by reordering v3's input, and that one sample per patch can't see a 10-point effect, so we built multi-sample evaluation with noise floors. Trains in the overnight window; results pending. |
| **v5** | Queued: preference training (DPO). v4 writes 4 reviews per patch, a judge ranks them against Linus' real reply, and training pushes toward the better one. | Not yet run. The aim: the first version with a signal for "this review is wrong", not just "this doesn't sound like him". |

Numbers and details: [`docs/REPORT.md`](docs/REPORT.md),
[`docs/PLAN-v4.md`](docs/PLAN-v4.md), [`docs/PLAN-v5.md`](docs/PLAN-v5.md).

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
