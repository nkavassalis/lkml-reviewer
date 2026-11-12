# reviewer: run the LoRA on a real pull request

```bash
# GitHub PR (needs the gh CLI, logged in)
python -m reviewer --pr 123 --repo owner/name --adapter path/to/adapter-final

# No GitHub at all
python -m reviewer --local --base origin/main --adapter ...   # your branch vs base
python -m reviewer --diff-file my.patch --adapter ...         # any unified diff
git diff | python -m reviewer --diff-file - --adapter ...

# See exactly what the model will be given; no GPU, no model
python -m reviewer --pr 123 --repo owner/name --dry-run

# Post it (as a comment; asks first unless --yes)
python -m reviewer --pr 123 --repo owner/name --adapter ... --post
```

`LKML_REVIEWER_ADAPTER` can stand in for `--adapter`. Big PRs exceed the
model's ~3,000-token input budget and lose the end of the diff (you get a
warning); `--per-file` reviews each changed file separately instead.

**This is a style imitation of a real person, trained on public LKML threads.
It is not Linus Torvalds, and its technical claims are often wrong.** A posted
review always starts with a disclaimer saying so. Posting is always a plain
comment: the reviewer never approves and never requests changes.

## What it does

1. **Gathers the change.** `gh pr view` + `gh pr diff`, or `git diff`, or a file.
   All the cheap checks (gh installed, authenticated, PR exists, diff not empty)
   run *before* the 55 GB model loads.
2. **Builds the prompt in the exact training format** (`linuspr/template.py`):
   `Subject: Re: [PATCH] <title>`, the PR description as the earlier-in-thread
   message from its author, then the diff. A test fails if this drifts from what
   `train/sft_lora.py` trained on. Drift between training and inference
   templates loses quality silently.
3. **Generates with sampling** (T=0.7, top_p 0.95, repetition penalty 1.1).
   Greedy decoding loops on ~44% of this adapter's outputs, so it isn't offered.

## gh problems, and what you'll see

| situation | message (exit 2) |
| --- | --- |
| gh not installed | install it from https://cli.github.com, or use --diff-file / --local |
| not logged in | run `gh auth login` (or set GH_TOKEN), or use --diff-file / --local |
| wrong PR number | no such pull request (check the number and --repo) |
| wrong / private repo | repository not found, or no access to it |
| outside a checkout, no --repo | not inside a GitHub checkout, so pass --repo owner/name |
| can read but not post | permission denied (your token can read but not write here?) |

Model or GPU problems exit 3, bad input exits 4. `tests/test_reviewer.py` runs
every row above against a fake `gh` on `PATH`.

## Backends

### cuda (now)

```bash
pip install -r reviewer/requirements-cuda.txt
```

BF16 needs ~60 GB of GPU memory (a DGX Spark / GB10, H100 80 GB, ...).
`--load-in-4bit` (NF4 via bitsandbytes) needs ~22 GB, at some quality cost
nobody has measured yet. `flash-linear-attention` provides the Gated DeltaNet
kernels the hybrid base uses; without it inference still works, slowly.

### mlx: Apple silicon, 64 GB M4 Pro Mac mini (planned after v3)

Not built yet. Here's the plan, so whoever builds it knows where the risk is:

- **Merge, then quantize.** mlx-lm can't load a PEFT adapter onto this base
  directly. On the GPU box, `merge_and_unload()` the LoRA into the BF16 base,
  then convert the merged model with `mlx_lm.convert` (`-q --q-bits 8`, or 4).
  Merging *before* quantizing avoids applying a LoRA trained against BF16
  weights to a quantized base.
- **Memory.** BF16 (54 GB) doesn't leave a 64 GB machine enough room. 8-bit is
  ~29 GB and 4-bit ~15 GB, both with room for the KV cache. The M4 Pro's
  ~273 GB/s puts generation at roughly 8–9 tok/s at 8-bit and ~15–17 at 4-bit
  (bandwidth arithmetic, not measured).
- **The big unknown:** whether mlx-lm (or, as a fallback, llama.cpp) supports
  this hybrid Gated-DeltaNet architecture (`qwen3_5`). Check that first; it
  decides everything else.
- **Validate it like any other model:** run the fixed 100 held-out patches and
  compare against the CUDA outputs with `train/score.py` and `train/judge.py`.
  Quantization can erode exactly the style this adapter learned.
