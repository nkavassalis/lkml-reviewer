"""The prompt format, in one place.

Training (`train/sft_lora.py`), export (`ReviewPair.as_sft`) and inference
(`reviewer/`) must all build exactly the same prompt. If they drift, the model
quietly gets worse and no metric names the cause. `train/sft_lora.py` can't
import this module (it runs alone in a container), so it keeps its own copy of
SYSTEM and TRUNC_MARK, and `tests/test_reviewer.py` fails if the copies differ.

Stdlib only. The tokenizer is passed in; nothing here imports transformers.
"""
from __future__ import annotations

SYSTEM = (
    "You are reviewing a patch posted to the Linux kernel mailing list. "
    "Reply as the upstream maintainer would: blunt, specific, and about the code."
)
TRUNC_MARK = "\n[... truncated ...]"
THREAD_HEADER = "--- Earlier in this thread (most recent last) ---"
PATCH_HEADER = "--- The patch ---"


# Layouts. "thread-first" is what v3 was trained on (and what the published v3
# model expects). "patch-first" is v4's: v3 answered the discussion instead of
# the code (it named something from the patch 8% of the time when a thread was
# present, 31% without), and with the thread first, the trainer's tail
# truncation cut the *patch*. Patch-first gives each section its own budget.
LAYOUTS = ("thread-first", "patch-first")
PATCH_BUDGET_CHARS = 7900    # ~2,400 tokens
THREAD_BUDGET_CHARS = 2000   # ~600 tokens


def fit_patch(patch: str, budget: int = PATCH_BUDGET_CHARS) -> str:
    """Shrink a diff to `budget` chars, keeping the head of *every* file
    (headers and first hunks) rather than cutting whole files off the end."""
    if len(patch) <= budget:
        return patch
    import re

    files = re.split(r"(?m)^(?=diff --git )", patch)
    files = [f for f in files if f]
    if len(files) <= 1:
        return patch[:budget].rstrip() + TRUNC_MARK
    share = max(budget // len(files), 200)
    out = []
    for f in files:
        out.append(f if len(f) <= share else f[:share].rstrip() + TRUNC_MARK + "\n")
    joined = "".join(out)
    return joined if len(joined) <= budget else joined[:budget].rstrip() + TRUNC_MARK


def fit_thread(thread: str, budget: int = THREAD_BUDGET_CHARS) -> str:
    """Keep the most recent messages (they come last) within `budget` chars."""
    if len(thread) <= budget:
        return thread
    blocks = thread.split("\n\n[")
    kept: list[str] = []
    for i, b in enumerate(reversed(blocks)):
        piece = b if i == len(blocks) - 1 else "[" + b
        if sum(len(k) for k in kept) + len(piece) > budget:
            break
        kept.insert(0, piece)
    return "\n\n".join(kept) if kept else thread[-budget:]


def build_input(subject: str, patch: str, thread: str = "", *, layout: str = "thread-first") -> str:
    """The user turn.

    thread-first (v3): subject, thread, patch; over-long inputs are cut from
    the tail, so a long patch loses its end.
    patch-first (v4): subject, patch, thread, each fitted to its own budget,
    so neither crowds the other out.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    parts = [f"Subject: {subject}"]
    if layout == "patch-first":
        if thread:
            parts.append(PATCH_HEADER)
        parts.append(fit_patch(patch))
        if thread:
            parts.append(f"{THREAD_HEADER}\n{fit_thread(thread)}")
        return "\n\n".join(parts)
    if thread:
        parts.append(f"{THREAD_HEADER}\n{thread}")
        parts.append(PATCH_HEADER)
    parts.append(patch)
    return "\n\n".join(parts)


def chat_messages(user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def prompt_text(tok, user: str) -> str:
    """Rendered prompt, thinking off: the model was trained with an empty
    <think></think> block, so inference must use the same."""
    return tok.apply_chat_template(
        chat_messages(user), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
