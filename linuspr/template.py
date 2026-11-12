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


def build_input(subject: str, patch: str, thread: str = "") -> str:
    """The user turn. Order is subject, thread, patch: over-long inputs are cut
    from the tail, so a long patch loses its end, never the discussion."""
    parts = [f"Subject: {subject}"]
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
