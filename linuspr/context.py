"""Thread context: what was said in the thread before Linus replied.

Many of his replies answer the *discussion*, not just the patch ("No, Al is
right, ...", "that's what I said in v2"). Without the discussion those targets
can't be predicted from the input, and a model learns to emit vague replies.
v3 puts a bounded excerpt of the earlier messages in the input.

Selection is pure (`select_context`), and fetching goes through `Fetcher`
(polite, cached). Only messages *before* Linus' reply in the same subject group
are used, so nothing from the future, including his own later messages, leaks
into the input.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .parse import parse_index
from .resolve import normalise_subject

THREAD_BUDGET_CHARS = 3300   # ~1,000 tokens
PER_MESSAGE_CHARS = 1200
MAX_MESSAGES = 6
# A message carrying a diff is a patch posting; the patch is already in the input.
# (resolve.patch_quality also scores plain length, so long prose would look like
# a patch to it.)
_DIFF_RE = re.compile(r"^(diff --git |@@ -\d+|--- a/)", re.M)


@dataclass
class ThreadMessage:
    author: str
    text: str


def earlier_in_thread(entries, subject: str, before_mid: str) -> list:
    """Index entries sharing `subject` that precede `before_mid`, oldest first.

    hypermail mids are document order within a bucket, so a smaller mid is an
    earlier message.
    """
    key = normalise_subject(subject)
    return [e for e in entries if normalise_subject(e.subject) == key and e.mid < before_mid]


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit].rstrip() + " [...]"


def select_context(
    messages: list[ThreadMessage],
    *,
    budget: int = THREAD_BUDGET_CHARS,
    per_message: int = PER_MESSAGE_CHARS,
    max_messages: int = MAX_MESSAGES,
) -> str:
    """Most recent messages last, newest kept first when the budget runs out.

    Patch postings are skipped (the patch is already in the input), as are
    empty messages.
    """
    chosen: list[str] = []
    used = 0
    for m in reversed(messages):
        if len(chosen) >= max_messages:
            break
        if not m.text.strip() or _DIFF_RE.search(m.text):
            continue
        block = f"[{m.author.strip() or 'unknown'}]:\n{_clip(m.text, per_message)}"
        if used + len(block) > budget:
            break
        chosen.append(block)
        used += len(block) + 2
    return "\n\n".join(reversed(chosen))


class ContextBuilder:
    """Builds thread context per pair; caches month indexes in memory."""

    def __init__(self, fetcher) -> None:
        self.fetcher = fetcher
        self._index: dict[str, list] = {}
        self.fetched = 0

    def entries(self, month: str) -> list:
        if month not in self._index:
            self._index[month] = parse_index(self.fetcher.get(month, "index.html"))
        return self._index[month]

    def uncached(self, month: str, subject: str, mid: str) -> int:
        """How many pages this pair would need from the network (dry run)."""
        cache = self.fetcher.cache_dir
        prior = earlier_in_thread(self.entries(month), subject, mid)
        if cache is None:
            return len(prior)
        return sum(not (cache / month / e.filename).exists() for e in prior)

    def build(self, month: str, subject: str, mid: str) -> str:
        from .pipeline import fetch_message_or_none  # noqa: PLC0415

        prior = earlier_in_thread(self.entries(month), subject, mid)
        msgs: list[ThreadMessage] = []
        for e in prior[-(MAX_MESSAGES * 2):]:  # a few spare for skipped postings
            m = fetch_message_or_none(self.fetcher, month, e.mid)
            self.fetched += 1
            if m is not None:
                msgs.append(ThreadMessage(author=e.author, text=m.own))
        return select_context(msgs)
