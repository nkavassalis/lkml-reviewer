"""Turn raw Linus messages into (context, review) training pairs + the filter.

The pair unit is deliberately conservative: `context` is what Linus quoted
(the patch hunk / the prose he is answering) and `review` is what he wrote.
Both come from the same message, so a pair is only ever as good as Linus'
own quoting habit — which is why `code_evidence` is a hard gate: a reply that
quotes no code-like text is usually an admin reply ("applied", "pulled"), not
a review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parse import Message

_QUOTE_PREFIX_RE = re.compile(r"^>\s?", re.M)
_ATTR_RE = re.compile(r"^On .{10,140} wrote:$", re.M)
_HUNK_RE = re.compile(r"^@@ -\d+(,\d+)? \+\d+(,\d+)? @@", re.M)
_DIFF_HDR_RE = re.compile(r"^(diff --git |--- a/|\+\+\+ b/)", re.M)
_CODE_RE = re.compile(r"[{};]\s*$|\(\s*\)|->|\bif \(|\breturn\b|=\s*\w+\(", re.M)
_IDENT_RE = re.compile(r"\b[a-z][a-z0-9_]{2,}_[a-z0-9_]{2,}\b")  # snake_case symbols
_APPLIED_RE = re.compile(r"^\s*(\[\s*PATCH\b.*)?applied\b", re.I | re.M)
_PULL_ONLY_RE = re.compile(r"^\s*(pull\b|pulled\b|merged?\b)", re.I)

# Things Linus says when he is not reviewing.
_OFF_TOPIC_RE = re.compile(
    r"\b(birth|baby|daughter|son is due|vacation|holiday|get.*back on [A-Z])\b", re.I
)
_RELEASE_RE = re.compile(r"^Linux \d+\.\d+", re.I)

# Bookkeeping replies: "Ack.", "Applied, thanks", "I took this directly", "which
# tree is this in?". They answer the thread, not the patch, so a model trained
# on them learns to emit vague acks. Dropped only when short AND free of any
# technical token, so "Ack, but the lock in foo_bar() is wrong" survives.
_BOOKKEEPING_RE = re.compile(
    r"^\W*(ack(ed)?|applied|pulled|merged|thanks|thank you|ok(ay)?|good|looks good|"
    r"i took|i'?ve (applied|taken|merged|pulled)|will apply|queued|which tree|"
    r"is this in|not in my tree)\b",
    re.I,
)
_TECH_TOKEN_RE = re.compile(
    r"[A-Za-z_]\w*\(\)|\b[a-z]+_[a-z0-9_]+\b|->|#define|\bstruct\b|\block(ing)?\b|"
    r"\brace\b|\bbug\b|\bregression\b|\bbreaks?\b"
)
# Anywhere in a *very* short reply: questions about tree state, not the code.
_TREE_STATE_RE = re.compile(
    r"\b(in my tree|already (applied|merged|upstream)|which tree|my queue)\b", re.I
)
BOOKKEEPING_MAX_CHARS = 400
TREE_STATE_MAX_CHARS = 200
# [^\S\n] = any whitespace except newline: hypermail indents with &nbsp; too.
_SIGNOFF_RE = re.compile(r"\n[^\S\n]*Linus[^\S\n]*\Z")

# The hypermail mirror replaces the *domain* of every local address with x's:
# `Peter Zijlstra <peterz@xxxxxxxxxxxxx>`, bare `zhouzhouyi@xxxxxxxxx`. Six
# consecutive x's inside a domain is not something real text produces, and
# requiring it (rather than "any run of x") keeps `xxxx` in prose and
# `xxxxx@domain.com`-style placeholders from being eaten. Scrubbed at export
# only — the corpus on disk keeps the raw text, so the transform is reversible.
MANGLED_EMAIL_RE = re.compile(
    r"(?:<[ \t]*)?[\w.+-]+@x[\w-]*x(?![\w.-])(?:[ \t]*>)?"  # x-mangled domain, no tld
)

# Mailing-list trailers appended by vger and Google Groups. They are not Linus'
# words, and ~40% of the archive's messages carry one, so leaving them in teaches
# the model to sign every review with "To unsubscribe from this list...".
# Stripped at export only, like the mangled addresses.
LIST_FOOTER_RES = (
    re.compile(
        r"(?m)(?:^--[ \t]*\n)?^To unsubscribe from this list:[^\n]*\n"
        r"[^\n]*\bmajordomo@[^\n]*"
        r"(?:\nMore majordomo info at[^\n]*)?(?:\nPlease read the FAQ at[^\n]*)?"
    ),
    re.compile(
        r"(?m)(?:^--[ \t]*\n)?^You received this message because you are subscribed "
        r"to the Google Groups[^\n]*(?:\nTo (?:unsubscribe|post|view|visit)[^\n]*)*"
    ),
    # Stragglers: any single line of the vger trailer on its own (partial copies,
    # or a trailer cut in half by a length budget).
    re.compile(
        r"(?m)^[ \t>]*(?:To unsubscribe from this list:|the body of a message to majordomo@|"
        r"More majordomo info at|Please read the FAQ at)[^\n]*\n?"
    ),
)


@dataclass
class ReviewPair:
    mid: str
    month: str
    subject: str
    date: str
    context: str
    review: str
    authors: list[str] = field(default_factory=list)
    has_diff: bool = False
    code_hits: int = 0
    commit: str = ""
    resolved_context: str = ""
    thread_context: str = ""  # earlier messages in the thread (v3), see context.py

    def as_sft(self, *, keep_signoff: bool = False) -> dict[str, str]:
        """Shape used for chat/SFT export: patch context in, review out.

        Mangled mirror emails are scrubbed here and only here — the corpus and
        review_pairs.jsonl keep the raw text so the transform is reversible.

        Input order is subject, thread, patch: the trainer truncates over-long
        inputs from the tail, so a long patch loses its end, never the thread.
        """
        from .template import build_input  # noqa: PLC0415  (one source of truth)

        context = self.resolved_context or self.context
        user = build_input(self.subject, context, self.thread_context)
        review = strip_attributions(strip_list_footers(self.review))
        if not keep_signoff:
            review = strip_signoff(review)
        return {
            "input": scrub_mangled(strip_list_footers(user)),
            "output": scrub_mangled(review),
        }


LORE_LINK_RE = re.compile(
    r"https?://lore\.kernel\.org/[^\s]*@x[\w-]*x(?![\w-])/?"  # link whose msgid died
)


# `On <date>, <name> wrote:` — possibly wrapped onto a second line — is the
# header of a quote that `pair_from` has already moved into the input. Left in
# the output, it taught run-1 to open 94% of reviews with an invented date and
# the wrong person (44 of 55 named someone other than the real author).
ATTRIBUTION_RE = re.compile(r"(?m)^On [^\n]{6,160}?(?:\n[^\n]{0,80}?)?\bwrote:[ \t>]*$")


def strip_attributions(text: str) -> str:
    """Remove attribution lines, plus one of the blank lines around each cut."""
    out: list[str] = []
    pos = 0
    for m in ATTRIBUTION_RE.finditer(text):
        start, end = m.start(), m.end()
        if end < len(text) and text[end] == "\n":
            end += 1  # the line's own newline
        if text.startswith("\n", end) and (start == 0 or text[start - 2 : start] == "\n\n"):
            end += 1  # and the blank line it was separated from the review by
        out.append(text[pos:start])
        pos = end
    out.append(text[pos:])
    return "".join(out)


def strip_signoff(text: str) -> str:
    """Drop a trailing bare "Linus" signature. It is his style, but a model that
    signs every review with a real person's name is an impersonation tool."""
    text = text.rstrip()
    while True:  # some messages carry the signature twice
        stripped = _SIGNOFF_RE.sub("", text).rstrip()
        if stripped == text:
            return text
        text = stripped


def strip_list_footers(text: str) -> str:
    """Remove vger/Google Groups trailers; everything else is left byte-identical."""
    for pattern in LIST_FOOTER_RES:
        text = pattern.sub("", text)
    return text.rstrip()


def _cut(pattern: re.Pattern[str], text: str) -> str:
    """Delete every match plus the horizontal whitespace around it, keeping at
    most one space where the cut joined two words.

    Only whitespace *adjacent to a cut* is touched. An earlier version collapsed
    every run of spaces/tabs in the whole text, which silently flattened the
    indentation of every diff and code block in the export.
    """
    out: list[str] = []
    pos = 0
    for m in pattern.finditer(text):
        start, end = m.start(), m.end()
        while start > pos and text[start - 1] in " \t":
            start -= 1
        while end < len(text) and text[end] in " \t":
            end += 1
        line_start = start == 0 or text[start - 1] == "\n"
        at_break = end == len(text) or text[end] in "\n,.;:)"
        joins_words = start < m.start() and end > m.end()
        out.append(text[pos:start])
        if joins_words and not line_start and not at_break:
            out.append(" ")
        pos = end
    out.append(text[pos:])
    return "".join(out)


def scrub_mangled(text: str) -> str:
    """Drop hypermail's `xxxxxxxx`-mangled addresses without eating the name."""
    out = _cut(LORE_LINK_RE, text)  # dead lore links first: they contain one
    out = _cut(MANGLED_EMAIL_RE, out)
    return out.lstrip("\n").rstrip()  # leading indentation is content, keep it


def patch_key(msg) -> str:
    """Key into patch_texts.json. `mid` alone is a per-bucket counter (`02009`)
    that repeats in every one of the 1,472 buckets; keying on it let a 2010
    block-layer patch overwrite, and then be attached to, a 2024 eventfs review."""
    return f"{msg.month}/{msg.mid}"


def unquote(quoted: str) -> str:
    """Strip the leading `>` from hypermail's quoted lines."""
    return _QUOTE_PREFIX_RE.sub("", quoted).strip()


def attributed_authors(review: str) -> list[str]:
    """Names from `On <date>, <name> wrote:` lines Linus left in his reply."""
    out: list[str] = []
    for m in _ATTR_RE.finditer(review):
        line = m.group(0)
        name = re.sub(r"^On .*,\s*", "", line).removesuffix(":").removesuffix(" wrote")
        name = name.split("<")[0].strip()  # the mirror xxxx's emails out anyway
        if name and name not in out:
            out.append(name)
    return out


def code_evidence(context: str) -> int:
    """Rough count of 'this is real code being discussed' signals."""
    n = len(_HUNK_RE.findall(context)) + len(_DIFF_HDR_RE.findall(context))
    n += len(_CODE_RE.findall(context))
    n += len(_IDENT_RE.findall(context)) // 3
    return n


def is_review(msg: Message) -> bool:
    """Gate: is this message plausibly a code review rather than admin noise?"""
    if not msg.is_reply or not msg.is_patch_thread:
        return False
    if _RELEASE_RE.match(msg.subject):
        return False
    review = msg.own.strip()
    if not review or _OFF_TOPIC_RE.search(review):
        return False
    noise = _APPLIED_RE.match(review) or _PULL_ONLY_RE.match(review)
    return not noise  # "applied" / "pulled" is bookkeeping, not review


def is_bookkeeping(review: str) -> bool:
    """Short ack/applied/which-tree reply with no technical content."""
    text = ATTRIBUTION_RE.sub("", review).strip()
    if _TECH_TOKEN_RE.search(text):
        return False
    if len(text) < TREE_STATE_MAX_CHARS and _TREE_STATE_RE.search(text):
        return True
    return len(text) < BOOKKEEPING_MAX_CHARS and bool(_BOOKKEEPING_RE.match(text))


def pair_from(msg: Message, *, has_patch: bool = False, legacy: bool = False) -> ReviewPair | None:
    """Build a pair, or None when the message fails the quality gates.

    v3 gate: a reply qualifies if Linus quoted code (code_evidence >= 3, the v2
    rule) OR we recovered the real patch it answers. The v2 gate looked only at
    the quote, so it threw away 4,975 replies whose patch we already had.
    `legacy=True` reproduces the v2 gate exactly (for rebuilding v2's data).
    """
    if not is_review(msg):
        return None
    context = unquote(msg.quoted)
    review = msg.own.strip()
    hits = code_evidence(context)
    quote_ok = hits >= 3 and len(context) >= 80
    if legacy:
        if not quote_ok or len(review) < 120:
            return None
    else:
        if not (quote_ok or has_patch) or len(review) < 120 or is_bookkeeping(review):
            return None
    return ReviewPair(
        mid=msg.mid,
        month=msg.month,
        subject=msg.subject,
        date=msg.date,
        context=context,
        review=review,
        authors=attributed_authors(review),
        has_diff=bool(_HUNK_RE.search(context) or _DIFF_HDR_RE.search(context)),
        code_hits=hits,
    )


def build_pairs(
    msgs: list[Message], *, patches: dict[str, str] | None = None, legacy: bool = False
) -> list[ReviewPair]:
    """Gate a corpus into pairs, attaching recovered patch text when we have it."""
    patches = patches or {}
    out: list[ReviewPair] = []
    seen: set[tuple[str, str]] = set()
    for m in msgs:
        p = pair_from(m, has_patch=patch_key(m) in patches, legacy=legacy)
        if p is not None and patch_key(m) in patches:
            p.resolved_context = patches[patch_key(m)]
        if p is None:
            continue
        key = (p.subject.lower().strip(), p.review[:200])
        if key in seen:  # hypermail occasionally mirrors a month into two buckets
            continue
        seen.add(key)
        out.append(p)
    return out
