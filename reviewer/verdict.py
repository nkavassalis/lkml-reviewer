"""Turn a free-text review into a GitHub verdict: approve, request-changes, or
comment. Stdlib only.

Linus-style reviews aren't structured, so this reads the signals he actually
uses ("NAK", "I'm not taking this", "Ack", "Looks good"). It is deliberately
conservative: mixed or absent signals give `comment`, never a guessed approval.
`--verdict` on the CLI overrides it, and `--live` always shows the verdict and
asks before submitting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

APPROVE, REQUEST_CHANGES, COMMENT = "approve", "request-changes", "comment"

_REJECT = re.compile(
    r"\bNAK\b|\bnack(ed)?\b|\bI'?m not (taking|applying|pulling|merging)\b|"
    r"\b(won'?t|will not|can'?t|cannot) (take|apply|merge|pull)\b|\bnot (ok|acceptable)\b|"
    r"\bthis is (wrong|broken|buggy)\b|\bthis (breaks|is a regression)\b|\bplease don'?t\b|"
    r"\bstop doing this\b|\bno, (please|we|this|that|don'?t)\b|^\W*no\.",
    re.I | re.M,
)
_APPROVE = re.compile(
    r"\back(ed)?(-by)?\b|\breviewed-by\b|\blooks? (good|sane|fine|ok|right)\b|"
    r"\blgtm\b|\bapplied\b|\bI'?ll (take|apply|pull) (it|this)\b|\bfine by me\b|"
    r"\bobviously ?correct\b|\bI like (it|this)\b",
    re.I,
)


@dataclass
class Verdict:
    event: str
    reason: str


def infer(review: str) -> Verdict:
    rejects = [m.group(0) for m in _REJECT.finditer(review)]
    approves = [m.group(0) for m in _APPROVE.finditer(review)]
    if rejects and approves:
        return Verdict(COMMENT, f"mixed signals: {rejects[0]!r} vs {approves[0]!r}")
    if rejects:
        return Verdict(REQUEST_CHANGES, f"rejects: {rejects[0]!r}")
    if approves:
        return Verdict(APPROVE, f"approves: {approves[0]!r}")
    return Verdict(COMMENT, "no clear approve/reject signal")
