"""Keep Linus' judgement, drop his abuse.

The project's stated spirit is that we are training on his *reviewing*, not his
swearing — and the corpus is full of both. This module is deliberately blunt
about its own limits: it is a vocabulary heuristic, not a classifier, and it is
tuned to *drop* rather than to *rewrite*. Rewriting his prose would put our words
in the training set; dropping a pair keeps only his words or none at all.

Three outcomes per pair:

  keep    no abuse detected
  scrub   abuse confined to whole sentences, which are cut; the technical body
          of the review survives verbatim
  drop    the review is nothing but abuse, or it attacks a *person* rather than
          the code

`drop` is preferred whenever we are unsure: a lost pair costs us nothing, a
pair that teaches "this is braindead code" as a *style* costs the whole project.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .pairs import ReviewPair

# Attacks on the author, not on the patch. These drop the pair outright.
PERSON_ATTACKS = (
    "braindead",
    "brain damage",
    "braindamaged",
    "idiot",
    "idiocy",
    "moron",
    "stupid",
    "dumb",
    "clueless",
    "clue",
    "incompetent",
    "incompetence",
    "you shouldn't be allowed",
    "what are you smoking",
    "do you even",
    "are you crazy",
    "are you insane",
    "have you any clue",
    "no business",
    "should be shot",
    "fire them",
)

# Attacks on the code/patch. Scrubbable when the review also has substance.
CODE_ATTACKS = (
    "garbage",
    "crap",
    "crapware",
    "ugly",
    "disgusting",
    "obscene",
    "insane",
    "crazy",
    "ridiculous",
    "absurd",
    "nonsense",
    "bullshit",
    "shit",
    "damn",
    "hell",
    "wtf",
    "no sane person",  # bare "sane" is usually approval: "looks sane to me"
    "trash",
    "junk",
    "half-arsed",
    "half-assed",
    "lazy",
)

_CLAUSE_SPLIT_RE = re.compile(r"[,;]|\bbut\b|\band then\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?:\s+|$)|\n\s*\n")
_TARGET_IS_CODE_RE = re.compile(
    r"\b(this|that|these|those|such|the|a|an|your|it)\b[^.!?]{0,40}\b"
    r"(patch|code|interface|api|function|struct|approach|idea|series|change|"
    r"commit|hack|solution|implementation|logic|scheme|name|naming|layer|helper)\b",
    re.I,
)
_TECHNICAL_RE = re.compile(
    r"\b(alloc|lock|race|abi|uapi|regression|revert|layering|interface|overhead|"
    r"performance|cache|atomic|leak|use-after-free|null pointer|panic|oops|"
    r"deadlock|lifetime|refcount|userspace|kernel|syscall|driver|subsystem|"
    r"maintainer|tree|patchset|we (should|need|have to|can|must)|you (should|"
    r"need to|have to|can|must)|because|since|instead|rather than|breaks|works)\b",
    re.I,
)
MIN_KEPT_TECHNICAL_CHARS = 150
_BLANK_RUN_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")


def _segments(text: str) -> list[tuple[str, str]]:
    """Split into (sentence, separator) spans whose concatenation is `text`."""
    out: list[tuple[str, str]] = []
    pos = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        if m.end() <= pos:
            continue
        out.append((text[pos : m.start()], m.group(0)))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], ""))
    return out


@dataclass
class Verdict:
    action: str  # keep | scrub | drop
    review: str
    hits: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.action != "drop"


def _compiled(words: tuple[str, ...]) -> re.Pattern[str]:
    """Whole-word match. Substring matching produced real false positives:
    'hell' hit *Christoph Hellwig* and 'sane' hit *filldir64()/unsafe*."""
    return re.compile(
        r"\b(?:" + "|".join(re.escape(w) for w in sorted(words)) + r")\b", re.I
    )


_PERSON_RE = _compiled(PERSON_ATTACKS)
_CODE_RE = _compiled(CODE_ATTACKS)


def _hits(text: str, words: tuple[str, ...]) -> tuple[str, ...]:
    pattern = _PERSON_RE if set(words) == set(PERSON_ATTACKS) else _CODE_RE
    return tuple(dict.fromkeys(m.group(0).lower() for m in pattern.finditer(text)))


def targets_code(sentence: str) -> bool:
    """Does the abuse in this sentence point at the code rather than a person?"""
    return bool(_TARGET_IS_CODE_RE.search(sentence))


def has_substance(review: str) -> bool:
    """Enough technical content to be worth keeping once the abuse is cut."""
    return bool(_TECHNICAL_RE.search(review)) and len(review) >= MIN_KEPT_TECHNICAL_CHARS


def classify(review: str) -> Verdict:
    """Decide what to do with one review."""
    person = _hits(review, PERSON_ATTACKS)
    code = _hits(review, CODE_ATTACKS)
    hits = tuple(dict.fromkeys(list(person) + list(code)))
    if not hits:
        return Verdict("keep", review)

    if person:
        # Person-directed abuse is never scrubbed — it is the thing we refuse to
        # teach, and sentence surgery on an insult is where rewrites creep in.
        return Verdict("drop", "", hits)

    # Work on spans of the original text and emit them verbatim, separators
    # included. An earlier version stripped and re-joined sentences, which threw
    # away the indentation of every code block in a scrubbed review.
    kept: list[str] = []
    for sentence, sep in _segments(review):
        if not sentence.strip() or not _hits(sentence, CODE_ATTACKS):
            kept.append(sentence + sep)
            continue
        # The sentence mixes abuse with content. Try cutting the abusive
        # *clause* (comma/semicolon separated) before throwing the sentence —
        # "This code is garbage, but the allocation must use GFP_KERNEL here"
        # is worth keeping; "This code is garbage" alone is not.
        clauses = [
            c.strip()
            for c in _CLAUSE_SPLIT_RE.split(sentence)
            if c.strip() and not _hits(c, CODE_ATTACKS)
        ]
        if not clauses and targets_code(sentence):
            kept.append(sentence + sep)  # abuse aimed at the code, and it is the point
        elif clauses:
            kept.append(", ".join(clauses) + sep)
        elif "\n" in sep:
            kept.append("\n")  # sentence cut; keep the line break it ended with
    scrubbed = _BLANK_RUN_RE.sub("\n\n", "".join(kept)).lstrip("\n").rstrip()
    if has_substance(scrubbed):
        return Verdict("scrub", scrubbed, hits)
    return Verdict("drop", "", hits)


def filter_pairs(pairs: list[ReviewPair]) -> tuple[list[ReviewPair], dict[str, int]]:
    """Apply `classify` to a corpus; returns (kept pairs, tally)."""
    kept: list[ReviewPair] = []
    tally = {"keep": 0, "scrub": 0, "drop": 0}
    for pair in pairs:
        verdict = classify(pair.review)
        tally[verdict.action] += 1
        if verdict.ok:
            pair.review = verdict.review
            kept.append(pair)
    return kept, tally
