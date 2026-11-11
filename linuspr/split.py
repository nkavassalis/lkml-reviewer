"""Held-out split by calendar year.

Nothing should be tuned on the eval years. Splitting by *year* rather than
randomly is deliberate: the same subsystem, the same maintainers and often the
same patch series appear within months of each other, so a random split leaks
near-duplicates between train and eval and flatters every number you read off
it.

Default eval year is 2024 — recent enough to be representative, far enough from
the bulk of the corpus, and a year we have already eyeballed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .pairs import ReviewPair

_YEAR_RE = re.compile(r"\b(19|20)\d\d\b")
DEFAULT_EVAL_YEARS = (2024,)


def pair_year(pair: ReviewPair) -> int | None:
    """Calendar year of a pair, from the date hypermail printed on the message."""
    m = _YEAR_RE.search(pair.date or "")
    return int(m.group(0)) if m else None


@dataclass
class Split:
    train: list[ReviewPair] = field(default_factory=list)
    eval: list[ReviewPair] = field(default_factory=list)
    undated: list[ReviewPair] = field(default_factory=list)

    @property
    def eval_years(self) -> tuple[int, ...]:
        return tuple(sorted({pair_year(p) for p in self.eval if pair_year(p)}))


def split_by_year(
    pairs: list[ReviewPair], *, eval_years: tuple[int, ...] = DEFAULT_EVAL_YEARS
) -> Split:
    """Partition pairs; undated pairs go to `undated`, never silently to train."""
    split = Split()
    for pair in pairs:
        year = pair_year(pair)
        if year is None:
            split.undated.append(pair)
        elif year in eval_years:
            split.eval.append(pair)
        else:
            split.train.append(pair)
    return split
