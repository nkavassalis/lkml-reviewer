"""Eval-split tests: year-based, no silent leakage into train."""
from __future__ import annotations

from linuspr.pairs import ReviewPair
from linuspr.split import DEFAULT_EVAL_YEARS, pair_year, split_by_year


def pair(date: str) -> ReviewPair:
    return ReviewPair(
        mid="1", month="1901.0", subject="Re: [PATCH] x", date=date,
        context="c", review="r",
    )


DATED = "Mon,  7 Jan 2019 10:00:00 -0500 (EST)"
DATED_2024 = "Sat,  1 Jun 2024 00:36:02 -0400 (EDT)"


def test_pair_year_reads_hypermail_dates():
    assert pair_year(pair(DATED)) == 2019
    assert pair_year(pair(DATED_2024)) == 2024
    assert pair_year(pair("")) is None
    assert pair_year(pair("no date at all")) is None


def test_default_eval_year_is_2024():
    assert DEFAULT_EVAL_YEARS == (2024,)


def test_split_partitions_by_year():
    split = split_by_year([pair(DATED), pair(DATED_2024)])
    assert [p.date for p in split.train] == [DATED]
    assert [p.date for p in split.eval] == [DATED_2024]
    assert split.undated == []
    assert split.eval_years == (2024,)


def test_undated_pairs_are_quarantined_not_trained_on():
    split = split_by_year([pair("garbage")])
    assert len(split.undated) == 1
    assert split.train == [] and split.eval == []


def test_multiple_eval_years():
    split = split_by_year(
        [pair(DATED), pair(DATED_2024)], eval_years=(2019, 2024)
    )
    assert len(split.eval) == 2 and split.train == []


def test_no_pair_appears_in_both_sides():
    pairs = [pair(DATED), pair(DATED_2024), pair("x")]
    split = split_by_year(pairs)
    total = len(split.train) + len(split.eval) + len(split.undated)
    assert total == len(pairs)
