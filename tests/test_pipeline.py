"""Corpus-building tests: month listing, filtering, JSONL round-trip."""
from __future__ import annotations

import json

import pytest

from linuspr.parse import IndexEntry, Message, parse_index
from linuspr.pipeline import (
    MonthCorpus,
    crawl_month,
    linus_entries,
    list_months,
    read_jsonl,
    write_jsonl,
)

ROOT_INDEX = """
<a href="2406.1/index.html">2406.1</a>
<a href="1901.0/index.html">1901.0</a>
<a href="2406.0/index.html">2406.0</a>
<a href="9603.0/index.html">9603.0</a>
"""


class FakeFetcher:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.requested: list[tuple[str, ...]] = []

    def get(self, *parts):
        self.requested.append(parts)
        return self.pages["/".join(p.strip("/") for p in parts)]


def test_list_months_sorts_199x_before_20xx():
    assert list_months(FakeFetcher({"": ROOT_INDEX})) == [
        "9603.0",  # 1996 — must sort before anything 20xx
        "1901.0",  # 2019
        "2406.0",  # 2024
        "2406.1",
    ]


def _entries():
    return [
        IndexEntry("00001", "[PATCH 1/1] thing", "Alice", "d"),
        IndexEntry("00002", "Re: [PATCH 1/1] thing", "Linus Torvalds", "d"),
        IndexEntry("00003", "Re: [GIT PULL] fixes", "Linus Torvalds", "d"),
        IndexEntry("00004", "Linux 6.10-rc2", "Linus Torvalds", "d"),
        IndexEntry("00005", "Re: [PATCH 1/1] thing", "Bob ", "d"),
    ]


def test_linus_entries_matches_exact_author():
    got = [e.mid for e in linus_entries(_entries())]
    assert got == ["00002", "00003", "00004"]


def test_linus_entries_patch_only_keeps_only_patch_replies():
    got = [e.mid for e in linus_entries(_entries(), patch_only=True)]
    assert got == ["00002"]


def test_crawl_month_fetches_index_then_only_linus_pages(index_html, msg_html):
    f = FakeFetcher({"2406.0/index.html": index_html})
    # every non-Linus page 404s (KeyError) — so a fetch of one would blow up
    with pytest.raises(KeyError):
        crawl_month(f, "2406.0")
    requested = [p[1] for p in f.requested[1:]]
    assert len(requested) == 1  # stopped at the first Linus page
    assert requested == ["01784.html"]


def test_crawl_month_patch_only_narrows_the_fetches(index_html, msg_html):
    f = FakeFetcher({"2406.0/index.html": index_html})
    with pytest.raises(KeyError):
        crawl_month(f, "2406.0", patch_only=True)
    assert [p[1] for p in f.requested[1:]] == ["01784.html"]


def test_crawl_month_returns_parsed_messages(index_html, msg_html):
    pages = {"2406.0/index.html": index_html}
    mids = [e.mid for e in linus_entries(parse_index(index_html), patch_only=True)]
    assert len(mids) == 5  # the Re: [PATCH] rows hypermail lists for 2406.0
    for mid in mids:
        pages[f"2406.0/{mid}.html"] = msg_html
    corpus = crawl_month(FakeFetcher(pages), "2406.0", patch_only=True)
    assert corpus.total_messages == 330
    assert [m.mid for m in corpus.linus_messages] == mids
    assert corpus.linus_messages[0].author == "Paul E. McKenney"  # fixture body


def test_write_and_read_jsonl_roundtrip(tmp_path):
    msgs = [
        Message(
            mid="00002",
            month="2406.0",
            subject="Re: [PATCH] x",
            author="Linus Torvalds",
            date="d",
            body="b",
            msgid="m",
            prev_mid="00001",
            prev_author="Alice",
            quoted="> hi",
            own="no.",
            lore_links=[("all", "abc")],
        )
    ]
    path = tmp_path / "corpus.jsonl"
    assert write_jsonl([MonthCorpus("2406.0", 10, msgs)], path) == 1
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["mid"] == "00002"
    back = read_jsonl(path)
    assert back[0] == msgs[0]


def test_write_jsonl_merges_instead_of_clobbering(tmp_path):
    """A second crawl of one month must not delete months crawled earlier."""
    one = Message(mid="1", month="2406.0", subject="s", author="Linus Torvalds",
                  date="d", body="b", own="o", quoted="q")
    two = Message(mid="2", month="1906.0", subject="s", author="Linus Torvalds",
                  date="d", body="b", own="o", quoted="q")
    path = tmp_path / "c.jsonl"
    write_jsonl([MonthCorpus("2406.0", 1, [one])], path)
    write_jsonl([MonthCorpus("1906.0", 1, [two])], path)
    assert {m.mid for m in read_jsonl(path)} == {"1", "2"}
    # re-crawling 2406.0 with new content replaces only that month
    one_v2 = Message(mid="1b", month="2406.0", subject="s", author="Linus Torvalds",
                     date="d", body="b", own="o", quoted="q")
    assert write_jsonl([MonthCorpus("2406.0", 1, [one_v2])], path) == 2
    assert {m.mid for m in read_jsonl(path)} == {"1b", "2"}


def test_bucket_year_handles_pre_2000_buckets():
    from linuspr.pipeline import bucket_year

    assert bucket_year("9603.0") == 1996  # the mirror's oldest bucket
    assert bucket_year("1901.0") == 2019  # the trap: NOT January 1919
    assert bucket_year("0001.0") == 2000
    assert bucket_year("2406.1") == 2024
    assert bucket_year("2611.1") == 2026


def test_buckets_of_year_matches_the_live_id_scheme():
    from linuspr.pipeline import buckets_of_year

    fake = ["9603.0", "9603.1", "1901.0", "1909.3", "2406.0", "2406.3"]
    assert buckets_of_year(fake, 2019) == ["1901.0", "1909.3"]
    assert buckets_of_year(fake, 1996) == ["9603.0", "9603.1"]
    assert buckets_of_year(fake, 2024) == ["2406.0", "2406.3"]


def test_read_jsonl_ignores_blank_lines_and_unknown_keys(tmp_path):
    path = tmp_path / "c.jsonl"
    row = {"mid": "1", "month": "m", "subject": "s", "author": "a", "date": "d",
           "body": "b", "bogus": "x"}
    path.write_text(json.dumps(row) + "\n\n", encoding="utf-8")
    assert read_jsonl(path)[0].mid == "1"
