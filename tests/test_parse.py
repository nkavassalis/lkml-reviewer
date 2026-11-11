"""Parser tests against real saved hypermail pages."""
from __future__ import annotations

from linuspr.parse import parse_index, parse_message


def test_parse_index_counts_every_message(index_html):
    entries = parse_index(index_html)
    assert len(entries) == 330  # trimmed fixture: 25 rows + every special one
    assert entries[0].mid == "03261"
    assert all(e.mid.isdigit() or "." in e.mid for e in entries)


def test_parse_index_finds_linus_rows(index_html):
    linus = [e for e in parse_index(index_html) if e.author == "Linus Torvalds"]
    assert len(linus) == 14
    assert any(e.subject.startswith("Re: [PATCH") for e in linus)
    assert all(e.date for e in linus)


def test_parse_index_subjects_are_unescaped(index_html):
    # the raw fixture holds `&amp;idpf` and `arch&#39;s`; parsed subjects do not
    assert "&amp;" in index_html and "&#x" in index_html
    entries = parse_index(index_html)
    assert not any("&amp;" in e.subject or "&#x" in e.subject for e in entries)
    assert any("arch's" in e.subject for e in entries)
    assert any("&idpf" in e.subject for e in entries)


def test_parse_message_head_fields(msg_html):
    m = parse_message(msg_html, mid="00001", month="2406.0")
    assert m.author == "Paul E. McKenney"
    assert m.subject.startswith("Re: [PATCH V2] rcutorture")
    assert m.msgid == "cd3709dc-2c08-4ac2-a19b-d8edb66195e3@paulmck-laptop"
    assert m.date.startswith("Sat,  1 Jun 2024")
    assert m.prev_mid == "00000"
    assert m.prev_author == "Michael Kelley"
    assert m.is_reply and m.is_patch_thread


def test_parse_message_splits_quoted_from_own(msg_html):
    m = parse_message(msg_html, mid="00001", month="2406.0")
    assert m.quoted.startswith(">")
    assert "Why not just use" in m.own  # McKenney's own commentary
    assert "Why not just use" not in m.quoted
    assert len(m.quoted) > 1000


def test_parse_message_strips_nav_and_footer(msg_html):
    m = parse_message(msg_html, mid="00001", month="2406.0")
    for junk in ("Messages sorted by", "Linux-Kernel Archive", "Next message"):
        assert junk not in m.body
        assert junk not in m.own


def test_parse_message_extracts_lore_links(msg_html):
    m = parse_message(msg_html, mid="00001", month="2406.0")
    assert ("all", "059d36ce-6453-42be-a31e-895abd35d590@paulmck-laptop") in m.lore_links


def test_parse_linus_nak_message():
    from pathlib import Path

    html = (
        Path(__file__).parent / "fixtures" / "msg_2406_1_01812_linus.html"
    ).read_text(encoding="utf-8", errors="replace")
    m = parse_message(html, mid="01812", month="2406.1")
    assert m.author == "Linus Torvalds"
    assert "runtime constant" in m.subject
    # the substance lives in `own`, the patch context in `quoted`
    assert "We're not adding a stupid config variable" in m.own
    assert "@@ -1492,6 +1492,9 @@" in m.quoted
    assert "Peter Zijlstra" in m.own  # attribution line survives
