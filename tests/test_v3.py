"""v3 data changes: gate on the recovered patch, bookkeeping filter, sign-off
stripping, thread context. Each rule is traceable to a v2 finding."""
from __future__ import annotations

from linuspr.context import ThreadMessage, earlier_in_thread, select_context
from linuspr.pairs import (
    build_pairs,
    is_bookkeeping,
    pair_from,
    patch_key,
    strip_signoff,
)
from linuspr.parse import IndexEntry, Message

LONG_REVIEW = (
    "No, please let's not do this. We have O_NOFOLLOW, and we can't get rid of it. "
    "So adding RESOLVE_NO_TRAILING_SYMLINKS isn't a cleanup, it's extra complexity "
    "for absolutely zero gain."
)


def prose_reply(own: str = LONG_REVIEW, quoted: str = "> I think we should add a flag.") -> Message:
    return Message(mid="00042", month="1906.0", subject="Re: [PATCH 01/14] VFS: add flags",
                   author="Linus Torvalds", date="d", body=own, quoted=quoted, own=own)


# --- gate -------------------------------------------------------------------

def test_thin_quote_with_recovered_patch_now_qualifies():
    m = prose_reply()
    assert pair_from(m) is None                       # no code in quote, no patch
    assert pair_from(m, has_patch=True) is not None   # v3: we have the real patch


def test_legacy_gate_reproduces_v2():
    assert pair_from(prose_reply(), has_patch=True, legacy=True) is None


def test_build_pairs_uses_patch_presence():
    m = prose_reply()
    pairs = build_pairs([m], patches={patch_key(m): "diff --git a/x b/x\n"})
    assert len(pairs) == 1 and pairs[0].resolved_context.startswith("diff --git")


# --- bookkeeping ------------------------------------------------------------

def test_pure_acks_are_bookkeeping():
    assert is_bookkeeping("Ack. Changed here locally.\n\n\t\tLinus")
    assert is_bookkeeping("On Tue, 4 Feb 2025 at 05:22, A B wrote:\n\nApplied, thanks.")
    assert is_bookkeeping("I'm not seeing this in my tree? Is it already applied?")


def test_ack_with_technical_content_is_kept():
    assert not is_bookkeeping("Ack, but the lock in foo_bar() is taken in the wrong order.")
    assert not is_bookkeeping(LONG_REVIEW)


def test_long_reply_starting_with_ack_is_kept():
    assert not is_bookkeeping("Ack. " + "This changes the whole approach to the problem. " * 12)


def test_bookkeeping_pairs_are_gated_out():
    m = prose_reply(own="Applied, thanks. " + "I'll push it out later today with the rest. " * 3)
    assert pair_from(m, has_patch=True) is None


# --- sign-off ---------------------------------------------------------------

def test_strip_signoff_removes_trailing_name_only():
    assert strip_signoff("This is wrong.\n\n\t\tLinus") == "This is wrong."
    assert strip_signoff("This is wrong.\n\n             Linus\n") == "This is wrong."
    assert strip_signoff("Ask Linus about it.") == "Ask Linus about it."


def test_as_sft_strips_signoff_by_default_keeps_on_request():
    p = pair_from(prose_reply(own=LONG_REVIEW + "\n\n\t\tLinus"), has_patch=True)
    assert not p.as_sft()["output"].endswith("Linus")
    assert p.as_sft(keep_signoff=True)["output"].endswith("Linus")


# --- thread context ---------------------------------------------------------

def entry(mid: str, subject: str, author: str = "A") -> IndexEntry:
    return IndexEntry(mid=mid, subject=subject, author=author, date="d")


def test_earlier_in_thread_is_same_subject_and_strictly_before():
    es = [entry("00001", "[PATCH] mm: x"), entry("00005", "Re: [PATCH] mm: x"),
          entry("00007", "Re: [PATCH] other"), entry("00009", "Re: [PATCH] mm: x")]
    got = [e.mid for e in earlier_in_thread(es, "Re: [PATCH] mm: x", "00009")]
    assert got == ["00001", "00005"]          # nothing from the future, no other thread


def test_select_context_skips_patch_postings_and_keeps_newest_within_budget():
    msgs = [ThreadMessage("Poster", "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n"),
            ThreadMessage("Old", "old " * 400),
            ThreadMessage("Al Viro", "That breaks the RCU walk.")]
    out = select_context(msgs, budget=200)
    assert "diff --git" not in out
    assert out.endswith("[Al Viro]:\nThat breaks the RCU walk.")
    assert "[Old]" not in out                 # over budget, the oldest goes first


def test_long_prose_is_not_mistaken_for_a_patch():
    out = select_context([ThreadMessage("Al", "word " * 300)], budget=5000)
    assert out.startswith("[Al]:")


def test_as_sft_puts_thread_before_patch():
    p = pair_from(prose_reply(), has_patch=True)
    p.resolved_context = "diff --git a/x b/x"
    p.thread_context = "[Al]:\nThat breaks the RCU walk."
    text = p.as_sft()["input"]
    assert text.index("Earlier in this thread") < text.index("The patch") < text.index("diff --git")


# --- leftovers found in the first v3 export ---------------------------------

def test_strip_signoff_handles_nbsp_indentation():
    assert strip_signoff("Tough.\n\n\xa0\xa0\xa0\xa0Linus") == "Tough."


def test_attribution_with_stray_quote_marker_is_stripped():
    from linuspr.pairs import strip_attributions
    assert strip_attributions("On Thu, Feb 14, 2019 at 10:10 AM Kees Cook wrote:>\n\nBetter.") == "Better."


def test_half_a_vger_trailer_is_still_removed():
    from linuspr.pairs import strip_list_footers
    text = "Hmm?\n\n\t\tLinus\n\nMore majordomo info at  http://vger.kernel.org/majordomo-info.html\n"
    assert "majordomo" not in strip_list_footers(text)


def test_double_signature_is_fully_stripped():
    assert strip_signoff("etc.\n\n      Linus\n\n      Linus") == "etc."


def test_quoted_vger_trailer_lines_are_removed():
    from linuspr.pairs import strip_list_footers
    text = " #else\n\n> --\n> To unsubscribe from this list: send the line x in\n> the body of a message to majordomo@x\n"
    assert "majordomo" not in strip_list_footers(text) and "#else" in strip_list_footers(text)


def test_signature_before_an_attached_patch_is_stripped_patch_kept():
    text = "Are there other callers?\n\n\t\tLinus\n---\n fs/x.c | 2 ++\n\ndiff --git a/fs/x.c b/fs/x.c\n"
    out = strip_signoff(text)
    assert "Linus" not in out
    assert out.startswith("Are there other callers?") and "diff --git a/fs/x.c" in out


def test_linus_inside_a_sentence_is_untouched():
    text = "Ask Linus about this.\nIt matters."
    assert strip_signoff(text) == text
