"""Quality-gate tests for (context, review) pair extraction."""
from __future__ import annotations

from pathlib import Path

from linuspr.pairs import (
    MANGLED_EMAIL_RE,
    attributed_authors,
    build_pairs,
    code_evidence,
    is_review,
    pair_from,
    patch_key,
    scrub_mangled,
    strip_attributions,
    strip_list_footers,
    unquote,
)
from linuspr.parse import Message, parse_message

CODE = "\n".join(
    [
        "--- a/mm/slub.c",
        "+++ b/mm/slub.c",
        "@@ -112,6 +112,9 @@",
        " struct kmem_cache {",
        "-       unsigned long flags;",
        "+       unsigned int flags;",
        "+       if (!s)",
        "+               return NULL;",
    ]
)


def msg(
    *,
    subject: str = "Re: [PATCH 1/1] mm: shrink slub",
    quoted: str = "\n".join("> " + ln for ln in CODE.splitlines()),
    own: str = (
        "On Mon, 3 Jun 2024 at 09:00, Alice Author <alice@xxxxxxxxxxxxx> wrote:\n\n"
        "No. You can't just widen the field like that, the on-disk format "
        "of struct kmem_cache is exposed to the slab shrinker tracepoints, "
        "so this breaks every existing tool that parses them. Either keep "
        "the type and fix the overflow at the source, or bump the format "
        "version and update the users. " + "z" * 120
    ),
) -> Message:
    return Message(
        mid="00001",
        month="2406.0",
        subject=subject,
        author="Linus Torvalds",
        date="d",
        body=own,
        quoted=quoted,
        own=own,
    )


def test_unquote_strips_markers():
    assert unquote("> foo\n>\tbar") == "foo\nbar"


def test_code_evidence_counts_diff_and_code():
    assert code_evidence(CODE) >= 3
    assert code_evidence("This looks great, thanks for the patch!") == 0


def test_attributed_authors_extracts_names():
    assert attributed_authors(msg().own) == ["Alice Author"]
    assert attributed_authors("no attribution here") == []


def test_clean_review_makes_a_pair():
    p = pair_from(msg())
    assert p is not None
    assert p.has_diff
    assert p.authors == ["Alice Author"]
    assert "widen the field" in p.review
    assert "@@ -112,6 +112,9 @@" in p.context
    assert not p.context.startswith(">")


def test_sft_shape():
    s = pair_from(msg()).as_sft()
    assert s["input"].startswith("Subject: Re: [PATCH 1/1]")
    assert "@@ -112" in s["input"]
    assert s["output"].startswith("No. You can't just widen the field")  # header stripped


def test_non_reply_is_not_a_review():
    assert not is_review(msg(subject="[PATCH 1/1] mm: shrink slub"))


def test_release_announcement_is_not_a_review():
    assert not is_review(msg(subject="Linux 6.10-rc2"))


def test_applied_reply_is_not_a_review():
    assert not is_review(msg(own="Applied, thanks.\n" + "x" * 150))


def test_off_topic_is_not_a_review():
    assert not is_review(
        msg(own="I'm on vacation until next week, my daughter " + "y" * 150)
    )


def test_prose_without_code_is_rejected():
    p = pair_from(
        msg(
            quoted="> " + "Some discussion about process and maintainership. " * 8,
        )
    )
    assert p is None


def test_short_review_is_rejected():
    p = pair_from(msg(own="Nak for the reason above."))
    assert p is None  # fine context, but under the review-length floor


def test_short_context_is_rejected():
    p = pair_from(msg(quoted="> if (!s) return NULL;"))
    assert p is None


def test_build_pairs_dedupes_same_review_same_subject():
    msgs = [msg(), msg()]
    assert len(build_pairs(msgs)) == 1
    msgs[1].subject = "Re: [PATCH 1/1] other thing"
    assert len(build_pairs(msgs)) == 2


def test_real_linus_nak_becomes_a_pair():
    html = (Path(__file__).parent / "fixtures" / "msg_2406_1_01812_linus.html").read_text(
        encoding="utf-8", errors="replace"
    )
    m = parse_message(html, mid="01812", month="2406.1")
    p = pair_from(m)
    assert p is not None
    assert "stupid config variable" in p.review
    assert "Peter Zijlstra" in p.authors
    assert p.code_hits > 5


# --- mangled-address scrubbing (mirror replaces email domains with x's) ----

def test_scrub_removes_angle_bracket_address_keeps_name():
    out = scrub_mangled(
        "On Mon, 10 Jun 2024 at 03:43, Peter Zijlstra <peterz@xxxxxxxxxxxxx> wrote:"
    )
    assert out == "On Mon, 10 Jun 2024 at 03:43, Peter Zijlstra wrote:"


def test_scrub_removes_bare_mangled_address():
    assert scrub_mangled("Signed-off-by: Zhouyi Zhou zhouzhouyi@xxxxxxxxx") == (
        "Signed-off-by: Zhouyi Zhou"
    )


def test_scrub_guts_dead_lore_links():
    out = scrub_mangled(
        "see https://lore.kernel.org/all/20240427-1-zhou@xxxxxxxxx/ for details"
    )
    assert out == "see for details"


def test_scrub_leaves_real_addresses_and_prose_alone():
    text = "mail real.author@example.com or see <a@b.c>; xxxx is not an address"
    assert scrub_mangled(text) == text


def test_scrub_is_applied_only_at_export():
    p = pair_from(msg())
    assert "@xxxxxxxxxxxxx" in p.review          # corpus keeps the mirror's text
    assert "@xxxxxxxxxxxxx" not in p.as_sft()["output"]


def test_mangled_regex_needs_the_domain_to_be_x_mangled():
    assert MANGLED_EMAIL_RE.search("a@xxxxxxxx")
    assert not MANGLED_EMAIL_RE.search("a@xxxxxxxx.yyyyy")  # real-looking tld
    assert not MANGLED_EMAIL_RE.search("a@kernel.org")


# --- export must not reformat code (found by reading sft.jsonl by hand) -----

def test_scrub_keeps_diff_indentation_byte_identical():
    diff = " \tu32 *fh = fid->raw;\n\n \tif (fh_len < 5 || fh_type != 3)\n-\t\treturn NULL;\n"
    assert scrub_mangled("Subject: x\n\n" + diff) == "Subject: x\n\n" + diff.rstrip()


def test_scrub_address_on_its_own_line_leaves_no_stray_space():
    out = scrub_mangled("On Thu, Jun 30, 2011 at 5:19 PM, Suresh Siddha\n"
                        "<suresh.b.siddha@xxxxxxxxx> wrote:")
    assert out == "On Thu, Jun 30, 2011 at 5:19 PM, Suresh Siddha\nwrote:"


def test_scrub_before_punctuation_closes_the_gap():
    assert scrub_mangled("Peter Zijlstra <peterz@xxxxxxxxxxxxx>, Ingo") == "Peter Zijlstra, Ingo"


VGER_FOOTER = (
    "--\nTo unsubscribe from this list: send the line \"unsubscribe linux-kernel\" in\n"
    "the body of a message to majordomo@xxxxxxxxxxxxxxx\n"
    "More majordomo info at  http://vger.kernel.org/majordomo-info.html\n"
    "Please read the FAQ at  http://www.tux.org/lkml/"
)
GROUPS_FOOTER = (
    "--\nYou received this message because you are subscribed to the Google Groups "
    "\"Clang Built Linux\" group.\nTo unsubscribe from this group and stop receiving "
    "emails from it, send an email to clang-built-linux+unsubscribe@xxxxxxxxxxxxxxxx.\n"
    "To view this discussion on the web visit https://groups.google.com/d/msgid/x."
)


def test_strip_list_footers_removes_vger_trailer():
    assert strip_list_footers("Looks good.\n\n\t\tLinus\n" + VGER_FOOTER) == (
        "Looks good.\n\n\t\tLinus"
    )


def test_strip_list_footers_removes_google_groups_trailer():
    assert strip_list_footers("+\treturn 0;\n\n" + GROUPS_FOOTER) == "+\treturn 0;"


def test_strip_list_footers_keeps_text_after_a_mid_message_trailer():
    text = "patch\n" + VGER_FOOTER + "\n\n#\n# Automatically generated file"
    assert strip_list_footers(text) == "patch\n\n\n#\n# Automatically generated file"


def test_as_sft_has_no_list_footer():
    p = pair_from(msg())
    p.review += "\n" + VGER_FOOTER
    p.resolved_context = "diff\n" + GROUPS_FOOTER
    sft = p.as_sft()
    assert "unsubscribe" not in sft["output"] and "majordomo" not in sft["output"]
    assert "Google Groups" not in sft["input"]


def test_patches_are_keyed_by_month_and_mid_not_mid_alone():
    """mid is a per-bucket counter; '02009' exists in 2010, 2016 and 2024."""
    a = msg(subject="Re: [PATCH] block: partition merge")
    b = msg(subject="Re: [PATCH] eventfs: stop using dcache_readdir")
    a.month, b.month = "1012.0", "2401.0"
    assert a.mid == b.mid
    patches = {patch_key(a): "BLOCK PATCH", patch_key(b): "EVENTFS PATCH"}
    by_subject = {p.subject: p.resolved_context for p in build_pairs([a, b], patches=patches)}
    assert by_subject[a.subject] == "BLOCK PATCH"
    assert by_subject[b.subject] == "EVENTFS PATCH"
    assert build_pairs([b], patches={"00001": "BLOCK PATCH"})[0].resolved_context == ""


# --- attribution headers (run-1 learned to invent them) ---------------------

def test_strip_attributions_removes_leading_header():
    text = "On Tue, 4 Feb 2025 at 05:22, Oleg Nesterov <oleg@xxxxxxxxxx> wrote:\n\nSo I really don't like it."
    assert strip_attributions(text) == "So I really don't like it."


def test_strip_attributions_handles_wrapped_header():
    text = "On Wed, Jul 14, 2010 at 10:06 AM, Mathieu Desnoyers\nwrote:\n\nThis is horrible."
    assert strip_attributions(text) == "This is horrible."


def test_strip_attributions_removes_mid_reply_headers_keeps_paragraphs():
    text = ("First point.\n\nOn Wed, 6 Mar 2024 at 11:46, Paul E. McKenney wrote:\n\n"
            "Second point.\n\n\t\tLinus")
    assert strip_attributions(text) == "First point.\n\nSecond point.\n\n\t\tLinus"


def test_strip_attributions_leaves_prose_starting_with_on_alone():
    text = "On x86 we already do this in the entry code, and nobody wrote: anything."
    assert strip_attributions(text) == text


def test_as_sft_output_has_no_attribution():
    assert not pair_from(msg()).as_sft()["output"].startswith("On ")
