"""Abuse-filter tests. The rule under test: keep the judgement, lose the abuse."""
from __future__ import annotations

import pytest

from linuspr.moderate import (
    classify,
    filter_pairs,
    has_substance,
    targets_code,
)
from linuspr.pairs import ReviewPair

TECH = (
    "The locking is wrong: you drop the mutex before the list walk, so a "
    "concurrent rename can free the entry under us. Take the lock in the "
    "caller instead and document the lifetime rule for the caller."
)


def pair(review: str) -> ReviewPair:
    return ReviewPair(
        mid="1",
        month="1901.0",
        subject="Re: [PATCH] x",
        date="d",
        context="ctx",
        review=review,
    )


def test_clean_review_is_kept_untouched():
    v = classify(TECH)
    assert v.action == "keep" and v.review == TECH and v.hits == ()


def test_person_attack_drops_the_pair():
    v = classify(TECH + " You are an idiot.")
    assert v.action == "drop" and v.review == ""
    assert "idiot" in v.hits


def test_pure_abuse_with_no_content_is_dropped():
    assert classify("This is crap. Get out.").action == "drop"


def test_abusive_interjection_is_cut_and_the_technics_survive():
    v = classify("Utter garbage. " + TECH)
    assert v.action == "scrub"
    assert "garbage" not in v.review
    assert "drop the mutex" in v.review


def test_abuse_and_content_in_one_sentence_keeps_the_content_clause():
    v = classify(
        "This code is garbage, but the allocation must use GFP_KERNEL here "
        "because it can sleep in this path and that is a bug. " + TECH
    )
    assert v.action == "scrub"
    assert "garbage" not in v.review
    assert "GFP_KERNEL" in v.review


def test_scrub_never_rewrites_the_surviving_prose():
    """We cut text; we never paraphrase it. Our words must not enter the corpus."""
    original = "Nak. " + TECH
    v = classify(original)
    for sentence in v.review.split("\n\n"):
        assert sentence.strip() in original


def test_has_substance_needs_both_technical_vocab_and_length():
    assert has_substance(TECH)
    assert not has_substance("the code is bad")  # too short
    assert not has_substance("lorem ipsum " * 40)  # long but no technical signal


def test_targets_code_recognises_code_directed_abuse():
    assert targets_code("this patch is garbage")
    assert targets_code("the interface is insane")
    assert not targets_code("you are garbage")


def test_filter_pairs_tallies_and_mutates():
    pairs = [pair(TECH), pair("Utter garbage. " + TECH), pair("You idiot.")]
    kept, tally = filter_pairs(pairs)
    assert tally == {"keep": 1, "scrub": 1, "drop": 1}
    assert len(kept) == 2
    assert "garbage" not in kept[1].review


def test_case_insensitive_matching():
    assert classify("This is GARBAGE. " + TECH).action == "scrub"


@pytest.mark.parametrize("word", ["braindead", "moron", "clueless", "incompetent"])
def test_person_insults_are_always_dropped(word):
    assert classify(TECH + f" Total {word} nonsense.").action == "drop"


# --- false positives found by reading real corpus output --------------------

def test_author_surnames_are_not_insults():
    tech = "Christoph Hellwig wrote: yes, that code was correct and the " + TECH
    assert classify(tech).action == "keep"


def test_unsafe_is_not_unsane():
    tech = (
        "Convert filldir64() from __put_user() to unsafe_put_user, because "
        "the size check is done earlier in the caller. " + TECH
    )
    assert classify(tech).action == "keep"


def test_matching_is_whole_word():
    from linuspr.moderate import CODE_ATTACKS, _hits

    assert _hits("this is crap", CODE_ATTACKS) == ("crap",)
    assert _hits("crappy", CODE_ATTACKS) == ()          # not a listed word
    assert _hits("Christoph Hellwig", CODE_ATTACKS) == ()


def test_sane_is_approval_not_abuse():
    """'sane' scrubbed 86 reviews, mostly verdicts like 'Looks sane to me'."""
    assert classify("Looks sane to me. " + TECH).action == "keep"
    assert classify("No sane person would do this. " + TECH).action == "scrub"


def test_scrub_keeps_code_indentation_and_line_breaks():
    code = "   void arch_set_restart_data(restart)\n   {\n      restart->arch_data = x;\n  }\n"
    review = "That is garbage.\n\nNo, I'd suggest this:\n\n" + code + "\n" + TECH
    v = classify(review)
    assert v.action == "scrub"
    assert "garbage" not in v.review
    assert v.review == "No, I'd suggest this:\n\n" + code + "\n" + TECH
