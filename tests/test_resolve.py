"""Resolver tests: msgid extraction, subject matching, SFT context choice."""
from __future__ import annotations

import pytest

from linuspr.pairs import ReviewPair
from linuspr.resolve import (
    SubjectIndex,
    candidate_msgids,
    load_index,
    match_commit,
    message_year,
    normalise_subject,
    resolve_pair,
    resolve_pairs,
    subject_overlap,
    write_index,
)

MSGID = "20190526012345.12345-1-brauner@kernel.org"
PATCH_SUBJ = "Re: [PATCH 1/2] fork: add clone6 system call"


def pair(context: str = "", subject: str = PATCH_SUBJ, **kw) -> ReviewPair:
    base = {
        "mid": "00001",
        "month": "1905.3",
        "subject": subject,
        "date": "d",
        "context": context,
        "review": "No, this is not acceptable.",
    }
    base.update(kw)
    return ReviewPair(**base)


class FakeRepo:
    def __init__(self, subjects=None, patches=None, fail=()):
        self.subjects = subjects or {}
        self.patches = patches or {}
        self.fail = set(fail)
        self.calls: list[str] = []

    def subject_of(self, commit):
        if commit in self.fail:
            raise RuntimeError("git died")
        return self.subjects[commit]

    def show_patch(self, commit):
        self.calls.append(commit)
        return self.patches.get(commit, f"commit {commit}\n\ndiff --git a/x b/x\n")


def test_normalise_subject_drops_brackets_and_re():
    assert normalise_subject("Re: [PATCH v2 3/7] fs: fix Thing") == "fs: fix thing"
    assert normalise_subject("re: re: [PATCH] a b") == "a b"


def test_subject_overlap_is_word_jaccard():
    a = "Re: [PATCH] fork: add clone6 system call"
    assert subject_overlap(a, "fork: add clone6 system call") == 1.0
    assert subject_overlap(a, "mm: fix slab accounting") < 0.1
    assert subject_overlap("", "anything") == 0.0


def test_candidate_msgids_reads_quoted_lore_links():
    p = pair(context=f"see https://lore.kernel.org/all/{MSGID}/ for the patch")
    assert candidate_msgids(p) == [MSGID]


def test_candidate_msgids_ignores_prose():
    assert candidate_msgids(pair(context="just some discussion")) == []


def test_resolves_when_msgid_hits_and_subject_agrees():
    p = pair(context=f"Link: https://lore.kernel.org/all/{MSGID}/")
    index = {MSGID: ["deadbeef"]}
    repo = FakeRepo(subjects={"deadbeef": "fork: add clone6 system call"})
    res = resolve_pair(p, index, repo)
    assert res.ok and res.commit == "deadbeef" and res.reason == "matched"
    assert "deadbeef" in res.patch


def test_rejects_when_subject_disagrees():
    p = pair(context=f"Link: https://lore.kernel.org/all/{MSGID}/")
    repo = FakeRepo(subjects={"deadbeef": "mm: rewrite the slab allocator"})
    res = resolve_pair(p, {MSGID: ["deadbeef"]}, repo)
    assert not res.ok and "subject overlap" in res.reason


def test_picks_the_best_commit_when_a_msgid_maps_to_several():
    p = pair(context=f"Link: https://lore.kernel.org/all/{MSGID}/")
    repo = FakeRepo(
        subjects={"aaa": "mm: something else entirely", "bbb": "fork: add clone6 system call"}
    )
    res = resolve_pair(p, {MSGID: ["aaa", "bbb"]}, repo)
    assert res.commit == "bbb"


def test_missing_msgid_is_reported_not_guessed():
    res = resolve_pair(pair(context="no links"), {}, FakeRepo())
    assert not res.ok and res.reason.startswith("no msgid")


def test_msgid_absent_from_index_is_reported():
    p = pair(context=f"Link: https://lore.kernel.org/all/{MSGID}/")
    res = resolve_pair(p, {}, FakeRepo())
    assert not res.ok and res.reason == "msgid not in the commit index"


def test_resolve_pairs_tallies_and_mutates_in_place():
    good = pair(context=f"Link: https://lore.kernel.org/all/{MSGID}/", mid="g")
    bad = pair(context="nothing", mid="b")
    repo = FakeRepo(subjects={"c1": "fork: add clone6 system call"})
    tally = resolve_pairs([good, bad], {MSGID: ["c1"]}, repo)
    assert tally == {
        "via_msgid": 1, "via_subject": 0, "patch_only": 0, "unresolved": 1
    }
    assert good.commit == "c1" and good.resolved_context
    assert bad.commit == "" and bad.resolved_context == ""


def test_sft_prefers_resolved_context_over_quote():
    p = pair(context="> thin quote", commit="c1", resolved_context="diff --git a/f b/f")
    assert "diff --git" in p.as_sft()["input"]
    p2 = pair(context="> thin quote")
    assert "thin quote" in p2.as_sft()["input"]


def test_index_roundtrip(tmp_path):
    path = tmp_path / "msgid_index.json"
    with pytest.raises(FileNotFoundError):
        load_index(path)
    write_index(path, {"m1@x": ["a", "b"]})
    assert load_index(path) == {"m1@x": ["a", "b"]}
    assert not list(tmp_path.glob("*.tmp"))  # written atomically


# --- subject-based commit matching (the route that actually works) ---------

ROWS = [
    ("c1", "2019-01-08", "arch/openrisc: Fix issues with access_ok()"),
    ("c2", "2015-03-01", "arch/openrisc: Fix issues with access_ok()"),
    ("c3", "2019-02-01", "mm: rewrite the slab allocator completely"),
    ("c4", "2019-01-09", "fork: add clone6 system call"),
]


def subject_index():
    return SubjectIndex(ROWS)


def test_message_year_extracts_from_hypermail_dates():
    assert message_year("Mon,  1 Jan 2019 10:00:00 -0500 (EST)") == 2019
    assert message_year("nonsense") is None
    assert message_year("") is None


def test_subject_index_indexes_every_commit():
    si = subject_index()
    assert len(si) == 4
    assert si.date_of("c1") == "2019-01-08"
    assert si.subject_of("c4").startswith("fork:")


def test_candidates_come_from_the_rarest_word():
    got = subject_index().candidates("fork: add clone6 system call")
    assert "c4" in got


def test_match_commit_prefers_the_year_near_the_message():
    p = pair(subject="Re: [PATCH] arch/openrisc: Fix issues with access_ok()")
    commit, score = match_commit(p, subject_index())
    assert commit == "c1" and score == 1.0  # c2 is the same subject, 4 years off


def test_match_commit_rejects_unrelated_subjects():
    p = pair(subject="Re: [PATCH] mm: nothing like the others at all")
    commit, score = match_commit(p, subject_index())
    assert commit == "" and score < 0.6


def test_match_commit_needs_both_subject_and_year():
    p = pair(subject="Re: [PATCH] fork: add clone6 system call", date="x 1999 x")
    assert match_commit(p, subject_index())[0] == ""


def test_match_commit_matches_a_plain_patch_subject():
    p = pair(subject="[PATCH 1/2] fork: add clone6 system call")
    assert match_commit(p, subject_index())[0] == "c4"
