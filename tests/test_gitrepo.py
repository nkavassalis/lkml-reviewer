"""Commit-index tests. git is invoked through GitRepo.run, faked here."""
from __future__ import annotations

import pytest

from linuspr.gitrepo import FIELD_SEP, RECORD_SEP, GitRepo, NoRepo, msgids_in


def repo_with(run):
    r = GitRepo(path=__import__("pathlib").Path("/nonexistent"))
    r.run = run  # type: ignore[method-assign]
    return r


def test_msgids_in_extracts_lore_links():
    body = (
        "fs: fix the thing\n\n"
        "Link: https://lore.kernel.org/all/20240427005626.1365935-1-zhou@kernel.org/\n"
        "Signed-off-by: Nobody\n"
    )
    assert msgids_in(body) == ["20240427005626.1365935-1-zhou@kernel.org"]


def test_msgids_in_accepts_message_id_tag_and_dedupes():
    body = (
        "Message-Id: <abc-123@laptop>\n"
        "Link: https://lore.kernel.org/lkml/abc-123@laptop/\n"
    )
    assert msgids_in(body) == ["abc-123@laptop"]


def test_msgids_in_strips_trailing_punctuation_from_urls():
    got = msgids_in("Link: https://lore.kernel.org/all/2024@host),")
    assert got == ["2024@host"]


def test_msgids_in_rejects_non_msgids():
    assert msgids_in("no ids here at all") == []
    assert msgids_in("two@a@b") == []  # two @ signs
    # a bare angle-bracketed id with no lore url / Message-Id: anchor is NOT
    # guessed at — false positives cost more than the rare miss
    assert msgids_in("see <2024@host>,") == []


def test_all_commit_msgids_builds_inverted_index():
    log = (
        f"aaa{FIELD_SEP}subj\n\nLink: https://lore.kernel.org/all/m1@x{RECORD_SEP}"
        f"bbb{FIELD_SEP}subj2\n\nMessage-Id: <m2@y>{RECORD_SEP}"
        f"ccc{FIELD_SEP}subj3\n\nLink: https://lore.kernel.org/all/m1@x/{RECORD_SEP}"
    )
    r = repo_with(lambda *a, **k: log)
    r.require = lambda: r  # type: ignore[method-assign]
    index = r.all_commit_msgids()
    assert index == {"m1@x": ["aaa", "ccc"], "m2@y": ["bbb"]}


def test_show_patch_and_subject_use_git_show():
    seen = []

    def run(*args, **kw):
        seen.append(args)
        return "x"

    r = repo_with(run)
    r.require = lambda: r  # type: ignore[method-assign]
    r.show_patch("deadbeef")
    r.subject_of("deadbeef")
    assert seen[0][:3] == ("show", "--format=full", "--no-color")
    assert seen[1][:3] == ("show", "--no-patch", "--format=%s")


def test_require_raises_without_a_repo(tmp_path):
    r = GitRepo(tmp_path / "nope")
    assert not r.exists()
    with pytest.raises(NoRepo, match="git clone --bare"):
        r.require()


def test_exists_looks_for_head_or_objects(tmp_path):
    (tmp_path / "objects").mkdir()
    assert GitRepo(tmp_path).exists()
