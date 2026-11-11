"""Thread-walking and patch-blob tests (no network: fetchers are faked)."""
from __future__ import annotations

from linuspr.parse import Message, patch_text
from linuspr.resolve import (
    ThreadMap,
    find_patch_for_message,
    find_patch_in_thread,
    patch_quality,
    walk_thread_for_patch,
)

FULL_PATCH = """Some prose about why the thing is broken.

---
 arch/x86/kernel/process.c | 12 ++++++------
 1 file changed, 6 insertions(+), 6 deletions(-)

diff --git a/arch/x86/kernel/process.c b/arch/x86/kernel/process.c
--- a/arch/x86/kernel/process.c
+++ b/arch/x86/kernel/process.c
@@ -100,7 +100,7 @@ void arch_cpu_idle(void)
-       native_safe_halt();
+       stop_this_machine();
--
2.20.1
"""


SUBJ = "Re: [PATCH] mm: fix the idle loop accounting"


def msg(body: str = "", prev: str | None = None, mid: str = "00001",
        month="1901.0", subject: str = SUBJ):
    return Message(
        mid=mid,
        month=month,
        subject=subject,
        author="Linus Torvalds",
        date="Mon,  1 Jan 2019 10:00:00 -0500 (EST)",
        body=body,
        prev_mid=prev,
    )


class FakeFetcher:
    def __init__(
        self,
        pages: dict[str, str],
        subject: str = SUBJ,
        *,
        step: int = 1,
        prev_map: dict[str, str] | None = None,
    ):
        self.pages = pages  # mid -> rendered body
        self.subject = subject
        self.step = step  # default nav-link direction: +1 chains upwards
        self.prev_map = prev_map or {}  # explicit overrides win
        self.requested: list[str] = []

    def get(self, month, filename):
        """Serve `NNNNN.html`; its "previous message" is the next number down,
        so a chain is just consecutive ids and a gap is a missing page."""
        mid = filename.removesuffix(".html")
        self.requested.append(mid)
        if mid not in self.pages:
            return "<html><title>404 Not Found</title></html>"
        prev = self.prev_map.get(mid, f"{int(mid) + self.step:05d}")
        prev_link = (
            f'Previous message: </strong> <a href="{prev}.html"> Someone: '
            f'&quot;{self.subject}&quot;</a>'
            if prev in self.pages or prev in self.prev_map
            else ""
        )
        return (
            f"<!--X-Subject: {self.subject} -->"
            f'<!--X-Body-of-Message-->{self.pages[mid]}<!--X-Body-of-Message-End-->'
            f"{prev_link}"
        )


def test_patch_text_extracts_the_diff_from_a_posting():
    m = msg(body="prose line\n\n" + FULL_PATCH)
    got = patch_text(m)
    assert got.startswith("diff --git")
    assert "@@ -100,7 +100,7 @@" in got
    assert "Some prose" not in got


def test_patch_text_empty_for_plain_reply():
    assert patch_text(msg(body="No. This is not acceptable.")) == ""


def test_patch_text_accepts_a_bare_diff():
    m = msg(body="notes\n\ndiff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-x\n+y\n")
    assert patch_text(m).startswith("diff --git")


def test_patch_quality_rewards_real_patches():
    assert patch_quality(FULL_PATCH) >= 4
    assert patch_quality("") == 0
    assert patch_quality("> a fragment") < 2


def test_walk_stops_at_the_first_good_parent():
    f = FakeFetcher({"00000": FULL_PATCH})
    patch, mid = walk_thread_for_patch(
        msg(mid="99999", prev="00000"), f, subjects={"00000": SUBJ}
    )
    assert mid == "00000" and patch.startswith("diff --git")
    assert f.requested == ["00000"]


def test_walk_climbs_past_a_thin_quote():
    f = FakeFetcher({"00000": "> just a two-line quote", "00001": FULL_PATCH})
    patch, mid = walk_thread_for_patch(
        msg(mid="99999", prev="00000"), f, subjects={"00000": SUBJ, "00001": SUBJ}
    )
    assert mid == "00001"
    assert f.requested == ["00000", "00001"]


def test_walk_gives_up_on_a_dead_end():
    f = FakeFetcher({})
    assert walk_thread_for_patch(msg(prev="9999"), f) == ("", "")


def test_walk_respects_max_hops():
    chain = {f"{i:05d}": "> nothing useful" for i in range(10)}
    chain["00002"] = FULL_PATCH  # two hops up from the first parent
    f = FakeFetcher(chain)
    patch, mid = walk_thread_for_patch(
        msg(mid="99999", prev="00000"), f, max_hops=2,
        subjects=dict.fromkeys(chain, SUBJ),
    )
    assert patch == "" and mid == ""
    assert len(f.requested) == 2


def test_walk_returns_best_partial_when_nothing_is_complete():
    partial = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n-a\n+b\n"
    f = FakeFetcher({"00000": partial})
    patch, mid = walk_thread_for_patch(
        msg(mid="99999", prev="00000"), f, subjects={"00000": SUBJ}
    )
    assert mid == "00000" and "diff --git" in patch


def test_walk_refuses_a_parent_from_an_unrelated_thread():
    """The regression this gate exists for: the document-order predecessor is
    often the tail of a *different* thread, and its diff would be attached to
    the wrong review."""
    f = FakeFetcher({"00000": FULL_PATCH})
    patch, mid = walk_thread_for_patch(
        msg(mid="99999", prev="00000"), f,
        subjects={"00000": "Re: [PATCH] tty: n_hdlc fix"},
    )
    assert (patch, mid) == ("", "")


# --- subject-group route (the one that actually feeds the dataset) ---------

INDEX_HTML = (
    '<html><body><ul>'
    '<li><strong><a name="00010" href="00010.html">[PATCH] mm: fix the idle loop '
    'accounting</a> </strong><em>Alice </em><small>(Mon Jan 1 2019)</small>'
    '<li><strong><a name="00011" href="00011.html">Re: [PATCH] mm: fix the idle '
    'loop accounting</a> </strong><em>Bob </em><small>(Mon Jan 1 2019)</small>'
    '<li><strong><a name="00012" href="00012.html">Re: [PATCH] mm: fix the idle '
    'loop accounting</a> </strong><em>Linus Torvalds </em><small>(Mon Jan 1 2019)</small>'
    '<li><strong><a name="00013" href="00013.html">Re: [PATCH] totally unrelated'
    '</a> </strong><em>Eve </em><small>(Mon Jan 1 2019)</small>'
    "</ul></body></html>"
)


class IndexFetcher(FakeFetcher):
    def get(self, month, filename):
        if filename == "index.html":
            return INDEX_HTML
        return super().get(month, filename)


def test_thread_map_groups_by_subject():
    tm = ThreadMap(IndexFetcher({"00010": FULL_PATCH}))
    group = tm.group("1901.0", "Re: [PATCH] mm: fix the idle loop accounting")
    assert [mid for mid, _ in group] == ["00010", "00011", "00012"]


def test_find_patch_in_thread_takes_the_posting_not_the_reply():
    pages = {"00010": FULL_PATCH, "00011": "> only a quote"}
    tm = ThreadMap(IndexFetcher(pages))
    patch, mid = find_patch_in_thread(
        msg(mid="00012", subject=SUBJ), IndexFetcher(pages), tm
    )
    assert mid == "00010" and patch.startswith("diff --git")


def test_find_patch_in_thread_ignores_other_subjects():
    pages = {"00013": FULL_PATCH}  # the patch belongs to a *different* subject
    tm = ThreadMap(IndexFetcher(pages))
    assert find_patch_in_thread(msg(mid="00012", subject=SUBJ), IndexFetcher(pages), tm) == (
        "",
        "",
    )


def test_find_patch_falls_back_to_the_nav_walk():
    # patch sits at the nav-link parent, outside the subject group
    # the reply page (00078) itself must exist: its nav link names the parent
    pages = {"00077": FULL_PATCH, "00078": "No. This is not acceptable."}
    f = FakeFetcher(pages, prev_map={"00078": "00077"})
    tm = ThreadMap(FakeFetcher({}))  # empty index -> no subject group
    patch, mid = find_patch_for_message(
        msg(mid="00078", prev="00077"), f, tm
    )
    assert mid == "00077" and patch.startswith("diff --git")
