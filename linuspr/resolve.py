"""Attach the *real* patch to each pair by resolving it through the git tree.

Why not just fetch lore: lore.kernel.org (and patchwork, and git.kernel.org's
web UI) all sit behind Techaro Anubis and hand stateless HTTP a JS challenge.
The git protocol is open, so the diff comes from the tree instead.

How a pair gets a diff:

  1. Linus quoted the patch, and his quote usually still contains the
     `https://lore.kernel.org/all/<msgid>` line the posting left behind.
  2. Merged commits carry `Link: https://lore.kernel.org/all/<msgid>` in their
     commit message, so one `git log` pass builds a msgid -> commit index.
  3. Subject agreement is checked before trusting the match — a msgid that
     resolves to a commit titled something else is rejected, not guessed at.

A pair whose msgid never resolved keeps its quoted context; `resolved_context`
stays empty and `export` falls back to `context`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .gitrepo import GitRepo, msgids_in
from .pairs import ReviewPair

# `name@xxxxxxxxxxxx` — a msgid whose domain the mirror ate. The local part
# survives untouched, which is what makes recovery possible.
RAW_MSGID_RE = re.compile(r"[\w.+-]+@[\w.-]+")
X_RUN_RE = re.compile(r"x+")

BRACKET_RE = re.compile(r"\[[^\]]*\]")
RE_PREFIX_RE = re.compile(r"^\s*(?:re:\s*)+", re.I)
MIN_SUBJECT_OVERLAP = 0.5
WORD_RE = re.compile(r"[a-z0-9_./-]{3,}")


@dataclass
class Resolution:
    commit: str = ""
    patch: str = ""
    reason: str = ""
    candidates: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.commit)


class DomainIndex:
    """msgid index re-cut by domain, so mangled domains can be recovered.

    hypermail replaces a domain's characters with x's but leaves the local part
    alone. `git log` knows ~450k real msgids, so we can ask "which real domains
    have a msgid with exactly this local part?" — a near-unique fingerprint.
    """

    def __init__(self, index: dict[str, list[str]]) -> None:
        self.by_domain: dict[str, dict[str, list[str]]] = {}
        for msgid, commits in index.items():
            local, _, domain = msgid.partition("@")
            if not domain:
                continue
            self.by_domain.setdefault(domain, {}).setdefault(local, []).extend(commits)
        # longest domains first: they are the most specific fingerprints
        self.domains = sorted(self.by_domain, key=len, reverse=True)

    def domains_matching(self, mangled: str, *, tolerance: int = 3) -> list[str]:
        """Real domains whose x-pattern is compatible with the mangled one."""
        if "x" not in mangled:
            return [d for d in self.domains if d == mangled]
        stripped = X_RUN_RE.sub("#", mangled)
        out = []
        for domain in self.domains:
            if abs(len(domain) - len(mangled)) > tolerance:
                continue
            if X_RUN_RE.sub("#", domain) == stripped:
                out.append(domain)
        return out

    def commits_for(self, local: str, domains: list[str]) -> list[str]:
        commits: list[str] = []
        for domain in domains:
            for commit in self.by_domain.get(domain, {}).get(local, []):
                if commit not in commits:
                    commits.append(commit)
        return commits


def raw_msgid_tokens(text: str) -> list[str]:
    """Everything that *looks* like a msgid, mangled domains included."""
    out: list[str] = []
    for match in RAW_MSGID_RE.finditer(text):
        token = match.group(0).strip("<>").rstrip(".,;)")
        if token.count("@") == 1 and token not in out:
            out.append(token)
    return out


def recover_msgid(token: str, domain_index: DomainIndex) -> str:
    """`brauner@xxxxxxxxx` -> the real msgid, or "" when nothing matches.

    Only ever returns a msgid whose local part matches exactly, so a wrong
    domain shape cannot produce a plausible-looking wrong patch.
    """
    local, _, mangled = token.partition("@")
    if not mangled or "x" not in mangled:
        return "" if "x" not in token else ""
    for domain in domain_index.domains_matching(mangled):
        if local in domain_index.by_domain.get(domain, {}):
            return f"{local}@{domain}"
    return ""


class SubjectIndex:
    """commit -> (date, subject) with an inverted word index over subjects.

    Built once from `git log --all` with \x1e/\x1f field separators and cached as
    JSON. 1.5M commits, ~200 MB in memory — that is deliberate: the alternative
    (a git grep per pair) is two orders of magnitude slower.
    """

    def __init__(self, rows: list[tuple[str, str, str]]) -> None:
        self.commits = {c: (d, subj) for c, d, subj in rows}
        self.inverted: dict[str, list[str]] = {}
        for commit, _date, subject in rows:
            for word in set(WORD_RE.findall(normalise_subject(subject))):
                self.inverted.setdefault(word, []).append(commit)

    def __len__(self) -> int:
        return len(self.commits)

    def candidates(self, subject: str, *, limit: int = 400) -> list[str]:
        words = set(WORD_RE.findall(normalise_subject(subject)))
        if not words:
            return []
        # start from the rarest word so the candidate set stays small
        postings = sorted(
            (self.inverted.get(w, []) for w in words), key=len
        )
        out: list[str] = []
        seen: set[str] = set()
        for commit in postings[0]:
            if commit in seen:
                continue
            seen.add(commit)
            out.append(commit)
            if len(out) >= limit:
                break
        return out

    def date_of(self, commit: str) -> str:
        return self.commits.get(commit, ("", ""))[0]

    def subject_of(self, commit: str) -> str:
        return self.commits.get(commit, ("", ""))[1]


def message_year(date: str) -> int | None:
    m = re.search(r"\b(19|20)\d\d\b", date or "")
    return int(m.group(0)) if m else None


def match_commit(
    pair: ReviewPair,
    subject_index: SubjectIndex,
    *,
    min_overlap: float = 0.6,
    year_slack: int = 2,
) -> tuple[str, float]:
    """Best commit whose subject matches this patch, or ("", 0.0).

    Subject overlap alone is not enough — kernel subjects repeat across years
    ("fix a typo in mm/..."), so the commit must also land within `year_slack`
    years of the message. Both conditions are required; a near miss returns
    nothing rather than a guess.
    """
    best, best_commit = 0.0, ""
    for commit in subject_index.candidates(pair.subject):
        date = subject_index.date_of(commit)
        year = message_year(date)
        msg_year = message_year(pair.date)
        if year and msg_year and abs(year - msg_year) > year_slack:
            continue
        score = subject_overlap(pair.subject, subject_index.subject_of(commit))
        if score > best:
            best, best_commit = score, commit
    return (best_commit, best) if best >= min_overlap else ("", best)


def candidate_msgids(pair: ReviewPair) -> list[str]:
    """Msgids reachable from a pair: quoted lore links plus any bare ones."""
    # NB: never scrub before this — the mangled addresses *are* the msgid of
    # patches posted from a mangled domain, and scrubbing would delete them.
    return msgids_in(f"{pair.subject}\n{pair.context}")


def normalise_subject(subject: str) -> str:
    """`Re: [PATCH v2 3/7] fs: fix thing` -> `fs: fix thing`."""
    s = BRACKET_RE.sub(" ", subject)
    s = RE_PREFIX_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def subject_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the significant words of two subjects."""
    wa = set(WORD_RE.findall(normalise_subject(a)))
    wb = set(WORD_RE.findall(normalise_subject(b)))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def resolve_pair(
    pair: ReviewPair,
    index: dict[str, list[str]],
    repo: GitRepo,
    *,
    min_overlap: float = MIN_SUBJECT_OVERLAP,
    domain_index: DomainIndex | None = None,
) -> Resolution:
    """Find the commit that landed this patch, or explain why we did not.

    Two routes, in order of trustworthiness: a msgid quoted verbatim, and a
    msgid whose domain the mirror mangled but whose local part still matches a
    real commit exactly. Both then have to survive the subject check.
    """
    msgids = candidate_msgids(pair)
    if not msgids and domain_index is not None:
        for token in raw_msgid_tokens(f"{pair.subject}\n{pair.context}"):
            recovered = recover_msgid(token, domain_index)
            if recovered and recovered not in msgids:
                msgids.append(recovered)
    if not msgids:
        return Resolution(reason="no msgid in the quoted context")
    commits: list[str] = []
    for msgid in msgids:
        for commit in index.get(msgid, []):
            if commit not in commits:
                commits.append(commit)
    if not commits:
        return Resolution(reason="msgid not in the commit index", candidates=msgids)
    scored = []
    for commit in commits:
        try:
            commit_subject = repo.subject_of(commit)
        except Exception as exc:  # pragma: no cover - git failure is environmental
            return Resolution(reason=f"git failed: {exc}")
        scored.append((subject_overlap(pair.subject, commit_subject), commit))
    best, commit = max(scored)
    if best < min_overlap:
        return Resolution(
            reason=f"subject overlap {best:.2f} < {min_overlap}", candidates=commits
        )
    return Resolution(commit=commit, patch=repo.show_patch(commit), reason="matched")


def resolve_pairs(
    pairs: list[ReviewPair],
    index: dict[str, list[str]],
    repo: GitRepo,
    *,
    min_overlap: float = MIN_SUBJECT_OVERLAP,
    domain_index: DomainIndex | None = None,
    subject_index: SubjectIndex | None = None,
) -> dict[str, int]:
    """Resolve in place; returns a tally for the caller to print."""
    if domain_index is None:
        domain_index = DomainIndex(index)
    tally = {
        "via_msgid": 0,
        "via_subject": 0,
        "patch_only": 0,
        "unresolved": 0,
    }
    for pair in pairs:
        res = resolve_pair(
            pair, index, repo, min_overlap=min_overlap, domain_index=domain_index
        )
        if res.ok:
            pair.commit = res.commit
            if res.patch:
                pair.resolved_context = res.patch
            tally["via_msgid"] += 1
            continue
        if subject_index is not None:
            commit, _score = match_commit(pair, subject_index)
            if commit:
                pair.commit = commit
                tally["via_subject"] += 1
                continue
        if pair.resolved_context:
            tally["patch_only"] += 1
        else:
            tally["unresolved"] += 1
    return tally


def load_index(path: Path, *, rebuild: bool = False) -> dict[str, list[str]]:
    if path.exists() and not rebuild:
        return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"{path} — build it with `python -m linuspr index`")


def write_index(path: Path, index: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def patch_quality(patch: str) -> int:
    """How much of a real patch a recovered blob looks like. 0 = not a patch."""
    if not patch:
        return 0
    score = 0
    score += 2 * len(re.findall(r"^diff --git ", patch, re.M))
    score += len(re.findall(r"^@@ -\d+", patch, re.M))
    score += 2 if re.search(r"^--- a/", patch, re.M) else 0
    score += len(patch) // 500
    return score


MIN_PATCH_QUALITY = 4
#: A parent two hops up has usually lost the `[PATCH 3/7]` tag but keeps the
#: title words, so this is deliberately looser than the commit gate.
MIN_THREAD_OVERLAP = 0.34


def walk_thread_for_patch(
    msg,
    fetcher,
    *,
    max_hops: int = 4,
    min_quality: int = MIN_PATCH_QUALITY,
    min_subject_overlap: float = MIN_THREAD_OVERLAP,
    subjects: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Walk a reply's parent chain and return the *best* patch blob we find.

    Stops early once a parent looks like a complete posting; otherwise keeps
    climbing and returns the best partial. hypermail's "Previous message" link
    is a document-order predecessor, not a true parent, so the chain is a
    heuristic — which is why the caller must quality-check the result.

    Returns (patch_text, mid_of_that_message); ("", "") when nothing qualified.
    """
    from .parse import patch_text  # noqa: PLC0415
    from .pipeline import fetch_message_or_none  # noqa: PLC0415

    # hypermail's "Previous message" is document order, not parentage: the page
    # before Linus' reply is very often the tail of an unrelated thread, and its
    # patch would be attached to the wrong review. The subject gate below is
    # what keeps those out — measured: without it, ~1 in 5 recovered patches
    # came from a different subject line entirely.
    best = ("", "")
    best_score = 0
    current = msg
    seen: set[str] = {msg.mid}
    for _ in range(max_hops):
        if not current.prev_mid or current.prev_mid in seen:
            break
        seen.add(current.prev_mid)
        parent = fetch_message_or_none(fetcher, current.month, current.prev_mid)
        if parent is None:
            break
        blob = patch_text(parent)
        score = patch_quality(blob)
        # Subjects come from the month index, not the nav link: the link renders
        # `Author: "Subject"` and the subject itself contains colons, which the
        # nav-link regex cannot delimit reliably.
        parent_subject = (subjects or {}).get(parent.mid) or parent.subject
        if score and subject_overlap(msg.subject, parent_subject) < min_subject_overlap:
            score = 0  # right shape, wrong thread
        if score > best_score:
            best, best_score = (blob, parent.mid), score
        current = parent
    return best if best_score >= 2 else ("", "")


class ThreadMap:
    """subject -> messages, across every bucket we have crawled.

    hypermail's own thread view groups by subject line, so the patch a reply
    answers is nearly always the *earliest* message sharing its subject — and
    unlike the "Previous message" nav link, this never wanders into another
    thread. Built lazily per bucket and cached in memory; buckets are cheap
    because `Fetcher` keeps them on disk.
    """

    def __init__(self, fetcher) -> None:
        self.fetcher = fetcher
        self._by_bucket: dict[str, dict[str, list[tuple[str, str]]]] = {}

    def group(self, month: str, subject: str) -> list[tuple[str, str]]:
        """[(mid, subject)] sharing `subject`, in document order, same bucket."""
        key = normalise_subject(subject)
        table = self._by_bucket.get(month)
        if table is None:
            from .parse import parse_index  # noqa: PLC0415

            table = {}
            for entry in parse_index(self.fetcher.get(month, "index.html")):
                table.setdefault(normalise_subject(entry.subject), []).append(
                    (entry.mid, entry.subject)
                )
            self._by_bucket[month] = table
        return table.get(key, [])


def find_patch_in_thread(
    msg,
    fetcher,
    thread_map: ThreadMap,
    *,
    min_quality: int = MIN_PATCH_QUALITY,
    max_candidates: int = 12,
) -> tuple[str, str]:
    """Best patch blob among the messages sharing this reply's subject line.

    Falls back to the nav-link walk when the subject group is empty (the patch
    was posted in an earlier month, or the subject drifted).
    """
    from .parse import patch_text  # noqa: PLC0415
    from .pipeline import fetch_message_or_none  # noqa: PLC0415

    group = thread_map.group(msg.month, msg.subject)
    best, best_score, best_mid = "", 0, ""
    for mid, _subject in group[:max_candidates]:
        if mid == msg.mid:
            continue
        parent = fetch_message_or_none(fetcher, msg.month, mid)
        if parent is None:
            continue
        blob = patch_text(parent)
        score = patch_quality(blob)
        if score > best_score:
            best, best_score, best_mid = blob, score, mid
    return (best, best_mid) if best_score >= min_quality else ("", "")


def find_patch_for_message(
    msg, fetcher, thread_map: ThreadMap, *, max_hops: int = 4
) -> tuple[str, str]:
    """Recover the patch a reply answers: subject group first, nav link second.

    The subject group is the trustworthy route; the nav-link walk is the
    fallback for threads that cross a month boundary, and it carries its own
    subject gate so it cannot smuggle in a stranger's diff.
    """
    patch, mid = find_patch_in_thread(msg, fetcher, thread_map)
    if patch:
        return patch, mid
    return walk_thread_for_patch(msg, fetcher, max_hops=max_hops)
