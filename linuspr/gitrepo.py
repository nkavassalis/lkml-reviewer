"""Bare torvalds/linux clone: the authoritative source of patch diffs.

lore.kernel.org is behind Anubis and refuses stateless HTTP, but the git
protocol on git.kernel.org is open, so the diffs come from the tree instead.
A bare clone (`git clone --bare ... data/git/linux.git`) is ~4 GB and holds
every commit message ever written — which is what makes msgid -> commit
resolution possible at all, since merged patches carry
`Link: https://lore.kernel.org/all/<msgid>`.

Run git through `run()` so tests can substitute a fake without monkeypatching
subprocess wholesale.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

LORE_MSGID_RE = re.compile(r"lore\.kernel\.org/(?:all|lkml|[^/]+)/([^/\s>\"\[]+)")
MSGID_TAG_RE = re.compile(
    r"^\s*(?:Link|Message-Id|Message-ID|lore)\s*:\s*<?([^>\s]+)>?\s*$", re.M
)
FIELD_SEP = "\x1e"  # record separator between commits
RECORD_SEP = "\x1f"  # unit separator between fields


class NoRepo(RuntimeError):
    pass


@dataclass
class GitRepo:
    path: Path

    def exists(self) -> bool:
        return (self.path / "HEAD").exists() or (self.path / "objects").exists()

    def require(self) -> GitRepo:
        if not self.exists():
            raise NoRepo(
                f"{self.path} is not a git repo — run "
                f"`git clone --bare https://git.kernel.org/pub/scm/linux/kernel/git/"
                f"torvalds/linux.git {self.path}` first"
            )
        return self

    def run_bytes(self, *args: str) -> bytes:
        """Raw git stdout. Commit messages are *not* guaranteed to be valid
        UTF-8 (author names in Latin-1 etc.), so we decode with replace rather
        than letting subprocess raise on byte 0xf6 at minute one."""
        proc = subprocess.run(
            ["git", "-C", str(self.path), *args],
            capture_output=True,
            check=True,
        )
        return proc.stdout

    def run(self, *args: str) -> str:
        return self.run_bytes(*args).decode("utf-8", errors="replace")

    def all_commit_msgids(self) -> dict[str, list[str]]:
        """msgid -> [commit, ...] over every commit message in the tree.

        One `git log` pass, NUL-delimited, msgid tags pulled by regex. This is
        the expensive step (~minutes over 1.2M commits), so callers cache it.
        """
        self.require()
        fmt = f"%H{FIELD_SEP}%B{RECORD_SEP}"
        out = self.run("log", "--all", f"--format={fmt}")
        index: dict[str, list[str]] = {}
        for record in out.split(RECORD_SEP):
            record = record.strip("\n")
            if not record:
                continue
            commit, _, body = record.partition(FIELD_SEP)
            for msgid in msgids_in(body):
                commits = index.setdefault(msgid, [])
                if commit not in commits:
                    commits.append(commit)
        return index

    def all_commit_subjects(self) -> list[tuple[str, str, str]]:
        """[(commit, author-date ISO, subject)] for the whole tree.

        One `git log` pass; ~1.5M rows, so this is cached as JSON next to the
        msgid index rather than rebuilt per run.
        """
        self.require()
        fmt = f"%H{FIELD_SEP}%ad{FIELD_SEP}%s{RECORD_SEP}"
        out = self.run("log", "--all", "--date=iso-strict", f"--format={fmt}")
        rows = []
        for record in out.split(RECORD_SEP):
            record = record.strip("\n")
            if not record:
                continue
            commit, _, rest = record.partition(FIELD_SEP)
            date, _, subject = rest.partition(FIELD_SEP)
            rows.append((commit, date[:10], subject))
        return rows

    def patch_of(self, commit: str) -> str:
        """Full commit message + unified diff, as `git show` prints it."""
        self.require()
        return self.run("show", "--format=fuller", "--no-color", commit)

    def show_patch(self, commit: str) -> str:
        """Commit message + diff, without the `commit <sha>`/Merge: noise."""
        self.require()
        return self.run("show", "--format=full", "--no-color", commit)

    def subject_of(self, commit: str) -> str:
        self.require()
        return self.run("show", "--no-patch", "--format=%s", commit).strip()


def msgids_in(text: str) -> list[str]:
    """Every plausible lore msgid referenced in a commit message."""
    found: list[str] = []
    for match in MSGID_TAG_RE.finditer(text):
        value = match.group(1)
        msgid = value.rsplit("/", 1)[-1] if "lore.kernel.org" in value else value
        _add(found, msgid)
    for match in LORE_MSGID_RE.finditer(text):
        _add(found, match.group(1))
    return found


def _add(found: list[str], msgid: str) -> None:
    msgid = msgid.strip("<>").rstrip(".,;)")
    # a msgid has exactly one @ and no spaces; url fragments do not qualify
    if msgid.count("@") == 1 and " " not in msgid and msgid not in found:
        found.append(msgid)
