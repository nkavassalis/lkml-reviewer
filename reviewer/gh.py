"""Thin, defensive wrapper around the GitHub CLI (`gh`). Stdlib only.

Every failure becomes a `GhError` with a message a person can act on:
gh missing, not logged in, the PR or repo not found, no permission to post.
Nothing here posts anything unless `post_review` is called explicitly.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field

INSTALL_HINT = "install it from https://cli.github.com, or skip GitHub entirely with --diff-file / --local"
AUTH_HINT = "run `gh auth login` (or set GH_TOKEN), or skip GitHub entirely with --diff-file / --local"
TIMEOUT = 60


class GhError(RuntimeError):
    """A gh problem with an actionable message."""


@dataclass
class PullRequest:
    number: int
    title: str
    body: str
    url: str
    author: str
    base: str
    head: str
    files: list[str] = field(default_factory=list)
    diff: str = ""


def _gh(args: list[str], *, stdin: str | None = None, gh: str = "gh") -> subprocess.CompletedProcess:
    exe = shutil.which(gh)
    if exe is None:
        raise GhError(f"the GitHub CLI (`{gh}`) is not installed or not on PATH: {INSTALL_HINT}")
    try:
        return subprocess.run(
            [exe, *args], input=stdin, capture_output=True, text=True, timeout=TIMEOUT,
            env={**os.environ, "GH_PROMPT_DISABLED": "1", "NO_COLOR": "1"},
        )
    except subprocess.TimeoutExpired as e:
        raise GhError(f"`gh {' '.join(args[:2])}` timed out after {TIMEOUT}s (network?)") from e


def _explain(stderr: str, what: str) -> str:
    """Map gh's error text to something actionable."""
    s = stderr.strip()
    low = s.lower()
    if "gh auth login" in low or "not logged" in low or "authentication" in low or "401" in low:
        return f"gh is not authenticated: {AUTH_HINT}"
    if "could not resolve to a pullrequest" in low or ("not found" in low and "pull" in low):
        return f"{what}: no such pull request (check the number and --repo)"
    if "could not resolve to a repository" in low or "404" in low:
        return f"{what}: repository not found, or no access to it (check --repo owner/name)"
    if "no git remotes" in low or "not a git repository" in low or "none of the git remotes" in low:
        return f"{what}: not inside a GitHub checkout, so pass --repo owner/name"
    if "403" in low or "resource not accessible" in low or "must have" in low:
        return f"{what}: permission denied (your token can read but not write here?)"
    if "rate limit" in low:
        return f"{what}: GitHub API rate limit hit; wait and retry"
    return f"{what}: gh failed: {s.splitlines()[-1] if s else 'no error output'}"


def check_auth(*, gh: str = "gh") -> str:
    """Return the logged-in account (or 'GH_TOKEN'), or raise GhError."""
    if os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"):
        return "GH_TOKEN"
    r = _gh(["auth", "status"], gh=gh)
    if r.returncode != 0:
        raise GhError(f"gh is not authenticated: {AUTH_HINT}")
    out = r.stdout + r.stderr
    for line in out.splitlines():
        if "Logged in to" in line and "account" in line:
            return line.split("account", 1)[1].split()[0].strip("()")
    return "unknown account"


def _repo_args(repo: str | None) -> list[str]:
    return ["--repo", repo] if repo else []


def fetch_pr(number: int, repo: str | None = None, *, gh: str = "gh") -> PullRequest:
    """PR metadata + unified diff. Raises GhError with a remedy on any failure."""
    check_auth(gh=gh)
    what = f"PR #{number}" + (f" in {repo}" if repo else "")
    fields = "number,title,body,url,author,baseRefName,headRefName,files"
    r = _gh(["pr", "view", str(number), *_repo_args(repo), "--json", fields], gh=gh)
    if r.returncode != 0:
        raise GhError(_explain(r.stderr, what))
    try:
        meta = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise GhError(f"{what}: gh returned something that isn't JSON") from e
    d = _gh(["pr", "diff", str(number), *_repo_args(repo)], gh=gh)
    if d.returncode != 0:
        raise GhError(_explain(d.stderr, f"{what} (diff)"))
    return PullRequest(
        number=int(meta.get("number", number)),
        title=meta.get("title", ""),
        body=meta.get("body") or "",
        url=meta.get("url", ""),
        author=(meta.get("author") or {}).get("login", "unknown"),
        base=meta.get("baseRefName", ""),
        head=meta.get("headRefName", ""),
        files=[f.get("path", "") for f in meta.get("files") or []],
        diff=d.stdout,
    )


def post_review(number: int, body: str, repo: str | None = None, *, gh: str = "gh") -> None:
    """Post as a plain review *comment*: never approve, never request changes."""
    check_auth(gh=gh)
    what = f"posting to PR #{number}"
    r = _gh(["pr", "review", str(number), *_repo_args(repo), "--comment", "--body-file", "-"],
            stdin=body, gh=gh)
    if r.returncode != 0:
        raise GhError(_explain(r.stderr, what))
