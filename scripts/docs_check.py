"""Documentation drift check, run by scripts/gate.sh. Stdlib only.

Docs drift silently: a README that still calls an old version "current", a
plan that says "nothing is scheduled" after it was queued, a test count from
three sessions ago. These checks catch the mechanical part; AGENTS.md
"Documenting as we go" covers the rest.

  1. Paths and links in the docs exist (a renamed script leaves a dead
     reference). docs/LEARNINGS.md is history and may name removed files.
  2. "N hermetic tests" claims match what pytest collects.
  3. Every docs/PLAN-v<N>.md has a **Status:** line near the top with a
     known state, and v<N> has a row in README's journey table.
  4. HANDOFF.md's "Status (YYYY-MM-DD)" is no older than STALE_DAYS before
     the newest commit that touched code (handoffs rot first).
  5. docs/LEARNINGS.md entries are "## YYYY-MM-DD: title", in date order,
     each with **Evidence:** and **Changed:** lines (append-only log).
  6. Every "## Decision: ..." section in a docs/PLAN-*.md says what the next
     step optimizes (**Optimizes:**) and how the decision measures that
     (**Measured by:**). A rule that picked v5's starting model without
     measuring correctness, the thing v5 was for, is why (2026-11-14).

    python3 scripts/docs_check.py           # exit 1 on any failure
"""
from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STALE_DAYS = 1
PLAN_STATES = ("draft", "proposal", "queued", "running", "done", "superseded", "abandoned")
# Top-level directories whose backticked paths must exist. data/, runs/ and
# home paths are machine-local and are not checked.
CHECKED_DIRS = ("docs", "train", "scripts", "tests", "linuspr", "reviewer", "ft27b", "examples")
CODE_DIRS = ("train", "scripts", "linuspr", "reviewer", "ft27b")
HISTORY = {"docs/LEARNINGS.md"}


def tracked_docs() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True)
    files = out.stdout.split() if out.returncode == 0 else [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.md")]
    return [ROOT / f for f in files if not f.startswith((".venv", "data/"))]


def check_paths(docs: list[Path]) -> list[str]:
    errs = []
    dirs = "|".join(CHECKED_DIRS)
    pat = re.compile(rf"`((?:{dirs})/[A-Za-z0-9_./-]+)`")
    link = re.compile(r"\]\(((?!https?:|#|mailto:)[^)\s#]+)")
    for doc in docs:
        rel = str(doc.relative_to(ROOT))
        if rel in HISTORY:
            continue
        for n, line in enumerate(doc.read_text().splitlines(), 1):
            for m in pat.finditer(line):
                p = m.group(1).rstrip(".,:;")
                if any(c in p for c in "*{}<>") or p.endswith("/"):
                    continue
                if not (ROOT / p).exists():
                    errs.append(f"{rel}:{n}: `{p}` does not exist")
            for m in link.finditer(line):
                target = (doc.parent / m.group(1)).resolve()
                if not target.exists():
                    errs.append(f"{rel}:{n}: link to {m.group(1)} is dead")
    return errs


def collected_tests() -> int | None:
    py = ROOT / ".venv" / "bin" / "python"
    out = subprocess.run([str(py) if py.exists() else sys.executable, "-m", "pytest", "--collect-only", "-q"],
                         cwd=ROOT, capture_output=True, text=True)
    counts = re.findall(r"^\S+\.py: (\d+)$", out.stdout, re.M)
    if counts:
        return sum(map(int, counts))
    ids = [ln for ln in out.stdout.splitlines() if "::" in ln]
    return len(ids) or None


def check_test_counts(docs: list[Path]) -> list[str]:
    claims = []
    for doc in docs:
        if str(doc.relative_to(ROOT)) in HISTORY:
            continue
        for n, line in enumerate(doc.read_text().splitlines(), 1):
            if re.search(r"\(?20\d\d-\d\d-\d\d", line) or "this session" in line:
                continue                      # dated statements are history
            for m in re.finditer(r"\b(\d+) hermetic tests\b", line):
                claims.append((doc, n, int(m.group(1))))
    if not claims:
        return []
    actual = collected_tests()
    if actual is None:
        return ["could not count tests (pytest --collect-only failed)"]
    return [f"{d.relative_to(ROOT)}:{n}: says {c} hermetic tests, pytest collects {actual}"
            for d, n, c in claims if c != actual]


def check_plans() -> list[str]:
    errs = []
    readme = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""
    for plan in sorted((ROOT / "docs").glob("PLAN-v*.md")):
        v = re.match(r"PLAN-(v\d+)\.md", plan.name)
        if not v:
            continue
        head = "\n".join(plan.read_text().splitlines()[:15])
        m = re.search(r"\*\*Status:\*\*\s*([a-z]+)", head, re.I)
        if not m or m.group(1).lower() not in PLAN_STATES:
            errs.append(f"docs/{plan.name}: needs '**Status:** <{'|'.join(PLAN_STATES)}> ...' in its first 15 lines")
        if not re.search(rf"^\|\s*\*\*{v.group(1)}\*\*\s*\|", readme, re.M):
            errs.append(f"README.md: the journey table has no row for {v.group(1)} (docs/{plan.name} exists)")
    return errs


def check_handoff() -> list[str]:
    h = ROOT / "HANDOFF.md"
    if not h.exists():
        return []
    m = re.search(r"^## .*Status \((\d{4}-\d\d-\d\d|YYYY-MM-DD)\)", h.read_text(), re.M)
    if not m:
        return ["HANDOFF.md: needs a '## ... Status (YYYY-MM-DD)' heading"]
    if m.group(1) == "YYYY-MM-DD":
        return []                             # an unfilled template
    status = dt.date.fromisoformat(m.group(1))
    dirs = [d for d in CODE_DIRS if (ROOT / d).exists()]
    out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", *dirs], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        return []
    last_code = dt.date.fromisoformat(out.stdout.strip())
    if (last_code - status).days > STALE_DAYS:
        return [f"HANDOFF.md: status dated {status}, but code changed on {last_code}; "
                "rewrite the status section (where things stand now)"]
    return []


def check_learnings() -> list[str]:
    f = ROOT / "docs" / "LEARNINGS.md"
    if not f.exists():
        return ["docs/LEARNINGS.md is missing (the append-only log of what we learned)"]
    errs, last = [], None
    entries = re.split(r"^(?=## )", f.read_text(), flags=re.M)[1:]
    for e in entries:
        title = e.splitlines()[0]
        m = re.match(r"## (\d{4}-\d\d-\d\d): \S", title)
        if not m:
            errs.append(f"docs/LEARNINGS.md: heading '{title}' should be '## YYYY-MM-DD: title'")
            continue
        d = dt.date.fromisoformat(m.group(1))
        if last and d < last:
            errs.append(f"docs/LEARNINGS.md: '{title}' is out of date order (append new entries at the end)")
        last = d
        for field in ("**Evidence:**", "**Changed:**"):
            if field not in e:
                errs.append(f"docs/LEARNINGS.md: '{title}' has no {field} line")
    return errs


def check_decisions() -> list[str]:
    errs = []
    for plan in sorted((ROOT / "docs").glob("PLAN-*.md")):
        text = plan.read_text()
        for m in re.finditer(r"^(#{2,3}) Decision:.*$", text, re.M):
            level = len(m.group(1))
            nxt = re.compile(rf"^#{{1,{level}}} ", re.M).search(text, m.end())
            body = text[m.end(): nxt.start() if nxt else len(text)]
            for field in ("**Optimizes:**", "**Measured by:**"):
                if field not in body:
                    errs.append(f"docs/{plan.name}: '{m.group(0).strip()}' has no {field} line "
                                "(what the next step is for, and how this decision measures it)")
    return errs


def main() -> int:
    docs = tracked_docs()
    errs = check_paths(docs) + check_test_counts(docs) + check_plans() + check_handoff() + check_learnings() + check_decisions()
    for e in errs:
        print(f"docs: {e}")
    print(f"docs-check: {'FAIL' if errs else 'clean'} ({len(docs)} docs)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
