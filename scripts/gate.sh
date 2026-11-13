#!/usr/bin/env bash
# gate.sh — the REAL-project push gate. Copy into <project>/scripts/gate.sh.
# The pre-push hook runs this; run it yourself before you hand anything back.
# Every step must pass. A step that cannot run must SKIP loudly, not silently.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
fail=0
step() { printf '\n=== %s ===\n' "$*"; }

step "unit / hermetic tier"
"$PY" -m pytest || fail=1

step "browser tier (Playwright + Chromium)"
# exit 5 == nothing carries the `ui` marker. That is a SKIP, not a failure —
# this project is a data pipeline with no web UI. A real failure still fails.
"$PY" -m pytest -m ui; rc=$?
[ $rc -eq 0 ] || [ $rc -eq 5 ] || fail=1
[ $rc -eq 5 ] && echo "SKIP — no tests marked ui (no browser tier in this project)"

step "lint"
if [ -x .venv/bin/ruff ]; then .venv/bin/ruff check . || fail=1
elif [ -f ruff.toml ] || grep -q '^\[tool.ruff' pyproject.toml 2>/dev/null; then
  ruff check . || fail=1
elif command -v ruff >/dev/null; then
  ruff check . || fail=1
else
  echo "SKIP — ruff not installed (add requirements-dev.txt to the venv)"
fi

step "syntax"
"$PY" -m compileall -q . -x '\.venv' || fail=1

step "docs (drift check: dead paths, stale counts, plan status, handoff date, learnings format)"
"$PY" scripts/docs_check.py || fail=1

step "secret scan (staged + branch vs origin/main)"
scripts/secret-scan.sh --staged || fail=1
git rev-parse --verify -q origin/main >/dev/null && { scripts/secret-scan.sh --range origin/main..HEAD || fail=1; }

printf '\n=== gate: %s ===\n' "$([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
