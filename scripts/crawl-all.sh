#!/usr/bin/env bash
# Crawl every bucket the mirror publishes (1996 .. current). Resumable: pages
# are cached under data/raw and the corpus merges per bucket, so re-running
# after an interruption costs only the missing pages.
#
# This box has no systemd — an orphaned process gets killed, so start this with
# nohup from a shell that stays open, or from the login bootstrap:
#   nohup scripts/crawl-all.sh > data/crawl-all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
"$PY" -u -m linuspr crawl --patch-only --delay "${DELAY:-0.2}"
"$PY" -u -m linuspr thread  --delay "${DELAY:-0.2}"
"$PY" -u -m linuspr pairs
"$PY" -u -m linuspr resolve
"$PY" -u -m linuspr progress
