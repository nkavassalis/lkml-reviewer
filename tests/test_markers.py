"""Pass markers (`*.ok`) are made with `touch`: empty files. Checking one with
`-s` (exists AND non-empty) can never succeed. lkml-reviewer, 2026-11-15: the
DPO smoke test passed, and the queue step reported it failed, which stopped the
queue for a night."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_marker_is_checked_for_size():
    bad = [f"{p.relative_to(ROOT)}:{n}" for p in (ROOT / "scripts").rglob("*.sh")
           for n, line in enumerate(p.read_text().splitlines(), 1)
           if re.search(r"-s\s+\S*\.ok\b", line)]
    assert not bad, f"use -f for touch-made markers: {bad}"
