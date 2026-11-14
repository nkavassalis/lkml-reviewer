"""scripts/docs_check.py catches the drift it exists for (hermetic: tmp dirs)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "docs_check", Path(__file__).resolve().parent.parent / "scripts" / "docs_check.py")
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    return tmp_path


def test_plan_needs_status_and_a_journey_row(repo):
    (repo / "docs" / "PLAN-v7.md").write_text("# Plan: v7\n\nno status here\n")
    (repo / "README.md").write_text("| **v6** | x | y |\n")
    errs = dc.check_plans()
    assert any("Status" in e for e in errs) and any("no row for v7" in e for e in errs)
    (repo / "docs" / "PLAN-v7.md").write_text("# Plan: v7\n\n**Status:** queued (2026-11-13)\n")
    (repo / "README.md").write_text("| **v7** | x | y |\n")
    assert dc.check_plans() == []


def test_learnings_format_and_order(repo):
    f = repo / "docs" / "LEARNINGS.md"
    good = "## 2026-11-12: a\n**Evidence:** e\n**Changed:** c\n"
    f.write_text("# L\n\n" + good + "## 2026-11-11: older\n**Evidence:** e\n")
    errs = dc.check_learnings()
    assert any("out of date order" in e for e in errs) and any("**Changed:**" in e for e in errs)
    f.write_text("# L\n\n" + good)
    assert dc.check_learnings() == []


def test_dead_paths_and_links(repo):
    (repo / "train").mkdir()
    (repo / "train" / "real.py").write_text("")
    doc = repo / "docs" / "X.md"
    doc.write_text("see `train/real.py`, `train/gone.py`, `data/local.json`, `train/*.py`\n"
                   "and [ok](../train/real.py), [dead](missing.md), [web](https://example.com)\n")
    errs = dc.check_paths([doc])
    assert len(errs) == 2 and "train/gone.py" in errs[0] and "missing.md" in errs[1]


def test_undated_test_counts_must_match(repo, monkeypatch):
    monkeypatch.setattr(dc, "collected_tests", lambda: 42)
    doc = repo / "AGENTS.md"
    doc.write_text("pytest  # 41 hermetic tests\nVerified (2026-11-10): 12 hermetic tests\n")
    errs = dc.check_test_counts([doc])
    assert len(errs) == 1 and "says 41" in errs[0]


def test_decision_sections_say_what_they_optimize_and_how_they_measure_it(repo):
    plan = repo / "docs" / "PLAN-v7.md"
    plan.write_text("# Plan: v7\n\n## Decision: which model v7 starts from\n\nv6 unless worse.\n\n"
                    "## Costs\n\n**Optimizes:** not in the decision section, doesn't count\n")
    errs = dc.check_decisions()
    assert len(errs) == 2 and "Optimizes" in errs[0] and "Measured by" in errs[1]
    plan.write_text("# Plan: v7\n\n## Decision: which model v7 starts from\n\n"
                    "**Optimizes:** correctness\n\n### 2026-11-14: dated note\n\n"
                    "**Measured by:** the blind judge, as decide.py's target\n\n## Costs\n")
    assert dc.check_decisions() == []
