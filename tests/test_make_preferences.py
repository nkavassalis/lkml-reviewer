"""make_preferences.py score: a dead judge fails the step and a rerun retries;
an unparseable verdict is recorded (and its patch dropped from pairs)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import make_preferences as mp  # noqa: E402


def cands(tmp_path):
    p = tmp_path / "cands.jsonl"
    p.write_text("".join(json.dumps({"eval_index": i, "input": "patch", "reference": "ref",
                                     **{f"cand_sampled_s{k}": f"review {k}" for k in range(4)}}) + "\n"
                         for i in range(3)))
    return p


def score(c, out):
    sys.argv = ["make_preferences.py", "score", str(c), "--arm", "cand", "--out", str(out), "--workers", "2"]
    mp.main()


def test_a_dead_judge_fails_and_a_rerun_retries(tmp_path, monkeypatch):
    c, out = cands(tmp_path), tmp_path / "scores.jsonl"
    monkeypatch.setattr(mp, "ask", lambda *a: {"score": None, "error": "URLError"})
    with pytest.raises(SystemExit, match="12/12 verdicts failed"):
        score(c, out)
    monkeypatch.setattr(mp, "ask", lambda p, r, t: {"score": float(t[-1]) * 3, "false_claims": 0})
    score(c, out)
    assert sum(json.loads(x).get("score") is not None for x in out.read_text().splitlines()) == 12


def test_an_unparseable_verdict_is_recorded(tmp_path, monkeypatch):
    c, out = cands(tmp_path), tmp_path / "scores.jsonl"
    monkeypatch.setattr(mp, "ask", lambda *a: {"score": None, "error": "IndexError"})
    score(c, out)
    assert len(out.read_text().splitlines()) == 12
