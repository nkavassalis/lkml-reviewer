"""train/judge_audit.py: a seeded reading pack, half flagged verdicts, half clean."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from judge_audit import pick, render  # noqa: E402


def test_pick_balances_flagged_and_clean_and_is_seeded():
    scores = ([{"eval_index": i, "key": "v3_sampled", "score": 2, "false_claims": 1} for i in range(10)]
              + [{"eval_index": i, "key": "v3_sampled", "score": 3, "false_claims": 0} for i in range(10, 20)]
              + [{"eval_index": 99, "key": "base_sampled", "score": 9, "false_claims": 0},
                 {"eval_index": 98, "key": "v3_sampled", "score": None}])
    a = pick(scores, ["v3_sampled"], 6, seed=1)
    assert len(a) == 6 and sum(s["false_claims"] > 0 for s in a) == 3
    assert all(s["key"] == "v3_sampled" and s["score"] is not None for s in a)
    assert a == pick(scores, ["v3_sampled"], 6, seed=1)


def test_render_shows_everything_needed_to_check_the_judge():
    rows = {1: {"input": "Subject: Re: [PATCH] x\n" + "y" * 3000, "reference": "Linus said", "v3_sampled": "review"}}
    md = render(rows, [{"eval_index": 1, "key": "v3_sampled", "score": 2, "false_claims": 1, "reason": "bad"}])
    for part in ("score 2/10, false claims 1: bad", "review", "Linus said", "[...]"):
        assert part in md
