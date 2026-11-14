"""train/decide.py: automate a decision only when the evidence agrees."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from decide import NEEDS_PERSON, Signal, decide, judge_signal, main, paired_signal, sign_test_p  # noqa: E402

RES = {"a": "v3", "b": "v4",
       "grounded": {"diff": [0.10, 0.04, 0.16]},       # v4 grounds more, beyond noise
       "loops": {"diff": [0.01, -0.02, 0.03]}}         # no difference


def judge_rows(base: int, tuned: int, tie: int = 0) -> list[dict]:
    return [{"final": "base"}] * base + [{"final": "tuned"}] * tuned + [{"final": "tie"}] * tie


def test_sign_test():
    assert sign_test_p(10, 10) == 1.0
    assert sign_test_p(45, 28) == pytest.approx(0.0606, abs=0.002)
    assert sign_test_p(50, 20) < 0.001
    assert sign_test_p(0, 0) == 1.0


def test_paired_signal_direction():
    g = paired_signal(RES, "grounded", "higher")
    assert g.significant and g.favors == "v4"
    lp = paired_signal(RES, "loops", "lower")
    assert not lp.significant and lp.favors is None
    worse = {"a": "v3", "b": "v4", "loops": {"diff": [0.05, 0.02, 0.08]}}   # v4 loops more
    assert paired_signal(worse, "loops", "lower").favors == "v3"


def test_agreeing_evidence_decides_automatically():
    ok, lines = decide("v4", [paired_signal(RES, "grounded", "higher"), judge_signal(judge_rows(20, 40), "v3", "v4")])
    assert ok and lines[-1].startswith("evidence agrees")


def test_a_significant_dissenting_signal_needs_a_person():
    # what happened with v5's starting model, with a clearer judge result
    ok, lines = decide("v4", [paired_signal(RES, "grounded", "higher"), judge_signal(judge_rows(50, 20), "v3", "v4")])
    assert not ok and any(line.startswith("CONFLICT: judge favours v3") for line in lines)


def test_an_inconclusive_dissent_does_not_block():
    ok, _ = decide("v4", [Signal("x", None, False, "inside the noise")])
    assert ok


def test_a_missing_required_signal_needs_a_person():
    ok, lines = decide("v4", [paired_signal(RES, "grounded", "higher")], required=("judge",))
    assert not ok and any("MISSING" in line for line in lines)


def test_cli_exit_codes(tmp_path):
    (tmp_path / "res.json").write_text(json.dumps(RES))
    (tmp_path / "j.jsonl").write_text("\n".join(json.dumps(r) for r in judge_rows(50, 20)))
    base = ["--rule", "v4", "--paired", f"{tmp_path}/res.json:grounded:higher"]
    assert main(base) == 0
    assert main(base + ["--judge", f"{tmp_path}/j.jsonl:v3:v4"]) == NEEDS_PERSON
    assert main(base + ["--require", "judge"]) == NEEDS_PERSON
    assert main(base + ["--judge", f"{tmp_path}/missing.jsonl:v3:v4", "--require", "judge"]) == NEEDS_PERSON


def test_the_real_case_needs_the_target_rule():
    # 45-28 is p = 0.06: a significance test alone lets it through ...
    j = judge_signal(judge_rows(45, 28, 18), "v3", "v4")
    assert not j.significant and j.lean == "v3"
    assert decide("v4", [paired_signal(RES, "grounded", "higher"), j])[0]
    # ... declaring the judge the target (v5 is for correctness) stops it.
    ok, lines = decide("v4", [paired_signal(RES, "grounded", "higher"), j], targets=("judge",))
    assert not ok and any(line.startswith("TARGET DISAGREES: judge") for line in lines)


def test_a_target_that_agrees_passes_and_a_missing_one_blocks():
    assert decide("v4", [judge_signal(judge_rows(28, 45), "v3", "v4")], targets=("judge",))[0]
    assert not decide("v4", [], targets=("judge",))[0]
    assert not decide("v4", [judge_signal(judge_rows(30, 30), "v3", "v4")], targets=("judge",))[0]


def test_aliases_map_file_labels_to_options(tmp_path):
    res = {"a": "v3_pf", "b": "v4", "g": {"diff": [-0.1, -0.15, -0.05]}}   # v3_pf grounds more
    (tmp_path / "r.json").write_text(json.dumps(res))
    spec = ["--rule", "v3", "--paired", f"{tmp_path}/r.json:g:higher"]
    assert main(spec) == NEEDS_PERSON                  # "v3_pf" != "v3": a false conflict
    assert main(spec + ["--alias", "v3_pf=v3"]) == 0
