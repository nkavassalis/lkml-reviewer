"""v5's starting-model rule (scripts/q/choose_base.py), set before the data."""
from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "choose_base", Path(__file__).resolve().parent.parent / "scripts" / "q" / "choose_base.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)


def res(g, lo, hi, loops=(0.0, -0.01, 0.01)):
    return {"a": "v3_pf", "b": "v4", "grounded in primary": {"diff": [g, lo, hi]}, "loops": {"diff": list(loops)}}


def test_default_is_v4_even_when_slightly_worse_within_noise():
    assert cb.decide(res(-0.04, -0.09, 0.01))[0] == "v4"


def test_v3_when_v4_clearly_grounds_less():
    base, why = cb.decide(res(-0.08, -0.13, -0.03))
    assert base == "v3" and any("grounds in the patch less" in w for w in why)


def test_v3_when_v4_clearly_loops_more():
    assert cb.decide(res(0.02, -0.03, 0.07, loops=(0.05, 0.02, 0.08)))[0] == "v3"


def test_better_v4_stays_v4():
    assert cb.decide(res(0.10, 0.05, 0.15))[0] == "v4"
