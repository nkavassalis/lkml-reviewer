"""Automate a decision only when the evidence agrees. Stdlib only.

A rule written before the data ("v4 unless it is clearly worse on X") picks an
option. This module then checks that choice against EVERY signal available,
not just the ones the rule was built on, and refuses to decide alone when:

  - a signal significantly favours another option (a conflict),
  - a TARGET signal (one measuring what the next step is for, --target) leans
    towards another option at all, significant or not: the next step is about
    that metric, so a rule that loses on it needs a person, or
  - a target or required signal is missing (no evidence is not agreement).

Then the step exits 76 and the improvement queue stops at a human gate
(train/workqueue.py), emailing this report. On lkml-reviewer the rule for v5's
starting model watched patch grounding and looping and would have picked v4,
while the blind judge, the only correctness signal, preferred v3 45-28. That
is p = 0.06 by a sign test, NOT significant at 0.05: a significance check
alone would have let it through. Hence --target.

    python3 train/decide.py --rule v4 --options v3,v4 \\
        --paired v3pf-vs-v4.json:"grounded in primary":higher \\
        --paired v3pf-vs-v4.json:loops:lower \\
        --judge judge-v4-vs-v3.jsonl:v3:v4 --target judge
    # exit 0: the evidence agrees with the rule; 76: a person must decide

--paired FILE:METRIC:higher|lower  a train/paired_stats.py --json result; the
    difference is b minus a, significant when its 95% CI excludes zero.
--judge FILE:BASE:TUNED  a judge.py output (final = base | tuned | tie |
    invalid); significant when a two-sided sign test on the decisive verdicts
    gives p < --alpha.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass

NEEDS_PERSON = 76


@dataclass
class Signal:
    name: str
    favors: str | None        # the option it points to beyond noise, None = no preference
    significant: bool
    detail: str
    lean: str | None = None   # the option its point estimate points to, significant or not


def sign_test_p(wins_a: int, wins_b: int) -> float:
    """Two-sided exact binomial (p = 0.5) on the decisive verdicts."""
    n, k = wins_a + wins_b, min(wins_a, wins_b)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_signal(res: dict, metric: str, better: str) -> Signal:
    """From paired_stats.py --json: diff = b - a with a 95% CI."""
    mean, lo, hi = res[metric]["diff"]
    a, b = res["a"], res["b"]
    sig = lo > 0 or hi < 0
    b_better = (mean > 0) == (better == "higher")
    lean = None if mean == 0 else (b if b_better else a)
    return Signal(metric, lean if sig else None, sig,
                  f"{b} minus {a}: {mean:+.3f} (95% CI {lo:+.3f}..{hi:+.3f})", lean)


def judge_signal(rows: list[dict], base: str, tuned: str, alpha: float = 0.05) -> Signal:
    t = {k: sum(r.get("final") == k for r in rows) for k in ("base", "tuned", "tie", "invalid")}
    p = sign_test_p(t["base"], t["tuned"])
    sig = p < alpha and t["base"] != t["tuned"]
    lean = None if t["base"] == t["tuned"] else (base if t["base"] > t["tuned"] else tuned)
    return Signal("judge", lean if sig else None, sig,
                  f"{base} {t['base']}, {tuned} {t['tuned']}, tie {t['tie']}, invalid {t['invalid']} "
                  f"(sign test p = {p:.3f})", lean)


def decide(rule_choice: str, signals: list[Signal], required: tuple[str, ...] = (),
           targets: tuple[str, ...] = ()) -> tuple[bool, list[str]]:
    """(automatic decision allowed, report lines)."""
    lines, ok = [f"rule's choice: {rule_choice}"], True
    have = {s.name for s in signals}
    for name in dict.fromkeys(required + targets):
        if name not in have:
            ok = False
            lines.append(f"MISSING: required signal '{name}' is not available; no evidence is not agreement")
    for s in signals:
        if s.significant and s.favors and s.favors != rule_choice:
            ok = False
            lines.append(f"CONFLICT: {s.name} favours {s.favors}: {s.detail}")
        elif s.name in targets and s.lean != rule_choice:
            ok = False
            lines.append(f"TARGET DISAGREES: {s.name} (what the next step is for) "
                         f"{'leans to ' + s.lean if s.lean else 'is a dead heat'}: {s.detail}")
        else:
            lines.append(f"agrees or inconclusive: {s.name}: {s.detail}")
    lines.append("evidence agrees: decided automatically" if ok
                 else "a person must decide (the rule's choice stands only if they confirm it)")
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rule", required=True, help="the option the pre-set rule chose")
    ap.add_argument("--paired", action="append", default=[], help="FILE:METRIC:higher|lower")
    ap.add_argument("--judge", action="append", default=[], help="FILE:BASE:TUNED")
    ap.add_argument("--require", action="append", default=[], help="signal names that must be present")
    ap.add_argument("--target", action="append", default=[],
                    help="signals measuring what the next step optimizes: must be present and lean to the rule's choice")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--alias", action="append", default=[],
                    help="LABEL=OPTION, when a file names an option differently (v3_pf=v3)")
    args = ap.parse_args(argv)
    alias = dict(a.split("=", 1) for a in args.alias)
    signals: list[Signal] = []
    for spec in args.paired:
        f, metric, better = spec.rsplit(":", 2)
        try:
            with open(f) as fh:
                signals.append(paired_signal(json.load(fh), metric, better))
        except (OSError, KeyError, ValueError) as e:
            print(f"(paired {spec}: unavailable: {e})")
    for spec in args.judge:
        f, base, tuned = spec.rsplit(":", 2)
        try:
            with open(f) as fh:
                signals.append(judge_signal([json.loads(x) for x in fh if x.strip()], base, tuned, args.alpha))
        except (OSError, ValueError) as e:
            print(f"(judge {spec}: unavailable: {e})")
    for sg in signals:
        sg.favors, sg.lean = alias.get(sg.favors, sg.favors), alias.get(sg.lean, sg.lean)
    ok, lines = decide(args.rule, signals, tuple(args.require), tuple(args.target))
    print("\n".join(lines))
    return 0 if ok else NEEDS_PERSON


if __name__ == "__main__":
    sys.exit(main())
