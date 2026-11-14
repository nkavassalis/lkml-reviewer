"""Pick v5's starting model from the multi-sample comparison (docs/PLAN-v5.md,
"Decision: which model v5 starts from"). The rule was written before the data existed:

    v4, unless v4 is clearly worse than v3 given the same patch-first input:
    its patch grounding is lower with the whole 95% CI below zero, or it
    loops more with the whole CI above zero. Then v3 (patch-first input).

    python3 scripts/q/choose_base.py v3pf-vs-v4.json      # prints v4 or v3, then reasons
"""
from __future__ import annotations

import json
import sys


def decide(res: dict) -> tuple[str, list[str]]:
    assert res["a"].startswith("v3") and res["b"] == "v4", "expects a = v3 patch-first, b = v4"
    g_mean, g_lo, g_hi = res["grounded in primary"]["diff"]
    l_mean, l_lo, l_hi = res["loops"]["diff"]
    reasons = [f"patch grounding, v4 minus v3: {g_mean:+.1%} (95% CI {g_lo:+.1%}..{g_hi:+.1%})",
               f"loops, v4 minus v3: {l_mean:+.1%} (95% CI {l_lo:+.1%}..{l_hi:+.1%})"]
    worse = []
    if g_hi < 0:
        worse.append("v4 grounds in the patch less, beyond noise")
    if l_lo > 0:
        worse.append("v4 loops more, beyond noise")
    if worse:
        return "v3", reasons + worse
    return "v4", reasons + ["v4 is not clearly worse: v5 starts from v4 (the default)"]


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        base, reasons = decide(json.load(f))
    print(base)
    print("\n".join(reasons))
