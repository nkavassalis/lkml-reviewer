"""A reading pack for checking the judge by eye. Stdlib only.

The planted-wrong check (judge_check.py) only shows the judge catches obvious
errors. Whether its "false claims" on subtle points are right needs a person.
This picks N rubric verdicts (half that report false claims, half that
don't), seeded, and lays each out: the patch, the maintainer's real reply, the
review, and the judge's score, false-claim count and reason.

    python3 judge_audit.py compare-all.jsonl rubric.jsonl --arms v3_sampled,v4_sampled -n 10 > AUDIT.md

For each one, ask: is every claimed false claim really false? Did it miss a
false claim? Would you give roughly that score?
"""
from __future__ import annotations

import argparse
import json
import random


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def pick(scores: list[dict], arms: list[str], n: int, seed: int) -> list[dict]:
    ok = [s for s in scores if s.get("score") is not None and s["key"] in arms]
    flagged = [s for s in ok if (s.get("false_claims") or 0) > 0]
    clean = [s for s in ok if not (s.get("false_claims") or 0)]
    rng = random.Random(seed)
    rng.shuffle(flagged)
    rng.shuffle(clean)
    half = n // 2
    chosen = flagged[: n - min(half, len(clean))] + clean[:half]
    return chosen[:n]


def render(rows: dict[int, dict], chosen: list[dict], patch_chars: int = 2500) -> str:
    out = ["# Judge audit: is the rubric judge right?", "",
           "For each review: is every claimed false claim really false? Did the judge miss one?",
           "Would you give roughly that score (1-10; accuracy, then overlap with the reply, then specificity)?",
           "Note your verdict per item (agree / too harsh / too lenient / wrong about a claim).", ""]
    for i, s in enumerate(chosen, 1):
        r = rows[s["eval_index"]]
        patch = r["input"]
        cut = len(patch) > patch_chars
        note = f" and thread, first {patch_chars} chars" if cut else ""
        out += [f"## {i}. {patch.splitlines()[0][:100]}", "",
                f"**Judge:** score {s['score']:.0f}/10, false claims {s.get('false_claims', '?')}: "
                f"{s.get('reason', '')}", "",
                f"**Review being judged** ({s['key']}):", "", "```", r[s["key"]].strip(), "```", "",
                "**The maintainer's real reply:**", "", "```", r["reference"].strip(), "```", "",
                f"**The input (patch{note}):**", "",
                "```", patch[:patch_chars].strip() + ("\n[...]" if cut else ""), "```", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("compare")
    ap.add_argument("scores")
    ap.add_argument("--arms", required=True)
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)
    rows = {r["eval_index"]: r for r in read_jsonl(args.compare)}
    chosen = pick(read_jsonl(args.scores), args.arms.split(","), args.n, args.seed)
    print(render(rows, chosen))


if __name__ == "__main__":
    main()
