"""Preference pairs for DPO (docs/PLAN-v5.md, step 2). Stdlib only. DRAFT: not run.

Input: compare.py output on TRAINING patches with K samples per patch
(`--samples K`: keys <arm>_sampled_s0..s{K-1}). The judge scores every
candidate against Linus' real reply (accuracy first), then each patch yields
one (chosen, rejected) pair: the best and worst candidates, kept only when the
judge separates them by at least --min-gap. Ties, unparseable verdicts and
judge failures drop the patch; nothing is guessed.

    JUDGE_URL=... python3 make_preferences.py score cands.jsonl --arm v4 --out scores.jsonl
    python3 make_preferences.py pairs cands.jsonl scores.jsonl --min-gap 3 --out pairs.jsonl

`score` is resumable (it skips candidates already scored).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = os.environ.get("JUDGE_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("JUDGE_MODEL", "qwen3.8-flash-next")
PROMPT = """You are grading one code review of a Linux kernel patch.

PATCH (may be truncated):
<<<
{patch}
>>>

REFERENCE: the review the upstream maintainer actually wrote:
<<<
{reference}
>>>

CANDIDATE:
<<<
{candidate}
>>>

Score the CANDIDATE from 1 to 10. Judge, in order:
1. accuracy: every false claim about the code costs heavily;
2. overlap with the substantive points the reference makes;
3. specificity to this patch rather than generic advice.
Ignore tone, rudeness, formatting and length. Repetition or invented facts
score low. Saying little is not a virtue: a review with no substance scores
low even if it contains no false claims.

Answer with one JSON object on the last line:
{{"score": <1-10>, "false_claims": <count>, "reason": "<one sentence>"}}"""


def ask(patch: str, reference: str, candidate: str) -> dict:
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 24000,
                       "messages": [{"role": "user", "content": PROMPT.format(
                           patch=patch[:6000], reference=reference[:3000],
                           candidate=candidate[:5000])}]}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            text = json.load(resp)["choices"][0]["message"]["content"] or ""
        found = re.findall(r"\{[^{}]*\"score\"[^{}]*\}", text)
        v = json.loads(found[-1])
        v["score"] = float(v["score"])
        return v
    except Exception as e:  # noqa: BLE001  (a failed verdict drops the candidate)
        return {"score": None, "error": type(e).__name__}


def candidates(row: dict, arm: str) -> list[tuple[str, str]]:
    keys = sorted(k for k in row if re.fullmatch(rf"{re.escape(arm)}_sampled_s\d+", k))
    return [(k, row[k]) for k in keys]


def cmd_score(args) -> None:
    with open(args.cands) as f:
        rows = [json.loads(line) for line in f]
    done = set()
    if Path(args.out).exists():
        with open(args.out) as f:
            done = {(r["eval_index"], r["key"]) for r in map(json.loads, f)}
    jobs = [(r, k, t) for r in rows for k, t in candidates(r, args.arm) if (r["eval_index"], k) not in done]
    print(f"{len(jobs)} candidates to score ({len(done)} already done)", flush=True)

    def one(job):
        r, k, t = job
        return {"eval_index": r["eval_index"], "key": k, **ask(r["input"], r["reference"], t)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(args.out, "a") as out:
        for n, res in enumerate(pool.map(one, jobs), 1):
            out.write(json.dumps(res) + "\n")
            out.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(jobs)} scored", flush=True)


def cmd_pairs(args) -> None:
    with open(args.cands) as f:
        rows = {r["eval_index"]: r for r in map(json.loads, f)}
    scores: dict[int, dict[str, float]] = {}
    with open(args.scores) as f:
        for s in map(json.loads, f):
            if s.get("score") is not None:
                scores.setdefault(s["eval_index"], {})[s["key"]] = s["score"]
    kept = dropped = 0
    with open(args.out, "w") as out:
        for i, row in rows.items():
            sc = scores.get(i, {})
            if len(sc) < 2:
                dropped += 1
                continue
            best = max(sc, key=sc.get)
            worst = min(sc, key=sc.get)
            if sc[best] - sc[worst] < args.min_gap:
                dropped += 1
                continue
            out.write(json.dumps({"eval_index": i, "input": row["input"], "chosen": row[best],
                                  "rejected": row[worst], "chosen_score": sc[best],
                                  "rejected_score": sc[worst]}) + "\n")
            kept += 1
    print(f"{kept} pairs kept, {dropped} patches dropped (fewer than 2 scores, or gap < {args.min_gap})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("cands")
    s.add_argument("--arm", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("pairs")
    p.add_argument("cands")
    p.add_argument("scores")
    p.add_argument("--min-gap", type=float, default=3.0)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"score": cmd_score, "pairs": cmd_pairs}[args.cmd](args)


if __name__ == "__main__":
    main()
