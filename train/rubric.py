"""Score models with the NEXT step's own yardstick. Stdlib only.

A decision between models ("which one does preference training start from?")
should be measured with the metric that step will optimize, not a proxy. For
preference training that is the pointwise rubric that scores its candidates:
accuracy first, then overlap with the reference, then specificity, and
"saying little is not a virtue". A pairwise judge without that last clause
rewards cautious, vague reviews (lkml-reviewer, 2026-11-14), so it measures
something different.

    JUDGE_URL=... python3 rubric.py score compare-all.jsonl --arms v3_sampled,v4_sampled --out rubric.jsonl
    python3 rubric.py paired compare-all.jsonl rubric.jsonl --a v3_sampled:v3 --b v4_sampled:v4 \\
        --json rubric-v3-vs-v4.json            # then: decide.py --paired ...:"rubric score":higher --target "rubric score"

The scorer is make_preferences.ask (same prompt, same parsing) when that
module is next to this one, so the decision and the training pairs use one
yardstick; otherwise the copy below. Keep the two prompts identical.
`score` is resumable. `paired` writes paired_stats.py's --json format (b minus
a, bootstrap 95% CI) for "rubric score" (higher is better) and "false claims"
(lower is better), over rows where both arms were scored.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_stats import bootstrap_ci  # noqa: E402

URL = os.environ.get("JUDGE_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("JUDGE_MODEL", "qwen3.8-flash-next")
PROMPT = """You are grading one code review of a patch.

PATCH (may be truncated):
<<<
{patch}
>>>

REFERENCE: the review the maintainer actually wrote:
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


def _ask(patch: str, reference: str, candidate: str) -> dict:
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 24000,
                       "messages": [{"role": "user", "content": PROMPT.format(
                           patch=patch[:6000], reference=reference[:3000], candidate=candidate[:5000])}]}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            text = json.load(resp)["choices"][0]["message"]["content"] or ""
        v = json.loads(re.findall(r"\{[^{}]*\"score\"[^{}]*\}", text)[-1])
        v["score"] = float(v["score"])
        return v
    except Exception as e:  # noqa: BLE001  (a failed verdict drops the row)
        return {"score": None, "error": type(e).__name__}


try:                                   # the preference scorer itself, if this project has one
    from make_preferences import ask  # noqa: E402
    SOURCE = "make_preferences.ask"
except ImportError:
    ask = _ask
    SOURCE = "rubric.py's copy"


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def cmd_score(args) -> None:
    rows = read_jsonl(args.compare)
    arms = args.arms.split(",")
    done = set()
    if Path(args.out).exists():
        done = {(r["eval_index"], r["key"]) for r in read_jsonl(args.out)}
    jobs = [(r, a) for r in rows for a in arms if a in r and (r["eval_index"], a) not in done]
    print(f"{len(jobs)} reviews to score with {SOURCE} ({len(done)} already done)", flush=True)

    def one(job):
        r, a = job
        return {"eval_index": r["eval_index"], "key": a, **ask(r["input"], r["reference"], r[a])}

    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(args.out, "a") as out:
        for n, res in enumerate(pool.map(one, jobs), 1):
            out.write(json.dumps(res) + "\n")
            out.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(jobs)} scored", flush=True)


def paired(scores: list[dict], a: str, b: str) -> dict:
    """paired_stats --json shaped result for arms a and b (keys in `scores`)."""
    by: dict[int, dict[str, dict]] = {}
    for s in scores:
        if s.get("score") is not None:
            by.setdefault(s["eval_index"], {})[s["key"]] = s
    both = [v for v in by.values() if a in v and b in v]
    out: dict = {"n": len(both)}
    for metric, field in (("rubric score", "score"), ("false claims", "false_claims")):
        xa = [float(v[a].get(field) or 0) for v in both]
        xb = [float(v[b].get(field) or 0) for v in both]
        if not both:
            continue
        out[metric] = {"a": statistics.fmean(xa), "b": statistics.fmean(xb),
                       "diff": list(bootstrap_ci([y - x for x, y in zip(xa, xb, strict=True)]))}
    return out


def cmd_paired(args) -> None:
    scores = read_jsonl(args.scores)
    (ka, la), (kb, lb) = (s.split(":", 1) if ":" in s else (s, s) for s in (args.a, args.b))
    res = {"a": la, "b": lb, **paired(scores, ka, kb)}
    print(f"rubric ({SOURCE}), {res['n']} patches scored for both\n")
    print(f"| metric | {la} | {lb} | {lb} minus {la} (95% CI) |\n|---|---|---|---|")
    for m in ("rubric score", "false claims"):
        if m in res:
            d, lo, hi = res[m]["diff"]
            print(f"| {m} | {res[m]['a']:.2f} | {res[m]['b']:.2f} | {d:+.2f} ({lo:+.2f}..{hi:+.2f}) |")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("compare")
    s.add_argument("--arms", required=True, help="comma-separated keys, e.g. v3_sampled,v4_sampled")
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("paired")
    p.add_argument("compare")
    p.add_argument("scores")
    p.add_argument("--a", required=True, help="KEY[:LABEL]")
    p.add_argument("--b", required=True, help="KEY[:LABEL]")
    p.add_argument("--json")
    args = ap.parse_args(argv)
    {"score": cmd_score, "paired": cmd_paired}[args.cmd](args)


if __name__ == "__main__":
    main()
