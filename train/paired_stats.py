"""Paired comparison of two models over several samples per input, with the
noise floor. Stdlib only.

One sample per input can't separate a real 10-point effect from luck: on
lkml-reviewer, two samples from the SAME model on byte-identical inputs
scored 46% vs 23% on a 13-input slice. So:

- each arm has K samples per input (compare*.py --samples K: KEY_s0..KEY_s{K-1});
- a metric is averaged over an arm's samples per input, then compared input by
  input (paired), with a bootstrap 95% CI over inputs;
- the **noise floor** compares an arm's sample 0 with its sample 1. Its true
  difference is zero, so its CI shows the smallest effect this evaluation can
  detect. A difference inside that band is not a finding.

    python3 train/paired_stats.py A.jsonl:v3_tf B.jsonl:v3_pf --primary-marker '--- The patch ---'
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections.abc import Callable

STRICT = re.compile(r"\b(?:[a-z_][a-z0-9]*_[a-z0-9_]+|[A-Za-z_]\w*\(\))")


def tokens(text: str) -> set[str]:
    return {m.group(0) for m in STRICT.finditer(text)}


def make_metrics(primary_marker: str | None, context_marker: str | None) -> dict[str, Callable]:
    def primary(inp: str) -> str:
        if not primary_marker or primary_marker not in inp:
            return inp
        p = inp.split(primary_marker, 1)[1]
        return p.split(context_marker, 1)[0] if context_marker and context_marker in p else p

    def loops(t: str) -> bool:
        lines = [ln.strip() for ln in t.splitlines() if len(ln.strip()) > 20]
        return any(lines.count(ln) >= 3 for ln in set(lines))

    return {
        "grounded in primary": lambda out, row: float(bool(tokens(out) & tokens(primary(row["input"])))),
        "length (chars)": lambda out, row: float(len(out)),
        "loops": lambda out, row: float(loops(out)),
    }


def load(spec: str) -> tuple[str, dict[int, dict], list[str]]:
    path, prefix = spec.rsplit(":", 1)
    with open(path) as f:
        rows = {r["eval_index"]: r for r in map(json.loads, f)}
    first = next(iter(rows.values()))
    keys = sorted(k for k in first if k == prefix or re.fullmatch(re.escape(prefix) + r"_s\d+", k))
    if not keys:
        raise SystemExit(f"{path}: no '{prefix}' or '{prefix}_sN' keys")
    return prefix, rows, keys


def bootstrap_ci(diffs: list[float], n: int = 10000, seed: int = 7) -> tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return statistics.fmean(diffs), means[int(0.025 * n)], means[int(0.975 * n)]


def per_input(rows, keys, fn, idx) -> list[float]:
    return [statistics.fmean(fn(rows[i][k], rows[i]) for k in keys) for i in idx]


def compare(a, b, metrics, only=None):
    (na, ra, ka), (nb, rb, kb) = a, b
    idx = sorted(set(ra) & set(rb))
    if only:
        idx = [i for i in idx if only(ra[i])]
    out = {"n": len(idx), "samples": (len(ka), len(kb))}
    for name, fn in metrics.items():
        xa, xb = per_input(ra, ka, fn, idx), per_input(rb, kb, fn, idx)
        out[name] = {na: statistics.fmean(xa), nb: statistics.fmean(xb),
                     "diff": bootstrap_ci([y - x for x, y in zip(xa, xb, strict=True)])}
        if len(ka) >= 2:  # noise floor: same arm, sample 0 vs sample 1
            s0, s1 = per_input(ra, ka[:1], fn, idx), per_input(ra, ka[1:2], fn, idx)
            out[name]["noise"] = bootstrap_ci([y - x for x, y in zip(s0, s1, strict=True)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="file.jsonl:KEY (the baseline)")
    ap.add_argument("b", help="file.jsonl:KEY (the candidate)")
    ap.add_argument("--primary-marker")
    ap.add_argument("--context-marker")
    args = ap.parse_args()
    metrics = make_metrics(args.primary_marker, args.context_marker)
    a, b = load(args.a), load(args.b)
    slices = {"all inputs": None}
    if args.context_marker:
        slices[f"with '{args.context_marker}'"] = lambda r: args.context_marker in r["input"]
    for label, only in slices.items():
        res = compare(a, b, metrics, only)
        print(f"\n## {label}: n={res['n']} inputs, samples per input {res['samples']}\n")
        print(f"| metric | {a[0]} | {b[0]} | difference (95% CI) | noise floor (95% CI) | real? |")
        print("|---|---|---|---|---|---|")
        for name in metrics:
            r = res[name]
            d, lo, hi = r["diff"]
            noise = r.get("noise")
            fmt = (lambda v: f"{v:.0f}") if "length" in name else (lambda v: f"{v:.0%}")
            ns = f"{fmt(noise[0])} ({fmt(noise[1])}..{fmt(noise[2])})" if noise else "n/a (1 sample)"
            real = "yes" if (lo > 0 or hi < 0) else "no: CI includes 0"
            print(f"| {name} | {fmt(r[a[0]])} | {fmt(r[b[0]])} | {fmt(d)} ({fmt(lo)}..{fmt(hi)}) | {ns} | {real} |")


if __name__ == "__main__":
    main()
