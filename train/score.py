"""Cheap, deterministic metrics over compare.jsonl. Stdlib + linuspr only.

These are proxies, not quality: they say whether the model *behaves* like the
reference (specific, terse, plain-text, not abusive), not whether its review
is technically right. Read the side-by-sides for that.

    PYTHONPATH=. .venv/bin/python train/score.py data/train/run-1/compare.jsonl
"""
from __future__ import annotations

import json
import re
import statistics

from linuspr.moderate import classify

IDENT_RE = re.compile(r"\b(?:[a-z_][a-z0-9]*_[a-z0-9_]+|[A-Za-z_]\w*\(\))")
ASSISTANT_RE = re.compile(
    r"(^\s*[-*]\s|\*\*|^#+\s|\bgreat (question|patch)\b|\bI hope\b|\blet me know\b|"
    r"\bcertainly\b|\boverall,|\bhere's (a|my)\b|\bsuggestions?:|\bin summary\b)",
    re.I | re.M,
)
ATTRIB_RE = re.compile(r"^\s*On .{0,120}wrote:", re.M)
# Anywhere, not just at the end: he often signs and then attaches a patch.
SIGN_RE = re.compile(r"^[^\S\n]*Linus[^\S\n]*$", re.M)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def loops(text: str) -> bool:
    """Same non-trivial line 3+ times: greedy decoding stuck in a loop."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
    return any(lines.count(ln) >= 3 for ln in set(lines))


def idents(text: str) -> set[str]:
    return {m.group(0).removesuffix("()") for m in IDENT_RE.finditer(text)}


def metrics(rows: list[dict], key: str) -> dict[str, float]:
    n = len(rows)
    out = [r[key] for r in rows]
    grounded = sum(bool(idents(o) & idents(r["input"])) for o, r in zip(out, rows, strict=True))
    return {
        "median length (chars)": statistics.median(len(o) for o in out),
        "names an identifier from the patch": grounded / n,
        "assistant-style formatting/phrases": sum(bool(ASSISTANT_RE.search(o)) for o in out) / n,
        "flagged by abuse filter (scrub|drop)": sum(classify(o).action != "keep" for o in out) / n,
        "starts with 'On ... wrote:'": sum(bool(ATTRIB_RE.match(o)) for o in out) / n,
        "signs 'Linus' (anywhere)": sum(bool(SIGN_RE.search(o)) for o in out) / n,
        "contains an email address": sum(bool(EMAIL_RE.search(o)) for o in out) / n,
        "repeats a line 3+ times (loop)": sum(loops(o) for o in out) / n,
    }


def read_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def targets_column(path: str) -> dict[str, float]:
    """The same metrics on the TRAINING targets: the rate the data teaches.
    A model rarely does something more often than its targets do, so this is
    the ceiling to read every model column against."""
    return metrics(read_rows(path), "output")


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="proxy metrics per arm; the training targets' rate alongside")
    ap.add_argument("compare", nargs="?", help="compare.py output (held-out rows, one column per arm)")
    ap.add_argument("--train", help="training file: adds a 'training targets' column (the data's rate)")
    ap.add_argument("--targets", help="training file alone: the metrics on its targets, nothing else")
    args = ap.parse_args(argv)
    if args.targets:
        col = targets_column(args.targets)
        print(f"training targets, n = {len(read_rows(args.targets))}\n\n| metric | training targets |\n|---|---|")
        for m, v in col.items():
            print(f"| {m} | {v:.0f} |" if m.startswith("median") else f"| {m} | {v:.0%} |")
        return
    rows = read_rows(args.compare)
    names = {"reference": "Linus (reference)", "base": "base, greedy",
             "tuned": "fine-tuned, greedy", "base_sampled": "base, sampled",
             "tuned_sampled": "fine-tuned, sampled"}
    # Known arms in a fixed order, then any others (v2, v3, v3_sampled, ...).
    extra = [k for k in rows[0] if k not in names and k not in ("input", "eval_index")]
    for k in extra:
        names[k] = k.replace("_sampled", ", sampled")
    keys = [k for k in names if k in rows[0]]
    cols = {k: metrics(rows, k) for k in keys}
    if args.train:   # the data's rate, next to Linus' and the models'
        names["training targets"] = "training targets"
        keys.insert(1, "training targets")
        cols["training targets"] = targets_column(args.train)
    print(f"n = {len(rows)} held-out pairs\n")
    print("| metric | " + " | ".join(names[k] for k in keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for m in cols["reference"]:
        cells = []
        for k in keys:
            v = cols[k][m]
            cells.append(f"{v:.0f}" if m.startswith("median") else f"{v:.0%}")
        print(f"| {m} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
