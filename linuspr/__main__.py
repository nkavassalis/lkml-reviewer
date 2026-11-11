"""CLI: `python -m linuspr months | crawl | stats`.

Examples
--------
    python -m linuspr months
    python -m linuspr crawl --month 2406.0 --month 2407.0
    python -m linuspr crawl --last 12
    python -m linuspr pairs
    python -m linuspr sample -n 3
    python -m linuspr export
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .fetch import Fetcher
from .pairs import build_pairs
from .pipeline import (
    bucket_year,
    crawl_months,
    list_months,
    read_jsonl,
    write_jsonl,
)

DATA = Path(__file__).resolve().parent.parent / "data"
GIT = DATA / "git" / "linux.git"
MSGID_INDEX = DATA / "msgid_index.json"
RAW = DATA / "raw"
CORPUS = DATA / "linus_messages.jsonl"
COMMIT_SUBJECTS = DATA / "commit_subjects.json"
PATCHES = DATA / "patch_texts.json"
THREADS = DATA / "thread_context.json"
PAIRS = DATA / "review_pairs.jsonl"

_ATTR_RE = re.compile(r"^On .{10,120} wrote:$", re.M)
_QUOTE_PREFIX_RE = re.compile(r"^>\s?", re.M)
_HUNK_RE = re.compile(r"^@@ -\d+", re.M)
_DIFF_RE = re.compile(r"^diff --git ", re.M)
_TECH_RE = re.compile(
    r"\b(alloc|lock|race|abi|uapi|regression|revert|layering|abstraction|"
    r"interface|overhead|performance|cache|atomic|leak|use-after-free|"
    r"null pointer|segfault|panic|oops|deadlock|lifetime|refcount|"
    r"userspace|kernel|syscall|driver|subsystem|maintainer|tree|patchset)\b",
    re.I,
)


def _expand_years(spec: str) -> set[int]:
    a, _, b = spec.partition("-")
    return set(range(int(a), int(b or a) + 1))


def cmd_months(args: argparse.Namespace) -> int:
    f = Fetcher(cache_dir=RAW)
    months = list_months(f)
    print(f"{len(months)} months: {months[0]} .. {months[-1]}")
    return 0


def cmd_crawl(args: argparse.Namespace) -> int:
    f = Fetcher(cache_dir=RAW, delay=args.delay)
    months = list(args.month or [])
    if not months:
        all_months = list_months(f)
        if args.years:
            years = {y for spec in args.years for y in _expand_years(spec)}
            all_months = [m for m in all_months if bucket_year(m) in years]
        if args.last:
            all_months = all_months[-args.last :]
        months = all_months
    print(f"crawling {len(months)} months ({months[0]}..{months[-1]})")
    corpora = crawl_months(f, months, patch_only=args.patch_only)
    n = write_jsonl(corpora, CORPUS)
    total = sum(c.total_messages for c in corpora)
    print(f"messages seen {total}, linus messages {n}, requests {f.requests_made}")
    print(f"wrote {CORPUS}")
    return 0


def cmd_thread(args: argparse.Namespace) -> int:
    """Walk each reply's thread back to the posting that carries the diff."""
    from .pairs import patch_key
    from .pipeline import read_jsonl as _r
    from .resolve import ThreadMap, find_patch_for_message

    msgs = _r(CORPUS)
    if not msgs:
        print("no corpus yet — run `crawl` first", file=sys.stderr)
        return 1
    f = Fetcher(cache_dir=RAW, delay=args.delay)
    patches = json.loads(PATCHES.read_text(encoding="utf-8")) if PATCHES.exists() else {}
    threads = ThreadMap(f)
    found = 0
    for i, m in enumerate(msgs, 1):
        if patch_key(m) in patches:
            found += 1
            continue
        text, _mid = find_patch_for_message(m, f, threads, max_hops=args.max_hops)
        if text:
            patches[patch_key(m)] = text
            found += 1
        if i % 100 == 0:
            print(f"  {i}/{len(msgs)} messages, {found} patches recovered, "
                  f"{f.requests_made} requests")
            PATCHES.write_text(json.dumps(patches, ensure_ascii=False), encoding="utf-8")
    PATCHES.write_text(json.dumps(patches, ensure_ascii=False), encoding="utf-8")
    print(f"patches recovered for {found}/{len(msgs)} Linus messages -> {PATCHES}")
    return 0


def cmd_pairs(args: argparse.Namespace) -> int:
    msgs = read_jsonl(CORPUS)
    if not msgs:
        print("no corpus yet — run `crawl` first", file=sys.stderr)
        return 1
    patches = {}
    if PATCHES.exists():
        patches = json.loads(PATCHES.read_text(encoding="utf-8"))
    pairs = build_pairs(msgs, patches=patches, legacy=args.legacy_gate)
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with PAIRS.open("w", encoding="utf-8") as fh:
        for pr in pairs:
            fh.write(json.dumps(asdict(pr), ensure_ascii=False) + "\n")
    n = len(pairs)
    with_diff = sum(1 for pr in pairs if pr.has_diff)
    with_patch = sum(1 for pr in pairs if pr.resolved_context)
    named = sum(1 for pr in pairs if pr.authors)
    rev_len = sorted(len(pr.review) for pr in pairs) or [0]
    ctx_len = sorted(len(pr.context) for pr in pairs) or [0]
    print(f"messages {len(msgs)} -> pairs {n} ({100.0*n/len(msgs):.1f}%)")
    print(f"  context has a real diff  {with_diff:5d} ({100.0*with_diff/max(n,1):.1f}%)")
    print(f"  names the author quoted  {named:5d} ({100.0*named/max(n,1):.1f}%)")
    print(f"  review len  median {rev_len[len(rev_len)//2]}, p90 {rev_len[int(len(rev_len)*.9)]}")
    print(f"  with a resolved patch      {with_patch:5d} ({100.0*with_patch/max(n,1):.1f}%)")
    print(f"  context len median {ctx_len[len(ctx_len)//2]}, p90 {ctx_len[int(len(ctx_len)*.9)]}")
    print(f"wrote {PAIRS}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Write the chat/SFT-ready JSONL: {"input": ..., "output": ...}."""
    pairs = _load_pairs()
    if not pairs:
        print("no pairs yet — run `pairs` first", file=sys.stderr)
        return 1
    from .moderate import filter_pairs

    tally = {"keep": len(pairs), "scrub": 0, "drop": 0}
    if not args.no_moderate:
        pairs, tally = filter_pairs(pairs)
        print(
            f"moderation: kept {tally['keep']}, scrubbed {tally['scrub']}, "
            f"dropped {tally['drop']}"
        )
    from .split import split_by_year

    if THREADS.exists() and not args.no_thread:
        threads = json.loads(THREADS.read_text(encoding="utf-8"))
        for pr in pairs:
            pr.thread_context = threads.get(f"{pr.month}/{pr.mid}", "")
        print(f"thread context attached to {sum(bool(p.thread_context) for p in pairs)}"
              f"/{len(pairs)} pairs")
    split = split_by_year(pairs, eval_years=tuple(args.eval_year))
    train, held = split.train, split.eval
    if split.undated:
        print(
            f"WARNING: {len(split.undated)} undated pairs quarantined "
            "(not written to either side)"
        )
    for held_out, path in ((False, DATA / "sft.jsonl"), (True, DATA / "sft_eval.jsonl")):
        rows = (held if held_out else train)[: args.limit]
        with path.open("w", encoding="utf-8") as fh:
            for pr in rows:
                fh.write(json.dumps(pr.as_sft(keep_signoff=args.keep_signoff),
                                    ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} pairs to {path}")
    print(f"eval years {list(args.eval_year)} -> {len(held)} held-out pairs")
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    """Collect earlier thread messages for every pair (v3). Resumable."""
    from .context import ContextBuilder

    pairs = _load_pairs()
    if not pairs:
        print("no pairs yet — run `pairs` first", file=sys.stderr)
        return 1
    f = Fetcher(cache_dir=RAW, delay=args.delay)
    cb = ContextBuilder(f)
    if args.dry_run:
        need = sum(cb.uncached(p.month, p.subject, p.mid) for p in pairs)
        print(f"{len(pairs)} pairs; {need} thread pages not cached "
              f"(~{need * max(args.delay, 0.2) / 60:.0f} min of polite fetching, "
              "before network latency)")
        return 0
    threads = json.loads(THREADS.read_text(encoding="utf-8")) if THREADS.exists() else {}
    for i, p in enumerate(pairs, 1):
        key = f"{p.month}/{p.mid}"
        if key not in threads:
            threads[key] = cb.build(p.month, p.subject, p.mid)
        if i % 200 == 0:
            THREADS.write_text(json.dumps(threads, ensure_ascii=False), encoding="utf-8")
            print(f"  {i}/{len(pairs)} pairs, {f.requests_made} requests", flush=True)
    THREADS.write_text(json.dumps(threads, ensure_ascii=False), encoding="utf-8")
    have = sum(bool(v) for v in threads.values())
    print(f"thread context for {have}/{len(pairs)} pairs -> {THREADS}")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """Eyeball a deterministic slice of the corpus (seeded, so reviews repeat)."""
    import random

    pairs = _load_pairs()
    if not pairs:
        print("no pairs yet — run `pairs` first", file=sys.stderr)
        return 1
    random.Random(args.seed).shuffle(pairs)
    for pr in pairs[: args.n]:
        print("=" * 72)
        print(f"[{pr.month}/{pr.mid}] {pr.subject}  diff={pr.has_diff}")
        print("--- context ---")
        print(pr.context[: args.width])
        print("--- linus ---")
        print(pr.review[: args.width])
    return 0


def _load_pairs():
    from dataclasses import fields

    from .pairs import ReviewPair

    if not PAIRS.exists():
        return []
    known = {f.name for f in fields(ReviewPair)}
    return [
        ReviewPair(**{k: v for k, v in json.loads(ln).items() if k in known})
        for ln in PAIRS.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def cmd_index(args: argparse.Namespace) -> int:
    """Build msgid -> commit from every commit message in the linux tree."""
    from .gitrepo import GitRepo
    from .resolve import write_index

    repo = GitRepo(args.repo)
    repo.require()
    print(f"scanning {args.repo} for lore msgids (a minute or two)...")
    index = repo.all_commit_msgids()
    write_index(MSGID_INDEX, index)
    print(f"{len(index)} msgids -> {MSGID_INDEX}")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    """Attach the real merged diff to each pair, via the commit index."""
    from .gitrepo import GitRepo
    from .resolve import load_index, resolve_pairs

    pairs = _load_pairs()
    if not pairs:
        print("no pairs yet — run `pairs` first", file=sys.stderr)
        return 1
    try:
        index = load_index(MSGID_INDEX, rebuild=args.rebuild_index)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    repo = GitRepo(args.repo)
    repo.require()
    subject_index = None
    if not args.no_subject_match:
        from .resolve import SubjectIndex

        if not COMMIT_SUBJECTS.exists():
            print("building the commit-subject index (a minute)...")
            rows = repo.all_commit_subjects()
            COMMIT_SUBJECTS.write_text(json.dumps(rows), encoding="utf-8")
        else:
            rows = json.loads(COMMIT_SUBJECTS.read_text(encoding="utf-8"))
        subject_index = SubjectIndex(rows)
    tally = resolve_pairs(
        pairs, index, repo, min_overlap=args.min_overlap, subject_index=subject_index
    )
    with PAIRS.open("w", encoding="utf-8") as fh:
        for pr in pairs:
            fh.write(json.dumps(asdict(pr), ensure_ascii=False) + "\n")
    n = len(pairs)
    print(f"pairs {n}: via msgid {tally['via_msgid']}, "
          f"via subject {tally['via_subject']}, "
          f"patch text only {tally['patch_only']}, unresolved {tally['unresolved']}")
    return 0


def cmd_progress(args: argparse.Namespace) -> int:
    """How far the crawl has got: buckets cached, messages, pairs."""
    raw = RAW
    if not raw.exists():
        print("nothing cached yet")
        return 0
    buckets = [d.name for d in raw.iterdir() if d.is_dir() and d.name != "git"]
    msgs = read_jsonl(CORPUS) if CORPUS.exists() else []
    pairs = _load_pairs()
    resolved = sum(1 for pr in pairs if pr.commit)
    print(f"buckets cached {len(buckets)}, linus messages {len(msgs)}, "
          f"pairs {len(pairs)} ({resolved} with a real diff)")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Show moderation decisions for hand review — percentages prove nothing."""
    from .moderate import classify

    pairs = _load_pairs()
    if not pairs:
        print("no pairs yet — run `pairs` first", file=sys.stderr)
        return 1
    import random

    random.Random(args.seed).shuffle(pairs)
    shown = {"keep": 0, "scrub": 0, "drop": 0}
    for pr in pairs:
        verdict = classify(pr.review)
        if shown[verdict.action] >= args.n:
            continue
        shown[verdict.action] += 1
        print("=" * 72)
        print(f"{verdict.action.upper():6} hits={list(verdict.hits)}  {pr.subject[:60]}")
        print("--- was ---")
        print(pr.review[: args.width])
        if verdict.action == "scrub":
            print("--- kept ---")
            print(verdict.review[: args.width])
        if all(v >= args.n for v in shown.values()):
            break
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    msgs = read_jsonl(CORPUS)
    if not msgs:
        print("no corpus yet — run `crawl` first", file=sys.stderr)
        return 1
    n = len(msgs)
    replies = [m for m in msgs if m.is_reply]
    patch = [m for m in replies if m.is_patch_thread]
    with_quote = [m for m in patch if len(m.quoted) >= args.min_quote]
    substantive = [m for m in patch if len(m.own) >= args.min_own]
    good = [m for m in patch if len(m.quoted) >= args.min_quote and len(m.own) >= args.min_own]
    with_diff = [m for m in good if _HUNK_RE.search(_QUOTE_PREFIX_RE.sub("", m.quoted))]
    with_attr = [m for m in good if _ATTR_RE.search(m.own)]
    tech = [m for m in good if _TECH_RE.search(m.own)]
    short = [m for m in patch if 0 < len(m.own) < 80]

    def pct(k: int, base: int = n) -> str:
        return f"{k:6d} ({100.0 * k / base:5.1f}%)"

    print(f"linus messages                {pct(n)}")
    print(f"  replies                     {pct(len(replies), n)}")
    print(f"  replies in [PATCH] threads  {pct(len(patch), n)}")
    print(f"  patch reply, quoted>={args.min_quote:3d}   {pct(len(with_quote), n)}")
    print(f"  patch reply, own>={args.min_own:3d}       {pct(len(substantive), n)}")
    print(f"  BOTH (candidate pairs)      {pct(len(good), n)}")
    print(f"    ... quoted contains diff  {pct(len(with_diff), len(good) or 1)}")
    print(f"    ... has 'X wrote:' attrib {pct(len(with_attr), len(good) or 1)}")
    print(f"    ... own has tech vocab    {pct(len(tech), len(good) or 1)}")
    print(f"    ... own < 80 chars (LGTM) {pct(len(short), n)}")

    subj_prefix = Counter(
        re.match(r"^\s*(re:\s*)*(\[[^\]]*\])?", m.subject, re.I)[2] or ""
        for m in good
    )
    print("\ntop subject tags in candidates:")
    for tag, c in subj_prefix.most_common(12):
        print(f"  {c:5d}  {tag!r}")
    own_len = sorted(len(m.own) for m in good)
    if own_len:
        q = own_len[len(own_len) // 2]
        print(f"\nown-length median {q} chars, "
              f"p90 {own_len[int(len(own_len)*0.9)]}, max {own_len[-1]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="linus-pr")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("months").set_defaults(func=cmd_months)

    c = sub.add_parser("crawl")
    c.add_argument("--month", action="append", help="explicit month bucket, repeatable")
    c.add_argument("--last", type=int, help="crawl the last N month buckets")
    c.add_argument(
        "--year",
        dest="years",
        action="append",
        metavar="YYYY[-YYYY]",
        help="restrict to calendar year(s), e.g. --year 2024 --year 2019-2021",
    )
    c.add_argument("--delay", type=float, default=0.35)
    c.add_argument(
        "--patch-only",
        action="store_true",
        help="only fetch Linus replies whose subject is Re: [PATCH ...]",
    )
    c.set_defaults(func=cmd_crawl)

    pr = sub.add_parser("pairs")
    pr.add_argument("--legacy-gate", action="store_true",
                    help="v2 gate: quote must carry code; no bookkeeping filter")
    pr.set_defaults(func=cmd_pairs)

    cx = sub.add_parser("context", help="earlier thread messages per pair (v3)")
    cx.add_argument("--dry-run", action="store_true",
                    help="count uncached pages the mirror would serve; fetch nothing")
    cx.add_argument("--delay", type=float, default=0.2)
    cx.set_defaults(func=cmd_context)

    e = sub.add_parser("export")
    e.add_argument("--limit", type=int, default=10**9)
    e.add_argument(
        "--no-moderate",
        action="store_true",
        help="skip the abuse filter (training runs should not do this)",
    )
    e.add_argument("--keep-signoff", action="store_true",
                   help='keep the trailing "Linus" signature (v2 behaviour)')
    e.add_argument("--no-thread", action="store_true",
                   help="ignore data/thread_context.json (v2 behaviour)")
    e.add_argument(
        "--eval-year",
        action="append",
        type=int,
        default=[2024],
        help="calendar year(s) held out of training (repeatable; default 2024)",
    )
    e.set_defaults(func=cmd_export)

    sm = sub.add_parser("sample")
    sm.add_argument("-n", type=int, default=3)
    sm.add_argument("--seed", type=int, default=0)
    sm.add_argument("--width", type=int, default=1200)
    sm.set_defaults(func=cmd_sample)

    i = sub.add_parser("index")
    i.add_argument("--repo", type=Path, default=GIT)
    i.set_defaults(func=cmd_index)

    r = sub.add_parser("resolve")
    r.add_argument("--repo", type=Path, default=GIT)
    r.add_argument("--min-overlap", type=float, default=0.5)
    r.add_argument("--rebuild-index", action="store_true")
    r.add_argument("--no-subject-match", action="store_true")
    r.set_defaults(func=cmd_resolve)

    t = sub.add_parser("thread")
    t.add_argument("--max-hops", type=int, default=4)
    t.add_argument("--delay", type=float, default=0.25)
    t.set_defaults(func=cmd_thread)

    a = sub.add_parser("audit")
    a.add_argument("-n", type=int, default=3, help="examples per verdict")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--width", type=int, default=700)
    a.set_defaults(func=cmd_audit)

    sub.add_parser("progress").set_defaults(func=cmd_progress)

    s = sub.add_parser("stats")
    s.add_argument("--min-quote", type=int, default=400)
    s.add_argument("--min-own", type=int, default=200)
    s.set_defaults(func=cmd_stats)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
