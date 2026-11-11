"""Build the raw corpus: every Linus reply in a month range, as JSONL."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .fetch import Fetcher
from .parse import IndexEntry, Message, parse_index, parse_message

LINUS = "Linus Torvalds"
_MONTH_RE = re.compile(r'href="(\d{4}\.\d)/index\.html"')


#: Bucket ids are plain **YYMM.N** (two-digit year, two-digit month, part
#: counter 0..3): 2406 = April 2024, 1901 = January 2019, 9603 = March 1996.
#: The mirror publishes 9603 (Mar 1996) through 2611 (Nov 2026), so the first
#: two digits are only ever 96..99 or 00..26 — the pivot is therefore anywhere
#: in 27..95 and 50 is the safe middle.
#:
#: Both natural-looking guesses here were wrong and each produced a
#: plausible-looking but wrong crawl: `yy >= 93` read "1901" as January *1919*
#: (a `--year 2019` crawl silently harvested 48 pre-kernel months and reported
#: 355k messages seen), and reading the id as YYYM turned 2024 into 2140.
#: Any change must keep test_bucket_year_* green.
PRE_2000_THRESHOLD = 50


def bucket_year(bucket: str) -> int:
    """Calendar year of a `YYMM.N` bucket id."""
    yy = int(bucket[:2])
    return 1900 + yy if yy >= PRE_2000_THRESHOLD else 2000 + yy


def buckets_of_year(all_buckets: list[str], year: int) -> list[str]:
    """Every bucket belonging to one calendar year."""
    return [b for b in all_buckets if bucket_year(b) == year]


def list_months(fetcher: Fetcher) -> list[str]:
    """All month buckets the mirror publishes, oldest first (e.g. '2406.0')."""
    html = fetcher.get("")
    return sorted(set(_MONTH_RE.findall(html)), key=lambda b: (bucket_year(b), b))


def linus_entries(
    entries: list[IndexEntry], *, patch_only: bool = False
) -> list[IndexEntry]:
    """Linus' rows of a month index; optionally only replies in [PATCH] threads."""
    out = [e for e in entries if e.author.strip() == LINUS]
    if patch_only:
        subj = lambda e: e.subject.lower()  # noqa: E731
        out = [e for e in out if subj(e).startswith("re:") and "[patch" in subj(e)]
    return out


@dataclass
class MonthCorpus:
    month: str
    total_messages: int
    linus_messages: list[Message]


def crawl_month(
    fetcher: Fetcher, month: str, *, patch_only: bool = False
) -> MonthCorpus:
    """Fetch one month index, then every Linus message page in it."""
    entries = parse_index(fetcher.get(month, "index.html"))
    msgs = [
        parse_message(fetcher.get(month, e.filename), mid=e.mid, month=month)
        for e in linus_entries(entries, patch_only=patch_only)
    ]
    return MonthCorpus(month=month, total_messages=len(entries), linus_messages=msgs)


def fetch_message_or_none(fetcher: Fetcher, month: str, mid: str) -> Message | None:
    """Fetch a thread parent; None when the mirror 404s or errors out."""
    from .fetch import FetchError

    try:
        raw = fetcher.get(month, f"{mid}.html")
    except FetchError:
        return None
    if "<title>404 Not Found" in raw[:2000]:
        return None
    return parse_message(raw, mid=mid, month=month)


def fetch_message(fetcher: Fetcher, month: str, mid: str) -> Message:
    """One message page by id — used to walk a thread back to its patch."""
    return parse_message(fetcher.get(month, f"{mid}.html"), mid=mid, month=month)


def crawl_months(
    fetcher: Fetcher, months: list[str], *, patch_only: bool = False
) -> list[MonthCorpus]:
    return [crawl_month(fetcher, m, patch_only=patch_only) for m in months]


def write_jsonl(corpora: list[MonthCorpus], path: Path) -> int:
    """Merge-on-write: replace the months we just crawled, keep everything else.

    Crawls are long and resumable, so a second run must never throw away the
    months an earlier run already harvested.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    refreshed = {c.month for c in corpora}
    rows: list[dict] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("month") not in refreshed:
                rows.append(row)
    for c in corpora:
        rows.extend(asdict(m) for m in c.linus_messages)
    # atomic swap: `stats` may read the corpus while a crawl is still running
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return len(rows)


def read_jsonl(path: Path) -> list[Message]:
    out: list[Message] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        known = set(Message.__dataclass_fields__)
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["lore_links"] = [tuple(pair) for pair in kwargs.get("lore_links", [])]
        out.append(Message(**kwargs))
    return out
