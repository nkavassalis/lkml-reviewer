"""Is the judge competent enough to trust? A planted-wrong test.

For each of the first N patches, candidate "on" is a real review of that patch
and candidate "off" is a fluent review of a *different* patch. A judge that
reads the code must prefer "on" in both orderings nearly every time; one that
rewards confident prose will not. Same prompt and endpoint as judge.py.

    JUDGE_URL=... python3 train/judge_check.py data/train/run-2/compare.jsonl 20
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from judge import ask


def check(args: tuple[int, list[dict]]) -> dict:
    i, rows = args
    row, other = rows[i], rows[(i + 7) % len(rows)]  # offset: a different subject
    on, off = row["tuned_sampled"], other["base_sampled"]
    v1 = ask(row["input"], row["reference"], on, off)   # on = A
    v2 = ask(row["input"], row["reference"], off, on)   # on = B
    ok = v1.get("winner") == "A" and v2.get("winner") == "B"
    invalid = "invalid" in (v1.get("winner"), v2.get("winner"))
    return {"subject": row["input"].split("\n")[0], "decoy": other["input"].split("\n")[0],
            "order1": v1, "order2": v2, "passed": ok, "invalid": invalid}


def main(src: str, n: str, dst: str) -> None:
    with open(src) as f:
        rows = [json.loads(line) for line in f]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(check, [(i, rows) for i in range(int(n))]))
    with open(dst, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    valid = [r for r in results if not r["invalid"]]
    passed = sum(r["passed"] for r in valid)
    print(f"planted-wrong check: {passed}/{len(valid)} passed "
          f"({len(results) - len(valid)} invalid of {len(results)})")


if __name__ == "__main__":
    main(*sys.argv[1:4])
