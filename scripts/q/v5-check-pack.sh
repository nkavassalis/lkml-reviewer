#!/usr/bin/env bash
# v5 step 2b: a reading pack for the human check (PLAN-v5: read 20 pairs; is
# "chosen" really the better review, judged against Linus' reply?).
. "$(dirname "$0")/lib.sh"
d="$QROOT/data/v5"
python3 - "$d" <<'PY' > "$d/CHECK.md"
import json, random, statistics, sys
d = sys.argv[1]
pairs = [json.loads(l) for l in open(f"{d}/pairs.jsonl")]
cands = {r["eval_index"]: r for r in map(json.loads, open(f"{d}/cands.jsonl"))}
sc = [json.loads(l) for l in open(f"{d}/scores.jsonl")]
vals = [x["score"] for x in sc if x.get("score") is not None]
print("# v5 preference pairs: human check\n")
print(f"{len(pairs)} pairs from {len(cands)} patches; {len(vals)} scored candidates, "
      f"median score {statistics.median(vals):.1f}, {len(sc) - len(vals)} judge failures.")
print(f"Chosen median length {statistics.median(len(p['chosen']) for p in pairs):.0f} chars, "
      f"rejected {statistics.median(len(p['rejected']) for p in pairs):.0f} "
      "(a big gap means the judge may be rewarding length).\n")
print("For each: is CHOSEN the better review of the patch, given what Linus said?")
print("Approve with `python3 train/workqueue.py approve v5-human-check` if ~16/20 or better.\n")
for n, p in enumerate(random.Random(7).sample(pairs, min(20, len(pairs))), 1):
    r = cands[p["eval_index"]]
    print(f"\n---\n## {n}. pair {p['eval_index']} (scores {p['chosen_score']:.0f} vs {p['rejected_score']:.0f})\n")
    print("### Patch (start)\n```\n" + r["input"][:1500] + "\n```")
    print("### Linus\n```\n" + r["reference"][:1500] + "\n```")
    print("### CHOSEN\n```\n" + p["chosen"][:2000] + "\n```")
    print("### REJECTED\n```\n" + p["rejected"][:2000] + "\n```")
PY
head -8 "$d/CHECK.md"
"$QROOT/scripts/notify.sh" "[lkml-reviewer] v5: 20 preference pairs to read (data/v5/CHECK.md)" < "$d/CHECK.md"
