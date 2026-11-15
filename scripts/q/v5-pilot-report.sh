#!/usr/bin/env bash
# v5 pilot report (Nick, 2026-11-14: "Pilot first"): from 50 patches x 4
# reviews by the starting model, how many preference pairs survive each
# minimum score gap, and what the full run would yield. Emailed; the queue then
# waits at v5-settings for the full run's N and gap.
. "$(dirname "$0")/lib.sh"
d="$QROOT/data/v5/pilot"
for g in 2 3 4; do
  python3 "$QROOT/train/make_preferences.py" pairs "$d/cands.jsonl" "$d/scores.jsonl" --min-gap $g \
    --out "$d/pairs-gap$g.jsonl" > /dev/null || exit 1
done
python3 - "$d" <<'PY' > "$d/REPORT.md" || exit 1
import json, statistics, sys
d = sys.argv[1]
cands = [json.loads(l) for l in open(f"{d}/cands.jsonl")]
sc = [json.loads(l) for l in open(f"{d}/scores.jsonl")]
vals = [x["score"] for x in sc if x.get("score") is not None]
fc = [x.get("false_claims") or 0 for x in sc if x.get("score") is not None]
by = {}
for x in sc:
    if x.get("score") is not None:
        by.setdefault(x["eval_index"], []).append(x["score"])
spread = [max(v) - min(v) for v in by.values() if len(v) >= 2]
n = len(cands)
print(f"# v5 pilot: preference pairs from {n} patches x 4 reviews (starting model: rev)\n")
print(f"Scored {len(vals)} of {sum(1 for r in cands for k in r if '_sampled_s' in k)} candidates "
      f"({len(sc) - len(vals)} unparseable). Median score {statistics.median(vals):.1f}, "
      f"false claims per review {statistics.fmean(fc):.2f}. "
      f"Best-minus-worst per patch: median {statistics.median(spread):.1f}.\n")
print("| min gap | pairs | yield | at N = 800 | at N = 1,718 (all review-only patches) |\n|---|---|---|---|---|")
for g in (2, 3, 4):
    k = sum(1 for _ in open(f"{d}/pairs-gap{g}.jsonl"))
    print(f"| {g} | {k} | {k / n:.0%} | ~{round(800 * k / n)} | ~{round(1718 * k / n)} |")
print("\nPLAN-v5 assumed ~700 pairs from N = 800 (gap 3). Choose the full run's N and gap, then:")
print("  echo 'V5_N=... V5_GAP=...' > data/v5/SETTINGS; python3 train/workqueue.py approve v5-settings")
PY
"$QROOT/scripts/notify.sh" "[lkml-reviewer] v5 pilot: preference pair yield" < "$d/REPORT.md"
cat "$d/REPORT.md"
