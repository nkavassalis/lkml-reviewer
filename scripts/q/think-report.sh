#!/usr/bin/env bash
# Thinking test, step 3: v5's rubric on the base's reviews with and without
# reasoning (same 30 patches), paired; emailed. Does reasoning cut false claims?
. "$(dirname "$0")/lib.sh"
d="$QROOT/data/think"; mkdir -p "$d"
for f in think-off think-on; do
  n=$(on_mac "wc -l < ~/lkml-mac/$f.jsonl 2>/dev/null" | tr -d ' '); [ "${n:-0}" -ge 30 ] || { say "$f: ${n:-0}/30 rows"; exit 1; }
  from_mac "lkml-mac/$f.jsonl" "$d/$f.jsonl" || exit 1
done
python3 - "$d" <<'PY' || exit 1
import json, statistics, sys
d = sys.argv[1]
off = {r["eval_index"]: r for r in map(json.loads, open(f"{d}/think-off.jsonl"))}
on = {r["eval_index"]: r for r in map(json.loads, open(f"{d}/think-on.jsonl"))}
with open(f"{d}/compare.jsonl", "w") as out:
    for i, r in off.items():
        if i in on:
            out.write(json.dumps({"eval_index": i, "input": r["input"], "reference": r["reference"],
                                  "base_nothink": r["base_nothink"], "base_think": on[i]["base_think"]}) + "\n")
closed = [r.get("base_think_think_closed") for r in on.values()]
chars = [r.get("base_think_think_chars", 0) for r in on.values()]
with open(f"{d}/STATS.md", "w") as s:
    s.write(f"reasoning finished before the token limit: {sum(map(bool, closed))}/{len(closed)}; "
            f"median reasoning {statistics.median(chars):.0f} chars\n\n"
            f"median review length: without {statistics.median(len(r['base_nothink']) for r in off.values()):.0f}, "
            f"with {statistics.median(len(r['base_think']) for r in on.values()):.0f} chars\n")
PY
: "${SERVE_URL:?SERVE_URL missing from .status.env}"
export JUDGE_URL="${SERVE_URL%/models}/chat/completions"
python3 "$QROOT/train/rubric.py" score "$d/compare.jsonl" --arms base_nothink,base_think --out "$d/rubric.jsonl" || exit 1
python3 "$QROOT/train/rubric.py" paired "$d/compare.jsonl" "$d/rubric.jsonl" --a base_nothink:without --b base_think:with \
  --json "$d/rubric.json" > "$d/RUBRIC.md" || exit 1
{ echo "# Does reasoning make the base model's reviews more accurate? (30 held-out patches)"; echo
  cat "$d/RUBRIC.md"; echo; cat "$d/STATS.md"; echo
  echo "Reviews whose reasoning hit the token limit are scored as empty (the worst case for thinking)."
} > "$d/REPORT.md"
"$QROOT/scripts/notify.sh" "[lkml-reviewer] Thinking test: base model with vs without reasoning" < "$d/REPORT.md"
cat "$d/REPORT.md"
