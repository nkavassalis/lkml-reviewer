#!/usr/bin/env bash
# Score v3, v4 and the base on the 100 held-out patches with v5's own rubric
# (make_preferences.ask via train/rubric.py): accuracy first, and saying
# little is not a virtue. This is the target signal for v5's starting model
# (PLAN-v5, "Decision: which model v5 starts from"). Uses the serving vLLM;
# no downtime. Resumable.
. "$(dirname "$0")/lib.sh"
d="$QROOT/data/train/run-4"
[ -s "$d/compare-all.jsonl" ] || { say "no $d/compare-all.jsonl (v4-fetch copies it)"; exit 1; }
: "${SERVE_URL:?SERVE_URL missing from .status.env}"
export JUDGE_URL="${SERVE_URL%/models}/chat/completions"
python3 "$QROOT/train/rubric.py" score "$d/compare-all.jsonl" --arms v3_sampled,v4_sampled,base_sampled \
  --out "$d/rubric.jsonl" --workers "${SCORE_WORKERS:-8}" || exit 1
for pair in "v3_sampled:v3 v4_sampled:v4 v3-vs-v4" "base_sampled:base v4_sampled:v4 base-vs-v4"; do
  set -- $pair
  python3 "$QROOT/train/rubric.py" paired "$d/compare-all.jsonl" "$d/rubric.jsonl" --a "$1" --b "$2" \
    --json "$d/rubric-$3.json" || exit 1
done | tee "$d/RUBRIC.md"
n=$(python3 -c "import json;print(json.load(open('$d/rubric-v3-vs-v4.json'))['n'])")
[ "$n" -ge 80 ] || { say "only $n of 100 patches scored for both v3 and v4; too few to decide on"; exit 1; }
