#!/usr/bin/env bash
# v5 step 2: the judge scores every candidate against Linus' real reply
# (make_preferences.py score, resumable), then pairs best vs worst per patch.
# Runs against the cluster's vLLM inside the window; exits 75 if cut off.
. "$(dirname "$0")/lib.sh"
d="$QROOT/data/v5"
# One candidates file; eval_index renumbered so it is unique across chunks.
python3 - "$d" <<'PY'
import json, sys
d = sys.argv[1]
with open(f"{d}/cands.jsonl", "w") as out:
    for c in range(8):
        for line in open(f"{d}/out/out-{c}.jsonl"):
            r = json.loads(line); r["eval_index"] += 100 * c
            out.write(json.dumps(r) + "\n")
PY
export JUDGE_URL="${SERVE_URL%/models}/chat/completions"
mins=$(( ${WINDOW_MINUTES_LEFT:-60} - 15 ))
[ "$mins" -ge 20 ] || exit 75
timeout "${mins}m" python3 "$QROOT/train/make_preferences.py" score "$d/cands.jsonl" --arm cand \
  --out "$d/scores.jsonl" --workers "${SCORE_WORKERS:-8}"; rc=$?
[ $rc -eq 124 ] && { say "cut off at the window's end; resumes next window"; exit 75; }
[ $rc -eq 0 ] || exit $rc
python3 - "$d" <<'PY' || exit 75
import json, sys
d = sys.argv[1]
n = sum(1 for _ in open(f"{d}/cands.jsonl")) * 4
s = [json.loads(l) for l in open(f"{d}/scores.jsonl")]
ok = sum(1 for x in s if x.get("score") is not None)
print(f"scored {ok} of {n} candidates ({len(s) - ok} judge failures)")
sys.exit(0 if len(s) >= n else 1)
PY
python3 "$QROOT/train/make_preferences.py" pairs "$d/cands.jsonl" "$d/scores.jsonl" --min-gap 3 --out "$d/pairs.jsonl"
