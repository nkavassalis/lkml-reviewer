#!/usr/bin/env bash
# Loop rate under greedy decoding, v3 / v4 / v5 (train/score.py's loop test:
# the same non-trivial line 3+ times). Emailed.
. "$(dirname "$0")/lib.sh"
d="$QROOT/data/train/greedy"; mkdir -p "$d"
for m in v3 v4 v5; do from_mac "lkml-mac/greedy-$m.jsonl" "$d/"; done
python3 - "$d" "$QROOT/train" <<'PY' > "$d/GREEDY.md" || exit 1
import json, sys
sys.path.insert(0, sys.argv[2])
from score import loops
print("# Greedy decoding: how often each model loops (100 held-out patches)\n")
print("Loop = the same non-trivial line 3+ times. Reference points (v2 eval, REPORT §6): "
      "base greedy 0%, v2 greedy 44%, v2 sampled 0%.\n")
print("| model | loops | rows |\n|---|---|---|")
for m in ("v3", "v4", "v5"):
    rows = [json.loads(l) for l in open(f"{sys.argv[1]}/greedy-{m}.jsonl")]
    assert len(rows) == 100, f"{m}: {len(rows)} rows"
    n = sum(loops(r[f"{m}_greedy"]) for r in rows)
    print(f"| {m} | {n}% | {len(rows)} |")
PY
cat "$d/GREEDY.md"
"$QROOT/scripts/notify.sh" "[lkml-reviewer] Greedy-decoding loop rate: v3 / v4 / v5" < "$d/GREEDY.md"
