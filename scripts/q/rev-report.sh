#!/usr/bin/env bash
# Review-only test report: v5's rubric on the new model's 100 held-out reviews
# (v3/v4/base scores reused), the proxies, and each criterion from PLAN-v5
# 'Decision: is "review-only" data the fix?' checked as written. Emailed.
. "$(dirname "$0")/lib.sh"
d="$QROOT/data/rev"; r4="$QROOT/data/train/run-4"
[ -s "$d/compare-master.jsonl" ] || { say "no $d/compare-master.jsonl (rev-train fetches it)"; exit 1; }
python3 - "$r4/compare-all.jsonl" "$d/compare-master.jsonl" "$d/compare-all.jsonl" <<'PY' || exit 1
import json, sys
rev = {r["eval_index"]: r for r in map(json.loads, open(sys.argv[2]))}
n = 0
with open(sys.argv[3], "w") as out:
    for r in map(json.loads, open(sys.argv[1])):
        if r["eval_index"] in rev and "rev_sampled" in rev[r["eval_index"]]:
            out.write(json.dumps(r | {"rev_sampled": rev[r["eval_index"]]["rev_sampled"]}) + "\n"); n += 1
print(f"joined {n} patches"); sys.exit(0 if n >= 90 else 1)
PY
: "${SERVE_URL:?SERVE_URL missing from .status.env}"
export JUDGE_URL="${SERVE_URL%/models}/chat/completions"
[ -f "$d/rubric.jsonl" ] || cp "$r4/rubric.jsonl" "$d/rubric.jsonl"     # v3/v4/base already scored
python3 "$QROOT/train/rubric.py" score "$d/compare-all.jsonl" --arms rev_sampled,v3_sampled,v4_sampled \
  --out "$d/rubric.jsonl" --workers "${SCORE_WORKERS:-8}" || exit 1
for other in v3 v4; do
  python3 "$QROOT/train/rubric.py" paired "$d/compare-all.jsonl" "$d/rubric.jsonl" \
    --a "${other}_sampled:$other" --b rev_sampled:rev --json "$d/rubric-$other-vs-rev.json" || exit 1
done > "$d/RUBRIC.md"
PYTHONPATH="$QROOT" python3 "$QROOT/train/score.py" "$d/compare-all.jsonl" > "$d/PROXIES.md" || exit 1
python3 - "$d" > "$d/VERDICT.md" <<'PY' || exit 1
import json, sys, re
d = sys.argv[1]
r = json.load(open(f"{d}/rubric-v3-vs-rev.json"))
s, lo, hi = r["rubric score"]["diff"]; f, flo, fhi = r["false claims"]["diff"]
prox = open(f"{d}/PROXIES.md").read()
def cell(metric):   # the rev column is last
    row = next(l for l in prox.splitlines() if l.startswith(f"| {metric}"))
    return row.strip("|").split("|")[-1].strip()
gates = {"signs 'Linus'": cell("signs 'Linus' (anywhere)"), "email addresses": cell("contains an email address"),
         "loops": cell("repeats a line 3+ times (loop)")}
c1 = lo > 0; c2 = flo <= 0; c3 = all(v == "0%" for v in gates.values())
print("| criterion (set before training) | met? | measured |\n|---|---|---|")
print(f"| rubric beats v3, 95% CI above 0 | {'yes' if c1 else '**no**'} | v3 {r['rubric score']['a']:.2f} -> rev {r['rubric score']['b']:.2f}, {s:+.2f} ({lo:+.2f}..{hi:+.2f}) |")
print(f"| false claims don't rise beyond noise | {'yes' if c2 else '**no**'} | {r['false claims']['a']:.2f} -> {r['false claims']['b']:.2f}, {f:+.2f} ({flo:+.2f}..{fhi:+.2f}) |")
print(f"| gates: 0 signatures, 0 emails, 0 loops | {'yes' if c3 else '**no**'} | {', '.join(f'{k} {v}' for k, v in gates.items())} |")
print(f"\n**Verdict:** {'review-only data IS the fix by the criteria: v5 starts from it' if c1 and c2 and c3 else 'NOT the fix by the criteria'}")
PY
{ echo "# Review-only SFT test (PLAN-v5)"; echo; cat "$d/VERDICT.md"; echo
  echo "Held-out loss: $(grep -E 'eval loss at step' "$d/train.log" | tail -1)"; echo
  echo "## v5's rubric, 100 held-out patches"; echo; cat "$d/RUBRIC.md"; echo
  echo "## Proxies"; echo; cat "$d/PROXIES.md"; echo
  echo "Pre-mortem checks (PLAN-v5): longer only? (median length above), more claims but more false? (false claims line), undertrained? (loss vs v4's 1.885)."
} > "$d/REPORT.md"
"$QROOT/scripts/notify.sh" "[lkml-reviewer] Review-only test: $(grep -o 'IS the fix\|NOT the fix' "$d/VERDICT.md")" < "$d/REPORT.md"
cat "$d/REPORT.md"
