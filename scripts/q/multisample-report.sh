#!/usr/bin/env bash
# Paired comparisons with the noise floor: v3 (as trained) vs v3 patch-first vs v4.
. "$(dirname "$0")/lib.sh"
d="$QROOT/data/train/multisample"; mkdir -p "$d"
for f in ms-v3-threadfirst ms-v3-patchfirst ms-v4; do
  n=$(on_mac "wc -l < ~/lkml-mac/$f.jsonl"); say "$f: $n / 200 patches"
  [ "$n" -eq 200 ] || { on_mac "tail -20 ~/lkml-mac/$f.log 2>/dev/null || tail -20 ~/lkml-mac/multi-sample.log"; exit 1; }
done
for f in ms-v3-threadfirst.jsonl ms-v3-patchfirst.jsonl ms-v4.jsonl; do
  from_mac "lkml-mac/$f" "$d/"
done
args=(--primary-marker '--- The patch ---' --context-marker '--- Earlier in this thread')
{
  echo "# Multi-sample evaluation: 200 held-out patches x 3 samples (MLX 8-bit)"; echo
  echo "Paired by patch, bootstrap 95% CIs, with each baseline's noise floor."
  echo "A difference whose CI includes 0 is not a finding."
  for pair in "ms-v3-threadfirst.jsonl:v3_tf ms-v4.jsonl:v4" "ms-v3-patchfirst.jsonl:v3_pf ms-v4.jsonl:v4" \
              "ms-v3-threadfirst.jsonl:v3_tf ms-v3-patchfirst.jsonl:v3_pf"; do
    set -- $pair
    echo; echo "## ${1##*:} vs ${2##*:}"
    python3 "$QROOT/train/paired_stats.py" "$d/$1" "$d/$2" "${args[@]}"
  done
} > "$d/MULTISAMPLE.md" || exit 1
# Which model v5 starts from (PLAN-v5; rule set before the data, in choose_base.py).
# v3 patch-first and v4 saw byte-identical inputs (data/v4/sft_eval.jsonl).
python3 "$QROOT/train/paired_stats.py" "$d/ms-v3-patchfirst.jsonl:v3_pf" "$d/ms-v4.jsonl:v4" "${args[@]}" \
  --json "$d/v3pf-vs-v4.json" > /dev/null || exit 1
mkdir -p "$QROOT/data/v5"
python3 "$QROOT/scripts/q/choose_base.py" "$d/v3pf-vs-v4.json" > "$d/v5-base.txt" || exit 1
base=$(head -1 "$d/v5-base.txt"); echo "$base" > "$QROOT/data/v5/BASE"
{ echo "## v5 starts from: $base"; echo; tail -n +2 "$d/v5-base.txt" | sed 's/^/- /'; echo; cat "$d/MULTISAMPLE.md"; } > "$d/MULTISAMPLE.email.md"
"$QROOT/scripts/notify.sh" "[lkml-reviewer] Multi-sample evaluation: v3 vs v4; v5 starts from $base" < "$d/MULTISAMPLE.email.md"
cat "$d/MULTISAMPLE.md"
