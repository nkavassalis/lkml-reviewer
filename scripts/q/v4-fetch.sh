#!/usr/bin/env bash
# Copy v4's results and adapter from the master to data/train/run-4/.
. "$(dirname "$0")/lib.sh"
on_master "docker exec linuspr-train chown -R 1000:1000 /work/run-4" || true
mkdir -p "$QROOT/data/train/run-4"
# Required: the adapter, the results, the generations. Nice to have: the rest.
scp -q -r "$MASTER:$W/run-4/adapter-final" "$MASTER:$W/run-4/compare-master.jsonl" "$MASTER:$W/RESULTS-v4.md" \
  "$QROOT/data/train/run-4/" || exit 1
for f in train.log compare-all.jsonl check-input.jsonl judge_check.jsonl judges.log 'judge-*.jsonl'; do
  scp -q "$MASTER:$W/run-4/$f" "$QROOT/data/train/run-4/" 2>/dev/null || say "not copied: $f"
done
ls -la "$QROOT/data/train/run-4/adapter-final/adapter_model.safetensors"
