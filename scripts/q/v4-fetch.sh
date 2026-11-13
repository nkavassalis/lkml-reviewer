#!/usr/bin/env bash
# Copy v4's results and adapter from the master to data/train/run-4/.
. "$(dirname "$0")/lib.sh"
on_master "docker exec linuspr-train chown -R 1000:1000 /work/run-4" || true
mkdir -p "$QROOT/data/train/run-4"
scp -q -r "$MASTER:$W/run-4/adapter-final" "$MASTER:$W/run-4/train.log" "$MASTER:$W/run-4/compare-master.jsonl" \
  "$MASTER:$W/run-4/compare-all.jsonl" "$MASTER:$W/RESULTS-v4.md" "$QROOT/data/train/run-4/"
scp -q "$MASTER:$W/run-4/judge-*.jsonl" "$MASTER:$W/run-4/judges.log" "$QROOT/data/train/run-4/" || true
ls -la "$QROOT/data/train/run-4/adapter-final/adapter_model.safetensors"
