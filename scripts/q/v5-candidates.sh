#!/usr/bin/env bash
# v5 step 1: 4 sampled reviews per training patch from v4, on 800 training
# patches (8 chunks of 100), two chunks at a time (one per node). Resumable:
# exits 75 while chunks remain, and the next window continues.
. "$(dirname "$0")/lib.sh"
d="$QROOT/data/v5"; mkdir -p "$d/in"
if [ ! -f "$d/in/chunk-7.jsonl" ]; then   # fixed, seeded selection of training patches
  python3 - "$QROOT/data/v4/sft.jsonl" "$d/in" <<'PY'
import json, random, sys
rows = open(sys.argv[1]).read().splitlines()
idx = sorted(random.Random(5).sample(range(len(rows)), 800))
for c in range(8):
    with open(f"{sys.argv[2]}/chunk-{c}.jsonl", "w") as f:
        for i in idx[c * 100:(c + 1) * 100]:
            f.write(json.dumps(json.loads(rows[i]) | {"train_index": i}) + "\n")
PY
fi
on_master "mkdir -p ~/$W/cands" && scp -q "$d"/in/chunk-*.jsonl "$MASTER:$W/cands/"
: "${PEER_FAST:?PEER_FAST (peer address as seen from the master) missing from .status.env}"
on_master "ssh -o BatchMode=yes $PEER_FAST mkdir -p $W/cands && scp -q ~/$W/cands/chunk-*.jsonl $PEER_FAST:$W/cands/"
cluster_plan night-plan-cands.sh; rc=$?
[ $rc -eq 0 ] || [ $rc -eq 75 ] || exit $rc
mkdir -p "$d/out"; scp -q "$MASTER:$W/cands/out-*.jsonl" "$d/out/" 2>/dev/null || true
n=$(ls "$d"/out/out-*.jsonl 2>/dev/null | wc -l); say "chunks done: $n / 8"
[ "$n" -eq 8 ] || exit 75
