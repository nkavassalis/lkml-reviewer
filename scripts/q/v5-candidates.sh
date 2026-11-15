#!/usr/bin/env bash
# v5 step 1: 4 sampled reviews per training patch from v5's starting model
# (data/v5/BASE: rev, Nick 2026-11-15), two chunks at a time (one per node).
# Resumable: exits 75 while chunks remain, and the next window continues.
#
#   V5_SET=pilot V5_N=50 V5_CHUNKS=2 scripts/q/v5-candidates.sh   # the pilot
#   scripts/q/v5-candidates.sh                                    # full: 800 in 8 chunks
#
# Patches come from the review-only training set (data/rev/sft.jsonl): the
# judge scores candidates against Linus' real reply, and a reference that isn't
# a review (two-thirds of v4's data) makes a noisy yardstick. The pilot's
# patches are excluded from the full set's draw only by seed; overlap is
# harmless (the same prompt, new samples).
. "$(dirname "$0")/lib.sh"
# Full run: the N and gap Nick chose after the pilot (data/v5/SETTINGS).
[ "${V5_SET:-full}" = full ] && [ -f "$QROOT/data/v5/SETTINGS" ] && . "$QROOT/data/v5/SETTINGS"
set_=${V5_SET:-full}; n=${V5_N:-800}; chunks=${V5_CHUNKS:-8}; size=$(( n / chunks ))
d="$QROOT/data/v5/$set_"; mkdir -p "$d/in"
src="$QROOT/data/rev/sft.jsonl"
if [ ! -f "$d/in/chunk-$((chunks - 1)).jsonl" ]; then   # fixed, seeded selection
  python3 - "$src" "$d/in" "$n" "$chunks" <<'PY'
import json, random, sys
rows = open(sys.argv[1]).read().splitlines()
n, chunks = int(sys.argv[3]), int(sys.argv[4]); size = n // chunks
idx = sorted(random.Random(5).sample(range(len(rows)), n))
for c in range(chunks):
    with open(f"{sys.argv[2]}/chunk-{c}.jsonl", "w") as f:
        for i in idx[c * size:(c + 1) * size]:
            f.write(json.dumps(json.loads(rows[i]) | {"train_index": i}) + "\n")
PY
fi
base=$(cat "$QROOT/data/v5/BASE" 2>/dev/null) || { say "no data/v5/BASE"; exit 1; }
case "$base" in rev|v4|v3) ;; *) say "data/v5/BASE is '$base', expected rev, v4 or v3"; exit 1 ;; esac
say "starting model: $base; set $set_: $n patches in $chunks chunks of $size"
cd_="cands-$set_"
printf 'CANDS=%s\nNCHUNKS=%s\nSUBSET=%s\n' "$cd_" "$chunks" "$size" > "$d/cands.env"
on_master "mkdir -p ~/$W/$cd_" && scp -q "$d"/in/chunk-*.jsonl "$QROOT/data/v5/BASE" "$MASTER:$W/$cd_/" \
  && scp -q "$d/cands.env" "$MASTER:$W/cands.env" || exit 1
: "${PEER_FAST:?PEER_FAST (peer address as seen from the master) missing from .status.env}"
on_master "ssh -o BatchMode=yes $PEER_FAST mkdir -p $W/$cd_ && scp -q ~/$W/$cd_/chunk-*.jsonl $PEER_FAST:$W/$cd_/"
deploy_train compare.py run-chain-2node.sh night-run.sh window-lib.sh || exit 1
cluster_plan night-plan-cands.sh; rc=$?
[ $rc -eq 0 ] || [ $rc -eq 75 ] || exit $rc
mkdir -p "$d/out"; scp -q "$MASTER:$W/$cd_/out-*.jsonl" "$d/out/" 2>/dev/null || true
got=$(ls "$d"/out/out-*.jsonl 2>/dev/null | wc -l); say "chunks done: $got / $chunks"
[ "$got" -eq "$chunks" ] || exit 75
