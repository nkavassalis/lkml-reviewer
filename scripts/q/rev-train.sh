#!/usr/bin/env bash
# Review-only SFT test (PLAN-v5, 'Decision: is "review-only" data the fix?'):
# train v3's recipe on data/rev/sft.jsonl on both nodes, then generate for the
# same 100 held-out patches. Resumable across windows (exit 75).
. "$(dirname "$0")/lib.sh"
f="$QROOT/data/rev/sft.jsonl"; [ -s "$f" ] || { say "no $f"; exit 1; }
: "${PEER_FAST:?PEER_FAST missing from .status.env}"
scp -q "$f" "$MASTER:$W/rev-sft.jsonl" && on_master "scp -q ~/$W/rev-sft.jsonl $PEER_FAST:$W/rev-sft.jsonl" || exit 1
deploy_train run-chain-2node.sh sft_lora.py compare.py night-run.sh window-lib.sh || exit 1
cluster_plan night-plan-rev.sh; rc=$?
[ $rc -eq 0 ] || exit $rc
on_master "test -f ~/$W/rev.done" || { say "the plan ended without finishing (see latest-run.log)"; exit 75; }
on_master "docker exec linuspr-train chown -R 1000:1000 /work/rev" || true
mkdir -p "$QROOT/data/rev"
scp -q "$MASTER:$W/rev/compare-master.jsonl" "$MASTER:$W/rev/train.log" "$QROOT/data/rev/" || exit 1
grep -E "eval loss|DONE" "$QROOT/data/rev/train.log" | tail -3
