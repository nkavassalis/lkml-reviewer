#!/usr/bin/env bash
# DPO smoke test on the cluster, before v5's real pairs exist: 16 synthetic
# pairs (chosen = Linus' reply, rejected = its first quarter), 2 steps on both
# nodes, a checkpoint, then evaluation loads the result. Tests the mechanics
# (memory, DDP, the flat adapter save, the peer copy), not quality.
. "$(dirname "$0")/lib.sh"
python3 - "$QROOT/data/v4/sft.jsonl" > "$QROOT/data/v5/smoke-pairs.jsonl" <<'PY' || exit 1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if 600 < len(r["output"]) < 2500][:16]
for r in rows:
    print(json.dumps({"input": r["input"], "chosen": r["output"], "rejected": r["output"][: len(r["output"]) // 4]}))
PY
[ "$(wc -l < "$QROOT/data/v5/smoke-pairs.jsonl")" -eq 16 ] || { say "could not build 16 smoke pairs"; exit 1; }
deploy_train dpo_lora.py dpo_core.py adapter_ckpt.py sft_lora.py compare.py run-chain-2node.sh night-run.sh window-lib.sh || exit 1
on_master "mkdir -p ~/$W/v5" && scp -q "$QROOT/data/v5/smoke-pairs.jsonl" "$MASTER:$W/v5/smoke-pairs.jsonl" || exit 1
cluster_plan night-plan-dpo-smoke.sh; rc=$?
[ $rc -eq 0 ] || exit $rc
on_master "cd ~/$W && test -s dpo-smoke.ok" || { say "smoke test did not pass (see dpo-smoke/train.log on the master)"; exit 1; }
on_master "cd ~/$W && tail -6 dpo-smoke/train.log && ls -la dpo-smoke/adapter-final"
