#!/bin/bash
# Train -> compare -> restore, on a GPU node that normally serves a model.
#
# The serving container is stopped for the run and restarted in an EXIT trap,
# so a crash, an OOM, or a kill can't leave the service down. Run it on the
# training node, detached:
#
#   RUN=run-2 EPOCHS=1 SERVE_PEERS="user@peer-node" \
#     setsid nohup ./run-chain.sh > /dev/null 2>&1 < /dev/null &
#
# Environment:
#   RUN           output dir name under $WORK (required)
#   EPOCHS        training epochs (default 1)
#   WORK          host dir mounted at /work in the training container (default .)
#   TRAIN_CTR     training container name (default linuspr-train)
#   SERVE_CTR     serving container to stop/start (default vllm_fp4local)
#   SERVE_PEERS   space-separated ssh targets that also run SERVE_CTR (tensor-
#                 parallel peers); stopped and restored together with this node
set -u
: "${RUN:?set RUN, e.g. RUN=run-3}"
EPOCHS=${EPOCHS:-1}
WORK=${WORK:-$(pwd)}
TRAIN_CTR=${TRAIN_CTR:-linuspr-train}
SERVE_CTR=${SERVE_CTR:-vllm_fp4local}
SERVE_PEERS=${SERVE_PEERS:-}
LOG=$WORK/chain-$RUN.log

say() { echo "[$(date +%T)] $*" >> "$LOG"; }

restore() {
  local status=$?
  say "restoring $SERVE_CTR (exit status $status)"
  docker start "$SERVE_CTR" >> "$LOG" 2>&1
  for peer in $SERVE_PEERS; do
    ssh -o BatchMode=yes "$peer" docker start "$SERVE_CTR" >> "$LOG" 2>&1
  done
  say "start issued on all nodes"
}
trap restore EXIT
set -e

say "stopping $SERVE_CTR"
docker stop -t 60 "$SERVE_CTR" >> "$LOG" 2>&1
for peer in $SERVE_PEERS; do
  ssh -o BatchMode=yes "$peer" docker stop -t 60 "$SERVE_CTR" >> "$LOG" 2>&1
done

say "training $RUN ($EPOCHS epoch(s))"
docker exec -w /work "$TRAIN_CTR" python3 sft_lora.py --train sft.jsonl \
  --eval sft_eval.jsonl --out "$RUN" --epochs "$EPOCHS" --grad-accum 8 \
  --n-samples 5 > "$WORK/$RUN.out" 2>&1

say "comparing base vs tuned"
docker exec -w /work "$TRAIN_CTR" python3 compare.py --adapter "$RUN/adapter-final" \
  --eval sft_eval.jsonl --out "$RUN/compare.jsonl" > "$WORK/compare-$RUN.out" 2>&1

say "CHAIN DONE"
