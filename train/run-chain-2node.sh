#!/bin/bash
# 2-node version of run-chain.sh: stop serving on both nodes -> data-parallel
# training (torchrun, one GPU per node) -> split evaluation across the nodes ->
# restore serving. Restore runs in an EXIT trap, so a crash, OOM or kill can't
# leave the service down. Run on the master node, detached:
#
#   RUN=run-3 PEER=user@peer MASTER_ADDR=<master ip on the fast link> \
#   IFACE=<fast-link interface> setsid nohup ./run-chain-2node.sh >/dev/null 2>&1 </dev/null &
#
# Environment:
#   RUN          output dir under $WORK (required)
#   PEER         ssh target of the second node (required)
#   MASTER_ADDR  this node's address on the inter-node link (required)
#   IFACE        that link's interface name, for NCCL/Gloo sockets (required)
#   EPOCHS       default 1
#   GRAD_ACCUM   per rank, default 4 (effective batch = 2 x GRAD_ACCUM)
#   TRAIN_EXTRA  extra trainer args (e.g. "--limit 200 --max-steps 4" for a smoke test)
#   TRAINER      sft (default: sft_lora.py on sft.jsonl) or dpo (dpo_lora.py; needs
#                SFT_ADAPTER, the adapter it starts from, and PAIRS, the preference
#                file, both relative to WORK; the chain copies them to the peer)
#   EVAL_ARMS_MASTER / EVAL_ARMS_PEER   compare.py arms per node, e.g. "base,v2" / "v3";
#                empty EVAL_ARMS_MASTER skips evaluation
#   EVAL_ADAPTERS_MASTER / EVAL_ADAPTERS_PEER   name=path lists for compare.py
#   EVAL_SUBSET  held-out rows to generate for (default 100)
#   EVAL_ONLY    1 = skip training and evaluate an existing $RUN/adapter-final
#   EVAL_FILE_MASTER / EVAL_FILE_PEER   eval inputs per node (default sft_eval.jsonl)
#   EVAL_EXTRA   extra compare.py args, e.g. "--samples 4 --max-new 600"
#   PEER_EVAL_LOSS   after its arms, the peer scores these adapters' held-out loss,
#                e.g. "v2=run-2-adapter base=" (empty path = the bare base)
#   WORK, TRAIN_CTR, SERVE_CTR   as in run-chain.sh
set -u
: "${RUN:?}" "${PEER:?}" "${MASTER_ADDR:?}" "${IFACE:?}"
EPOCHS=${EPOCHS:-1}
TRAINER=${TRAINER:-sft}
case "$TRAINER" in
  sft) ;;
  dpo) : "${SFT_ADAPTER:?TRAINER=dpo needs SFT_ADAPTER}" "${PAIRS:?TRAINER=dpo needs PAIRS}" ;;
  *) echo "TRAINER must be sft or dpo" >&2; exit 2 ;;
esac
GRAD_ACCUM=${GRAD_ACCUM:-4}
TRAIN_EXTRA=${TRAIN_EXTRA:-}
EVAL_ARMS_MASTER=${EVAL_ARMS_MASTER:-}
EVAL_ARMS_PEER=${EVAL_ARMS_PEER:-}
EVAL_ADAPTERS_MASTER=${EVAL_ADAPTERS_MASTER:-}
EVAL_ADAPTERS_PEER=${EVAL_ADAPTERS_PEER:-}
EVAL_SUBSET=${EVAL_SUBSET:-100}
WORK=${WORK:-$(pwd)}
TRAIN_CTR=${TRAIN_CTR:-linuspr-train}
SERVE_CTR=${SERVE_CTR:-vllm_fp4local}
LOG=$WORK/chain-$RUN.log
PORT=29511  # not the serving stack's port

say() { echo "[$(date +%T)] $*" >> "$LOG"; }
peer() { ssh -o BatchMode=yes "$PEER" "$@"; }

restore() {
  local status=$?
  say "restoring $SERVE_CTR on both nodes (exit status $status)"
  # If one rank died, the other would wait out the NCCL timeout holding its GPU,
  # and the serving container could not start there. Kill leftovers first.
  docker exec "$TRAIN_CTR" pkill -f "sft_lora.py|dpo_lora.py|compare.py" >> "$LOG" 2>&1 || true
  # Quoted as ONE remote command string: ssh re-parses it, and a bare | in the
  # pattern became a shell pipe on the peer ("compare.py: command not found").
  peer "docker exec $TRAIN_CTR pkill -f 'sft_lora.py|dpo_lora.py|compare.py'" >> "$LOG" 2>&1 || true
  sleep 5
  docker start "$SERVE_CTR" >> "$LOG" 2>&1
  peer docker start "$SERVE_CTR" >> "$LOG" 2>&1
  say "start issued on both nodes"
}
trap restore EXIT
# A stop signal (the window's hard stop, a kill) becomes a normal exit, so the
# EXIT trap above runs and serving is restored.
trap 'exit 143' TERM INT
set -e

NCCL_ENV=(-e "NCCL_SOCKET_IFNAME=$IFACE" -e "GLOO_SOCKET_IFNAME=$IFACE" -e NCCL_IB_DISABLE=1
          -e NCCL_DEBUG=WARN)
torchrun_cmd() {  # $1 = node rank
  local job
  if [ "$TRAINER" = dpo ]; then
    job="dpo_lora.py --sft-adapter $SFT_ADAPTER --pairs $PAIRS --out $RUN --epochs $EPOCHS --grad-accum $GRAD_ACCUM --save-every 25"
  else
    job="sft_lora.py --train sft.jsonl --eval sft_eval.jsonl --out $RUN --epochs $EPOCHS --grad-accum $GRAD_ACCUM --save-every 50"
  fi
  echo docker exec "${NCCL_ENV[@]}" -w /work "$TRAIN_CTR" torchrun --nnodes 2 \
    --nproc-per-node 1 --node-rank "$1" --master-addr "$MASTER_ADDR" --master-port $PORT \
    $job $TRAIN_EXTRA
}
# Copy WORK-relative $1 to the same place on the peer (a fresh copy: scp -r
# into an existing directory nests it).
to_peer() {
  local rel=$1 pw; pw=$(basename "$WORK")
  docker exec "$TRAIN_CTR" chown -R "$(id -u):$(id -g)" "/work/$rel" >> "$LOG" 2>&1 || true
  peer "rm -rf $pw/$rel && mkdir -p $pw/$(dirname "$rel")"
  scp -q -r "$WORK/$rel" "$PEER:$pw/$rel"
}

say "stopping $SERVE_CTR on both nodes"
docker stop -t 60 "$SERVE_CTR" >> "$LOG" 2>&1
peer docker stop -t 60 "$SERVE_CTR" >> "$LOG" 2>&1

if [ "${EVAL_ONLY:-0}" = 1 ]; then
  say "EVAL_ONLY: skipping training, evaluating existing $RUN/adapter-final"
else
say "training $RUN on 2 nodes ($TRAINER)"
if [ "$TRAINER" = dpo ]; then
  to_peer "$SFT_ADAPTER"; to_peer "$PAIRS"
fi
# Only rank 0 writes checkpoints. A resumed run needs the same checkpoint on the
# peer, or rank 1 restarts from step 0 and the ranks disagree on the steps left.
if [ -f "$WORK/$RUN/adapter-latest/STEP" ]; then
  to_peer "$RUN/adapter-latest"
  say "resuming: copied $RUN/adapter-latest (step $(cat "$WORK/$RUN/adapter-latest/STEP")) to the peer"
fi
peer "$(torchrun_cmd 1)" > "$WORK/$RUN.rank1.out" 2>&1 &
rank1=$!
$(torchrun_cmd 0) > "$WORK/$RUN.out" 2>&1
# Rank 1 can exit non-zero after a successful run: torchrun's exit barrier
# reports "connection closed" once rank 0 has already left. Rank 0's adapter
# is the source of truth for success.
if ! wait $rank1; then
  if [ -f "$WORK/$RUN/adapter-final/adapter_model.safetensors" ]; then
    say "rank 1 exited non-zero after rank 0 finished (exit-barrier noise); continuing"
  else
    say "rank 1 failed and rank 0 produced no adapter"; exit 1
  fi
fi
say "training done"
fi

if [ -n "$EVAL_ARMS_MASTER" ]; then
  # The peer needs the new adapter for its arms. The container writes as root
  # with owner-only permissions, so hand the outputs back before copying: on
  # lkml-reviewer run-3 the scp below failed on a -rw------- root file and the
  # chain stopped before evaluation.
  docker exec "$TRAIN_CTR" chown -R "$(id -u):$(id -g)" /work >> "$LOG" 2>&1
  # scp -r into an existing directory nests the copy inside it, and a stale
  # partial copy from an earlier failed run made the peer load config-only
  # (lkml-reviewer run-3b eval: "Repository Not Found ... adapter_model.bin").
  peer "rm -rf $(basename "$WORK")/$RUN-adapter-final"
  scp -q -r "$WORK/$RUN/adapter-final" "$PEER:$(basename "$WORK")/$RUN-adapter-final"
  say "evaluating: master arms [$EVAL_ARMS_MASTER], peer arms [$EVAL_ARMS_PEER]"
  cmp_cmd() {  # $1 = arms, $2 = adapters, $3 = out file, $4 = eval file
    echo docker exec -w /work "$TRAIN_CTR" python3 compare.py --eval "${4:-sft_eval.jsonl}" \
      --subset "$EVAL_SUBSET" --modes sampled --arms "$1" --adapters $2 --out "$3" ${EVAL_EXTRA:-}
  }
  if [ -n "$EVAL_ARMS_PEER" ]; then
    peer_job="$(cmp_cmd "$EVAL_ARMS_PEER" "$EVAL_ADAPTERS_PEER" "$RUN-compare-peer.jsonl" "${EVAL_FILE_PEER:-sft_eval.jsonl}")"
    for pair in ${PEER_EVAL_LOSS:-}; do  # name=adapter-path ("base=" for none)
      name=${pair%%=*}; path=${pair#*=}
      peer_job="$peer_job && docker exec -w /work $TRAIN_CTR python3 eval_loss.py --eval sft_eval.jsonl ${path:+--adapter $path} ${EVAL_LOSS_LIMIT:+--limit $EVAL_LOSS_LIMIT} --out $RUN-loss-$name.json"
    done
    peer "$peer_job" > "$WORK/$RUN.compare-peer.out" 2>&1 &
    cpeer=$!
  fi
  $(cmp_cmd "$EVAL_ARMS_MASTER" "$EVAL_ADAPTERS_MASTER" "$RUN/compare-master.jsonl" "${EVAL_FILE_MASTER:-sft_eval.jsonl}") \
    > "$WORK/$RUN.compare-master.out" 2>&1
  [ -n "$EVAL_ARMS_PEER" ] && wait $cpeer
  [ -n "$EVAL_ARMS_PEER" ] && scp -q "$PEER:$(basename "$WORK")/$RUN-compare-peer.jsonl" \
    "$WORK/$RUN/compare-peer.jsonl"
  for pair in ${PEER_EVAL_LOSS:-}; do
    scp -q "$PEER:$(basename "$WORK")/$RUN-loss-${pair%%=*}.json" "$WORK/$RUN/" || true
  done
  say "evaluation done"
fi
say "CHAIN DONE"
