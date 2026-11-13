#!/usr/bin/env bash
# Start (detached, on the Mac) greedy decoding for v3, v4 and v5 on the same 100
# held-out patches, to measure looping (v2 greedy: 44% vs base 0%; REPORT §6).
# Each model gets the eval layout it was trained on. greedy-loops-report reads it.
. "$(dirname "$0")/lib.sh"
on_mac "cd ~/lkml-mac && (git -C repo pull -q || true) && rm -f greedy-*.jsonl && nohup bash -c '
  for m in v3:sft_eval_v3.jsonl v4:sft_eval_v4.jsonl v5:sft_eval_v4.jsonl; do
    .venv/bin/python repo/train/compare_mlx.py --greedy --model \${m%%:*}/mlx-8bit --eval \${m#*:} \
      --subset 100 --key \${m%%:*}_greedy --out greedy-\${m%%:*}.jsonl
  done' > greedy.log 2>&1 < /dev/null &"
sleep 90
on_mac "pgrep -f '[c]ompare_mlx.py --greedy' >/dev/null" || { on_mac "tail -20 ~/lkml-mac/greedy.log"; exit 1; }
say "started on the Mac"
