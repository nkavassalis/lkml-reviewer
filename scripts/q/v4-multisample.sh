#!/usr/bin/env bash
# Start (detached, on the Mac) v4 as trained, patch-first: 3 samples x the same
# 200 held-out patches as v3's runs. Returns at once so the queue isn't held
# for the ~10 hours it takes; multisample-report waits for the Mac to be idle.
. "$(dirname "$0")/lib.sh"
to_mac "$QROOT/data/v4/sft_eval.jsonl" lkml-mac/sft_eval_v4.jsonl
on_mac "cd ~/lkml-mac && rm -f ms-v4.jsonl && nohup .venv/bin/python repo/train/compare_mlx.py --model v4/mlx-8bit --eval sft_eval_v4.jsonl --subset 200 --samples 3 --key v4 --out ms-v4.jsonl > ms-v4.log 2>&1 < /dev/null &"
sleep 90
on_mac "pgrep -f '[c]ompare_mlx.py.*ms-v4.jsonl' >/dev/null" || { on_mac "tail -20 ~/lkml-mac/ms-v4.log"; exit 1; }
say "started on the Mac; multisample-report runs when it finishes"
