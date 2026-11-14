#!/usr/bin/env bash
# Thinking test, step 2 (detached on the Mac): the base on the same 30
# held-out patches, without and then with reasoning. Same sampling; the
# thinking run gets room to reason (6,144 new tokens). Returns at once.
. "$(dirname "$0")/lib.sh"
to_mac "$QROOT/data/v4/sft_eval.jsonl" lkml-mac/sft_eval_v4.jsonl
on_mac "cd ~/lkml-mac && (git -C repo pull -q || true) && grep -q -- '--thinking' repo/train/compare_mlx.py" \
  || { say "the Mac's repo has no compare_mlx.py --thinking (push first)"; exit 1; }
on_mac "cd ~/lkml-mac && nohup bash -c '.venv/bin/python repo/train/compare_mlx.py --model base/mlx-8bit --eval sft_eval_v4.jsonl --subset 30 --key base_nothink --max-new 1024 --out think-off.jsonl && .venv/bin/python repo/train/compare_mlx.py --model base/mlx-8bit --eval sft_eval_v4.jsonl --subset 30 --key base_think --max-new 6144 --thinking --out think-on.jsonl' > think.log 2>&1 < /dev/null &"
sleep 90
on_mac "pgrep -f '[c]ompare_mlx.py.*think-o' >/dev/null" || { on_mac "tail -20 ~/lkml-mac/think.log"; exit 1; }
say "started on the Mac; think-report runs when both files have 30 rows"
