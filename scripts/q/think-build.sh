#!/usr/bin/env bash
# Thinking test, step 1: the untuned base as MLX 8-bit on the Mac, converted
# straight from the cached BF16 base (not convert-mlx.sh --merged, which would
# delete its input: here, the only copy of the base).
. "$(dirname "$0")/lib.sh"
on_mac "test -f ~/lkml-mac/base/mlx-8bit/config.json" && { say "base MLX build already there"; exit 0; }
on_mac "cd ~/lkml-mac && mkdir -p base && nohup .venv/bin/python -m mlx_lm convert --hf-path Qwen/Qwen3.8-27B --mlx-path \$HOME/lkml-mac/base/mlx-8bit -q --q-bits 8 --q-group-size 64 > base/convert.log 2>&1 < /dev/null &"
sleep 60
while on_mac "pgrep -f '[m]lx_lm convert --hf-path Qwen/Qwen3.8-27B' >/dev/null"; do sleep 60; done
on_mac "tail -3 ~/lkml-mac/base/convert.log; test -f ~/lkml-mac/base/mlx-8bit/config.json"
