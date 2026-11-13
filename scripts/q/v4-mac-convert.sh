#!/usr/bin/env bash
# Merge v4 into the BF16 base on the Mac and convert to MLX 8-bit (detached on
# the Mac, polled from here, so a dropped ssh session doesn't kill it).
. "$(dirname "$0")/lib.sh"
free=$(on_mac "df -g ~ | tail -1 | awk '{print \$4}'")
if [ "$free" -lt 90 ]; then   # merged BF16 (55 GB) + MLX (27 GB) at peak
  say "only ${free} GB free on the Mac: removing the stock mlx-community 8-bit test model (re-downloadable)"
  on_mac "rm -rf ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-8bit"
fi
on_mac "rm -rf ~/lkml-mac/v4-adapter"
to_mac "$QROOT/data/train/run-4/adapter-final" lkml-mac/v4-adapter
on_mac "cd ~/lkml-mac && (git -C repo pull -q || true) && PYTHON=\$HOME/lkml-mac/.venv/bin/python nohup repo/scripts/convert-mlx.sh --base Qwen/Qwen3.8-27B --adapter v4-adapter --out \$HOME/lkml-mac/v4 --bits 8 > convert-v4.log 2>&1 < /dev/null &"
sleep 60
while on_mac "pgrep -f '[c]onvert-mlx.sh.*v4-adapter' >/dev/null"; do sleep 120; done
on_mac "tail -5 ~/lkml-mac/convert-v4.log; test -f ~/lkml-mac/v4/mlx-8bit/config.json"
