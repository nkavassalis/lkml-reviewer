#!/usr/bin/env bash
# Readiness checks for QUEUE.toml `when` (exit 0 = ready).
#   ready.sh v4done   v4's results exist and nothing runs on the cluster
#   ready.sh v4adapter v4's final adapter exists (training finished)
#   ready.sh cluster  nothing runs on the cluster
#   ready.sh mac      nothing runs on the Mac
#   ready.sh v5mac    v5's MLX build exists on the Mac and nothing runs there
. "$(dirname "$0")/lib.sh" > /dev/null
case "${1:-}" in
  v4done) on_master "test -f ~/$W/RESULTS-v4.md" && master_idle ;;
  v4adapter) on_master "test -f ~/$W/run-4/adapter-final/adapter_model.safetensors" ;;
  cluster) master_idle ;;
  mac) mac_idle ;;
  v5mac) on_mac 'test -f ~/lkml-mac/v5/mlx-8bit/config.json' && mac_idle ;;
  *) echo "usage: $0 v4done|v4adapter|cluster|mac|v5mac" >&2; exit 2 ;;
esac
