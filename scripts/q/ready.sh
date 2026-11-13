#!/usr/bin/env bash
# Readiness checks for QUEUE.toml `when` (exit 0 = ready).
#   ready.sh v4done   v4's results exist and nothing runs on the cluster
#   ready.sh cluster  nothing runs on the cluster
#   ready.sh mac      nothing runs on the Mac
. "$(dirname "$0")/lib.sh" > /dev/null
case "${1:-}" in
  v4done) on_master "test -f ~/$W/RESULTS-v4.md" && master_idle ;;
  cluster) master_idle ;;
  mac) mac_idle ;;
  *) echo "usage: $0 v4done|cluster|mac" >&2; exit 2 ;;
esac
