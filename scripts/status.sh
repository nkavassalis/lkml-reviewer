#!/usr/bin/env bash
# One-shot, read-only status of a long training run. Changes nothing.
# Meant to be run by hand or by an hourly scheduled job (docs/OPERATIONS.md).
#
# Hosts and run names come from .status.env (gitignored; addresses never go in
# the repo). Example .status.env:
#   MASTER=user@master-host     PEER=user@peer-host     RUN=run-3b
#   WORKDIR=linuspr-train       SERVE_URL=http://master-host:8000/v1/models
#   WATCH_LOG=data/watch-run-3b.log
#   MAC="-i ~/.ssh/key user@mac-host"   (optional)
#
# Exit status: 0 all normal, 1 something needs attention (printed as ATTENTION).
set -u
root="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$root/.status.env" ] && . "$root/.status.env"
: "${MASTER:?set MASTER in .status.env}" "${RUN:?set RUN in .status.env}"
PEER=${PEER:-}; WORKDIR=${WORKDIR:-linuspr-train}; WATCH_LOG=${WATCH_LOG:-}
attention=0
flag() { echo "ATTENTION: $*"; attention=1; }
remote() { timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=15 "$@"; }

echo "== status at $(date '+%Y-%m-%d %H:%M %Z') =="

# --- training ----------------------------------------------------------------
t=$(remote "$MASTER" "
  L=~/$WORKDIR/$RUN/train.log; C=~/$WORKDIR/chain-$RUN.log
  echo \"chain: \$(tail -1 \$C 2>/dev/null)\"
  echo \"step:  \$(grep -E 'step [0-9]+/' \$L 2>/dev/null | tail -1)\"
  echo \"ckpt:  \$(grep -E 'checkpoint at step' \$L 2>/dev/null | tail -1)\"
  echo \"eval:  \$(grep -E 'eval loss' \$L 2>/dev/null | tail -1)\"
  echo \"age:   \$(( \$(date +%s) - \$(stat -c %Y \$L 2>/dev/null || echo 0) ))\"
") || flag "master unreachable over ssh"
if [ -n "${t:-}" ]; then
  echo "$t" | grep -v '^age:' | cut -c1-170
  age=$(echo "$t" | awk '/^age:/ {print $2}')
  case "$t" in *"start issued"*) echo "run: FINISHED (serving restore issued)" ;; *)
    [ "${age:-99999}" -gt 1800 ] && flag "training log hasn't moved for $((age / 60)) min" ;;
  esac
fi

# --- memory on each node --------------------------------------------------------
for host in "$MASTER" $PEER; do
  avail=$(remote "$host" "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo") \
    || { flag "$host unreachable"; continue; }
  echo "mem:   $host available ${avail} GB"
  [ "$avail" -lt 15 ] && flag "$host has only ${avail} GB available"
done

# --- serving stack ----------------------------------------------------------------
if [ -n "${SERVE_URL:-}" ]; then
  if curl -sf -m 8 "$SERVE_URL" >/dev/null; then echo "serve: UP"; else echo "serve: down"; fi
fi

# --- watcher ------------------------------------------------------------------------
if [ -n "$WATCH_LOG" ]; then
  if pgrep -f "watch-run.sh" >/dev/null; then
    echo "watch: running; last: $(tail -1 "$root/$WATCH_LOG" 2>/dev/null | cut -c1-120)"
  else
    case "${t:-}" in *"start issued"*) echo "watch: exited (run finished)" ;;
      *) flag "watcher is not running: $(tail -1 "$root/$WATCH_LOG" 2>/dev/null)" ;; esac
  fi
fi

# --- optional: Apple-silicon box --------------------------------------------------
if [ -n "${MAC:-}" ]; then
  # shellcheck disable=SC2086
  m=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 $MAC \
      'memory_pressure | grep -E "free percentage"; df -h ~ | tail -1 | awk "{print \$4\" free disk\"}"') \
    && echo "mac:   $(echo "$m" | tr '\n' ' ')" || echo "mac:   unreachable"
fi

[ $attention = 0 ] && echo "== all normal ==" || echo "== NEEDS ATTENTION =="
exit $attention
