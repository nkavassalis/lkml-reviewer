#!/usr/bin/env bash
# Shared helpers for queue items. Addresses come from .status.env (gitignored).
set -u
QROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_l() { while IFS= read -r line || [ -n "$line" ]; do case "$line" in ''|\#*) continue;; esac
       k=${line%%=*}; [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue; [ -n "${!k+x}" ] || eval "$line"; done < "$1"; }
[ -f "$QROOT/.status.env" ] && _l "$QROOT/.status.env"
: "${MASTER:?MASTER missing from .status.env}" "${MAC:?MAC missing from .status.env}"
W=${WORKDIR:-linuspr-train}
PEER_FAST=${PEER_FAST:-}        # peer address on the inter-node link, as seen from the master
on_master() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$MASTER" "$@"; }
# shellcheck disable=SC2086
on_mac() { ssh -o BatchMode=yes -o ConnectTimeout=20 $MAC "$@"; }
# MAC is "[ssh options] user@host": scp gets the same options.
MAC_HOST=${MAC##* }; MAC_OPTS=""; [ "$MAC_HOST" = "$MAC" ] || MAC_OPTS=${MAC% *}
# shellcheck disable=SC2086
to_mac() { scp -q -r -o BatchMode=yes $MAC_OPTS "$1" "$MAC_HOST:$2"; }
# shellcheck disable=SC2086
from_mac() { scp -q -r -o BatchMode=yes $MAC_OPTS "$MAC_HOST:$1" "$2"; }
say() { echo "[$(date '+%F %T')] $*"; }
# Nothing else running on the cluster: no launcher, chain or training process.
# (bracketed first letters: the pattern must not match the remote shell running pgrep)
master_idle() { ! on_master "pgrep -f '[n]ight-run.sh|[r]un-chain|[s]ft_lora.py|[c]ompare.py|[d]po_lora.py' >/dev/null"; }
mac_idle() { ! on_mac "pgrep -f '[c]ompare_mlx.py|[e]val_loss_mlx.py|[c]onvert-mlx.sh|[m]ulti-sample-queue.sh' >/dev/null"; }
# Run a cluster plan now, inside the window, with a hard stop sized to the time
# left (WINDOW_MINUTES_LEFT from the queue runner); waits for it to finish.
# Copy train/<files> to the master's and the peer's work dirs. Each lands under a
# temporary name and is renamed into place: bash reads a running script as it
# goes, and a chain may be running from the old copy (docs/LEARNINGS.md).
deploy_train() {
  : "${PEER_FAST:?PEER_FAST missing from .status.env}"
  local f
  for f in "$@"; do
    scp -q "$QROOT/train/$f" "$MASTER:$W/.$f.new" || return 1
    on_master "cd ~/$W && chmod --reference=$f .$f.new 2>/dev/null; mv .$f.new $f && \
      scp -q $f $PEER_FAST:$W/.$f.new && ssh -o BatchMode=yes $PEER_FAST 'cd $W && mv .$f.new $f'" || return 1
  done
  say "deployed to both nodes: $*"
}

cluster_plan() {
  local plan=$1 mins=$(( ${WINDOW_MINUTES_LEFT:-60} - 10 ))
  [ "$mins" -ge 30 ] || { say "not enough window left ($mins min)"; return 75; }
  scp -q "$QROOT/data/cluster/$plan" "$MASTER:$W/$plan"
  on_master "cd ~/$W && chmod +x $plan && GO_NOW=1 MAX_MINUTES=$mins PLAN=$plan PEER=$PEER_FAST setsid nohup ./night-run.sh > /dev/null 2>&1 < /dev/null &" < /dev/null
  sleep 60
  until on_master "grep -q 'night done' ~/$W/latest-run.log"; do sleep 120; done
  on_master "tail -30 ~/$W/latest-run.log"
  # A round that overran the hard stop is redone next window, not a failure.
  on_master "grep -q 'HARD STOP reached' ~/$W/latest-run.log" && return 75
  on_master "grep -q 'plan finished (exit status 0)' ~/$W/latest-run.log"
}
