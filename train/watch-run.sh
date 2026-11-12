#!/bin/bash
# Health watcher for a long cluster run. Runs OFF the cluster, so a frozen node
# can't take it down with it. Exits (which is the alert) on the first problem,
# or when the chain finishes. Written after run-3: rank 0's node ran out of
# memory and froze ~3 h into a 14 h run, and a wait-for-"done" watcher never
# fired, so the serving stack stayed down for 12 hours unnoticed.
#
#   MASTER=user@master PEER=user@peer RUN=run-3b WORKDIR=linuspr-train \
#     train/watch-run.sh
#
# Exit status: 0 = chain finished, 2 = node unreachable, 3 = low memory,
#              4 = stalled (no log activity), 5 = chain finished with an error
set -u
: "${MASTER:?}" "${PEER:?}" "${RUN:?}"
WORKDIR=${WORKDIR:-linuspr-train}
INTERVAL=${INTERVAL:-300}        # seconds between checks
STALL_MIN=${STALL_MIN:-30}       # no file under the work dir touched for this long
MIN_AVAIL_GB=${MIN_AVAIL_GB:-10} # available host memory floor, per node
fails_master=0; fails_peer=0

probe() {  # $1 = host -> prints "avail_gb newest_mtime_age_min chain_tail" or fails
  # timeout wraps ssh itself: `timeout probe` can't run a shell function, and
  # the first version reported a healthy node as unreachable because of it.
  timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" "
    avail=\$(awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo)
    newest=\$(find ~/$WORKDIR -maxdepth 3 -newermt '-$STALL_MIN minutes' -type f 2>/dev/null | head -1)
    tailline=\$(tail -1 ~/$WORKDIR/chain-$RUN.log 2>/dev/null)
    echo \"\$avail|\${newest:+fresh}|\$tailline\""
}

while true; do
  now=$(date +%T)
  m=$(probe "$MASTER") || m=""
  p=$(probe "$PEER") || p=""
  if [ -z "$m" ]; then fails_master=$((fails_master + 1)); else fails_master=0; fi
  if [ -z "$p" ]; then fails_peer=$((fails_peer + 1)); else fails_peer=0; fi
  if [ $fails_master -ge 2 ]; then echo "[$now] ALERT: master unreachable twice in a row"; exit 2; fi
  if [ $fails_peer -ge 2 ]; then echo "[$now] ALERT: peer unreachable twice in a row"; exit 2; fi
  if [ -n "$m" ] && [ -n "$p" ]; then
    m_avail=${m%%|*}; p_avail=${p%%|*}
    m_rest=${m#*|}; m_fresh=${m_rest%%|*}; m_tail=${m_rest#*|}
    echo "[$now] ok: master avail ${m_avail} GB, peer avail ${p_avail} GB, log ${m_fresh:-STALE}, chain: ${m_tail}"
    if [ "$m_avail" -lt "$MIN_AVAIL_GB" ] || [ "$p_avail" -lt "$MIN_AVAIL_GB" ]; then
      echo "[$now] ALERT: available memory under ${MIN_AVAIL_GB} GB"; exit 3
    fi
    case "$m_tail" in
      *"start issued"*)
        if ssh -o BatchMode=yes "$MASTER" "grep -q 'exit status 0' ~/$WORKDIR/chain-$RUN.log"; then
          echo "[$now] chain finished cleanly"; exit 0
        else
          echo "[$now] ALERT: chain finished with an error"; exit 5
        fi ;;
    esac
    if [ -z "$m_fresh" ]; then echo "[$now] ALERT: no log activity for ${STALL_MIN} min"; exit 4; fi
  fi
  sleep "$INTERVAL"
done
