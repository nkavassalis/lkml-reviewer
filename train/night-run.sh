#!/usr/bin/env bash
# Run a plan inside the cluster's maintenance window, with a hard stop.
#
#   PLAN=night-plan.sh PEER=user@peer setsid nohup train/night-run.sh > /dev/null 2>&1 < /dev/null &
#
# 1. Waits until the window opens (train/window-lib.sh: default 20:00-08:00,
#    cluster timezone). Launch it any time of day; set WAIT_FOR_NEXT_WINDOW=1
#    to skip a window that's already open (e.g. launching at 06:00 for tonight).
# 2. Runs PLAN (a bash script of steps, usually chain invocations) in its own
#    process group. The plan can use need_minutes/window_remaining_min to skip
#    steps that won't fit.
# 3. At the hard stop (window end minus STOP_MARGIN_MIN) it terminates the plan.
#    The chain scripts' EXIT traps kill leftover workers and restore serving;
#    long training resumes next night from its last checkpoint (--resume-auto).
# 4. Safety net, whatever happened: makes sure SERVE_CTR is running on this
#    node and on PEER before it exits.
#
# Run now instead ("go for it"): GO_NOW=1 starts immediately, inside the window
# or not, with a hard stop after MAX_HOURS (unset = no hard stop). Running
# outside the agreed window needs the machines' owner's OK; the log records
# that it was started by request. Reporting is identical either way.
#
# Logs to runs/night-YYYY-MM-DD.log (or ./night-*.log without runs/).
set -u
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/window-lib.sh"
: "${PLAN:?set PLAN to the plan script for the night}"
PEER=${PEER:-}
SERVE_CTR=${SERVE_CTR:-vllm_fp4local}
LOGDIR=${LOGDIR:-$( [ -d runs ] && echo runs || echo . )}
if [ "${GO_NOW:-0}" = 1 ]; then
  LOG="$LOGDIR/now-$(TZ=$WINDOW_TZ date +%F-%H%M).log"
else
  LOG="$LOGDIR/night-$(TZ=$WINDOW_TZ date +%F).log"
fi
# Reporters follow this link, so they track the current run in either mode.
ln -sf "$(basename "$LOG")" "$LOGDIR/latest-run.log"
say() { echo "[$(TZ=$WINDOW_TZ date '+%F %T')] $*" >> "$LOG"; }

ensure_serving() {
  for host in local $PEER; do
    if [ "$host" = local ]; then
      running=$(docker inspect -f '{{.State.Running}}' "$SERVE_CTR" 2>/dev/null)
      [ "$running" = true ] || { docker start "$SERVE_CTR" >> "$LOG" 2>&1; say "safety net: started $SERVE_CTR locally"; }
    else
      running=$(ssh -o BatchMode=yes "$host" "docker inspect -f '{{.State.Running}}' $SERVE_CTR" 2>/dev/null)
      [ "$running" = true ] || { ssh -o BatchMode=yes "$host" docker start "$SERVE_CTR" >> "$LOG" 2>&1; say "safety net: started $SERVE_CTR on $host"; }
    fi
  done
}

# WAIT_FOR_NEXT_WINDOW=1: don't use a window that is already open (launching
# in the morning for tonight would otherwise start at once, with minutes left).
if [ "${WAIT_FOR_NEXT_WINDOW:-0}" = 1 ]; then opens=$(window_next_start); else opens=$(window_next_open)
[ "${GO_NOW:-0}" = 1 ] && opens=$(date +%s); fi
if [ "$opens" -gt "$(date +%s)" ]; then
  wait_min=$(( (opens - $(date +%s)) / 60 ))
  say "waiting for the window to open at $WINDOW_START ($wait_min min)"
  sleep $(( opens - $(date +%s) ))
fi
deadline=$(window_deadline)
if [ "${GO_NOW:-0}" = 1 ]; then
  if [ -n "${MAX_MINUTES:-}" ]; then deadline=$(( $(date +%s) + MAX_MINUTES * 60 ))
  elif [ -n "${MAX_HOURS:-}" ]; then deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
  else deadline=$(( $(date +%s) + 365 * 86400 )); fi
  window_open || say "GO_NOW: running OUTSIDE the maintenance window, by request"
  say "GO_NOW: started now; hard stop at $(TZ=$WINDOW_TZ date -d @"$deadline" '+%F %T')${MAX_HOURS:+ (MAX_HOURS=$MAX_HOURS)}; running $PLAN"
else
  say "window open; hard stop at $(TZ=$WINDOW_TZ date -d @"$deadline" '+%F %T'); running $PLAN"
fi

export PLAN_DEADLINE=$deadline   # need_minutes in the plan uses this (window-lib.sh)
setsid bash "$PLAN" >> "$LOG" 2>&1 &
plan_pid=$!
while kill -0 "$plan_pid" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "HARD STOP reached: terminating the plan (its chains restore serving)"
    kill -TERM -- -"$plan_pid" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$plan_pid" 2>/dev/null || break; sleep 5; done
    kill -KILL -- -"$plan_pid" 2>/dev/null && say "plan didn't exit in 5 min; killed"
    break
  fi
  sleep 30
done
wait "$plan_pid" 2>/dev/null; say "plan finished (exit status $?)"
sleep 10
ensure_serving
say "night done; serving confirmed on all nodes"
