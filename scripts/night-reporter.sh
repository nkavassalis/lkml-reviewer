#!/usr/bin/env bash
# Watch a multi-night cluster plan from OFF the cluster and send reports:
#   PROBLEM   at once, when scripts/status.sh raises ATTENTION or the night's
#             log shows a failure (once per distinct problem per night)
#   Night     each morning, once the night's log says the night is done
#   FINISHED  when RESULTS_FILE appears on the master; then the reporter exits
#
#   scripts/night-reporter.sh start | stop | status | once
#
# `start` runs a supervised setsid loop (survives logout; relaunches itself if
# it dies), like a service on a box without cron or systemd. Settings come
# from .status.env: MASTER, WORKDIR, RUN, NOTIFY_CMD, and
#   RESULTS_FILE   path on the master, relative to ~/$WORKDIR (default RESULTS.md)
#   REPORT_EVERY   seconds between checks (default 300)
set -u
root="$(cd "$(dirname "$0")/.." && pwd)"
# .status.env supplies defaults only: anything already set in the environment
# wins (so a test can point at a scratch dir or NOTIFY_CMD=cat without editing it).
load_env() {
  local line key
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    key=${line%%=*}
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [ -n "${!key+x}" ] || eval "$line"
  done < "$1"
}
[ -f "$root/.status.env" ] && load_env "$root/.status.env"
export NOTIFY_CMD
: "${MASTER:?set MASTER in .status.env}"
WORKDIR=${WORKDIR:-linuspr-train}
RESULTS_FILE=${RESULTS_FILE:-RESULTS.md}
REPORT_EVERY=${REPORT_EVERY:-300}
STATE="${REPORTER_STATE:-$root/data/reporter}"; mkdir -p "$STATE"   # override for tests
PIDFILE="$STATE/reporter.pid"
notify() { "$root/scripts/notify.sh" "[lkml-reviewer] $1"; }
remote() { timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=20 "$MASTER" "$@"; }
sent() { [ -f "$STATE/sent-$1" ]; }
mark() { date '+%F %T' > "$STATE/sent-$1"; }
key() { printf '%s' "$1" | md5sum | cut -c1-12; }

# Per-run-log reports: GO_NOW start, plan failure, end of night.
# $1 = run id (night-DATE / now-DATE-HHMM), $2 = its log, $3 = status.sh output
check_log() {
  local nightlog=$2

  # 2b. a run started outside the window ("go for it"): say so at the start
  if echo "$nightlog" | grep -q "GO_NOW: started" && ! sent "started-$1"; then
    { echo "A run was started now, by request, rather than waiting for the window:"; echo;
      echo "$nightlog" | head -5; } | notify "Started: $1" && mark "started-$1"
  fi
  # Exact, case-sensitive markers: "hard stop at 14:00" in a healthy start line
  # must not count. Keyed on the FIRST failure line, so one failure = one email
  # even as the log grows ("FAILED" then "plan finished (exit status 1)").
  local fail_re='FAILED|HARD STOP reached|Traceback|exit status [1-9]'
  if echo "$nightlog" | grep -qE "$fail_re"; then
    local k; k="planfail-$1-$(key "$(echo "$nightlog" | grep -E "$fail_re" | head -1)")"
    if ! sent "$k"; then
      { echo "The night's plan logged a failure or hit the hard stop:"; echo;
        echo "$nightlog" | tail -40; } | notify "PROBLEM in tonight's plan ($1)" && mark "$k"
    fi
  fi

  # 4. Night report: once the launcher logged the end of the night
  if echo "$nightlog" | grep -q "night done" && ! sent "night-$1"; then
    { echo "Night of $1: done. vLLM restored on all nodes (checked by the launcher)."; echo;
      echo "--- plan log:"; echo "$nightlog" | tail -60; echo;
      echo "--- training ($RUN):"; remote "grep -E 'step [0-9]+/|eval loss|checkpoint at|DONE' ~/$WORKDIR/${RUN}/train.log 2>/dev/null | tail -6"; echo;
      echo "--- status now:"; echo "$3"; } | notify "Report: $1" && mark "night-$1"
  fi
}

check_once() {
  local night status rc
  # Follow the launcher's latest-run.log link: the same reporting for a
  # windowed night (night-DATE.log) and a "go for it" run (now-DATE-HHMM.log).
  night=$(remote "readlink ~/$WORKDIR/latest-run.log 2>/dev/null" | sed 's/\.log$//')
  [ -n "$night" ] || night="no-run-yet"

  # 1. FINISHED: results are in
  if remote "test -f ~/$WORKDIR/$RESULTS_FILE"; then
    if ! sent "finished"; then
      remote "cat ~/$WORKDIR/$RESULTS_FILE" | notify "FINISHED: results are in" && mark finished
    fi
    return 10
  fi

  # 2. PROBLEM: status.sh alarms (it knows when a run stage is active)
  status=$("$root/scripts/status.sh" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then
    local k; k="problem-$night-$(key "$(echo "$status" | grep ATTENTION)")"
    if ! sent "$k"; then
      { echo "status.sh reports a problem:"; echo; echo "$status"; echo;
        echo "--- tonight's plan log (tail):"; remote "tail -25 ~/$WORKDIR/latest-run.log" 2>&1; } \
        | notify "PROBLEM on the cluster ($night)" && mark "$k"
    fi
  fi

  # 3-4. Read the newest TWO run logs, not only the one latest-run.log points
  # at: a multi-night loop starts the next night seconds after "night done",
  # repointing the link long before the next check (a missed report, 2026-11-14).
  local log
  for log in $(remote "cd ~/$WORKDIR/ && ls -t night-*.log now-*.log 2>/dev/null | head -2"); do
    check_log "${log%.log}" "$(remote "cat ~/$WORKDIR/$log")" "$status"
  done
  return 0
}

case "${1:-status}" in
  once) check_once; echo "check rc=$?";;
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
    setsid nohup bash -c "
      while :; do
        bash -c '
          while :; do
            \"$root/scripts/night-reporter.sh\" once >> \"$STATE/reporter.log\" 2>&1
            grep -q \"check rc=10\" <(tail -1 \"$STATE/reporter.log\") && exit 0
            sleep $REPORT_EVERY
          done' && break
        echo \"\$(date) supervisor: reporter loop exited, relaunching\" >> \"$STATE/reporter.log\"
        sleep 30
      done; echo \"\$(date) reporter: results delivered, stopping\" >> \"$STATE/reporter.log\"" >/dev/null 2>&1 &
    echo $! > "$PIDFILE"; echo "reporter started (pid $!, every ${REPORT_EVERY}s)";;
  stop) [ -f "$PIDFILE" ] && kill -- -"$(cat "$PIDFILE")" 2>/dev/null; kill "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; rm -f "$PIDFILE"; echo stopped;;
  status) if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "running (pid $(cat "$PIDFILE"))"; tail -3 "$STATE/reporter.log"; else echo "not running"; fi;;
  *) echo "usage: $0 start|stop|status|once" >&2; exit 2;;
esac
