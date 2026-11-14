#!/usr/bin/env bash
# Run the improvement queue (QUEUE.toml) forever, one item at a time, on the
# monitoring box. Emails (scripts/notify.sh): Started, Done, FAILED (the queue
# then halts until `workqueue.py reset|skip`), and "Needs you" at a human gate.
#
#   scripts/queue-runner.sh start | stop | status | once
#
# Window items start only inside the maintenance window (train/window-lib.sh)
# and only if their estimate fits before the hard stop; they get
# WINDOW_MINUTES_LEFT in the environment to size their own hard stop
# (e.g. GO_NOW=1 MAX_HOURS=... night-run.sh on the cluster).
#
# Downtime allowed outside the window, for a while ("start everything now"):
#   echo "2026-11-15 07:40 EST" > $STATE_DIR/open-until
# Until then window items may start, sized to that time (or the window's own
# hard stop, if later). Expired or unreadable = ignored; delete it to cancel.
set -u
root="$(cd "$(dirname "$0")/.." && pwd)"
. "$root/train/window-lib.sh"
STATE_DIR="${QUEUE_RUNNER_STATE:-$root/data/queue/runner}"; mkdir -p "$STATE_DIR"
PIDFILE="$STATE_DIR/runner.pid"
EVERY=${QUEUE_EVERY:-300}
PY=${PY:-python3}
wq() { "$PY" "$root/train/workqueue.py" "$@"; }
notify() { "$root/scripts/notify.sh" "${NOTIFY_PREFIX:-[queue]} $1"; }
title() { "$PY" -c "import sys,tomllib;q=tomllib.load(open(sys.argv[1],'rb'));print(next(i['title'] for i in q['item'] if i['id']==sys.argv[2]))" "${QUEUE_FILE:-$root/QUEUE.toml}" "$1"; }

open_until() {
  local f="$STATE_DIR/open-until" u
  [ -f "$f" ] || return 1
  u=$(date -d "$(cat "$f")" +%s 2>/dev/null) || return 1
  [ "$u" -gt "$(_now)" ] && echo "$u"
}

once() {
  local open=0 left=0 out id u o
  window_open && open=1 && left=$(window_remaining_min)
  if u=$(open_until); then
    open=1; o=$(( (u - $(_now)) / 60 )); [ "$o" -gt "$left" ] && left=$o
  fi
  out=$(wq next --window-open $open --minutes-left "$left")
  for w in $(echo "$out" | awk '/^WAITING/ {print $2}'); do
    wq waiting "$w" > /dev/null
    { echo "The queue reached a step that needs a person: $(title "$w")."; echo;
      echo "When it's done:  python3 train/workqueue.py approve $w"; echo;
      wq status; } | notify "Needs you: $(title "$w")"
  done
  id=$(echo "$out" | awk '/^RUN/ {print $2}')
  [ -n "$id" ] || return 0
  { echo "Starting $(title "$id") (window open=$open, $left min left)."; echo; wq status; } | notify "Started: $(title "$id")"
  WINDOW_MINUTES_LEFT=$left wq run "$id"; rc=$?
  if [ $rc -eq 0 ]; then
    { echo "Done: $(title "$id")"; echo; tail -20 "${QUEUE_LOGS:-$root/data/queue/logs}/$id.log"; echo; wq status; } | notify "Done: $(title "$id")"
  elif [ $rc -eq 75 ]; then
    { echo "Paused at the window's end, resumes in the next window: $(title "$id")"; echo;
      tail -15 "${QUEUE_LOGS:-$root/data/queue/logs}/$id.log"; } | notify "Paused (resumes next window): $(title "$id")"
  else
    { echo "FAILED: $(title "$id"). The queue is halted until someone looks:"
      echo "  python3 train/workqueue.py reset $id   (retry)   or   skip $id"; echo
      tail -40 "${QUEUE_LOGS:-$root/data/queue/logs}/$id.log"; echo; wq status; } | notify "FAILED: $(title "$id")"
  fi
}

case "${1:-status}" in
  once) once ;;
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
    setsid nohup bash -c "
      while :; do
        bash -c 'while :; do \"$root/scripts/queue-runner.sh\" once >> \"$STATE_DIR/runner.log\" 2>&1; sleep $EVERY; done'
        echo \"\$(date) supervisor: runner loop exited, relaunching\" >> \"$STATE_DIR/runner.log\"; sleep 30
      done" > /dev/null 2>&1 &
    echo $! > "$PIDFILE"; echo "queue runner started (pid $!, every ${EVERY}s)" ;;
  stop) kill -- -"$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; rm -f "$PIDFILE"; echo stopped ;;
  status) if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "running (pid $(cat "$PIDFILE"))"; else echo "not running"; fi; wq status ;;
  *) echo "usage: $0 start|stop|status|once" >&2; exit 2 ;;
esac
