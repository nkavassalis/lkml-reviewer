#!/usr/bin/env bash
# The cluster's maintenance window, as shell functions. Source it:
#
#   . train/window-lib.sh
#   window_open && echo "inside the window"
#   need_minutes 600 "v4 training" || exit 0   # skip a step that can't finish in time
#
# The window is when the shared serving stack may be down (agreed with its
# owner): by default 20:00-08:00 in the cluster's timezone. Steps must finish,
# and the service must be back, by WINDOW_END minus STOP_MARGIN_MIN.
WINDOW_START=${WINDOW_START:-20:00}
WINDOW_END=${WINDOW_END:-08:00}
STOP_MARGIN_MIN=${STOP_MARGIN_MIN:-20}   # leaves time to restore before the end
WINDOW_TZ=${WINDOW_TZ:-America/New_York}

# WINDOW_NOW (epoch seconds) pretends it's that time: for tests only.
_now() { echo "${WINDOW_NOW:-$(date +%s)}"; }
_epoch() { TZ=$WINDOW_TZ date -d "$(TZ=$WINDOW_TZ date -d @"$(_now)" +%F) $1" +%s; }
_epoch_tomorrow() { TZ=$WINDOW_TZ date -d "$(TZ=$WINDOW_TZ date -d @"$(( $(_now) + 86400 ))" +%F) $1" +%s; }

# Epoch of the window's hard stop (end minus margin) for a window that is open
# now or opens next.
window_deadline() {
  local now start end
  now=$(_now)
  end=$(_epoch "$WINDOW_END")
  # if today's end has passed, the window that's open (or next) ends tomorrow
  if [ "$now" -ge "$end" ]; then end=$(_epoch_tomorrow "$WINDOW_END"); fi
  echo $(( end - STOP_MARGIN_MIN * 60 ))
}

# Epoch the window next opens (now, if it's open).
window_next_open() {
  if window_open; then _now; return; fi
  local start; start=$(_epoch "$WINDOW_START")
  [ "$start" -gt "$(_now)" ] || start=$(_epoch_tomorrow "$WINDOW_START")
  echo "$start"
}

# Epoch of the next window *start* strictly after now, even if a window is open
# right now (e.g. launched at 06:00 for tonight's run).
window_next_start() {
  local start; start=$(_epoch "$WINDOW_START")
  [ "$start" -gt "$(_now)" ] || start=$(_epoch_tomorrow "$WINDOW_START")
  echo "$start"
}

window_open() {
  local hm; hm=$(TZ=$WINDOW_TZ date -d @"$(_now)" +%H:%M)
  if [[ "$WINDOW_START" > "$WINDOW_END" ]]; then        # overnight, e.g. 20:00-08:00
    [[ ! "$hm" < "$WINDOW_START" || "$hm" < "$WINDOW_END" ]]
  else
    [[ ! "$hm" < "$WINDOW_START" && "$hm" < "$WINDOW_END" ]]
  fi
}

window_remaining_min() { echo $(( ($(window_deadline) - $(_now)) / 60 )); }

# need_minutes N "what": true if N minutes still fit before the hard stop.
need_minutes() {
  local left; left=$(window_remaining_min)
  if [ "$left" -ge "$1" ]; then return 0; fi
  echo "[$(TZ=$WINDOW_TZ date +%T)] skipping $2: needs ~$1 min, $left left before the window's hard stop"
  return 1
}
