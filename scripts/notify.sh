#!/usr/bin/env bash
# Send a notification: subject as $1, body on stdin. Pluggable and optional.
#
#   echo "body" | scripts/notify.sh "subject"
#
# NOTIFY_CMD (from .status.env) is any command that takes the subject as its
# last argument and reads the body on stdin, e.g. a Gmail sender:
#   NOTIFY_CMD='python3 /path/to/send_mail.py --to someone@example.com --subject'
# or ntfy, Slack, sendmail... Every notification is also appended to
# runs/notifications.log (or ./notifications.log), so nothing is lost when
# NOTIFY_CMD is unset or fails. Credentials never pass through this script.
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
subject="${1:?usage: notify.sh SUBJECT < body}"
body=$(cat)
logdir="$root/runs"; [ -d "$logdir" ] || logdir="$root/data"; [ -d "$logdir" ] || logdir="$root"
{ echo "=== $(date '+%F %T %Z') | $subject"; echo "$body"; echo; } >> "$logdir/notifications.log"
if [ -n "${NOTIFY_CMD:-}" ]; then
  # shellcheck disable=SC2086  (NOTIFY_CMD is a command line by design)
  if ! printf '%s\n' "$body" | $NOTIFY_CMD "$subject" > /dev/null 2>> "$logdir/notifications.log"; then
    echo "notify: NOTIFY_CMD failed (see $logdir/notifications.log)" >&2
    exit 1
  fi
fi
