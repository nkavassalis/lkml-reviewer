"""ensure_serving (train/window-lib.sh): running isn't serving. lkml-reviewer,
2026-11-15: vLLM was started, logged as "serving confirmed" 10 s later, and
exited at 2.5 min (a crashed trainer still held GPU memory). Down 7 hours."""
import os
import stat
import subprocess
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / "train" / "window-lib.sh"

# A docker stub: `start`/`restart` are recorded; `inspect` reports the state
# file, so a test can make the container die after it was started.
DOCKER = """#!/bin/sh
echo "$*" >> "$STUB/calls"
case "$1" in
  inspect) cat "$STUB/state" ;;
  start|restart) [ -f "$STUB/dies" ] && echo false > "$STUB/state" || echo true > "$STUB/state" ;;
esac
"""


def run(tmp_path, state, dies):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    d = bin_ / "docker"
    d.write_text(DOCKER)
    d.chmod(d.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "state").write_text(state + "\n")
    if dies:
        (tmp_path / "dies").write_text("")
    script = f'. "{LIB}"; say() {{ echo "$*" >> "$STUB/log"; }}; ensure_serving'
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "STUB": str(tmp_path),
           "SERVE_CTR": "vllm", "PEER": "", "SERVE_SETTLE_SEC": "0"}
    rc = subprocess.run(["bash", "-c", script], env=env).returncode
    log = (tmp_path / "log").read_text() if (tmp_path / "log").exists() else ""
    return rc, log, (tmp_path / "calls").read_text()


def test_a_server_that_starts_then_dies_is_reported_not_confirmed(tmp_path):
    rc, log, calls = run(tmp_path, "false", dies=True)
    assert rc == 1
    assert "did not stay up (try 2 of 2)" in log
    assert "start vllm" in calls and "restart vllm" in calls   # retried once


def test_a_running_server_is_left_alone(tmp_path):
    rc, log, calls = run(tmp_path, "true", dies=False)
    assert rc == 0 and log == ""
    assert "start" not in calls


def test_a_stopped_server_that_comes_up_passes(tmp_path):
    rc, log, _ = run(tmp_path, "false", dies=False)
    assert rc == 0 and "safety net: started vllm on local" in log
