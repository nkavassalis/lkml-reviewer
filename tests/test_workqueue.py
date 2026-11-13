"""The improvement queue: order, prerequisites, gates, window fit, halting."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WQ = ROOT / "train" / "workqueue.py"


def setup(tmp: Path, toml: str) -> dict:
    (tmp / "QUEUE.toml").write_text(toml)
    return {**os.environ, "QUEUE_FILE": str(tmp / "QUEUE.toml"),
            "QUEUE_STATE": str(tmp / "state.json"), "QUEUE_LOGS": str(tmp / "logs")}


def wq(env, *args) -> str:
    return subprocess.run([sys.executable, str(WQ), *args], env=env, capture_output=True,
                          text=True).stdout


def nxt(env, open_=1, left=600) -> str:
    return wq(env, "next", "--window-open", str(open_), "--minutes-left", str(left))


Q = '''
[[item]]
id = "a"
title = "first"
run = "echo a"
est_min = 10

[[item]]
id = "b"
title = "second, after a"
run = "echo b"
after = ["a"]
'''


def test_runs_in_order_respecting_prerequisites(tmp_path):
    env = setup(tmp_path, Q)
    assert nxt(env).strip() == "RUN a"
    wq(env, "run", "a")
    assert nxt(env).strip() == "RUN b"
    wq(env, "run", "b")
    assert nxt(env).strip() == ""                       # all done
    assert "exit 0" in (tmp_path / "logs" / "a.log").read_text()


def test_a_failure_halts_the_whole_queue(tmp_path):
    env = setup(tmp_path, Q.replace('run = "echo a"', 'run = "exit 3"'))
    wq(env, "run", "a")
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["a"]["status"] == "failed" and state["a"]["rc"] == 3
    assert nxt(env).strip() == ""                       # halted, b doesn't start
    wq(env, "skip", "a")
    assert nxt(env).strip() == "RUN b"                  # a person chose to move on


def test_human_gate_waits_then_runs_after_approve(tmp_path):
    env = setup(tmp_path, Q.replace('title = "second, after a"', 'title = "check"\ngate = "human"'))
    wq(env, "run", "a")
    assert nxt(env).strip() == "WAITING b"
    wq(env, "waiting", "b")
    assert nxt(env).strip() == ""                       # asked once, not every check
    wq(env, "approve", "b")
    assert nxt(env).strip() == "RUN b"


def test_window_items_need_the_window_and_must_fit(tmp_path):
    env = setup(tmp_path, Q.replace('est_min = 10', 'est_min = 120\nwindow = true'))
    assert nxt(env, open_=0).strip() == ""              # window closed
    assert nxt(env, open_=1, left=60).strip() == ""     # open, but 120 min won't fit
    assert nxt(env, open_=1, left=300).strip() == "RUN a"


def test_when_condition_gates_readiness(tmp_path):
    flag = tmp_path / "RESULTS-v4.md"
    env = setup(tmp_path, Q.replace('est_min = 10', f'est_min = 10\nwhen = "test -f {flag}"'))
    assert nxt(env).strip() == ""
    flag.write_text("done")
    assert nxt(env).strip() == "RUN a"


def test_bad_queue_files_are_rejected(tmp_path):
    env = setup(tmp_path, Q.replace('after = ["a"]', 'after = ["nope"]'))
    r = subprocess.run([sys.executable, str(WQ), "status"], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "unknown items" in r.stderr


def test_runner_emails_started_done_and_needs_you(tmp_path):
    """scripts/queue-runner.sh once, end to end, with a fake sender."""
    toml = Q + '''
[[item]]
id = "c"
title = "human check"
run = "echo c"
after = ["b"]
gate = "human"
'''
    env = setup(tmp_path, toml)
    sent = tmp_path / "sent.txt"
    fake = tmp_path / "fake.sh"
    fake.write_text(f'#!/bin/bash\necho "${{@: -1}}" >> {sent}\ncat > /dev/null\n')
    fake.chmod(0o755)
    env |= {"NOTIFY_CMD": str(fake), "QUEUE_RUNNER_STATE": str(tmp_path / "runner"),
            "NOTIFY_LOG_DIR": str(tmp_path),
            "WINDOW_START": "00:00", "WINDOW_END": "23:59", "STOP_MARGIN_MIN": "0"}
    runner = ROOT / "scripts" / "queue-runner.sh"
    for _ in range(4):
        subprocess.run(["bash", str(runner), "once"], env=env, capture_output=True, text=True)
    subjects = sent.read_text().splitlines()
    assert subjects == ["[queue] Started: first", "[queue] Done: first",
                        "[queue] Started: second, after a", "[queue] Done: second, after a",
                        "[queue] Needs you: human check"]


def test_exit_75_means_resume_later_not_failure(tmp_path):
    marker = tmp_path / "progress"
    run = f'n=$(cat {marker} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {marker}; [ $n -ge 3 ] && exit 0 || exit 75'
    env = setup(tmp_path, Q.replace('run = "echo a"', f"run = '{run}'"))
    for expected in ("pending", "pending", "done"):
        assert nxt(env).strip() == "RUN a"
        wq(env, "run", "a")
        st = json.loads((tmp_path / "state.json").read_text())["a"]
        assert st["status"] == expected
    assert st["passes"] == 3
    assert nxt(env).strip() == "RUN b"                  # never halted the queue
