"""An ongoing improvement queue: work items run in order, one at a time.
Stdlib only.

Items live in QUEUE.toml (edit it to add, reorder or remove work):

    [[item]]
    id = "v4-eval"                 # unique
    title = "Evaluate v4"          # what the emails say
    run = "scripts/eval-v4.sh"     # shell command, run from the repo root
    est_min = 90                   # estimate, used to fit the window
    after = ["v4-train"]           # these must be done first
    when = "ssh host test -f RESULTS-v4.md"   # optional readiness check (exit 0 = ready)
    window = true                  # only inside the maintenance window, and only if est fits
    gate = "auto"                  # "auto", or "human": waits for `approve` (emails "Needs you")

State (gitignored) records each item: pending, running, done, failed,
waiting-human, approved, needs-person, skipped. A failed item halts the queue until someone
looks (`reset` or `skip`): unattended retries of a broken step waste nights.

Exit code 75 (EX_TEMPFAIL) means "not finished, resume later": a step longer
than one window (e.g. generating thousands of candidates) stops at the hard
stop, exits 75, and runs again in the next window, picking up where it left
off. It goes back to pending (keeping its count of passes); it doesn't halt
the queue.

Exit code 76 means "a person must decide": a gate the step raises itself, e.g.
a decision script that finds the evidence disagreeing (train/decide.py). The
item becomes needs-person, its dependents wait, the runner emails "Needs you"
with the step's log, and `approve ID` marks it done without re-running it
(the person has made the call, e.g. by editing the file the step wrote).

    python3 train/workqueue.py status
    python3 train/workqueue.py next --window-open 1 --minutes-left 600   # id or nothing
    python3 train/workqueue.py run ID
    python3 train/workqueue.py approve ID | reset ID | skip ID
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = Path(os.environ.get("QUEUE_FILE", ROOT / "QUEUE.toml"))
STATE = Path(os.environ.get("QUEUE_STATE", ROOT / "data" / "queue" / "state.json"))
LOGS = Path(os.environ.get("QUEUE_LOGS", ROOT / "data" / "queue" / "logs"))


def load_items() -> list[dict]:
    items = tomllib.loads(QUEUE.read_text()).get("item", [])
    ids = [i["id"] for i in items]
    dup = {x for x in ids if ids.count(x) > 1}
    if dup:
        raise SystemExit(f"{QUEUE}: duplicate item ids {sorted(dup)}")
    for i in items:
        missing = [a for a in i.get("after", []) if a not in ids]
        if missing:
            raise SystemExit(f"{QUEUE}: item {i['id']} is after unknown items {missing}")
    return items


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE)


def status_of(state: dict, item_id: str) -> str:
    return state.get(item_id, {}).get("status", "pending")


def ready(item: dict) -> bool:
    cond = item.get("when")
    if not cond:
        return True
    return subprocess.run(["bash", "-c", cond], cwd=ROOT, capture_output=True,
                          timeout=120).returncode == 0


def next_item(items, state, *, window_open: bool, minutes_left: int) -> tuple[str | None, list[str]]:
    """(id to run now, ids newly waiting for a human). One failure halts all."""
    if any(status_of(state, i["id"]) in ("failed", "running") for i in items):
        return None, []
    waiting = []
    for item in items:
        st = status_of(state, item["id"])
        if st in ("done", "skipped", "needs-person"):
            continue
        if any(status_of(state, a) not in ("done", "skipped") for a in item.get("after", [])):
            continue
        if item.get("gate", "auto") == "human" and st != "approved":
            if st != "waiting-human":
                waiting.append(item["id"])
            continue
        if item.get("window") and (not window_open or item.get("est_min", 0) > minutes_left):
            continue
        if not ready(item):
            continue
        return item["id"], waiting
    return None, waiting


RESUME_LATER = 75  # EX_TEMPFAIL
NEEDS_PERSON = 76  # the step raised its own human gate


def run(item: dict, state: dict) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{item['id']}.log"
    passes = state.get(item["id"], {}).get("passes", 0)
    state[item["id"]] = {"status": "running", "started": time.strftime("%F %T"), "log": str(log),
                         "passes": passes}
    save_state(state)
    with open(log, "a") as fh:
        fh.write(f"=== {time.strftime('%F %T')} start: {item['title']}\n")
        fh.flush()
        rc = subprocess.run(["bash", "-c", item["run"]], cwd=ROOT, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
        fh.write(f"=== {time.strftime('%F %T')} exit {rc}\n")
    passes += 1
    status = {0: "done", RESUME_LATER: "pending", NEEDS_PERSON: "needs-person"}.get(rc, "failed")
    state[item["id"]].update(status=status, rc=rc, passes=passes, ended=time.strftime("%F %T"))
    save_state(state)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    n = sub.add_parser("next")
    n.add_argument("--window-open", type=int, default=0)
    n.add_argument("--minutes-left", type=int, default=0)
    for c in ("run", "approve", "reset", "skip", "waiting"):
        sub.add_parser(c).add_argument("id")
    args = ap.parse_args()
    items = load_items()
    by_id = {i["id"]: i for i in items}
    state = load_state()

    if args.cmd == "status":
        for i in items:
            st = state.get(i["id"], {})
            gate = " [human gate]" if i.get("gate") == "human" else ""
            win = " [window]" if i.get("window") else ""
            print(f"{st.get('status', 'pending'):<14} {i['id']:<24} ~{i.get('est_min', '?')} min{win}{gate}  {i['title']}")
        return 0
    if args.cmd == "next":
        nid, waiting = next_item(items, state, window_open=bool(args.window_open),
                                 minutes_left=args.minutes_left)
        for w in waiting:
            print(f"WAITING {w}")
        if nid:
            print(f"RUN {nid}")
        return 0
    if args.id not in by_id:
        print(f"no item {args.id!r}", file=sys.stderr)
        return 2
    if args.cmd == "run":
        return run(by_id[args.id], state)
    new = {"approve": "approved", "reset": "pending", "skip": "skipped", "waiting": "waiting-human"}[args.cmd]
    if args.cmd == "approve" and state.get(args.id, {}).get("status") == "needs-person":
        new = "done"          # the person decided; the step's own run already happened
    state.setdefault(args.id, {})["status"] = new
    save_state(state)
    print(f"{args.id}: {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
