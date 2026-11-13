# Running long jobs on shared machines

How this project runs multi-hour training on GPU nodes that also serve a
production model. Every rule here was paid for; see the incident at the end.
Use it as the default for future projects of this kind.

## The four rules

1. **Anything you stop, you restore, even if you crash.** Stop the serving
   stack inside a script whose `EXIT` trap restores it
   (`train/run-chain.sh`, `train/run-chain-2node.sh`). Before restoring, kill
   leftover workers. A data-parallel rank whose partner died otherwise sits out
   the NCCL timeout holding its GPU, and the service can't start on that node.
2. **Make memory exhaustion a process error, never a machine freeze.** On
   unified-memory machines (GB10 / DGX Spark, Apple silicon) the GPU allocator
   draws from system RAM. Cap it (`--mem-fraction`, ~75%) and reduce
   fragmentation (`expandable_segments:True`). A capped process raises a CUDA
   OOM, the trap runs and the service comes back. An uncapped one froze a whole
   node, and nothing ran after that, the trap included.
3. **Checkpoint for resume, not just for results.** Save the adapter *and* the
   optimizer state every N steps (`--save-every 50`). Make resuming replay the
   data order exactly (`--resume-from/--resume-step`; verified to match an
   uninterrupted run).
4. **Watch it from somewhere else.** Monitoring that runs on the node being
   watched dies with it. See below.

## Overnight maintenance windows (optional)

If the GPUs serve something by day, agree a window with whoever owns them,
then let a script keep to it instead of a person. lkml-reviewer's is 20:00–08:00
Eastern, cluster time.

- **`train/night-run.sh`** waits for the window, runs a plan (a bash script of
  steps, see `train/night-plan.example.sh`), and at the **hard stop** (window
  end minus a margin, default 20 min) terminates whatever is still running. The
  chains' EXIT traps restore serving, and the launcher then checks serving is
  up on every node itself, as a safety net. Configure it with environment variables (`WINDOW_START`, `WINDOW_END`, `WINDOW_TZ`, `STOP_MARGIN_MIN`).
- **Steps that can't fit are skipped, never started.** In a plan, guard each
  step with `need_minutes <estimate> "<what>"` (`train/window-lib.sh`).
- **Long runs span nights.** `sft_lora.py --resume-auto` continues from the
  last checkpoint (each records its step), and reuses the before-training
  eval loss instead of recomputing it. A 15-hour run becomes two nights.
- **Launching for tonight from the morning:** set `WAIT_FOR_NEXT_WINDOW=1`.
  An overnight window is technically still *open* at 06:00, and without it the
  plan would start at once with minutes left. (The tests caught this.)
- **Outside the window, ask.** The window is the agreement; anything beyond
  it needs the owner's OK, every time.
- Tested: `tests/test_window.py` checks the arithmetic at fixed times with a
  test clock (`WINDOW_NOW`), and the launcher's hard stop, trap and safety net
  were exercised end to end against a fake `docker`.

### "Go for it": running now instead

`GO_NOW=1 train/night-run.sh` starts immediately, inside the window or not,
with an optional hard stop after `MAX_HOURS`. Everything else is the same:
the plan, the stop handling, the safety net and the reporting. Running
outside the agreed window needs the owner's explicit OK; the log records
that the run was started by request, and the reporter emails a "Started"
notice for it.

### Unattended runs: reports by email (optional)

When nobody is watching, `scripts/night-reporter.sh start` runs a supervised
loop on the **monitoring box** (not the cluster, so a frozen node can't take
it down). Every 5 minutes it checks, and it sends:

| message | when |
| --- | --- |
| **PROBLEM** | `status.sh` raises an alarm, or the run's log shows a failure (`FAILED`, `HARD STOP reached`, a traceback, a non-zero exit). One email per distinct failure, not one per check. |
| **Started** | a run began outside the window (`GO_NOW`) |
| **Report** | the run ended (the launcher logged `night done`): plan log, training progress, status |
| **FINISHED** | the plan wrote its results file (`RESULTS_FILE`); the email carries it, then the reporter stops |

It follows the launcher's `latest-run.log` link, so windowed nights and go-now
runs report identically. Sending is pluggable: `NOTIFY_CMD` is any command
that takes the subject as its last argument and the body on stdin
(e.g. a Gmail sender script that reads its own credentials, ntfy, Slack, sendmail). Without one, messages go to
`notifications.log` only. Credentials never pass through these scripts.

**Settings in `.status.env` are defaults only.** Anything set in the
environment wins, so a test (`NOTIFY_CMD=… REPORTER_STATE=/tmp/x
night-reporter.sh once`) can never reach the real sender or mark real
messages as sent. On lkml-reviewer, the first version re-read `.status.env`
after the test's settings and quietly pointed at the real work directory.

**Test the whole sequence before leaving it alone.** Stage a scratch run
directory with log lines for each event (start, a failure, the end, a results
file), run `once` after each step with a fake sender, and check you get
exactly one message per event. That test found two reporter bugs: "hard stop
at 14:00" in a healthy log line matched as a failure, and one failure produced
several emails as the log grew. Then send one real message and check it
arrived.

## Monitoring: two layers, different jobs

| layer | what | cadence | purpose |
| --- | --- | --- | --- |
| **watcher** | `train/watch-run.sh`, a separate process off the cluster | every 5 min | **alarm.** Exits (and so alerts) on: node unreachable twice, available memory under 10 GB, no log activity for 30 min, chain finished |
| **status run** | `scripts/status.sh`, run hourly by a scheduled job | hourly | **report.** A short human update: step, ETA, last checkpoint, memory, serving state, watcher alive |

The watcher is the safety net; the hourly status run is what keeps a person
informed without them having to ask. Don't let either stand in for the other.
A status run every hour still means up to an hour of unnoticed outage, and a
watcher that only fires on failure says nothing when things are fine.

### `scripts/status.sh` contract

- **Read-only.** It never changes anything, so it's safe to run any time,
  from anywhere, by anyone.
- **One command**, output under ~15 lines, exit `0` when normal and `1` with
  `ATTENTION:` lines when not.
- **No addresses in the repo.** Hosts come from a gitignored `.status.env`
  (see the script header for the keys).
- Expected states aren't alarms: the serving stack being down *during* a
  run is reported, not flagged.

### Scheduling the hourly run

With Claude Code, an hourly `CronCreate` job runs the status script and
reports, e.g. at minute 7 (avoid :00):

> Hourly status run: run `scripts/status.sh`; report step / ETA, checkpoint,
> memory, watcher in 3–6 lines; on ATTENTION investigate and lead with the
> problem; when the run has finished, confirm the service is back, start the
> post-run steps, then delete this job.

**Limits to know:** these jobs live only in the running session (gone if it
exits), fire only while the session is idle, and expire after 7 days. That's
why the watcher, a real process, stays the alarm. If a status record must
survive the session, add a system cron entry on the monitoring box, e.g.
`7 * * * * cd ~/linus-pr && scripts/status.sh >> data/status.log 2>&1`.

## Launch checklist for a long run

- [ ] Smoke test the **whole chain**, evaluation and file copies included, at
      tiny sizes (a few steps, a few eval rows), and read its sample output.
      lkml-reviewer's run-3b trained for 10 h, then stopped at a copy step the
      smoke test had skipped (the container wrote the adapter as root, mode 600).
- [ ] Memory cap set; peak memory from the smoke test well under it.
- [ ] Checkpoints every ≤ 1 h of work, with optimizer state.
- [ ] Serving stack stopped *by the chain script*, restored by its trap.
- [ ] The smoke test runs against the **same paths and leftover state** as the
      real run. lkml-reviewer's eval smoke test used a fresh directory name and
      passed; the real run reused a directory a failed run had left half-full,
      and `scp -r` nested the copy inside it. Clean up, or test in place.
- [ ] Watcher started off-cluster, and its alert paths tested once.
- [ ] Status checks **every node's GPU**, not just the master's logs: a peer
      that crashes early sits idle while the master looks healthy.
- [ ] Hourly status job scheduled; `.status.env` points at this run.
- [ ] The person who owns the machines knows the expected downtime and ETA.

## Incident: run-3, 2026-11-11

The v3 run's rank-0 node ran out of memory ~3 h into a 14 h data-parallel
run. The NVIDIA driver logged `NV_ERR_NO_MEMORY`, sshd stopped accepting
connections ("no more sessions"), and the node froze with only 72 GB
*allocated*: the CUDA caching allocator's reserved pool plus fragmentation
had used up the unified memory the OS needed. The only monitoring waited for
the chain's "done" line, which a frozen node never writes, so the outage went
unnoticed for ~12 h, and the serving stack was down for ~15 h. Recovery took
a hard power cycle.

Fixes, now standard: the allocator cap (rule 2), resume from the step-100
checkpoint (rule 3: ~2.3 h of training saved), the off-cluster watcher, and the
hourly status run. Resumed, the run held 42–53 GB available on both nodes.
