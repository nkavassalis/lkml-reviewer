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

- [ ] Smoke test the exact command (a few steps) and read its sample output.
- [ ] Memory cap set; peak memory from the smoke test well under it.
- [ ] Checkpoints every ≤ 1 h of work, with optimizer state.
- [ ] Serving stack stopped *by the chain script*, restored by its trap.
- [ ] Watcher started off-cluster, and its alert paths tested once.
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
