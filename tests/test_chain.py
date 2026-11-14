"""train/run-chain-2node.sh with docker/ssh/scp replaced by recorders: which
trainer runs, what gets copied to the peer, and that serving is restored."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

CHAIN = Path(__file__).resolve().parent.parent / "train" / "run-chain-2node.sh"


def run_chain(tmp: Path, **env) -> tuple[int, list[str], str]:
    bin_ = tmp / "bin"
    bin_.mkdir(exist_ok=True)
    calls = tmp / "calls.log"
    for tool in ("docker", "ssh", "scp"):
        (bin_ / tool).write_text(f'#!/bin/bash\necho "{tool} $*" >> {calls}\n')
        (bin_ / tool).chmod(0o755)
    work = tmp / "work"
    work.mkdir(exist_ok=True)
    e = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "WORK": str(work), "RUN": "run-5",
         "PEER": "peer", "MASTER_ADDR": "10.0.0.1", "IFACE": "eth9", **env}
    p = subprocess.run(["bash", str(CHAIN)], env=e, capture_output=True, text=True)
    log = (work / "chain-run-5.log").read_text() if (work / "chain-run-5.log").exists() else ""
    return p.returncode, calls.read_text().splitlines() if calls.exists() else [], log


def test_sft_is_the_default_trainer(tmp_path):
    rc, calls, log = run_chain(tmp_path)
    train = [c for c in calls if "torchrun" in c]
    assert rc == 0 and len(train) == 2                  # one per node
    assert all("sft_lora.py --train sft.jsonl" in c for c in train)
    assert sum("start vllm_fp4local" in c for c in calls) == 2   # restored on both nodes


def test_dpo_runs_dpo_lora_and_copies_its_inputs_to_the_peer(tmp_path):
    rc, calls, log = run_chain(tmp_path, TRAINER="dpo", SFT_ADAPTER="run-4/adapter-final",
                               PAIRS="v5/pairs.jsonl")
    train = [c for c in calls if "torchrun" in c]
    assert rc == 0 and len(train) == 2
    assert all("dpo_lora.py --sft-adapter run-4/adapter-final --pairs v5/pairs.jsonl --out run-5" in c
               for c in train)
    copies = [c for c in calls if c.startswith("scp")]
    assert any("run-4/adapter-final" in c for c in copies) and any("v5/pairs.jsonl" in c for c in copies)
    assert any("dpo_lora.py" in c and "pkill" in c for c in calls)   # the restore trap kills it too
    assert sum("start vllm_fp4local" in c for c in calls) == 2


def test_dpo_without_its_inputs_refuses_before_stopping_anything(tmp_path):
    rc, calls, _ = run_chain(tmp_path, TRAINER="dpo")
    assert rc != 0 and not any("stop" in c for c in calls)


def test_a_resumed_run_copies_the_checkpoint_to_the_peer_first(tmp_path):
    ck = tmp_path / "work" / "run-5" / "adapter-latest"
    ck.mkdir(parents=True)
    (ck / "STEP").write_text("50")
    rc, calls, log = run_chain(tmp_path)
    first_train = next(i for i, c in enumerate(calls) if "torchrun" in c)
    copy = next(i for i, c in enumerate(calls) if c.startswith("scp") and "run-5/adapter-latest" in c)
    assert rc == 0 and copy < first_train
    assert "resuming: copied run-5/adapter-latest (step 50)" in log
