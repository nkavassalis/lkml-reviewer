"""train/score.py --targets / --train: the data's rate, next to the models'."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, str(ROOT / "train" / "score.py"), *args], env=env,
                          capture_output=True, text=True, check=True).stdout


def write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_targets_and_the_training_column(tmp_path):
    patch = "--- The patch ---\n+static int frob_widget(struct widget *w)"
    train = write(tmp_path / "sft.jsonl", [{"input": patch, "output": "frob_widget() leaks w"},
                                           {"input": patch, "output": "I hate this whole design."},
                                           {"input": patch, "output": "No. Just no."}])
    out = run("--targets", train)
    assert "| names an identifier from the patch | 33% |" in out     # one of three targets does the task
    cmp = write(tmp_path / "cmp.jsonl", [{"eval_index": 0, "input": patch, "reference": "frob_widget() is wrong",
                                          "v9_sampled": "meh"}])
    table = run(cmp, "--train", train)
    header = next(line for line in table.splitlines() if line.startswith("| metric"))
    assert header.split("|")[2:4] == [" Linus (reference) ", " training targets "]
