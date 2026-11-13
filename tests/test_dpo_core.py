"""The DPO objective against hand-computed values. Needs torch: skips where it
isn't installed (run on a training or Mac box)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from dpo_core import dpo_loss, response_logprob  # noqa: E402


def test_loss_matches_the_formula():
    loss, info = dpo_loss(torch.tensor(-10.0), torch.tensor(-12.0), torch.tensor(-11.0),
                          torch.tensor(-11.0), beta=0.1)
    margin = 0.1 * ((-10 + 11) - (-12 + 11))           # 0.2
    assert info["margin"] == pytest.approx(margin)
    assert float(loss) == pytest.approx(-math.log(1 / (1 + math.exp(-margin))))
    assert info["correct"] == 1.0


def test_at_the_start_policy_equals_reference_so_loss_is_log2():
    x = torch.tensor(-5.0)
    loss, info = dpo_loss(x, x - 1, x, x - 1)          # pi == ref: zero margin
    assert float(loss) == pytest.approx(math.log(2)) and info["margin"] == 0.0


def test_gradient_pushes_chosen_up_and_rejected_down():
    pc, pr = torch.tensor(-10.0, requires_grad=True), torch.tensor(-10.0, requires_grad=True)
    loss, _ = dpo_loss(pc, pr, torch.tensor(-10.0), torch.tensor(-10.0))
    loss.backward()
    assert pc.grad < 0 < pr.grad                        # descending raises pc, lowers pr


def test_response_logprob_scores_only_the_response():
    vocab, T, p = 5, 6, 3
    ids = torch.tensor([[0, 1, 2, 3, 4, 1]])
    logits = torch.zeros(1, T, vocab)
    logits[0, p - 1 :, :] = torch.log(torch.full((vocab,), 1e-9))
    for t in range(p - 1, T - 1):                       # make the true next token near-certain
        logits[0, t, ids[0, t + 1]] = 0.0
    lp = response_logprob(logits, ids, p)
    assert lp > -1e-3                                   # ~log(1) summed over 3 response tokens
    logits[0, 0, :] = -1e9                              # garbage in the prompt region: ignored
    assert response_logprob(logits, ids, p) == pytest.approx(float(lp))
