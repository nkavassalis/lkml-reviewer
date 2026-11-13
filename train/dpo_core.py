"""The DPO objective, as small torch functions (testable without a model).

Direct Preference Optimization (Rafailov et al., 2023): for a prompt x with a
preferred response y_w and a dispreferred y_l,

    loss = -log sigmoid( beta * [ (log pi(y_w|x) - log pi(y_l|x))
                                 - (log ref(y_w|x) - log ref(y_l|x)) ] )

where pi is the model being trained and ref the frozen starting model (v4).
Sequence log-probabilities are sums over the *response* tokens only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def response_logprob(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Sum of log p(token) over the response part of one sequence.

    logits: [1, T, V] from a forward pass over input_ids [1, T]. Position t
    predicts token t+1, so the response tokens input_ids[prompt_len:] are
    scored by logits[prompt_len-1 : T-1]. Only those rows go through
    log_softmax, which matters with a 248k vocabulary.
    """
    pred = logits[0, prompt_len - 1 : -1].float()
    tgt = input_ids[0, prompt_len:]
    return torch.log_softmax(pred, dim=-1).gather(-1, tgt[:, None]).squeeze(-1).sum()


def dpo_loss(pi_chosen: torch.Tensor, pi_rejected: torch.Tensor, ref_chosen: torch.Tensor,
             ref_rejected: torch.Tensor, beta: float = 0.1) -> tuple[torch.Tensor, dict]:
    """The DPO loss plus the numbers worth logging."""
    chosen_reward = beta * (pi_chosen - ref_chosen)
    rejected_reward = beta * (pi_rejected - ref_rejected)
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin)
    return loss, {"margin": float(margin), "chosen_reward": float(chosen_reward),
                  "rejected_reward": float(rejected_reward), "correct": float(margin > 0)}
