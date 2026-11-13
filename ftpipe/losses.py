import torch
import torch.nn.functional as F

from .data import IGNORE


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log-probs of the label tokens per sequence (ignoring masked positions)."""
    logits, labels = logits[:, :-1], labels[:, 1:]
    mask = labels != IGNORE
    safe = labels.masked_fill(~mask, 0)
    logp = torch.gather(logits.log_softmax(-1), 2, safe.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1)


def dpo_loss(policy_logps, ref_logps, beta: float):
    """policy_logps/ref_logps: [2B] with chosen in the first half, rejected in the second."""
    pc, pr = policy_logps.chunk(2)
    rc, rr = ref_logps.chunk(2)
    margin = beta * ((pc - rc) - (pr - rr))
    loss = -F.logsigmoid(margin).mean()
    stats = {
        "reward_acc": (margin > 0).float().mean().item(),
        "reward_margin": margin.mean().item() / beta,
    }
    return loss, stats
