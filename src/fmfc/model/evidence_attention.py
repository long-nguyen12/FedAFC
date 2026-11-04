"""Claim-conditioned attention with explicit evidence-presence masks."""

import math

import torch
from torch import Tensor, nn


class EvidenceAttention(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)

    def forward(self, claim: Tensor, evidence: Tensor, mask: Tensor) -> Tensor:
        if evidence.shape[1] == 0 or not mask.any():
            return claim.new_zeros(claim.shape)
        evidence = evidence.masked_fill(~mask.unsqueeze(-1), 0)
        scores = (self.query(claim).unsqueeze(1) * self.key(evidence)).sum(-1)
        scores = scores / math.sqrt(claim.shape[-1])
        present = mask.any(dim=1, keepdim=True)
        scores = scores.masked_fill(~mask, -torch.inf)
        scores = torch.where(present, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=1).masked_fill(~mask, 0)
        return (weights.unsqueeze(-1) * evidence).sum(dim=1)
