"""A masked gate over claim, text/image evidence and an optional table branch."""

import torch
from torch import Tensor, nn


class MaskedFusion(nn.Module):
    def __init__(self, dimension: int, include_table: bool = False) -> None:
        super().__init__()
        self.include_table = include_table
        self.gate = nn.Linear(dimension, 4 if include_table else 3)

    def forward(
        self, claim: Tensor, text: Tensor, image: Tensor,
        text_present: Tensor, image_present: Tensor,
        table: Tensor | None = None, table_present: Tensor | None = None,
    ) -> Tensor:
        values = [claim, text, image]
        presence = [torch.ones_like(text_present), text_present, image_present]
        if self.include_table:
            values.append(table)
            presence.append(table_present)
        elif table_present is not None and table_present.any():
            raise ValueError("Table evidence requires a table-enabled fusion")
        mask = torch.stack(presence, dim=1)
        logits = self.gate(claim).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(logits, dim=1)
        return (torch.stack(values, dim=1) * weights.unsqueeze(-1)).sum(dim=1)
