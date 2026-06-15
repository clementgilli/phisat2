from __future__ import annotations

import torch
import torch.nn as nn

class ComposedModel(nn.Module):
    
    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        return self.head(features)