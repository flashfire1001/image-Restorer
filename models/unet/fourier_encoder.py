#Fourier Encoder for better time perception

import torch
import torch.nn as nn
import math

class FourierEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2 == 0)
        self.half_dim = dim // 2
        self.freqs = nn.Parameter(torch.randn(1, self.half_dim)) # (1, half_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1) # (bs, 1)
        theta = t * self.freqs * 2 * math.pi # (bs, half_dim)
        sin_embed = torch.sin(theta) # (bs, half_dim)
        cos_embed = torch.cos(theta) # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # (bs, dim)
