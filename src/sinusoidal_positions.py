"""Fixed absolute sinusoidal position vectors, with no trainable parameters."""
import math

import torch
from torch import nn


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, context_length, embedding_dim):
        super().__init__()
        if any(type(size) is not int or size < 1 for size in (context_length, embedding_dim)):
            raise ValueError('context_length and embedding_dim must be positive integers')
        positions = torch.arange(context_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(0, embedding_dim, 2, dtype=torch.float32) / embedding_dim)
        angles = positions * frequencies
        table = torch.zeros(context_length, embedding_dim)
        table[:, 0::2] = torch.sin(angles)
        table[:, 1::2] = torch.cos(angles[:, :table[:, 1::2].shape[1]])
        self.register_buffer('table', table)

    def forward(self, positions):
        return self.table[positions]
