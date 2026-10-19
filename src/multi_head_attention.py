import torch
import torch.nn.functional as F
from torch import nn


class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim):
        super().__init__()

        self.query = nn.Linear(embedding_dim, head_dim, bias=False)

        self.key = nn.Linear(embedding_dim, head_dim, bias=False)

        self.value = nn.Linear(embedding_dim, head_dim, bias=False)

    def forward(self, x, attention_mask=None):
        # x shape: [batch, sequence, embedding_dim]

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        seq_len = x.shape[1]
        head_dim = K.shape[-1]

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (head_dim**0.5)

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )
        )

        if attention_mask is not None:
            if (attention_mask.dtype != torch.bool or attention_mask.shape != (seq_len, seq_len)
                    or attention_mask.device != x.device):
                raise ValueError("attention_mask must be a boolean [sequence, sequence] tensor on the input device")
            mask = mask & attention_mask
            if not mask.any(dim=-1).all():
                raise ValueError("attention_mask must leave a causal key for every query")

        scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        return weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        assert embedding_dim % num_heads == 0

        head_dim = embedding_dim // num_heads

        self.heads = nn.ModuleList(
            [AttentionHead(embedding_dim, head_dim) for _ in range(num_heads)]
        )

        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, attention_mask=None):
        outputs = [head(x, attention_mask=attention_mask) for head in self.heads]

        combined = torch.cat(outputs, dim=-1)

        return self.projection(combined)


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    sequence_length = 5
    embedding_dim = 12
    num_heads = 3

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
    )

    attention = MultiHeadAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
    )

    output = attention(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Number of heads:", num_heads)

    print("Trainable parameters:", sum(p.numel() for p in attention.parameters()))
