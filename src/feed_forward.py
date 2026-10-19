import torch
from torch import nn


class FeedForward(nn.Module):
    """Apply the same nonlinear feature transformation to each token."""

    def __init__(self, embedding_dim, expansion_factor=4):
        super().__init__()

        hidden_dim = embedding_dim * expansion_factor

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        # Linear layers transform only the final dimension: [B, T, D].
        return self.network(x)


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    sequence_length = 5
    embedding_dim = 12
    expansion_factor = 4

    x = torch.randn(batch_size, sequence_length, embedding_dim)
    feed_forward = FeedForward(embedding_dim, expansion_factor)
    output = feed_forward(x)

    print("Input shape:", x.shape)
    print("Hidden dimension:", embedding_dim * expansion_factor)
    print("Output shape:", output.shape)
    print("Trainable parameters:", sum(p.numel() for p in feed_forward.parameters()))
