import torch
from torch import nn

from feed_forward import FeedForward
from multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """A pre-normalized causal decoder block for [batch, sequence, features]."""

    def __init__(self, embedding_dim, num_heads, expansion_factor=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, expansion_factor)

    def forward(self, x, attention_mask=None):
        # Each branch learns an update to the residual stream.
        x = x + self.attention(self.norm1(x), attention_mask=attention_mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn(2, 5, 12)
    block = TransformerBlock(embedding_dim=12, num_heads=3)
    output = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Trainable parameters:", sum(p.numel() for p in block.parameters()))
