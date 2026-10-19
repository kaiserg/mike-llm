import torch
from torch import nn

from transformer_block import TransformerBlock
from sinusoidal_positions import SinusoidalPositionEmbedding


class GPT(nn.Module):
    """A small decoder-only language model returning [B, T, vocab_size] logits."""

    def __init__(
        self,
        vocab_size,
        context_length,
        embedding_dim=12,
        num_heads=3,
        num_layers=2,
        expansion_factor=4,
        positional_encoding="learned",
    ):
        super().__init__()
        dimensions = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "expansion_factor": expansion_factor,
        }
        for name, size in dimensions.items():
            if type(size) is not int or size <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        if positional_encoding not in ("learned", "sinusoidal"):
            raise ValueError("positional_encoding must be learned or sinusoidal")

        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        if positional_encoding == "learned":
            self.position_embedding = nn.Embedding(context_length, embedding_dim)
        else:
            self.position_embedding = SinusoidalPositionEmbedding(context_length, embedding_dim)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embedding_dim, num_heads, expansion_factor)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, token_ids, attention_mask=None):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.dtype != torch.long:
            raise ValueError("token_ids must use torch.long dtype")
        batch_size, sequence_length = token_ids.shape
        if batch_size == 0 or not 1 <= sequence_length <= self.context_length:
            raise ValueError("expected a nonempty batch and 1 <= sequence length <= context_length")

        # Positions broadcast across the batch; input and model share a device.
        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        if attention_mask is None:
            x = self.blocks(x)
        else:
            for block in self.blocks:
                x = block(x, attention_mask=attention_mask)
        return self.lm_head(self.final_norm(x))


if __name__ == "__main__":
    from tokenizer import CharacterTokenizer

    torch.manual_seed(42)
    tokenizer = CharacterTokenizer("hello")
    token_ids = torch.tensor(
        [tokenizer.encode("hello"), tokenizer.encode("olleh")],
        dtype=torch.long,
    )
    model = GPT(vocab_size=tokenizer.vocab_size, context_length=16)
    logits = model(token_ids)

    print("Input shape:", token_ids.shape)
    print("Logits shape:", logits.shape)
    print("Number of blocks:", len(model.blocks))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
