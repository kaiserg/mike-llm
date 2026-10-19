import torch
from torch import nn

from tokenizer import CharacterTokenizer

# Reproducible random values
torch.manual_seed(42)

text = "hello mike, welcome to language models!"

tokenizer = CharacterTokenizer(text)

tokens = tokenizer.encode("hello")

token_tensor = torch.tensor(tokens, dtype=torch.long)

# Configuration
embedding_dim = 4
context_length = 16

# Token embeddings
token_embedding = nn.Embedding(
    tokenizer.vocab_size,
    embedding_dim,
)

# Position embeddings
position_embedding = nn.Embedding(
    context_length,
    embedding_dim,
)

# Positions: 0, 1, 2, 3, 4
positions = torch.arange(len(tokens))

# Get both representations
token_vectors = token_embedding(token_tensor)
position_vectors = position_embedding(positions)

# Combine them
combined_vectors = token_vectors + position_vectors

print("Tokens:", tokens)
print("Positions:", positions.tolist())

print("\nToken embeddings:")
print(token_vectors)

print("\nPosition embeddings:")
print(position_vectors)

print("\nCombined embeddings:")
print(combined_vectors)

print("\nShape:", combined_vectors.shape)
