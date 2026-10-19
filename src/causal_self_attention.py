import torch
import torch.nn.functional as F
from torch import nn

from tokenizer import CharacterTokenizer

torch.manual_seed(42)

# Configuration
embedding_dim = 4
context_length = 16

# Prepare input
text = "hello mike, welcome to language models!"
tokenizer = CharacterTokenizer(text)

tokens = tokenizer.encode("hello")
token_tensor = torch.tensor(tokens, dtype=torch.long)

# Embeddings
token_embedding = nn.Embedding(tokenizer.vocab_size, embedding_dim)

position_embedding = nn.Embedding(context_length, embedding_dim)

positions = torch.arange(len(tokens))

x = token_embedding(token_tensor) + position_embedding(positions)

# Query, Key and Value projections
query = nn.Linear(embedding_dim, embedding_dim, bias=False)
key = nn.Linear(embedding_dim, embedding_dim, bias=False)
value = nn.Linear(embedding_dim, embedding_dim, bias=False)

Q = query(x)
K = key(x)
V = value(x)

# Scaled dot-product attention
head_size = K.shape[-1]

scores = Q @ K.T
scores = scores / (head_size**0.5)

# Causal mask
seq_len = len(tokens)

mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

scores = scores.masked_fill(~mask, float("-inf"))

# Attention weights
weights = F.softmax(scores, dim=-1)

# Weighted combination of values
output = weights @ V

print("Input shape:", x.shape)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

print("\nAttention weights:")
print(weights)

print("\nOutput:")
print(output)

print("\nOutput shape:", output.shape)

# Verify that future tokens are blocked
assert torch.all(weights.triu(diagonal=1) == 0)

print("\nCausal masking verified!")
