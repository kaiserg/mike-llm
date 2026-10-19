import torch
import torch.nn.functional as F

from tokenizer import CharacterTokenizer

torch.manual_seed(42)

# 1. Prepare text
text = "hello mike, welcome to language models!"
tokenizer = CharacterTokenizer(text)

tokens = tokenizer.encode("hello")

# 2. Convert tokens to tensor
token_tensor = torch.tensor(tokens, dtype=torch.long)

# 3. Configuration
embedding_dim = 4
context_length = 16

# 4. Create embeddings
token_embedding = torch.nn.Embedding(tokenizer.vocab_size, embedding_dim)

position_embedding = torch.nn.Embedding(context_length, embedding_dim)

# 5. Combine token and position embeddings
positions = torch.arange(len(tokens))

x = token_embedding(token_tensor) + position_embedding(positions)

# 6. Calculate similarity scores
scores = x @ x.T

# Number of tokens
seq_len = len(tokens)

# Create a lower triangular mask
mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

# Block future positions
masked_scores = scores.masked_fill(~mask, float("-inf"))

# 7 Normalize attention scores
weights = F.softmax(masked_scores, dim=-1)

print("\nCausal mask:")
print(mask.int())

print("\nMasked scores:")
print(masked_scores)

# 8. Combine information from tokens
output = weights @ x

print("Input shape:", x.shape)

print("\nSimilarity scores:")
print(scores)

print("\nAttention weights:")
print(weights)

print("\nOutput:")
print(output)

print("\nOutput shape:", output.shape)
