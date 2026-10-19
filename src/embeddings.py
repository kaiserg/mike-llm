import torch
from torch import nn

from tokenizer import CharacterTokenizer

# 1. Training text
text = "hello mike, welcome to language models!"

# 2. Create tokenizer
tokenizer = CharacterTokenizer(text)

# 3. Encode text
tokens = tokenizer.encode("hello")

# 4. Convert tokens into a PyTorch tensor
token_tensor = torch.tensor(tokens, dtype=torch.long)

# 5. Create an embedding layer
embedding = nn.Embedding(
    num_embeddings=tokenizer.vocab_size,
    embedding_dim=4,
)

# 6. Convert token IDs into vectors
vectors = embedding(token_tensor)

print("Tokens:", tokens)
print("\nEmbedding vectors:")
print(vectors)

print("\nShape:", vectors.shape)
