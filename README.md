# Mike-LLM v0.1

An educational project to build and train a small decoder-only Transformer from random weights, targeting Apple Silicon with PyTorch MPS.

## Current progress

The repository currently contains standalone learning demonstrations and reusable causal multi-head attention and feed-forward modules. These components are randomly initialized; a complete language model and training loop are still pending.

- Character-level tokenizer with reversible encoding for known characters.
- Token embeddings and learned absolute positional embeddings.
- Dot-product attention and causal masking.
- Learned Query, Key, and Value projections with scaled dot-product attention.
- Multi-head attention with independent heads, feature concatenation, and an output projection.
- Position-wise feed-forward network with configurable expansion and GELU.

## Setup on macOS

```bash
git clone https://github.com/kaiserg/mike-llm.git
cd mike-llm
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/check_device.py
```

The device check reports local backend availability. The current educational examples use CPU tensors by default.

## Run the demonstrations

From the repository root with the environment activated:

```bash
python src/tokenizer.py
python src/embeddings.py
python src/positional_embeddings.py
python src/attention.py
python src/causal_self_attention.py
python src/multi_head_attention.py
python src/feed_forward.py
```

The multi-head attention demonstration produces:

```text
Input shape: torch.Size([2, 5, 12])
Output shape: torch.Size([2, 5, 12])
Number of heads: 3
Trainable parameters: 588
```

This confirms dimensions and parameter count, not trained model quality. Batch size is two independent sequences; three attention heads process each sequence with their own QKV projections.

## Documentation

Start with the [English learning guide and notation](docs/README.md). Each chapter includes explanations, equations, runnable examples, Mermaid diagrams, and links to the existing implementation.

1. [Tokenization](docs/01-tokenization.md)
2. [Token embeddings](docs/02-token-embeddings.md)
3. [Positional embeddings](docs/03-positional-embeddings.md)
4. [Dot-product attention](docs/04-dot-product-attention.md)
5. [Causal masking](docs/05-causal-masking.md)
6. [Query, Key, Value](docs/06-query-key-value.md)
7. [Multi-head attention](docs/07-multi-head-attention.md)
8. [Feed-forward network](docs/08-feed-forward.md)

![Multi-head attention: three causal heads, concatenation, and output projection](docs/images/multi-head-attention.svg)

## Repository layout

```text
src/        Educational scripts and multi-head attention module
scripts/    Device availability check
configs/    Reserved for future configuration files
data/       Local corpus storage (contents ignored by Git)
docs/       Learning chapters and diagrams
```

## Roadmap

1. Assemble an appropriately licensed corpus and split training and validation data.
2. Decide whether to extend the current character tokenizer or train a subword tokenizer.
3. Combine attention and the FFN with normalization and residual connections into Transformer blocks.
4. Build a small GPT-style model; approximately 10–15M parameters is a target, not the current model size.
5. Implement training, checkpointing, reproducible seeds, and validation metrics.
6. Add autoregressive generation and evaluate loss, perplexity, samples, and speed.

Use appropriately licensed data and keep private data out of the corpus. Training from scratch requires suitable data and compute; no training results are reported yet.
