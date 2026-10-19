# Mike-LLM learning notes

These chapters follow the existing educational code from characters to a complete small GPT-style forward architecture. Early chapters introduce randomly initialized components; later chapters document training, evaluation, generation, and controlled experiments on small models.

## Reading order

1. [Tokenization](01-tokenization.md) — [`src/tokenizer.py`](../src/tokenizer.py)
2. [Token embeddings](02-token-embeddings.md) — [`src/embeddings.py`](../src/embeddings.py)
3. [Positional embeddings](03-positional-embeddings.md) — [`src/positional_embeddings.py`](../src/positional_embeddings.py)
4. [Dot-product attention](04-dot-product-attention.md) — [`src/attention.py`](../src/attention.py)
5. [Causal masking](05-causal-masking.md) — [`src/attention.py`](../src/attention.py)
6. [Query, Key, Value](06-query-key-value.md) — [`src/causal_self_attention.py`](../src/causal_self_attention.py)
7. [Multi-head attention](07-multi-head-attention.md) — [`src/multi_head_attention.py`](../src/multi_head_attention.py)
8. [Feed-forward network](08-feed-forward.md) — [`src/feed_forward.py`](../src/feed_forward.py)
9. [Transformer block: LayerNorm and residual connections](09-transformer-block.md) — [`src/transformer_block.py`](../src/transformer_block.py)
10. [GPT language model](10-gpt-model.md) — [`src/gpt.py`](../src/gpt.py)
11. [First reproducible training run](11-training.md) — [`src/train.py`](../src/train.py)
12. [Text generation and temperature](12-text-generation.md) — [`src/generate.py`](../src/generate.py)
13. [Checkpoint evaluation and perplexity](13-evaluation.md) — [`src/evaluate.py`](../src/evaluate.py)
14. [Real corpus preparation and held-out splits](14-real-corpus.md) — [`src/prepare_corpus.py`](../src/prepare_corpus.py)
15. [First training experiment on real books](15-real-books-training.md) — [`scripts/run_real_experiment.py`](../scripts/run_real_experiment.py)
16. [Real-book generation and temperature comparison](16-real-books-generation.md) — [`scripts/compare_temperatures.py`](../scripts/compare_temperatures.py)
17. [Context-length experiment: 16 versus 64](17-context-length-experiment.md) — [`scripts/run_context_experiment.py`](../scripts/run_context_experiment.py)
18. [Longer training for context 64](18-longer-context-training.md) — [`scripts/run_long_context_experiment.py`](../scripts/run_long_context_experiment.py)
19. [Matched-budget context comparison](19-matched-budget-contexts.md) — [`scripts/run_matched_budget_experiment.py`](../scripts/run_matched_budget_experiment.py)
20. [Context comparison across training seeds](20-multiseed-contexts.md) — [`scripts/run_multiseed_experiment.py`](../scripts/run_multiseed_experiment.py)
21. [Loss by position and available context](21-position-loss-diagnostics.md) — [`scripts/analyze_positions.py`](../scripts/analyze_positions.py)
22. [Match examples per absolute position](22-position-exposure-training.md) — [`scripts/run_position_exposure_experiment.py`](../scripts/run_position_exposure_experiment.py)
23. [Sinusoidal positional encoding experiment](23-sinusoidal-positions.md) — [`scripts/run_sinusoidal_experiment.py`](../scripts/run_sinusoidal_experiment.py)
24. [Learned versus sinusoidal position diagnostics](24-sinusoidal-position-diagnostics.md) — [`scripts/compare_position_diagnostics.py`](../scripts/compare_position_diagnostics.py)

## Running the examples

Run commands from the repository root with the project environment activated:

```bash
source .venv/bin/activate
python src/tokenizer.py
python src/embeddings.py
python src/positional_embeddings.py
python src/attention.py
python src/causal_self_attention.py
python src/multi_head_attention.py
python src/feed_forward.py
python src/transformer_block.py
python src/gpt.py
```

For the independent Python snippets in these chapters, start `PYTHONPATH=src python` from the repository root and paste a whole snippet, or save it and run `PYTHONPATH=src python path/to/example.py`. Install dependencies from the [requirements file](../requirements.txt) if the environment is not prepared.

Examples use CPU tensors and set a seed where randomness matters. Exact floating-point values can vary with PyTorch version and device; assertions focus on shapes, identities, and causality. The original embedding demonstration has no fixed seed. Current source demonstrations and chapter snippets were checked with Python 3.13 and PyTorch 2.14.1; later chapters report measured synthetic training results and a real-corpus preparation check.

## Notation

| Symbol | Meaning |
| --- | --- |
| B | Batch size: independent sequences |
| T | Sequence length |
| V | Vocabulary size (also the conventional symbol for values in attention equations) |
| D | Embedding width |
| C | Positional table capacity |
| H | Number of attention heads |
| d_h / d_k | Per-head width / key width |

Earlier scripts use `[T, D]`; the multi-head module uses `[B, T, D]`. Mermaid diagrams are editable directly in Markdown and render on GitHub. The [multi-head image](images/multi-head-attention.svg) is a standalone vector diagram with readable shapes and labels.

[Back to project README](../README.md)
