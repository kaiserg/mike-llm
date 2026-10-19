"""Generate character tokens from a saved GPT checkpoint on CPU."""

import argparse
import math
from pathlib import Path

import torch

from gpt import GPT
from tokenizer import CharacterTokenizer


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    mapping = checkpoint["tokenizer_stoi"]
    if (not mapping or any(not isinstance(c, str) or len(c) != 1 for c in mapping)
            or any(type(i) is not int for i in mapping.values())
            or sorted(mapping.values()) != list(range(len(mapping)))):
        raise ValueError("checkpoint must contain a contiguous character vocabulary")
    tokenizer = CharacterTokenizer("".join(mapping))
    if tokenizer.stoi != mapping:
        raise ValueError("checkpoint vocabulary does not match CharacterTokenizer ordering")
    if len(mapping) != checkpoint['model_config']['vocab_size']:
        raise ValueError("checkpoint vocabulary size does not match model")
    model = GPT(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, token_ids, max_new_tokens=100, temperature=1.0, generator=None, context_length=None):
    """Return the full prompt plus continuation; temperature zero uses argmax."""
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if context_length is None:
        context_length = model.context_length
    if type(context_length) is not int or not 1 <= context_length <= model.context_length:
        raise ValueError("context_length must be an integer from 1 through the model capacity")
    if (token_ids.ndim != 2 or token_ids.dtype != torch.long
            or token_ids.shape[0] == 0 or token_ids.shape[1] == 0):
        raise ValueError("token_ids must be a nonempty [batch, sequence] torch.long tensor")
    if token_ids.min() < 0 or token_ids.max() >= model.token_embedding.num_embeddings:
        raise ValueError("prompt token IDs must be within the model vocabulary")

    was_training = model.training
    model.eval()
    try:
        result = token_ids.clone()
        for _ in range(max_new_tokens):
            context = result[:, -context_length:]
            logits = model(context)[:, -1, :]
            if not torch.isfinite(logits).all():
                raise RuntimeError("model produced nonfinite logits")
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                # Subtracting the maximum first avoids positive overflow at low temperatures.
                scores = (logits.double() - logits.double().amax(dim=-1, keepdim=True)) / temperature
                probabilities = scores.softmax(dim=-1)
                next_id = torch.multinomial(probabilities, 1, generator=generator)
            result = torch.cat((result, next_id), dim=1)
        return result
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/first-training/checkpoint.pt'))
    parser.add_argument('--prompt', default='a model')
    parser.add_argument('--max-new-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--context-length', type=int, help='inference window; defaults to checkpoint capacity')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, tokenizer = load_checkpoint(args.checkpoint)
    if not args.prompt:
        parser.error('prompt must not be empty; this tokenizer has no beginning-of-sequence token')
    unknown = sorted(set(args.prompt) - set(tokenizer.stoi))
    if unknown:
        parser.error(f'prompt contains characters absent from checkpoint vocabulary: {unknown!r}')
    if args.max_new_tokens < 0 or not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error('max-new-tokens must be nonnegative and temperature finite and nonnegative')
    if args.context_length is not None and not 1 <= args.context_length <= model.context_length:
        parser.error(f'context-length must be from 1 through checkpoint capacity {model.context_length}')
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    rng = torch.Generator().manual_seed(args.seed)
    output = generate(model, ids, args.max_new_tokens, args.temperature, rng, args.context_length)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == '__main__':
    main()
