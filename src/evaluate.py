"""Evaluate every next-token target once with a sliding context window."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from generate import load_checkpoint
from train import DEMO_VALIDATION


@torch.no_grad()
def evaluate_text(model, tokenizer, text, batch_size=32):
    """Score N-1 targets using up to context_length preceding tokens each."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if len(text) < 2:
        raise ValueError("validation text needs at least two characters")
    unknown = sorted(set(text) - set(tokenizer.stoi))
    if unknown:
        raise ValueError(f"validation contains characters absent from checkpoint vocabulary: {unknown!r}")
    device = next(model.parameters()).device
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    total_nll = 0.0
    correct = 0
    count = 0
    was_training = model.training
    model.eval()
    try:
        for start in range(1, len(ids), batch_size):
            # Group by length: short initial prefixes need no padding or fake context.
            groups = {}
            for target in range(start, min(start + batch_size, len(ids))):
                length = min(target, model.context_length)
                groups.setdefault(length, []).append(target)
            for length, targets in groups.items():
                inputs = torch.stack([ids[t - length:t] for t in targets])
                labels = ids[targets]
                logits = model(inputs)[:, -1, :]
                losses = F.cross_entropy(logits.double(), labels, reduction="none")
                if not torch.isfinite(losses).all():
                    raise RuntimeError("evaluation produced nonfinite loss")
                # Sum per-token losses in stable target order, independent of batch grouping.
                total_nll += math.fsum(losses.tolist())
                correct += (logits.argmax(-1) == labels).sum().item()
                count += len(targets)
    finally:
        model.train(was_training)
    mean_nll = total_nll / count
    try:
        perplexity = math.exp(mean_nll)
    except OverflowError:
        perplexity = None
    return dict(scored_tokens=count, mean_nll=mean_nll, perplexity=perplexity,
                perplexity_overflow=perplexity is None, bits_per_character=mean_nll / math.log(2),
                next_token_accuracy=correct / count, total_nll=total_nll)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/first-training/checkpoint.pt'))
    parser.add_argument('--validation-file', type=Path)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', type=Path, default=Path('runs/first-training/evaluation.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, tokenizer = load_checkpoint(args.checkpoint)
    text = args.validation_file.read_text(encoding='utf-8') if args.validation_file else DEMO_VALIDATION
    try:
        metrics = evaluate_text(model, tokenizer, text, args.batch_size)
    except ValueError as error:
        parser.error(str(error))
    report = dict(
        checkpoint=str(args.checkpoint), checkpoint_sha256=file_sha256(args.checkpoint),
        validation_source=str(args.validation_file) if args.validation_file else 'built-in synthetic validation',
        validation_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
        validation_characters=len(text), vocabulary_size=tokenizer.vocab_size,
        context_length=model.context_length, batch_size=args.batch_size,
        protocol='each target once; up to context_length prior characters; positions reset per window',
        first_character_scored=False, device='cpu', threads=1,
        torch_version=str(torch.__version__), **metrics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
