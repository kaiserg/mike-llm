"""Reproducible CPU training demonstration; run from the repository root."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from gpt import GPT
from tokenizer import CharacterTokenizer


# Original synthetic text for this project; intentionally repetitive and tiny.
DEMO_TRAIN = (
    "a small model reads a token. the next token follows the last token.\n"
    "we learn from text. we test on text. a model learns a small pattern.\n"
) * 20
DEMO_VALIDATION = (
    "the model reads text. a small token follows the last token.\n"
    "we test a small model. we learn a pattern from text.\n"
) * 4


def prepare_data(train_text, validation_text, context_length):
    """Fit vocabulary on training text only; never create cross-split windows."""
    if len(train_text) <= context_length or len(validation_text) <= context_length:
        raise ValueError("each split needs at least context_length + 1 characters")
    tokenizer = CharacterTokenizer(train_text)
    unknown = sorted(set(validation_text) - set(tokenizer.stoi))
    if unknown:
        raise ValueError(f"validation contains characters absent from training: {unknown!r}")
    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    validation_ids = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)
    return tokenizer, train_ids, validation_ids


def sample_batch(data, batch_size, context_length, generator):
    starts = torch.randint(len(data) - context_length, (batch_size,), generator=generator)
    inputs = torch.stack([data[i:i + context_length] for i in starts.tolist()])
    targets = torch.stack([data[i + 1:i + context_length + 1] for i in starts.tolist()])
    return inputs, targets


def next_token_loss(model, inputs, targets):
    logits = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.no_grad()
def evaluate(model, batches):
    was_training = model.training
    model.eval()
    try:
        return sum(next_token_loss(model, x, y).item() for x, y in batches) / len(batches)
    finally:
        model.train(was_training)


def run_training(train_text, validation_text, *, steps=100, seed=42,
                 batch_size=8, context_length=16, learning_rate=0.003,
                 eval_interval=25, eval_batches=8, output_dir=Path("runs/first-training"),
                 on_step=None, positional_encoding="learned"):
    for name, value in dict(steps=steps, batch_size=batch_size, context_length=context_length,
                            eval_interval=eval_interval, eval_batches=eval_batches).items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    # This entry point deliberately uses CPU and one thread for repeatability.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    tokenizer, train_ids, validation_ids = prepare_data(train_text, validation_text, context_length)
    model_config = dict(vocab_size=tokenizer.vocab_size, context_length=context_length,
                        embedding_dim=24, num_heads=3, num_layers=2, expansion_factor=4)
    if positional_encoding != "learned":
        model_config["positional_encoding"] = positional_encoding
    model = GPT(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    train_rng = torch.Generator().manual_seed(seed + 1)
    eval_rng = torch.Generator().manual_seed(seed + 2)
    # Fixed evaluation windows, independent of training-batch randomness.
    evaluation = {
        name: [sample_batch(data, batch_size, context_length, eval_rng) for _ in range(eval_batches)]
        for name, data in (("train", train_ids), ("validation", validation_ids))
    }
    history = []

    def record(step):
        row = {"step": step, **{name + "_loss": evaluate(model, batches)
                               for name, batches in evaluation.items()}}
        history.append(row)
        print(f"step {step:4d} | train {row['train_loss']:.4f} | validation {row['validation_loss']:.4f}")

    model.train()
    record(0)
    for step in range(1, steps + 1):
        inputs, targets = sample_batch(train_ids, batch_size, context_length, train_rng)
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model, inputs, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % eval_interval == 0 or step == steps:
            record(step)
        if on_step is not None:
            on_step(step, model)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(steps=steps, seed=seed, batch_size=batch_size, learning_rate=learning_rate,
                  eval_interval=eval_interval, eval_batches=eval_batches, device="cpu",
                  threads=1, weight_decay=0.01, max_grad_norm=1.0)
    metadata = dict(config=config, model_config=model_config, torch_version=str(torch.__version__),
                    tokenizer_stoi=tokenizer.stoi, train_characters=len(train_text),
                    validation_characters=len(validation_text),
                    train_sha256=hashlib.sha256(train_text.encode()).hexdigest(),
                    validation_sha256=hashlib.sha256(validation_text.encode()).hexdigest(),
                    history=history)
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    torch.save({**metadata, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "step": steps,
                "torch_rng_state": torch.get_rng_state(), "train_rng_state": train_rng.get_state()},
               output_dir / "checkpoint.pt")
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--positional-encoding", choices=["learned", "sinusoidal"], default="learned")
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/first-training"))
    args = parser.parse_args()
    if bool(args.train_file) != bool(args.validation_file):
        parser.error("provide both --train-file and --validation-file, or neither")
    train_text = args.train_file.read_text(encoding="utf-8") if args.train_file else DEMO_TRAIN
    val_text = args.validation_file.read_text(encoding="utf-8") if args.validation_file else DEMO_VALIDATION
    run_training(train_text, val_text, steps=args.steps, seed=args.seed, output_dir=args.output_dir,
                 context_length=args.context_length, batch_size=args.batch_size,
                 positional_encoding=args.positional_encoding)


if __name__ == "__main__":
    main()
