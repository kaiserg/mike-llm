"""Train a tiny GPT from scratch and show measurable learning in a few minutes."""

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate import generate
from gpt import GPT
from tokenizer import CharacterTokenizer
from train import prepare_data, run_training


TRAIN_TEXT = (
    "red blue green yellow\n"
    "blue green yellow red\n"
    "green yellow red blue\n"
    "yellow red blue green\n"
) * 40
VALIDATION_TEXT = (
    "green yellow red blue\n"
    "red blue green yellow\n"
    "yellow red blue green\n"
    "blue green yellow red\n"
) * 8
PROMPT = "red "
MODEL_CONFIG = {
    "context_length": 16,
    "embedding_dim": 24,
    "num_heads": 3,
    "num_layers": 2,
    "expansion_factor": 4,
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_greedy(model, tokenizer, prompt, characters):
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    output = generate(model, ids, characters, temperature=0)
    return tokenizer.decode(output[0].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/quick-demo")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")

    tokenizer, _, _ = prepare_data(TRAIN_TEXT, VALIDATION_TEXT, MODEL_CONFIG["context_length"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    initial_model = GPT(vocab_size=tokenizer.vocab_size, **MODEL_CONFIG)
    before = decode_greedy(initial_model, tokenizer, PROMPT, 64)

    started = time.perf_counter()
    model, history = run_training(
        TRAIN_TEXT,
        VALIDATION_TEXT,
        steps=args.steps,
        seed=args.seed,
        batch_size=8,
        context_length=MODEL_CONFIG["context_length"],
        learning_rate=0.003,
        eval_interval=max(1, args.steps // 6),
        eval_batches=8,
        output_dir=args.output_dir,
    )
    elapsed = time.perf_counter() - started
    after = decode_greedy(model, tokenizer, PROMPT, 64)
    initial_loss = history[0]["validation_loss"]
    final_loss = history[-1]["validation_loss"]
    if not (math.isfinite(final_loss) and final_loss < initial_loss):
        raise RuntimeError("demo did not reduce validation loss")

    report = {
        "schema_version": 1,
        "purpose": "clean-clone learning demonstration; synthetic data only",
        "steps": args.steps,
        "seed": args.seed,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
        },
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_config": {**MODEL_CONFIG, "vocab_size": tokenizer.vocab_size},
        "train_characters": len(TRAIN_TEXT),
        "validation_characters": len(VALIDATION_TEXT),
        "initial_validation_loss": initial_loss,
        "final_validation_loss": final_loss,
        "validation_loss_reduction_percent": 100 * (initial_loss - final_loss) / initial_loss,
        "initial_validation_perplexity": math.exp(initial_loss),
        "final_validation_perplexity": math.exp(final_loss),
        "elapsed_seconds": elapsed,
        "prompt": PROMPT,
        "before_training": before,
        "after_training": after,
        "history": history,
        "data": {
            "kind": "original synthetic cyclic color language embedded in scripts/quick_demo.py",
            "train_sha256": hashlib.sha256(TRAIN_TEXT.encode()).hexdigest(),
            "validation_sha256": hashlib.sha256(VALIDATION_TEXT.encode()).hexdigest(),
        },
        "source_sha256": {
            path: sha256(ROOT / path)
            for path in ("scripts/quick_demo.py", "src/train.py", "src/generate.py", "src/gpt.py")
        },
        "limitations": [
            "The task is intentionally repetitive and tests pattern learning, not language understanding.",
            "Training and validation use the same synthetic grammar with different sequence order.",
            "Greedy output is a qualitative illustration; validation loss is the measured learning signal.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "demo-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\nBefore training:")
    print(before)
    print("\nAfter training:")
    print(after)
    print(f"\nValidation loss: {initial_loss:.4f} -> {final_loss:.4f} "
          f"({report['validation_loss_reduction_percent']:.1f}% reduction)")
    print(f"Validation perplexity: {math.exp(initial_loss):.2f} -> {math.exp(final_loss):.2f}")
    print(f"Elapsed training time: {elapsed:.2f} seconds on CPU")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
