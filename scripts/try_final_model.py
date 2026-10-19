"""Generate text with the frozen final checkpoint and its selected inference policy."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate import generate, load_checkpoint


def file_sha256(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_frozen_selection(path):
    selection = json.loads(path.read_text())
    selected = selection["selected"]
    checkpoint = ROOT / selected["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"selected checkpoint is unavailable: {checkpoint}")
    if file_sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("selected checkpoint hash differs from the frozen manifest")
    context_length = selected["evaluation_context_length"]
    if selected["inference"] != "crop last evaluation_context_length characters; reset IDs to 0..L-1; native causal attention":
        raise ValueError("unsupported frozen inference policy")
    return checkpoint, context_length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "docs/results/final-model-selection.json")
    parser.add_argument("--prompt", default="the creature ")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.prompt:
        parser.error("prompt must not be empty")
    if args.max_new_tokens < 0 or not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("max-new-tokens must be nonnegative and temperature finite and nonnegative")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        checkpoint, context_length = load_frozen_selection(args.selection)
        model, tokenizer = load_checkpoint(checkpoint)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    unknown = sorted(set(args.prompt) - set(tokenizer.stoi))
    if unknown:
        parser.error(f"prompt contains characters absent from checkpoint vocabulary: {unknown!r}")
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    rng = torch.Generator().manual_seed(args.seed)
    output = generate(model, ids, args.max_new_tokens, args.temperature, rng, context_length)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
