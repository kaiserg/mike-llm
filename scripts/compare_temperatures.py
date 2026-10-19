"""Generate a fixed, unfiltered temperature grid from the real-book checkpoint."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from evaluate import file_sha256
from generate import generate, load_checkpoint


def repetition_metrics(text):
    words = re.findall(r'[a-z]+', text)
    grams = [text[i:i + 4] for i in range(max(0, len(text) - 3))]
    return dict(word_count=len(words), unique_word_count=len(set(words)),
                repeated_character_4gram_fraction=1 - len(set(grams)) / len(grams) if grams else 0,
                most_common_words=Counter(words).most_common(5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/real-books-2000/primary/checkpoint.pt'))
    parser.add_argument('--output', type=Path, default=Path('runs/real-books-2000/temperature-comparison.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new --output')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, tokenizer = load_checkpoint(args.checkpoint)
    samples = []
    for prompt in ('the creature ', 'i felt '):
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        for temperature in (0.0, 0.5, 0.8, 1.0, 1.3):
            for seed in (42, 123):
                def sample():
                    return generate(model, ids, 200, temperature, torch.Generator().manual_seed(seed))
                output = sample()
                if not torch.equal(output, sample()):
                    raise RuntimeError('same-seed generation was not repeatable')
                full = tokenizer.decode(output[0].tolist())
                continuation = full[len(prompt):]
                samples.append(dict(prompt=prompt, temperature=temperature, seed=seed,
                                    continuation=continuation, text=full,
                                    metrics=repetition_metrics(continuation)))
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=file_sha256(args.checkpoint),
                  code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  python_version=platform.python_version(), torch_version=str(torch.__version__),
                  device='cpu', threads=1, context_length=model.context_length,
                  max_new_tokens=200, same_seed_repeats_identical=True, test_used=False, samples=samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for row in samples:
        print(f"\nprompt={row['prompt']!r} temperature={row['temperature']} seed={row['seed']}")
        print(row['text'])


if __name__ == '__main__':
    main()
