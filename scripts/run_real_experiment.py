"""Train twice on Frankenstein and evaluate all Alice targets; never read test text."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from evaluate import evaluate_text, file_sha256
from generate import load_checkpoint
from gpt import GPT
from train import prepare_data, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', type=Path, default=Path('data/prepared/gutenberg'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/real-books-2000'))
    parser.add_argument('--steps', type=int, default=2000)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('experiment output already exists; choose a new --output-dir')
    if args.steps < 1:
        parser.error('steps must be positive')
    train_text = (args.corpus_dir / 'train.txt').read_text(encoding='utf-8')
    validation_text = (args.corpus_dir / 'validation.txt').read_text(encoding='utf-8')
    corpus_report = json.loads((args.corpus_dir / 'report.json').read_text())
    for name, text in (('train', train_text), ('validation', validation_text)):
        if hashlib.sha256(text.encode()).hexdigest() != corpus_report['splits'][name]['sha256']:
            raise ValueError(f'{name} file differs from preparation report')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    tokenizer, _, _ = prepare_data(train_text, validation_text, 16)
    config = dict(vocab_size=tokenizer.vocab_size, context_length=16,
                  embedding_dim=24, num_heads=3, num_layers=2, expansion_factor=4)
    initial = GPT(**config)
    print('Evaluating untrained baseline on the full validation text...', flush=True)
    baseline = evaluate_text(initial, tokenizer, validation_text, batch_size=128)
    results = []
    for label in ('primary', 'repeat'):
        print(f'Training {label} run...', flush=True)
        start = time.perf_counter()
        model, history = run_training(train_text, validation_text, steps=args.steps, seed=42,
            eval_interval=200, output_dir=args.output_dir / label)
        results.append((model, history, time.perf_counter() - start))
    first, second = results[0][0], results[1][0]
    same_history = results[0][1] == results[1][1]
    same_weights = all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    if not same_history or not same_weights:
        raise RuntimeError('repeatability check failed')
    checkpoint = args.output_dir / 'primary/checkpoint.pt'
    restored, restored_tokenizer = load_checkpoint(checkpoint)
    print('Evaluating reloaded checkpoint on the full validation text...', flush=True)
    final = evaluate_text(restored, restored_tokenizer, validation_text, batch_size=128)
    ids = torch.tensor([tokenizer.encode('the model')], dtype=torch.long)
    with torch.no_grad():
        reload_exact = torch.equal(first(ids), restored(ids))
    report = dict(
        training_code_commit=__import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        python_version=platform.python_version(), torch_version=str(torch.__version__),
        platform=platform.platform(), device='cpu', threads=1, seed=42,
        steps=args.steps, batch_size=8, learning_rate=0.003, weight_decay=0.01,
        max_grad_norm=1.0, eval_interval=200, eval_batches=8, model_config=config,
        parameters=sum(p.numel() for p in first.parameters()),
        train_characters=len(train_text), validation_characters=len(validation_text),
        train_sha256=corpus_report['splits']['train']['sha256'],
        validation_sha256=corpus_report['splits']['validation']['sha256'],
        preparation_report_sha256=file_sha256(args.corpus_dir / 'report.json'),
        checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
        evaluation_protocol='each target once; up to 16 prior characters; positions reset per window',
        full_evaluation_batch_size=128, baseline=baseline, final=final,
        repeated_histories_identical=same_history, repeated_weights_identical=same_weights,
        checkpoint_reload_logits_identical=reload_exact,
        primary_training_seconds=results[0][2], repeat_training_seconds=results[1][2],
        history=results[0][1], test_evaluated=False,
    )
    (args.output_dir / 'experiment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
