"""Extend the exact context-16 training trajectory to 10000 updates, twice."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from evaluate import evaluate_text, file_sha256
from generate import load_checkpoint
from train import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', type=Path, default=Path('data/prepared/gutenberg'))
    parser.add_argument('--previous-checkpoint', type=Path, default=Path('runs/real-books-2000/primary/checkpoint.pt'))
    parser.add_argument('--context64-checkpoint', type=Path, default=Path('runs/context-64-10000/primary/checkpoint.pt'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/context-16-10000'))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output exists; choose a fresh --output-dir')
    prep = json.loads((args.corpus_dir / 'report.json').read_text())
    texts = {name: (args.corpus_dir / f'{name}.txt').read_text(encoding='utf-8')
             for name in ('train', 'validation')}
    import hashlib
    for name, text in texts.items():
        if hashlib.sha256(text.encode()).hexdigest() != prep['splits'][name]['sha256']:
            raise ValueError(f'{name} data hash differs from preparation')
    old, old_tokenizer = load_checkpoint(args.previous_checkpoint)
    previous = torch.load(args.previous_checkpoint, map_location='cpu', weights_only=True)
    for name in ('train', 'validation'):
        if previous[name + '_sha256'] != prep['splits'][name]['sha256']:
            raise ValueError('previous model used different data')
    expected_config = dict(vocab_size=28, context_length=16, embedding_dim=24,
                           num_heads=3, num_layers=2, expansion_factor=4)
    if previous['model_config'] != expected_config:
        raise ValueError('previous model differs from the fixed context-16 experiment')
    expected_training = dict(steps=2000, seed=42, batch_size=8, learning_rate=0.003,
                             weight_decay=0.01, max_grad_norm=1.0)
    if any(previous['config'][key] != value for key, value in expected_training.items()):
        raise ValueError('previous training differs from the fixed baseline')
    prefix_checks = []
    def verify_prefix(step, model):
        if step == 2000:
            same = all(torch.equal(value, previous['model_state_dict'][key])
                       for key, value in model.state_dict().items())
            if not same:
                raise RuntimeError('step-2000 state differs from previous context-16 checkpoint')
            prefix_checks.append(same)
            print('Verified exact match to previous checkpoint at update 2000.', flush=True)
    runs = []
    for label in ('primary', 'repeat'):
        print(f'Training context-16 for 10000 updates: {label}...', flush=True)
        start = time.perf_counter()
        model, history = run_training(texts['train'], texts['validation'], steps=10000, seed=42,
            batch_size=8, context_length=16, eval_interval=200, output_dir=args.output_dir / label, on_step=verify_prefix)
        runs.append((model, history, time.perf_counter() - start))
    weights_equal = all(torch.equal(v, runs[1][0].state_dict()[k]) for k, v in runs[0][0].state_dict().items())
    histories_equal = runs[0][1] == runs[1][1]
    if not weights_equal or not histories_equal:
        raise RuntimeError('reproducibility check failed')
    checkpoint = args.output_dir / 'primary/checkpoint.pt'
    model, tokenizer = load_checkpoint(checkpoint)
    if tokenizer.stoi != old_tokenizer.stoi:
        raise ValueError('models must use the same vocabulary')
    print('Evaluating all validation targets at native and common context caps...', flush=True)
    evaluations = {
        'previous_context16_2000': evaluate_text(old, tokenizer, texts['validation'], 128),
        'long_context16_10000': evaluate_text(model, tokenizer, texts['validation'], 128),
    }
    context64, context64_tokenizer = load_checkpoint(args.context64_checkpoint)
    reference64 = torch.load(args.context64_checkpoint, map_location='cpu', weights_only=True)
    expected64 = {**expected_config, 'context_length': 64}
    if reference64['model_config'] != expected64 or context64_tokenizer.stoi != tokenizer.stoi:
        raise ValueError('context-64 reference configuration or vocabulary differs')
    for name in ('train', 'validation'):
        if reference64[name + '_sha256'] != prep['splits'][name]['sha256']:
            raise ValueError('context-64 reference used different data')
    matched_settings = {**expected_training, 'steps': 10000, 'batch_size': 2}
    if any(reference64['config'][key] != value for key, value in matched_settings.items()):
        raise ValueError('context-64 reference training budget or optimizer differs')
    evaluations['long_context64_10000'] = evaluate_text(context64, tokenizer, texts['validation'], 128)
    evaluations['long_context64_capped16_10000'] = evaluate_text(context64, tokenizer, texts['validation'], 128, context_length=16)
    ids = torch.tensor([tokenizer.encode('the creature ')], dtype=torch.long)
    with torch.no_grad():
        reload_equal = torch.equal(model(ids), runs[0][0](ids))
    report = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={name: file_sha256(name) for name in ('scripts/run_matched_budget_experiment.py',
            'src/train.py', 'src/evaluate.py', 'src/gpt.py', 'src/transformer_block.py',
            'src/multi_head_attention.py', 'src/feed_forward.py', 'src/generate.py')},
        python_version=platform.python_version(), torch_version=str(torch.__version__),
        platform=platform.platform(), seed=42, device='cpu', threads=1,
        training_targets_per_run=1280000, updates_per_run=10000,
        old_batch_size=8, new_batch_size=8, old_context_length=16, new_context_length=16,
        old_parameters=sum(p.numel() for p in old.parameters()),
        new_parameters=sum(p.numel() for p in model.parameters()),
        previous_checkpoint=str(args.previous_checkpoint), previous_checkpoint_sha256=file_sha256(args.previous_checkpoint),
        checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
        train_sha256=prep['splits']['train']['sha256'], validation_sha256=prep['splits']['validation']['sha256'],
        evaluations=evaluations, history=runs[0][1], repeated_weights_identical=weights_equal,
        repeated_histories_identical=histories_equal, reload_logits_identical=reload_equal,
        primary_training_seconds=runs[0][2], repeat_training_seconds=runs[1][2],
        step2000_matches_previous=prefix_checks,
        periodic_prefix_matches_previous=all(
            [row for row in run[1] if row['step'] <= 2000] == previous['history'] for run in runs),
        context64_checkpoint=str(args.context64_checkpoint),
        context64_checkpoint_sha256=file_sha256(args.context64_checkpoint),
        context64_parameters=sum(p.numel() for p in context64.parameters()),
        context64_training_targets=reference64['config']['steps'] * reference64['config']['batch_size'] * reference64['model_config']['context_length'],
        test_used=False)
    (args.output_dir / 'experiment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
