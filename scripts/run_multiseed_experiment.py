"""Compare contexts across fixed independent training seeds at equal target budgets."""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from evaluate import evaluate_text, file_sha256
from generate import load_checkpoint
from train import run_training


def summarize(rows):
    conditions = ('context16', 'context64', 'context64_capped16')
    metrics = ('mean_nll', 'perplexity', 'bits_per_character', 'next_token_accuracy')
    summary = {}
    for condition in conditions:
        summary[condition] = {}
        for metric in metrics:
            values = [row['evaluations'][condition][metric] for row in rows]
            summary[condition][metric] = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values))
    for condition in ('context64', 'context64_capped16'):
        differences = [row['evaluations'][condition]['mean_nll'] - row['evaluations']['context16']['mean_nll']
                       for row in rows]
        summary[condition + '_minus_context16_nll'] = dict(
            by_seed={str(row['seed']): diff for row, diff in zip(rows, differences)},
            mean=statistics.mean(differences), sample_sd=statistics.stdev(differences),
            context16_wins=sum(d > 0 for d in differences), ties=sum(d == 0 for d in differences),
            context64_wins=sum(d < 0 for d in differences))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', type=Path, default=Path('data/prepared/gutenberg'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/multiseed-context-10000'))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output exists; choose a new --output-dir')
    prep = json.loads((args.corpus_dir / 'report.json').read_text())
    texts = {s: (args.corpus_dir / f'{s}.txt').read_text(encoding='utf-8') for s in ('train', 'validation')}
    for name, text in texts.items():
        if hashlib.sha256(text.encode()).hexdigest() != prep['splits'][name]['sha256']:
            raise ValueError('corpus checksum mismatch')
    args.output_dir.mkdir(parents=True)
    rows = []
    for seed in (42, 7, 123, 2026):
        row = dict(seed=seed, evaluations={}, models={}, generation={})
        for context, batch in ((16, 8), (64, 2)):
            label = f'seed-{seed}/context-{context}'
            directory = args.output_dir / label
            directory.mkdir(parents=True)
            if seed == 42:
                checkpoint = Path(f'runs/context-{context}-10000/primary/checkpoint.pt')
                repeat_checkpoint = Path(f'runs/context-{context}-10000/repeat/checkpoint.pt')
                print(f'Reusing and verifying {label}...', flush=True)
                a = torch.load(checkpoint, map_location='cpu', weights_only=True)
                b = torch.load(repeat_checkpoint, map_location='cpu', weights_only=True)
                equal = all(torch.equal(v, b['model_state_dict'][k]) for k, v in a['model_state_dict'].items())
                histories_equal = a['history'] == b['history']
                seconds = []
            else:
                seconds = []
                for repeat in ('primary', 'repeat'):
                    print(f'Training {label} {repeat}: 10000 updates...', flush=True)
                    start = time.perf_counter()
                    with (directory / f'{repeat}.log').open('w') as log, contextlib.redirect_stdout(log):
                        run_training(texts['train'], texts['validation'], seed=seed, steps=10000,
                                     batch_size=batch, context_length=context, eval_interval=200,
                                     output_dir=directory / repeat)
                    seconds.append(time.perf_counter() - start)
                checkpoint = directory / 'primary/checkpoint.pt'
                repeat_checkpoint = directory / 'repeat/checkpoint.pt'
                a = torch.load(checkpoint, map_location='cpu', weights_only=True)
                b = torch.load(repeat_checkpoint, map_location='cpu', weights_only=True)
                equal = all(torch.equal(v, b['model_state_dict'][k]) for k, v in a['model_state_dict'].items())
                histories_equal = a['history'] == b['history']
            if not equal or not histories_equal:
                raise RuntimeError('exact reproducibility check failed')
            expected_config = dict(vocab_size=28, context_length=context, embedding_dim=24,
                                   num_heads=3, num_layers=2, expansion_factor=4)
            expected_training = dict(steps=10000, seed=seed, batch_size=batch,
                                     learning_rate=0.003, weight_decay=0.01, max_grad_norm=1.0,
                                     eval_interval=200, eval_batches=8, device='cpu', threads=1)
            if a['model_config'] != expected_config or any(a['config'][k] != v for k, v in expected_training.items()):
                raise ValueError('checkpoint configuration differs')
            for split in ('train', 'validation'):
                if a[split + '_sha256'] != prep['splits'][split]['sha256']:
                    raise ValueError('checkpoint data differs')
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            model, tokenizer = load_checkpoint(checkpoint)
            print(f'Evaluating {label} on all validation targets...', flush=True)
            result = evaluate_text(model, tokenizer, texts['validation'], 128)
            row['evaluations'][f'context{context}'] = result
            if context == 64:
                row['evaluations']['context64_capped16'] = evaluate_text(model, tokenizer, texts['validation'], 128, 16)
            row['models'][str(context)] = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                repeat_checkpoint=str(repeat_checkpoint), repeat_checkpoint_sha256=file_sha256(repeat_checkpoint),
                histories_identical=histories_equal, weights_identical=equal,
                history=a['history'], training_seconds=seconds, reused_seed42=seed == 42,
                model_config=a['model_config'], training_config=a['config'])
            print(f'Generating full temperature grid for {label}...', flush=True)
            generation_path = directory / 'generation.json'
            subprocess.run([sys.executable, 'scripts/compare_temperatures.py', '--checkpoint', str(checkpoint),
                            '--output', str(generation_path)], check=True, capture_output=True, text=True)
            row['generation'][str(context)] = json.loads(generation_path.read_text())
            print(f"Completed {label}: validation perplexity {result['perplexity']:.6f}", flush=True)
            (directory / 'result.json').write_text(json.dumps(dict(seed=seed, context=context,
                evaluation=result, model=row['models'][str(context)], generation=row['generation'][str(context)]), indent=2) + '\n')
        rows.append(row)
    report = dict(training_seeds=[42, 7, 123, 2026], new_training_seeds=[7, 123, 2026],
        steps=10000, training_targets_per_model=1280000, sampling_seeds=[42, 123],
        temperatures=[0, 0.5, 0.8, 1, 1.3], prompts=['the creature ', 'i felt '], max_new_tokens=200,
        seed42_previously_inspected=True, seed_choice_fixed_before_new_results=True,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={name: file_sha256(name) for name in ('scripts/run_multiseed_experiment.py',
            'scripts/compare_temperatures.py', 'src/train.py', 'src/evaluate.py', 'src/generate.py',
            'src/gpt.py', 'src/transformer_block.py', 'src/multi_head_attention.py', 'src/feed_forward.py')},
        python_version=platform.python_version(), torch_version=str(torch.__version__),
        device='cpu', threads=1, train_sha256=prep['splits']['train']['sha256'],
        validation_sha256=prep['splits']['validation']['sha256'], rows=rows,
        all_four_summary=summarize(rows), new_seeds_summary=summarize(rows[1:]), test_used=False)
    (args.output_dir / 'experiment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'all_four_summary': report['all_four_summary'], 'new_seeds_summary': report['new_seeds_summary']}, indent=2))


if __name__ == '__main__':
    main()
