"""Analyze all four seed pairs without training or accessing test text."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from evaluate import file_sha256
from generate import load_checkpoint
from position_diagnostics import score_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('docs/results/multiseed-context-experiment.json'))
    parser.add_argument('--validation-file', type=Path, default=Path('data/prepared/gutenberg/validation.txt'))
    parser.add_argument('--output', type=Path, default=Path('runs/position-diagnostics/report.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose a new --output')
    study = json.loads(args.study.read_text())
    text = args.validation_file.read_text(encoding='utf-8')
    if hashlib.sha256(text.encode()).hexdigest() != study['validation_sha256']:
        raise ValueError('validation hash differs from the multi-seed study')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # Fixed positions spanning the full book; all lengths and checkpoints score these targets.
    targets = list(range(64, len(text), 64))
    rows = []
    for seed_row in study['rows']:
        row = dict(seed=seed_row['seed'], profiles={}, shifted16={}, checkpoints={})
        for context in (16, 64):
            source = seed_row['models'][str(context)]
            if file_sha256(source['checkpoint']) != source['checkpoint_sha256']:
                raise ValueError('checkpoint differs from study')
            model, tokenizer = load_checkpoint(source['checkpoint'])
            ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
            print(f"Scoring seed {row['seed']} context {context} at lengths 1..{context}...", flush=True)
            row['profiles'][str(context)] = score_positions(model, ids, targets, list(range(1, context + 1)))
            if context == 64:
                for offset in (0, 16, 32, 48):
                    row['shifted16'][str(offset)] = score_positions(model, ids, targets, [16], position_offset=offset)[0]
            row['checkpoints'][str(context)] = dict(path=source['checkpoint'], sha256=source['checkpoint_sha256'])
        rows.append(row)
    summary = dict(profiles={}, shifted16={})
    for context in (16, 64):
        summary['profiles'][str(context)] = []
        for index in range(context):
            values = [row['profiles'][str(context)][index]['mean_nll'] for row in rows]
            summary['profiles'][str(context)].append(dict(history_length=index + 1, final_position_id=index,
                mean_nll=statistics.mean(values), seed_sample_sd=statistics.stdev(values)))
    for offset in (0, 16, 32, 48):
        values = [row['shifted16'][str(offset)]['mean_nll'] for row in rows]
        summary['shifted16'][str(offset)] = dict(mean_nll=statistics.mean(values), seed_sample_sd=statistics.stdev(values))
    report = dict(study=str(args.study), study_sha256=file_sha256(args.study),
        validation_sha256=study['validation_sha256'], validation_characters=len(text),
        target_indices=targets, target_count=len(targets), target_stride=64,
        target_indices_sha256=hashlib.sha256(json.dumps(targets).encode()).hexdigest(),
        protocol='same targets for all lengths; natural position IDs 0..L-1; only last-position loss',
        shifted_protocol='same 16-token prefixes; shift every position ID by 0,16,32,48',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        source_sha256={name:file_sha256(name) for name in ('scripts/analyze_positions.py','src/position_diagnostics.py',
            'src/gpt.py','src/transformer_block.py','src/multi_head_attention.py','src/feed_forward.py','src/generate.py')},
        python_version=platform.python_version(), torch_version=str(torch.__version__),
        device='cpu', threads=1, rows=rows, summary=summary, test_used=False,
        training_position_examples_per_run={'context16':80000,'context64':20000})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Saved position diagnostics:',args.output,flush=True)


if __name__ == '__main__':
    main()
