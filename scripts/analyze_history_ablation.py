"""Compare full history with isolated last-16 history, keeping position IDs."""
import argparse
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
from history_ablation import score_history_ablation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/history-ablation/report.json'))
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose another path')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    validation = Path('data/prepared/gutenberg/validation.txt')
    text = validation.read_text()
    rows = []
    studies = {'learned':'docs/results/multiseed-context-experiment.json',
               'sinusoidal':'docs/results/sinusoidal-context-experiment.json'}
    for encoding, study_path in studies.items():
        study = json.loads(Path(study_path).read_text())
        if file_sha256(validation) != study['validation_sha256']:
            raise ValueError('validation hash differs')
        for row in study['rows']:
            source = row['models']['64']
            if file_sha256(source['checkpoint']) != source['checkpoint_sha256']:
                raise ValueError('checkpoint hash differs')
            model, tokenizer = load_checkpoint(source['checkpoint'])
            ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
            result = score_history_ablation(model, ids, args.batch_size)
            reference = row['evaluations']['context64']['mean_nll']
            if abs(result['full']['mean_nll'] - reference) > 1e-6:
                raise ValueError('native evaluation differs from archived reference')
            if file_sha256(source['checkpoint']) != source['checkpoint_sha256']:
                raise ValueError('checkpoint changed')
            rows.append(dict(encoding=encoding, seed=row['seed'], checkpoint=source['checkpoint'],
                checkpoint_sha256=source['checkpoint_sha256'], native_reference_nll=reference, **result))
            print(encoding, row['seed'], result, flush=True)
    summary = {}
    for encoding in studies:
        group = [r for r in rows if r['encoding'] == encoding]
        summary[encoding] = {name:dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values))
            for name, values in {
                'full_nll':[r['full']['mean_nll'] for r in group],
                'masked_nll':[r['masked']['mean_nll'] for r in group],
                'full_perplexity':[r['full']['perplexity'] for r in group],
                'masked_perplexity':[r['masked']['perplexity'] for r in group],
                'masked_minus_full_nll':[r['masked_minus_full_nll'] for r in group]}.items()}
    sources = ['scripts/analyze_history_ablation.py','src/history_ablation.py','src/gpt.py',
               'src/multi_head_attention.py','src/transformer_block.py','src/sinusoidal_positions.py',
               'src/feed_forward.py','src/generate.py']
    report = dict(protocol='Every validation target once; native window up to 64; isolate last 16 tokens in every layer; preserve all position IDs; score last query only',
        validation_sha256=file_sha256(validation), validation_characters=len(text), test_used=False,
        training_steps=10000, device='cpu', threads=1, deterministic_algorithms=True,
        batch_size=args.batch_size, python_version=platform.python_version(), torch_version=str(torch.__version__),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={p:file_sha256(p) for p in sources},
        study_sha256={p:file_sha256(p) for p in studies.values()}, rows=rows, summary=summary)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__':
    main()
