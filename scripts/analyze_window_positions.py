"""Score training-like full windows for the same four paired checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from evaluate import file_sha256
from generate import load_checkpoint
from window_diagnostics import score_window_grid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=Path('docs/results/multiseed-context-experiment.json'))
    p.add_argument('--validation-file',type=Path,default=Path('data/prepared/gutenberg/validation.txt'))
    p.add_argument('--output',type=Path,default=Path('runs/position-diagnostics/windows.json'))
    args=p.parse_args()
    if args.output.exists():p.error('output exists; choose a new --output')
    study=json.loads(args.study.read_text())
    text=args.validation_file.read_text(encoding='utf-8')
    if hashlib.sha256(text.encode()).hexdigest()!=study['validation_sha256']:raise ValueError('validation differs')
    starts=list(range(0,len(text)-64,64))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows=[]
    for source in study['rows']:
        row=dict(seed=source['seed'],contexts={})
        for ctx in ('16','64'):
            item=source['models'][ctx]
            if file_sha256(item['checkpoint'])!=item['checkpoint_sha256']:raise ValueError('checkpoint differs')
            model,tokenizer=load_checkpoint(item['checkpoint'])
            ids=torch.tensor(tokenizer.encode(text),dtype=torch.long)
            row['contexts'][ctx]=score_window_grid(model,ids,starts)
        rows.append(row)
    summary={}
    for ctx in ('16','64'):
        summary[ctx]=dict(mean_nll=statistics.mean(row['contexts'][ctx]['mean_nll'] for row in rows),
            positions=[dict(block_column=i,position_id=i%int(ctx),mean_nll=statistics.mean(row['contexts'][ctx]['positions'][i]['mean_nll'] for row in rows)) for i in range(64)])
    report=dict(study_sha256=file_sha256(args.study),validation_sha256=study['validation_sha256'],
        source_sha256={name:file_sha256(name) for name in ('scripts/analyze_window_positions.py','src/window_diagnostics.py')},
        block_starts=starts,scored_targets=len(starts)*64,
        protocol='nonoverlapping 64-input blocks; context16 resets every 16 inputs; same targets in both configurations',
        rows=rows,summary=summary,test_used=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Saved window-position diagnostics:',args.output)


if __name__=='__main__':main()
