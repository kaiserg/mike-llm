"""Mask added history per head in learned seed 7 and sinusoidal seed 123."""
import argparse
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import torch.nn.functional as F
from evaluate import file_sha256
from generate import load_checkpoint
from head_diagnostics import forward_head_masks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/head-history/report.json'))
    parser.add_argument('--batch-size',type=int,default=128)
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists; choose a fresh path')
    if args.batch_size<1:parser.error('batch size must be positive')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    archive_path=Path('docs/results/nonmonotonic-history-diagnostics.json')
    archive=json.loads(archive_path.read_text())
    validation=Path('data/prepared/gutenberg/validation.txt')
    if file_sha256(validation)!=archive['validation_sha256']:raise ValueError('validation changed')
    text=validation.read_text();rows=[]
    for encoding,seed in (('learned',7),('sinusoidal',123)):
        source=next(x for x in archive['rows'] if x['encoding']==encoding and x['seed']==seed)
        if file_sha256(source['checkpoint'])!=source['checkpoint_sha256']:raise ValueError('checkpoint changed')
        model,tokenizer=load_checkpoint(source['checkpoint'])
        if len(model.blocks)!=2 or any(len(b.attention.heads)!=3 for b in model.blocks):raise ValueError('expected two blocks with three heads')
        pairs=[(b,h) for b in range(2) for h in range(3)]
        conditions={'baseline16':[],**{f'block{b+1}_head{h+1}':[(b,h)] for b,h in pairs},
                    'block1_all':[(0,h) for h in range(3)],'block2_all':[(1,h) for h in range(3)],'all8':pairs}
        ids=torch.tensor(tokenizer.encode(text),dtype=torch.long)
        losses={k:[] for k in conditions};correct={k:0 for k in conditions}
        for start in range(1,len(ids),args.batch_size):
            groups={}
            for t in range(start,min(start+args.batch_size,len(ids))):groups.setdefault(min(t,64),[]).append(t)
            for length,targets in groups.items():
                inputs=torch.stack([ids[t-length:t] for t in targets])
                for name,blocked in conditions.items():
                    logits=forward_head_masks(model,inputs,blocked)
                    nll=F.cross_entropy(logits.double(),ids[targets],reduction='none')
                    if not torch.isfinite(nll).all():raise RuntimeError('nonfinite loss')
                    losses[name].extend(nll.tolist());correct[name]+=(logits.argmax(-1)==ids[targets]).sum().item()
        metrics={}
        baseline=torch.tensor(losses['baseline16'],dtype=torch.float64)
        for name,values in losses.items():
            mean=math.fsum(values)/len(values)
            delta=torch.tensor(values,dtype=torch.float64)-baseline
            metrics[name]=dict(mean_nll=mean,perplexity=math.exp(mean),accuracy=correct[name]/len(values),scored_tokens=len(values),
                nll_delta_vs_baseline16=delta.mean().item(),improvement_vs_baseline16=-delta.mean().item(),
                fraction_targets_improved=(delta<0).double().mean().item(),median_nll_delta=delta.median().item())
        for name,old in (('baseline16','16_16'),('all8','8_8'),('block1_all','8_16'),('block2_all','16_8')):
            if abs(metrics[name]['mean_nll']-source['metrics'][old]['mean_nll'])>1e-10:raise ValueError('archived control mismatch')
        if file_sha256(source['checkpoint'])!=source['checkpoint_sha256']:raise ValueError('checkpoint modified')
        ranking=sorted((k for k in conditions if '_head' in k),key=lambda k:metrics[k]['nll_delta_vs_baseline16'])
        row=dict(encoding=encoding,seed=seed,checkpoint=source['checkpoint'],checkpoint_sha256=source['checkpoint_sha256'],
                 metrics=metrics,head_ranking_most_helpful_to_mask=ranking,
                 conditions_blocked_zero_based={k:[list(pair) for pair in v] for k,v in conditions.items()},
                 original_16_minus_8_nll=source['layer_decomposition']['total_delta'],
                 nonadditivity={f'block{b+1}':dict(sum_individual_improvements=sum(metrics[f'block{b+1}_head{h+1}']['improvement_vs_baseline16'] for h in range(3)),
                    joint_block_improvement=metrics[f'block{b+1}_all']['improvement_vs_baseline16']) for b in range(2)})
        rows.append(row)
        print(encoding,seed,[(k,metrics[k]['improvement_vs_baseline16']) for k in ranking],flush=True)
    sources=['scripts/analyze_head_history.py','src/head_diagnostics.py','src/history_ablation.py','src/gpt.py',
             'src/transformer_block.py','src/multi_head_attention.py','src/feed_forward.py','src/sinusoidal_positions.py','src/generate.py']
    report=dict(protocol='All validation targets; final16 outer isolation in every head; restrict selected heads to latest8 in all queries; original position IDs; no training',
        validation_sha256=file_sha256(validation),validation_characters=len(text),test_used=False,
        archive_sha256=file_sha256(archive_path),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={p:file_sha256(p) for p in sources},python_version=platform.python_version(),torch_version=str(torch.__version__),
        device='cpu',threads=1,deterministic_algorithms=True,batch_size=args.batch_size,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
