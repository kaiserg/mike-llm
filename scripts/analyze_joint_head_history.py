"""Measure joint and conditional interactions among previously selected heads."""
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


def pair(name):
    block,head=name.split('_')
    return int(block.removeprefix('block'))-1,int(head.removeprefix('head'))-1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/joint-head-history/report.json'))
    parser.add_argument('--batch-size',type=int,default=128)
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists; choose a fresh path')
    if args.batch_size<1:parser.error('batch size must be positive')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    archive_path=Path('docs/results/head-history-diagnostics.json')
    archive=json.loads(archive_path.read_text())
    validation=Path('data/prepared/gutenberg/validation.txt')
    if file_sha256(validation)!=archive['validation_sha256']:raise ValueError('validation changed')
    text=validation.read_text();rows=[]
    for source in archive['rows']:
        if file_sha256(source['checkpoint'])!=source['checkpoint_sha256']:raise ValueError('checkpoint changed')
        model,tokenizer=load_checkpoint(source['checkpoint'])
        harmful=[name for name in source['head_ranking_most_helpful_to_mask']
                 if source['metrics'][name]['improvement_vs_baseline16']>0]
        selected=[pair(name) for name in harmful]
        conditions={'baseline16':[], 'joint_selected':selected,
                    'all8':[(b,h) for b in range(2) for h in range(3)]}
        conditions.update({f'without_{name}':[p for p in selected if p!=pair(name)] for name in harmful})
        conditions.update({f'prefix_{k}':selected[:k] for k in range(1,len(selected)+1)})
        conditions.update({f'selected_block{b+1}':[p for p in selected if p[0]==b] for b in range(2)})
        # Compute identical mask sets once, preserving all descriptive aliases.
        aliases={};unique={}
        for name,blocked in conditions.items():
            key=tuple(sorted(blocked))
            if key not in unique:unique[key]=name
            aliases[name]=unique[key]
        losses={name:[] for name in unique.values()};correct={name:0 for name in losses}
        ids=torch.tensor(tokenizer.encode(text),dtype=torch.long)
        for start in range(1,len(ids),args.batch_size):
            groups={}
            for t in range(start,min(start+args.batch_size,len(ids))):groups.setdefault(min(t,64),[]).append(t)
            for length,targets in groups.items():
                inputs=torch.stack([ids[t-length:t] for t in targets])
                for blocked,name in unique.items():
                    logits=forward_head_masks(model,inputs,blocked)
                    nll=F.cross_entropy(logits.double(),ids[targets],reduction='none')
                    if not torch.isfinite(nll).all():raise RuntimeError('nonfinite loss')
                    losses[name].extend(nll.tolist());correct[name]+=(logits.argmax(-1)==ids[targets]).sum().item()
        metrics={}
        baseline=torch.tensor(losses['baseline16'],dtype=torch.float64)
        for name,values in losses.items():
            mean=math.fsum(values)/len(values);delta=torch.tensor(values,dtype=torch.float64)-baseline
            metrics[name]=dict(mean_nll=mean,perplexity=math.exp(mean),accuracy=correct[name]/len(values),
                scored_tokens=len(values),improvement_vs_baseline16=-delta.mean().item(),
                fraction_targets_improved=(delta<0).double().mean().item())
        metrics={name:metrics[alias] for name,alias in aliases.items()}
        # Recompute both endpoints and any individual or whole-block archived mask.
        archive_matches={}
        old_sets={tuple(sorted(tuple(p) for p in blocked)):name for name,blocked in source['conditions_blocked_zero_based'].items()}
        for blocked,name in unique.items():
            if blocked in old_sets:
                old=old_sets[blocked];difference=metrics[name]['mean_nll']-source['metrics'][old]['mean_nll']
                if abs(difference)>1e-10:raise ValueError('archived condition differs')
                archive_matches[name]=dict(archived_condition=old,nll_difference=difference)
        individual={name:source['metrics'][name] for name in source['head_ranking_most_helpful_to_mask']}
        sum_individual=math.fsum(individual[name]['improvement_vs_baseline16'] for name in harmful)
        joint_gain=metrics['joint_selected']['improvement_vs_baseline16']
        conditional={name:dict(individual_gain=individual[name]['improvement_vs_baseline16'],
            gain_given_other_selected=metrics[f'without_{name}']['mean_nll']-metrics['joint_selected']['mean_nll']) for name in harmful}
        cumulative=[];previous=metrics['baseline16']['mean_nll']
        for k,name in enumerate(harmful,1):
            current=metrics[f'prefix_{k}']['mean_nll']
            cumulative.append(dict(step=k,head_added=name,mean_nll=current,incremental_gain=previous-current))
            previous=current
        interaction=joint_gain-metrics['selected_block1']['improvement_vs_baseline16']-metrics['selected_block2']['improvement_vs_baseline16']
        if file_sha256(source['checkpoint'])!=source['checkpoint_sha256']:raise ValueError('checkpoint changed')
        row=dict(encoding=source['encoding'],seed=source['seed'],checkpoint=source['checkpoint'],
            checkpoint_sha256=source['checkpoint_sha256'],selected_head_names=harmful,
            selection_rule='positive individual NLL improvement in archived full-validation experiment; no re-selection',
            conditions_blocked_zero_based={k:[list(p) for p in v] for k,v in conditions.items()},
            computed_unique_conditions=len(unique),aliases=aliases,metrics=metrics,archived_individual_metrics=individual,
            archive_control_checks=archive_matches,conditional_gains=conditional,cumulative_ranked_path=cumulative,
            joint_gain=joint_gain,sum_individual_gains=sum_individual,
            joint_minus_sum_individual_gain=joint_gain-sum_individual,
            cross_block_joint_minus_sum_block_gains=interaction,
            joint_minus_all8_nll=metrics['joint_selected']['mean_nll']-metrics['all8']['mean_nll'])
        rows.append(row)
        print(source['encoding'],source['seed'],'joint',metrics['joint_selected'],
              'joint-minus-sum gain',row['joint_minus_sum_individual_gain'],'conditional',conditional,flush=True)
    sources=['scripts/analyze_joint_head_history.py','src/head_diagnostics.py','src/history_ablation.py','src/gpt.py',
             'src/transformer_block.py','src/multi_head_attention.py','src/feed_forward.py','src/sinusoidal_positions.py','src/generate.py']
    report=dict(protocol='All validation targets; outer suffix16 isolation; jointly restrict previously harmful heads to suffix8; same tokens, IDs and frozen weights',
        validation_sha256=file_sha256(validation),validation_characters=len(text),test_used=False,
        archive_sha256=file_sha256(archive_path),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={p:file_sha256(p) for p in sources},python_version=platform.python_version(),torch_version=str(torch.__version__),
        device='cpu',threads=1,deterministic_algorithms=True,batch_size=args.batch_size,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
