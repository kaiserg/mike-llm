"""Evaluate all 64 six-head 8/16 access masks on the full validation book."""
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
from exhaustive_head_diagnostics import forward_all_head_masks,blocked_pairs,analyze_mask_cube


def mask_id(pairs):return sum(1<<(3*b+h) for b,h in pairs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/all-head-masks/report.json'))
    parser.add_argument('--batch-size',type=int,default=128)
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists; choose a fresh path')
    if args.batch_size<1:parser.error('batch size must be positive')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    paths=[Path('docs/results/head-history-diagnostics.json'),Path('docs/results/joint-head-history-diagnostics.json')]
    archives=[json.loads(p.read_text()) for p in paths]
    validation=Path('data/prepared/gutenberg/validation.txt')
    if any(file_sha256(validation)!=a['validation_sha256'] for a in archives):raise ValueError('validation changed')
    text=validation.read_text();rows=[]
    for source in archives[0]['rows']:
        if file_sha256(source['checkpoint'])!=source['checkpoint_sha256']:raise ValueError('checkpoint changed')
        model,tokenizer=load_checkpoint(source['checkpoint'])
        ids=torch.tensor(tokenizer.encode(text),dtype=torch.long)
        totals=torch.zeros(64,dtype=torch.float64);correct=torch.zeros(64,dtype=torch.long)
        better0=torch.zeros(64,dtype=torch.long);better63=torch.zeros(64,dtype=torch.long)
        count=0
        for start in range(1,len(ids),args.batch_size):
            groups={}
            for t in range(start,min(start+args.batch_size,len(ids))):groups.setdefault(min(t,64),[]).append(t)
            for length,targets in groups.items():
                inputs=torch.stack([ids[t-length:t] for t in targets])
                logits=forward_all_head_masks(model,inputs)
                labels=ids[targets,None].expand(-1,64)
                losses=F.cross_entropy(logits.double().reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none').reshape(-1,64)
                if not torch.isfinite(losses).all():raise RuntimeError('nonfinite loss')
                totals+=losses.sum(0)
                correct+=(logits.argmax(-1)==labels).sum(0)
                better0+=(losses<losses[:,0:1]).sum(0)
                better63+=(losses<losses[:,63:64]).sum(0)
                count+=len(targets)
            if (start-1)%(args.batch_size*128)==0:
                print(source['encoding'],source['seed'],'targets',count,'/',len(ids)-1,flush=True)
        metrics=[]
        for mask in range(64):
            mean=(totals[mask]/count).item()
            metrics.append(dict(mask=mask,blocked_heads=[f'B{b+1}H{h+1}' for b,h in blocked_pairs(mask)],
                blocked_zero_based=blocked_pairs(mask),masked_head_count=mask.bit_count(),
                scored_tokens=count,mean_nll=mean,perplexity=math.exp(mean),accuracy=correct[mask].item()/count,
                fraction_targets_improved_vs_baseline16=better0[mask].item()/count,
                fraction_targets_improved_vs_all8=better63[mask].item()/count))
        controls=[]
        for archive_index,archive in enumerate(archives):
            prior=next(x for x in archive['rows'] if x['encoding']==source['encoding'] and x['seed']==source['seed'])
            if prior['checkpoint_sha256']!=source['checkpoint_sha256']:raise ValueError('archive checkpoint differs')
            for name,pairs in prior['conditions_blocked_zero_based'].items():
                mask=mask_id(pairs);delta=metrics[mask]['mean_nll']-prior['metrics'][name]['mean_nll']
                if abs(delta)>1e-6:raise ValueError(f'archive mismatch {name}: {delta}')
                controls.append(dict(archive=str(paths[archive_index]),condition=name,mask=mask,nll_difference=delta))
        ranking=sorted(range(64),key=lambda m:metrics[m]['mean_nll'])
        nll=[m['mean_nll'] for m in metrics]
        best=ranking[0]
        best_by_count={str(k):min((m for m in range(64) if m.bit_count()==k),key=lambda m:nll[m]) for k in range(7)}
        local_minima=[m for m in range(64) if all(nll[m]<=nll[m^(1<<b)] for b in range(6))]
        joint=next(x for x in archives[1]['rows'] if x['encoding']==source['encoding'] and x['seed']==source['seed'])
        selected_mask=mask_id(joint['conditions_blocked_zero_based']['joint_selected'])
        row=dict(encoding=source['encoding'],seed=source['seed'],checkpoint=source['checkpoint'],checkpoint_sha256=source['checkpoint_sha256'],
            metrics=metrics,ranking=ranking,best_mask=best,best_by_masked_head_count=best_by_count,
            single_flip_local_minima=local_minima,previous_selected_mask=selected_mask,
            best_improvement_vs_baseline16=nll[0]-nll[best],best_improvement_vs_all8=nll[63]-nll[best],
            best_improvement_vs_previous_selected=nll[selected_mask]-nll[best],
            masks_within_1e4_nll_of_best=[m for m in range(64) if nll[m]-nll[best]<=1e-4],
            archive_control_checks=controls,max_absolute_archive_nll_difference=max(abs(c['nll_difference']) for c in controls),
            interactions=analyze_mask_cube(nll))
        if file_sha256(source['checkpoint'])!=source['checkpoint_sha256']:raise ValueError('checkpoint modified')
        rows.append(row)
        print('Completed',source['encoding'],source['seed'],'best',metrics[best],flush=True)
    sources=['scripts/analyze_all_head_masks.py','src/exhaustive_head_diagnostics.py','src/head_diagnostics.py',
             'src/history_ablation.py','src/gpt.py','src/transformer_block.py','src/multi_head_attention.py','src/feed_forward.py',
             'src/sinusoidal_positions.py','src/generate.py']
    report=dict(protocol='All validation targets; all 64 subsets restricting selected heads to suffix8 within common suffix16 isolation; original IDs; frozen weights',
        mask_bit_order=['B1H1','B1H2','B1H3','B2H1','B2H2','B2H3'],one_bit_means='restrict added-eight access in that head',
        validation_sha256=file_sha256(validation),validation_characters=len(text),test_used=False,
        archive_sha256={str(p):file_sha256(p) for p in paths},code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={p:file_sha256(p) for p in sources},python_version=platform.python_version(),torch_version=str(torch.__version__),
        device='cpu',threads=1,deterministic_algorithms=True,batch_size=args.batch_size,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
