"""Diagnose 8-to-16 history reversals using full-validation layer interventions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import torch.nn.functional as F
from evaluate import file_sha256
from generate import load_checkpoint
from nonmonotonic_diagnostics import forward_layer_histories


def summarize(losses,correct):
    mean=math.fsum(losses)/len(losses)
    return dict(mean_nll=mean,perplexity=math.exp(mean),accuracy=correct/len(losses),scored_tokens=len(losses))


def paired_summary(a,b,targets,text):
    delta=torch.tensor(b,dtype=torch.float64)-torch.tensor(a,dtype=torch.float64)
    n=len(delta)
    worst=torch.argsort(delta,descending=True)
    count=max(1,math.ceil(n*.01))
    chars={}
    for ch in sorted(set(text[t] for t in targets)):
        indices=[i for i,t in enumerate(targets) if text[t]==ch]
        values=delta[indices]
        chars[ch]=dict(count=len(indices),mean_delta=values.mean().item(),contribution_to_overall_nll=values.sum().item()/n)
    def examples(indices):
        return [dict(target_index=targets[i],history=text[max(0,targets[i]-64):targets[i]],
                     target=text[targets[i]],nll8=a[i],nll16=b[i],delta=delta[i].item()) for i in indices]
    return dict(mean_delta=delta.mean().item(),fraction_harmed=(delta>0).double().mean().item(),
        quantiles={str(q):torch.quantile(delta,q).item() for q in (0.,.01,.1,.5,.9,.99,1.)},
        positive_contribution=delta.clamp_min(0).sum().item()/n,
        negative_contribution=delta.clamp_max(0).sum().item()/n,
        worst_one_percent_contribution=delta[worst[:count]].sum().item()/n,
        remaining_99_percent_mean=delta[worst[count:]].mean().item(),
        by_target_character=chars,worst_examples=examples(worst[:5].tolist()),
        best_examples=examples(worst[-5:].tolist()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/nonmonotonic-history/report.json'))
    parser.add_argument('--batch-size',type=int,default=128)
    args=parser.parse_args()
    if args.output.exists(): parser.error('output exists; choose a new path')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    validation=Path('data/prepared/gutenberg/validation.txt')
    text=validation.read_text()
    archive_path=Path('docs/results/history-length-curve.json')
    archive=json.loads(archive_path.read_text())
    if file_sha256(validation)!=archive['validation_sha256']: raise ValueError('validation changed')
    variants={'8_8':(8,8),'16_8':(16,8),'8_16':(8,16),'16_16':(16,16)}
    rows=[]
    for source in archive['rows']:
        if file_sha256(source['checkpoint'])!=source['checkpoint_sha256']: raise ValueError('checkpoint changed')
        model,tokenizer=load_checkpoint(source['checkpoint'])
        if len(model.blocks)!=2: raise ValueError('two blocks required')
        ids=torch.tensor(tokenizer.encode(text),dtype=torch.long)
        losses={k:[] for k in variants}; correct={k:0 for k in variants}
        attention={k:torch.zeros(2,3,2,dtype=torch.float64) for k in variants}
        # Fixed stride-64 anchors are exploratory controls, not the primary estimates.
        control_losses=[]; control_targets=[]; control_original=[]
        for start in range(1,len(ids),args.batch_size):
            groups={}
            for t in range(start,min(start+args.batch_size,len(ids))):
                groups.setdefault(min(t,64),[]).append(t)
            for length,targets in groups.items():
                inputs=torch.stack([ids[t-length:t] for t in targets])
                for name,keeps in variants.items():
                    logits,observed=forward_layer_histories(model,inputs,keeps)
                    nll=F.cross_entropy(logits.double(),ids[targets],reduction='none')
                    if not torch.isfinite(nll).all(): raise RuntimeError('nonfinite loss')
                    losses[name].extend(nll.tolist())
                    correct[name]+=(logits.argmax(-1)==ids[targets]).sum().item()
                    attention[name]+=observed.double().sum(0)
                    if name=='16_16':
                        selected=[i for i,t in enumerate(targets) if t>=64 and t%64==0]
                        if selected:
                            logits,_=forward_layer_histories(model,inputs[selected],keeps,reverse_added=True)
                            control_losses.extend(F.cross_entropy(logits.double(),ids[[targets[i] for i in selected]],reduction='none').tolist())
                            control_targets.extend(targets[i] for i in selected)
                            control_original.extend(nll[selected].tolist())
        metrics={k:summarize(losses[k],correct[k]) for k in variants}
        for name,condition in (('8_8','8'),('16_16','16')):
            if abs(metrics[name]['mean_nll']-source[condition]['mean_nll'])>1e-10: raise ValueError('archive mismatch')
        n={k:v['mean_nll'] for k,v in metrics.items()}
        interaction=n['16_16']-n['16_8']-n['8_16']+n['8_8']
        block1=.5*((n['16_8']-n['8_8'])+(n['16_16']-n['8_16']))
        block2=.5*((n['8_16']-n['8_8'])+(n['16_16']-n['16_8']))
        row=dict(encoding=source['encoding'],seed=source['seed'],checkpoint=source['checkpoint'],
            checkpoint_sha256=source['checkpoint_sha256'],metrics=metrics,
            layer_decomposition=dict(block1_symmetrized_delta=block1,block2_symmetrized_delta=block2,
                interaction=interaction,total_delta=n['16_16']-n['8_8']),
            paired=paired_summary(losses['8_8'],losses['16_16'],list(range(1,len(ids))),text),
            attention=dict(statistics=['mass_on_added_eight','entropy_nats'],
                protocol='final-query attention averaged over every validation target; arrays [block][head][statistic]',
                means={k:(v/(len(ids)-1)).tolist() for k,v in attention.items()}),
            reversal_control=dict(target_count=len(control_targets),target_indices_sha256=hashlib.sha256(json.dumps(control_targets).encode()).hexdigest(),
                original_mean_nll=statistics.mean(control_original),reversed_mean_nll=statistics.mean(control_losses),
                reversed_minus_original=statistics.mean(b-a for a,b in zip(control_original,control_losses))))
        if file_sha256(source['checkpoint'])!=source['checkpoint_sha256']: raise ValueError('checkpoint modified')
        rows.append(row)
        print(source['encoding'],source['seed'],row['layer_decomposition'],row['reversal_control'],flush=True)
    sources=['scripts/analyze_nonmonotonic_history.py','src/nonmonotonic_diagnostics.py','src/history_ablation.py',
             'src/gpt.py','src/transformer_block.py','src/multi_head_attention.py','src/feed_forward.py','src/sinusoidal_positions.py','src/generate.py']
    report=dict(protocol='All validation targets; fixed native IDs; factorial 8/16 suffix isolation per block; frozen checkpoints',
        validation_sha256=file_sha256(validation),validation_characters=len(text),test_used=False,
        archive_sha256=file_sha256(archive_path),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={p:file_sha256(p) for p in sources},python_version=platform.python_version(),
        torch_version=str(torch.__version__),device='cpu',threads=1,batch_size=args.batch_size,
        deterministic_algorithms=True,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
