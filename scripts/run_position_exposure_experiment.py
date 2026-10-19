"""Match per-position examples: context64/40000 updates versus context16/10000."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import torch

from evaluate import evaluate_text, file_sha256
from generate import load_checkpoint
from train import run_training


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('docs/results/multiseed-context-experiment.json'))
    parser.add_argument('--corpus-dir',type=Path,default=Path('data/prepared/gutenberg'))
    parser.add_argument('--output-dir',type=Path,default=Path('runs/position-exposure-40000'))
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('output exists; choose a new --output-dir')
    study=json.loads(args.study.read_text())
    texts={name:(args.corpus_dir/f'{name}.txt').read_text(encoding='utf-8') for name in ('train','validation')}
    for name,text in texts.items():
        if hashlib.sha256(text.encode()).hexdigest()!=study[name+'_sha256']:
            raise ValueError('corpus differs from reference study')
    args.output_dir.mkdir(parents=True)
    rows=[]
    terminal=sys.stdout
    for reference in study['rows']:
        seed=reference['seed']
        directory=args.output_dir/f'seed-{seed}'
        directory.mkdir()
        source64=reference['models']['64']
        if file_sha256(source64['checkpoint'])!=source64['checkpoint_sha256']:
            raise ValueError('context64 reference checkpoint differs')
        previous=torch.load(source64['checkpoint'],map_location='cpu',weights_only=True)
        if previous['model_config']!=dict(vocab_size=28,context_length=64,embedding_dim=24,num_heads=3,num_layers=2,expansion_factor=4):
            raise ValueError('reference model configuration differs')
        expected=dict(steps=10000,seed=seed,batch_size=2,learning_rate=0.003,weight_decay=0.01,
                      max_grad_norm=1.0,eval_interval=200,eval_batches=8,device='cpu',threads=1)
        if any(previous['config'][k]!=v for k,v in expected.items()):raise ValueError('reference training settings differ')
        for name in ('train','validation'):
            if previous[name+'_sha256']!=study[name+'_sha256']:raise ValueError('reference corpus differs')
        prefix_checks=[]
        trained=[]
        for label in ('primary','repeat'):
            def observe(step,model):
                if step==10000:
                    equal=all(torch.equal(value,previous['model_state_dict'][k]) for k,value in model.state_dict().items())
                    if not equal:raise RuntimeError('10000-step prefix does not match earlier checkpoint')
                    prefix_checks.append(equal)
                if step%10000==0:
                    print(f'Seed {seed} {label}: completed {step}/40000 updates.',file=terminal,flush=True)
            print(f'Training seed {seed} {label}, context64, 40000 updates...',flush=True)
            start=time.perf_counter()
            with (directory/f'{label}.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log):
                model,history=run_training(texts['train'],texts['validation'],steps=40000,seed=seed,
                    batch_size=2,context_length=64,eval_interval=200,output_dir=directory/label,on_step=observe)
            trained.append((model,history,time.perf_counter()-start))
        identical_weights=all(torch.equal(v,trained[1][0].state_dict()[k]) for k,v in trained[0][0].state_dict().items())
        identical_histories=trained[0][1]==trained[1][1]
        identical_prefix=all([p for p in run[1] if p['step']<=10000]==previous['history'] for run in trained)
        if not (identical_weights and identical_histories and identical_prefix):raise RuntimeError('reproducibility check failed')
        checkpoint=directory/'primary/checkpoint.pt'
        model,tokenizer=load_checkpoint(checkpoint)
        print(f'Evaluating seed {seed} final checkpoint at caps 64 and 16...',flush=True)
        native=evaluate_text(model,tokenizer,texts['validation'],128)
        capped=evaluate_text(model,tokenizer,texts['validation'],128,16)
        source16=reference['models']['16']
        if file_sha256(source16['checkpoint'])!=source16['checkpoint_sha256']:raise ValueError('context16 reference differs')
        baseline,baseline_tokenizer=load_checkpoint(source16['checkpoint'])
        if baseline_tokenizer.stoi!=tokenizer.stoi:raise ValueError('vocabulary differs')
        baseline_result=evaluate_text(baseline,tokenizer,texts['validation'],128)
        if baseline_result!=reference['evaluations']['context16']:raise RuntimeError('reference metrics did not reproduce')
        ids=torch.tensor([tokenizer.encode('the creature ')],dtype=torch.long)
        with torch.no_grad():reload_equal=torch.equal(model(ids),trained[0][0](ids))
        print(f'Generating seed {seed} temperature grid...',flush=True)
        generation_path=directory/'generation.json'
        subprocess.run([sys.executable,'scripts/compare_temperatures.py','--checkpoint',str(checkpoint),
                        '--output',str(generation_path)],check=True,capture_output=True,text=True)
        row=dict(seed=seed,checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
            repeat_checkpoint=str(directory/'repeat/checkpoint.pt'),repeat_checkpoint_sha256=file_sha256(directory/'repeat/checkpoint.pt'),
            reference64_checkpoint=source64['checkpoint'],reference64_checkpoint_sha256=source64['checkpoint_sha256'],
            reference16_checkpoint=source16['checkpoint'],reference16_checkpoint_sha256=source16['checkpoint_sha256'],
            evaluations=dict(context16_10000=baseline_result,context64_10000=reference['evaluations']['context64'],
                             context64_40000=native,context64_40000_capped16=capped),
            history=trained[0][1],weights_identical=identical_weights,histories_identical=identical_histories,
            step10000_weights_match=prefix_checks,prefix_histories_match=identical_prefix,reload_logits_identical=reload_equal,
            training_seconds=[run[2] for run in trained],generation=json.loads(generation_path.read_text()),
            baseline_generation=reference['generation']['16'],previous64_generation=reference['generation']['64'])
        (directory/'result.json').write_text(json.dumps(row,indent=2)+'\n')
        rows.append(row)
        print(f"Completed seed {seed}: context64 PPL {native['perplexity']:.6f}; context16 reference {baseline_result['perplexity']:.6f}",flush=True)
    summary={}
    for condition in ('context16_10000','context64_10000','context64_40000','context64_40000_capped16'):
        summary[condition]={}
        for metric in ('mean_nll','perplexity','bits_per_character','next_token_accuracy'):
            values=[row['evaluations'][condition][metric] for row in rows]
            summary[condition][metric]=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values))
    for condition in ('context64_40000','context64_40000_capped16'):
        differences=[row['evaluations'][condition]['mean_nll']-row['evaluations']['context16_10000']['mean_nll'] for row in rows]
        summary[condition+'_minus_context16_nll']=dict(by_seed={str(row['seed']):d for row,d in zip(rows,differences)},
            mean=statistics.mean(differences),sample_sd=statistics.stdev(differences),context64_wins=sum(d<0 for d in differences),
            context16_wins=sum(d>0 for d in differences),ties=sum(d==0 for d in differences))
    report=dict(study=str(args.study),study_sha256=file_sha256(args.study),training_seeds=[row['seed'] for row in rows],
        context64_steps=40000,context16_steps=10000,context64_batch_size=2,context16_batch_size=8,
        context64_targets=5120000,context16_targets=1280000,examples_per_absolute_position=80000,
        context64_parameters=17232,context16_parameters=16080,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={name:file_sha256(name) for name in ('scripts/run_position_exposure_experiment.py','scripts/compare_temperatures.py',
            'src/train.py','src/evaluate.py','src/generate.py','src/gpt.py','src/transformer_block.py','src/multi_head_attention.py','src/feed_forward.py')},
        python_version=platform.python_version(),torch_version=str(torch.__version__),device='cpu',threads=1,
        train_sha256=study['train_sha256'],validation_sha256=study['validation_sha256'],rows=rows,summary=summary,test_used=False)
    (args.output_dir/'experiment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
