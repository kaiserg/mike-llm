"""Freeze a validation-only choice, then evaluate only that choice on test."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from evaluate import evaluate_text,file_sha256
from generate import load_checkpoint

ROOT=Path('docs/results')
CORPUS=Path('data/prepared/gutenberg')


def now():return datetime.now(timezone.utc).isoformat()


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def provenance():
    sources=['scripts/finalize_stage.py','src/evaluate.py','src/generate.py','src/gpt.py',
             'src/transformer_block.py','src/multi_head_attention.py','src/feed_forward.py','src/sinusoidal_positions.py']
    return dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={p:file_sha256(p) for p in sources},python_version=platform.python_version(),
        torch_version=str(torch.__version__),device='cpu',threads=1,deterministic_algorithms=True)


def choose(candidates):
    if not candidates:raise ValueError('no validation candidates')
    return min(candidates,key=lambda x:(x['validation']['mean_nll'],x['id']))


def select(args):
    if args.selection.exists():raise ValueError('selection exists; choose a new path')
    preparation=json.loads((CORPUS/'report.json').read_text())
    candidates=[]
    def add(source,seed,checkpoint,digest,encoding,steps,capacity,cap,metric,label):
        if metric['scored_tokens']!=135522:raise ValueError('incomparable target count')
        candidates.append(dict(id=label,source=source,seed=seed,checkpoint=checkpoint,checkpoint_sha256=digest,
            positional_encoding=encoding,training_steps=steps,training_targets=steps*128,
            model_context_length=capacity,evaluation_context_length=cap,validation=metric,
            inference='crop last evaluation_context_length characters; reset IDs to 0..L-1; native causal attention'))
    for filename,encoding in (('multiseed-context-experiment.json','learned'),('sinusoidal-context-experiment.json','sinusoidal')):
        study=json.loads((ROOT/filename).read_text())
        if study['validation_sha256']!=preparation['splits']['validation']['sha256']:raise ValueError('validation differs')
        for row in study['rows']:
            for capacity,cap,key in ((16,16,'context16'),(64,64,'context64'),(64,16,'context64_capped16')):
                model=row['models'][str(capacity)]
                add(filename,row['seed'],model['checkpoint'],model['checkpoint_sha256'],encoding,10000,capacity,cap,
                    row['evaluations'][key],f'{encoding}-10000-seed{row["seed"]}-capacity{capacity}-eval{cap}')
    filename='position-exposure-training.json';study=json.loads((ROOT/filename).read_text())
    for row in study['rows']:
        for cap,key in ((64,'context64_40000'),(16,'context64_40000_capped16')):
            add(filename,row['seed'],row['checkpoint'],row['checkpoint_sha256'],'learned',40000,64,cap,
                row['evaluations'][key],f'learned-40000-seed{row["seed"]}-capacity64-eval{cap}')
    filename='real-books-2000.json';study=json.loads((ROOT/filename).read_text())
    add(filename,42,study['checkpoint'],study['checkpoint_sha256'],'learned',2000,16,16,study['final'],'learned-2000-seed42-capacity16-eval16')
    filename='context-64-experiment.json';study=json.loads((ROOT/filename).read_text())
    for cap,key in ((64,'new_context64'),(16,'new_capped16')):
        add(filename,42,study['checkpoint'],study['checkpoint_sha256'],'learned',2000,64,cap,study['evaluations'][key],f'learned-2000-seed42-capacity64-eval{cap}')
    for candidate in candidates:
        if file_sha256(candidate['checkpoint'])!=candidate['checkpoint_sha256']:raise ValueError('candidate checkpoint changed')
    # Audit all archived full-validation diagnostic scores, including every mask.
    audit=[];archives={}
    def visit(value,path,source):
        if isinstance(value,dict):
            count=value.get('scored_tokens',value.get('scored_targets'))
            if count==135522 and isinstance(value.get('mean_nll'),(int,float)):
                audit.append(dict(source=source,json_path=path,mean_nll=value['mean_nll']))
            for key,child in value.items():visit(child,path+'/'+str(key),source)
        elif isinstance(value,list):
            for i,child in enumerate(value):visit(child,path+'/'+str(i),source)
    for path in sorted(ROOT.glob('*.json')):
        if path.name.startswith('final-'):continue
        archives[str(path)]=file_sha256(path)
        visit(json.loads(path.read_text()),'',str(path))
    selected=choose(candidates)
    if min(x['mean_nll'] for x in audit)<selected['validation']['mean_nll']-1e-6:
        raise ValueError('an archived full-validation diagnostic beats the candidate selection; review before test')
    model,tokenizer=load_checkpoint(selected['checkpoint'])
    checkpoint=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True)
    validation=(CORPUS/'validation.txt').read_text()
    if file_sha256(CORPUS/'validation.txt')!=preparation['splits']['validation']['sha256']:raise ValueError('validation changed')
    for split in ('train','validation'):
        if checkpoint[split+'_sha256']!=preparation['splits'][split]['sha256']:raise ValueError('checkpoint data differs')
    measured=evaluate_text(model,tokenizer,validation,batch_size=128,context_length=selected['evaluation_context_length'])
    if abs(measured['mean_nll']-selected['validation']['mean_nll'])>1e-10:raise ValueError('selected validation does not reproduce')
    report=dict(stage='validation-only frozen selection',selection_frozen_at_utc=now(),test_content_read=False,
        selection_rule='minimum archived full-validation mean NLL among existing operating configurations; tie by candidate ID; no retraining; no budget constraint',
        candidate_count=len(candidates),candidates=sorted(candidates,key=lambda x:(x['validation']['mean_nll'],x['id'])),
        selected=selected,selected_validation_recomputed=measured,
        selected_model_config=checkpoint['model_config'],selected_training_config=checkpoint['config'],
        selected_parameters=sum(p.numel() for p in model.parameters()),
        selection_bias='Repeated validation exploration and seed selection can overfit this book; 40000-step winner uses four times the targets of 10000-step comparisons.',
        diagnostic_audit=dict(full_validation_metric_records=len(audit),minimum=min(audit,key=lambda x:x['mean_nll']),
            scope='all archived full-validation scalar NLL records; subset-target profiles excluded; diagnostic masks cannot beat winner'),
        preparation_report_sha256=file_sha256(CORPUS/'report.json'),corpus_splits=preparation['splits'],
        archive_sha256=archives,**provenance())
    write_new(args.selection,report)
    print('Frozen selection:',selected['id'],'validation PPL',measured['perplexity'],'file',args.selection,flush=True)


def evaluate(args):
    if args.output.exists():raise ValueError('test report exists; refusing another test evaluation at this path')
    digest=file_sha256(args.selection);selection=json.loads(args.selection.read_text())
    if selection['stage']!='validation-only frozen selection' or selection['test_content_read']:
        raise ValueError('not a validation-only frozen selection')
    selected=selection['selected']
    if choose(selection['candidates'])['id']!=selected['id']:raise ValueError('frozen selection inconsistent')
    if file_sha256(selected['checkpoint'])!=selected['checkpoint_sha256']:raise ValueError('selected checkpoint changed')
    if file_sha256(CORPUS/'report.json')!=selection['preparation_report_sha256']:raise ValueError('preparation changed')
    for path,expected in selection['source_sha256'].items():
        if file_sha256(path)!=expected:raise ValueError('source changed since selection')
    started=now()
    # First test-content read is strictly after loading and validating frozen selection.
    test=(CORPUS/'test.txt').read_text()
    if file_sha256(CORPUS/'test.txt')!=selection['corpus_splits']['test']['sha256']:raise ValueError('test hash differs')
    model,tokenizer=load_checkpoint(selected['checkpoint'])
    metrics=evaluate_text(model,tokenizer,test,batch_size=128,context_length=selected['evaluation_context_length'])
    if len(test)!=selection['corpus_splits']['test']['characters']:raise ValueError('test size differs')
    if metrics['scored_tokens']!=len(test)-1:raise ValueError('test target count differs')
    if file_sha256(selected['checkpoint'])!=selected['checkpoint_sha256'] or file_sha256(args.selection)!=digest:
        raise ValueError('checkpoint or selection changed during test evaluation')
    report=dict(stage='final held-out test evaluation; selection unchanged',selection_file=str(args.selection),selection_sha256=digest,
        selection_frozen_at_utc=selection['selection_frozen_at_utc'],test_evaluation_started_at_utc=started,
        test_evaluation_completed_at_utc=now(),selected=selected,validation=selection['selected_validation_recomputed'],test=metrics,
        test_source=str(CORPUS/'test.txt'),test_sha256=selection['corpus_splits']['test']['sha256'],test_characters=len(test),
        protocol='each test next-character target once; preceding at most 16 characters; IDs reset 0..L-1; first character not scored; no cross-split context',
        test_checkpoints_evaluated=1,test_used_for_selection=False,selection_changed_after_test=False,
        test_status='held-out test has now been consumed for this finalized selection; no longer untouched for future iterations',
        nll_test_minus_validation=metrics['mean_nll']-selection['selected_validation_recomputed']['mean_nll'],
        batch_size=128,**provenance())
    write_new(args.output,report)
    print(json.dumps(dict(test=metrics,nll_test_minus_validation=report['nll_test_minus_validation']),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('select','evaluate'))
    parser.add_argument('--selection',type=Path,default=ROOT/'final-model-selection.json')
    parser.add_argument('--output',type=Path,default=ROOT/'final-test-evaluation.json')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    (select if args.phase=='select' else evaluate)(args)

if __name__=='__main__':main()
