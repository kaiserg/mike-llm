"""Compare learned and sinusoidal position/history diagnostics on identical targets."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluate import file_sha256


def describe(values):
    return dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values),
                positive=sum(v>0 for v in values),negative=sum(v<0 for v in values))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('runs/sinusoidal-position-diagnostics'))
    args=p.parse_args()
    output=args.output_dir/'comparison.json'
    if output.exists():p.error('comparison exists; choose a new --output-dir')
    inputs={}
    for encoding,study,prefix in [('learned','docs/results/multiseed-context-experiment.json','learned-'),
                                   ('sinusoidal','docs/results/sinusoidal-context-experiment.json','')]:
        inputs[encoding]={}
        for kind,script,suffix in [('prefix','scripts/analyze_positions.py','report.json'),
                                   ('windows','scripts/analyze_window_positions.py','windows.json')]:
            path=args.output_dir/(prefix+suffix)
            if not path.exists():
                subprocess.run([sys.executable,script,'--study',study,'--output',str(path)],check=True)
            report=json.loads(path.read_text())
            if report['study_sha256']!=file_sha256(study) or report['test_used']:
                raise ValueError('cached report does not match study')
            for name,digest in report['source_sha256'].items():
                if file_sha256(name)!=digest:raise ValueError('cached report source changed; use a new output directory')
            inputs[encoding][kind]=dict(path=str(path),sha256=file_sha256(path),data=report)
    lp=inputs['learned']['prefix']['data'];sp=inputs['sinusoidal']['prefix']['data']
    lw=inputs['learned']['windows']['data'];sw=inputs['sinusoidal']['windows']['data']
    if not (lp['target_indices']==sp['target_indices'] and lw['block_starts']==sw['block_starts']
            and [s+64 for s in sw['block_starts']]==sp['target_indices']):
        raise ValueError('diagnostics use different targets')
    if len({r['validation_sha256'] for encoding in inputs.values() for item in encoding.values() for r in [item['data']]})!=1:
        raise ValueError('validation data differs')
    rows=[]
    for l,s in zip(lp['rows'],sp['rows']):
        if l['seed']!=s['seed']:raise ValueError('seed ordering differs')
        row=dict(seed=l['seed'],encodings={})
        for label,r in [('learned',l),('sinusoidal',s)]:
            short=r['profiles']['16'];long=r['profiles']['64']
            row['encodings'][label]=dict(short16=short[15]['mean_nll'],long16=long[15]['mean_nll'],long64=long[63]['mean_nll'],
                extra_history_delta=long[63]['mean_nll']-long[15]['mean_nll'],
                common16_model_gap=long[15]['mean_nll']-short[15]['mean_nll'],
                native_model_gap=long[63]['mean_nll']-short[15]['mean_nll'],
                shift48_delta=r['shifted16']['48']['mean_nll']-r['shifted16']['0']['mean_nll'])
        rows.append(row)
    summary={encoding:{metric:describe([row['encodings'][encoding][metric] for row in rows])
                        for metric in ('extra_history_delta','common16_model_gap','native_model_gap','shift48_delta')}
             for encoding in ('learned','sinusoidal')}
    archived_prefix=json.loads(Path('docs/results/position-diagnostics.json').read_text())
    archived_windows=json.loads(Path('docs/results/window-position-diagnostics.json').read_text())
    same_prefix=all(a['profiles']==b['profiles'] and a['shifted16']==b['shifted16'] for a,b in zip(lp['rows'],archived_prefix['rows']))
    same_windows=all(a['contexts']==b['contexts'] for a,b in zip(lw['rows'],archived_windows['rows']))
    if not (same_prefix and same_windows):raise RuntimeError('learned diagnostics no longer reproduce archived values')
    report=dict(inputs={encoding:{kind:{k:v for k,v in item.items() if k!='data'} for kind,item in reports.items()} for encoding,reports in inputs.items()},
        source_sha256=file_sha256(__file__),validation_sha256=sp['validation_sha256'],target_count=sp['target_count'],
        targets_sha256=sp['target_indices_sha256'],rows=rows,summary=summary,
        learned_archived_prefix_identical=same_prefix,learned_archived_windows_identical=same_windows,
        learned_archives={name:file_sha256(name) for name in ('docs/results/position-diagnostics.json','docs/results/window-position-diagnostics.json')},
        test_used=False)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
