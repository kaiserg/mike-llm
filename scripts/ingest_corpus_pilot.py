"""Download and verify the explicit development-only v2 pilot registry."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from corpus_pilot import ingest_pilot,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,default=Path('configs/corpus_v2_sources.json'))
    parser.add_argument('--output-dir',type=Path,default=Path('data/raw/corpus-v2-pilot'))
    parser.add_argument('--report',type=Path,default=Path('docs/results/corpus-v2-ingestion-pilot.json'))
    parser.add_argument('--cache-dir',type=Path,help='optional local download cache; reviewed raw hashes are still required')
    args=parser.parse_args()
    def fetch(url,name):
        if args.cache_dir and (args.cache_dir/name).is_file():return (args.cache_dir/name).read_bytes()
        print('Retrieving:',url,flush=True)
        response=subprocess.run(['curl','--silent','--show-error','--fail','--location','--proto','=https',
            '--proto-redir','=https','--max-time','60','--max-filesize','12000000',url],check=True,capture_output=True)
        time.sleep(.5)
        return response.stdout
    report=ingest_pilot(args.sources,args.output_dir,fetch)
    # Source provenance lives outside the snapshot and does not alter its text bytes.
    report=dict(report,implementation_sha256={p:digest(Path(p).read_bytes()) for p in
        ('scripts/ingest_corpus_pilot.py','src/corpus_pilot.py','src/prepare_corpus.py')})
    serialized=json.dumps(report,indent=2)+'\n'
    if args.report.exists():
        if args.report.read_text()!=serialized:raise ValueError('report differs; choose a new --report path')
    else:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(serialized)
    print(json.dumps({k:report[k] for k in ('document_count','extracted_documents','quarantined_documents',
        'extracted_characters','source_families','domains','new_test_created','model_loaded')},indent=2))

if __name__=='__main__':main()
