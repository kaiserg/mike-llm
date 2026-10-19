"""Download only the three reviewed Gutenberg sources, retaining complete originals."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def verify_snapshot(output_dir, sources):
    records = json.loads((output_dir / 'manifest.json').read_text())['documents']
    if len(records) != len(sources) or len({r['id'] for r in records}) != len(records):
        raise ValueError('snapshot document list differs from sources')
    saved = {r['id']: r for r in records}
    for source in sources:
        record = saved[source['id']]
        for key, value in source.items():
            if key != 'expected_sha256' and record.get(key) != value:
                raise ValueError(f"snapshot metadata differs: {source['id']} / {key}")
        expected_path = source['id'] + '.txt'
        if record['path'] != expected_path:
            raise ValueError('snapshot file path differs')
        raw = (output_dir / expected_path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record['sha256'] or (source.get('expected_sha256') and digest != source['expected_sha256']):
            raise ValueError(f"snapshot checksum mismatch: {source['id']}")
        raw.decode('utf-8-sig')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, default=Path('configs/corpus_sources.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/raw/gutenberg'))
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())['documents']
    for row in sources:
        if row['id'] not in ('pg11', 'pg84', 'pg1342') or row['download_url'] != f"https://www.gutenberg.org/cache/epub/{row['id'][2:]}/pg{row['id'][2:]}.txt":
            parser.error('this downloader accepts only the three documented Gutenberg URLs')
    if args.output_dir.exists():
        try:
            verify_snapshot(args.output_dir, sources)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f'existing snapshot could not be verified: {error}; use a new --output-dir')
        print(f'Reusing verified snapshot: {args.output_dir}')
        return
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as temporary:
        staging = Path(temporary) / 'snapshot'
        staging.mkdir()
        records = []
        for row in sources:
            raw = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                                  '--proto', '=https', '--proto-redir', '=https', '--max-time', '60',
                                  row['download_url']], check=True, capture_output=True).stdout
            if row.get('expected_sha256') and hashlib.sha256(raw).hexdigest() != row['expected_sha256']:
                raise ValueError(f"source changed since reviewed snapshot: {row['id']}")
            raw.decode('utf-8-sig')  # Fail rather than silently repairing encoding.
            filename = row['id'] + '.txt'
            (staging / filename).write_bytes(raw)
            records.append({**row, 'path': filename, 'sha256': hashlib.sha256(raw).hexdigest(),
                            'retrieved_at': datetime.now(timezone.utc).isoformat()})
            print(f"Downloaded {row['id']}: {len(raw)} bytes")
        (staging / 'manifest.json').write_text(json.dumps({'documents': records}, indent=2) + '\n')
        staging.rename(args.output_dir)



if __name__ == '__main__':
    main()
