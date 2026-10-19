"""Normalize, deduplicate and split local documents by group before tokenization."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize(text, profile='unicode'):
    text = unicodedata.normalize('NFC', text.replace('\r\n', '\n').replace('\r', '\n'))
    if profile == 'ascii-letters':
        # Explicitly lossy educational profile: accents, case and punctuation are discarded.
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().lower()
        text = re.sub('[^a-z\\s]', ' ', text)
    elif profile != 'unicode':
        raise ValueError('unknown normalization profile')
    text = '\n'.join(re.sub(r'[^\S\n]+', ' ', line).strip() for line in text.split('\n'))
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def strip_gutenberg(text):
    start = re.search(r'^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*[^\n]*\n', text, re.M)
    end = re.search(r'^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*', text, re.M)
    if not start or not end or start.end() >= end.start():
        raise ValueError('missing or invalid Gutenberg boundary markers')
    return text[start.end():end.start()]


def prepare_corpus(manifest_path, output_dir, *, seed=42, validation_ratio=0.1,
                   test_ratio=0.1, profile='unicode', min_characters=100):
    if not (0 < validation_ratio < 1 and 0 < test_ratio < 1 and validation_ratio + test_ratio < 1):
        raise ValueError('ratios must be positive and sum to less than one')
    if type(min_characters) is not int or min_characters < 1:
        raise ValueError('min_characters must be positive')
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    raw_manifest = manifest_path.read_bytes()
    records = json.loads(raw_manifest)['documents']
    documents, seen_ids, hashes, duplicates = [], set(), {}, []
    for record in sorted(records, key=lambda r: r['id']):
        for key in ('id', 'group', 'path', 'sha256', 'title', 'author', 'url', 'license', 'license_url'):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f'missing provenance field: {key}')
        if record['id'] in seen_ids:
            raise ValueError('duplicate document ID')
        seen_ids.add(record['id'])
        raw = (manifest_path.parent / record['path']).read_bytes()
        if sha256(raw) != record['sha256']:
            raise ValueError(f"source checksum mismatch: {record['id']}")
        text = raw.decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
        if record.get('format', 'text') == 'gutenberg':
            text = strip_gutenberg(text)
        elif record.get('format', 'text') != 'text':
            raise ValueError('unsupported document format')
        text = normalize(text, profile)
        if len(text) < min_characters:
            raise ValueError(f"document too short after normalization: {record['id']}")
        digest = sha256(text.encode())
        if digest in hashes:
            previous = hashes[digest]
            # Conflicting groups could conceal related-document leakage; require correction.
            if previous['group'] != record['group']:
                raise ValueError('duplicate content has different groups; assign related documents to one group')
            duplicates.append({'id': record['id'], 'duplicate_of': previous['id']})
            continue
        doc = {**record, 'normalized_sha256': digest, 'text': text}
        hashes[digest] = doc
        documents.append(doc)
    groups = sorted({d['group'] for d in documents}, key=lambda g: (sha256(f'{seed}:{g}'.encode()), g))
    if len(groups) < 3:
        raise ValueError('at least three distinct groups are required')
    n_val, n_test = max(1, int(len(groups) * validation_ratio)), max(1, int(len(groups) * test_ratio))
    if n_val + n_test >= len(groups):
        raise ValueError('not enough groups for requested ratios')
    assignment = {g: 'validation' if i < n_val else 'test' if i < n_val + n_test else 'train'
                  for i, g in enumerate(groups)}
    splits = {name: [d for d in documents if assignment[d['group']] == name]
              for name in ('train', 'validation', 'test')}
    texts = {name: '\n\n'.join(d['text'] for d in docs) + '\n' for name, docs in splits.items()}
    train_vocab = set(texts['train'])
    report = dict(seed=seed, normalization=profile, min_characters=min_characters,
                  requested_ratios={'validation': validation_ratio, 'test': test_ratio},
                  manifest_sha256=sha256(raw_manifest), duplicates_removed=duplicates,
                  group_assignment=assignment, splits={})
    for name, docs in splits.items():
        report['splits'][name] = dict(documents=len(docs), groups=len({d['group'] for d in docs}),
            characters=len(texts[name]), sha256=sha256(texts[name].encode()),
            characters_absent_from_train=sorted(set(texts[name]) - train_vocab))
    # Compare the full expected artifact set before reusing an existing preparation.
    outputs = {'report.json': json.dumps(report, indent=2, ensure_ascii=False) + '\n'}
    for name, docs in splits.items():
        outputs[f'{name}.txt'] = texts[name]
        outputs[f'{name}.jsonl'] = ''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in docs)
    if output_dir.exists():
        for filename, content in outputs.items():
            path = output_dir / filename
            if not path.is_file() or path.read_bytes() != content.encode('utf-8'):
                raise ValueError(f'existing preparation differs or is incomplete ({filename}); use a new --output-dir')
        return report
    output_dir.mkdir(parents=True)
    for filename, content in outputs.items():
        (output_dir / filename).write_text(content, encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/raw/gutenberg/manifest.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/prepared/gutenberg'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--profile', choices=['unicode', 'ascii-letters'], default='unicode')
    parser.add_argument('--min-characters', type=int, default=100)
    args = parser.parse_args()
    try:
        report = prepare_corpus(args.manifest, args.output_dir, seed=args.seed,
            validation_ratio=args.validation_ratio, test_ratio=args.test_ratio,
            profile=args.profile, min_characters=args.min_characters)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
