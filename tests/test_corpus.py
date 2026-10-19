import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from prepare_corpus import normalize, prepare_corpus, strip_gutenberg


class CorpusTests(unittest.TestCase):
    def make_manifest(self, root):
        records = []
        for i in range(10):
            text = f'document {i} has its own contents\n' * 5
            raw = text.encode()
            (root / f'{i}.txt').write_bytes(raw)
            records.append(dict(id=str(i), group=str(i), path=f'{i}.txt', sha256=hashlib.sha256(raw).hexdigest(),
                                title=str(i), author='test', url='https://example.org',
                                license='fixture only', license_url='https://example.org/license'))
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps({'documents': records}))
        return manifest, records

    def test_reproducibility_disjoint_groups_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, records = self.make_manifest(root)
            a = prepare_corpus(manifest, root / 'a')
            b = prepare_corpus(manifest, root / 'b')
            self.assertEqual(a, b)
            self.assertEqual([a['splits'][s]['documents'] for s in ('train', 'validation', 'test')], [8, 1, 1])
            seen = set()
            for split in ('train', 'validation', 'test'):
                docs = [json.loads(line) for line in (root / 'a' / f'{split}.jsonl').read_text().splitlines()]
                ids = {d['id'] for d in docs}
                self.assertFalse(ids & seen)
                seen |= ids
                raw = (root / 'a' / f'{split}.txt').read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), a['splits'][split]['sha256'])
                self.assertEqual(raw, (root / 'b' / f'{split}.txt').read_bytes())
            self.assertEqual(len(seen), len(records))

    def test_duplicate_and_invalid_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, records = self.make_manifest(root)
            records.append({**records[0], 'id': 'duplicate'})
            manifest.write_text(json.dumps({'documents': records}))
            report = prepare_corpus(manifest, root / 'ok')
            self.assertEqual(len(report['duplicates_removed']), 1)
            records[-1]['group'] = 'other'
            manifest.write_text(json.dumps({'documents': records}))
            with self.assertRaises(ValueError):
                prepare_corpus(manifest, root / 'conflict')
            records.pop()
            records[0]['sha256'] = 'wrong'
            manifest.write_text(json.dumps({'documents': records}))
            with self.assertRaises(ValueError):
                prepare_corpus(manifest, root / 'bad')
            self.assertFalse((root / 'bad').exists())

    def test_existing_preparation_is_verified_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _ = self.make_manifest(root)
            output = root / 'prepared'
            first = prepare_corpus(manifest, output)
            times = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}
            self.assertEqual(prepare_corpus(manifest, output), first)
            self.assertEqual(times, {p.name: p.stat().st_mtime_ns for p in output.iterdir()})
            with self.assertRaises(ValueError):
                prepare_corpus(manifest, output, seed=7)
            (output / 'train.txt').write_text('modified')
            with self.assertRaises(ValueError):
                prepare_corpus(manifest, output)
            self.assertEqual((output / 'train.txt').read_text(), 'modified')

    def test_existing_download_snapshot_is_verified(self):
        from scripts.download_corpus import verify_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, records = self.make_manifest(root)
            verify_snapshot(root, records)
            (root / '0.txt').write_text('modified')
            with self.assertRaises(ValueError):
                verify_snapshot(root, records)

    def test_normalization_and_markers(self):
        self.assertEqual(normalize('  cafe\u0301\r\n\r\nHello  '), 'café\n\nHello')
        self.assertEqual(normalize('Café, HELLO!', 'ascii-letters'), 'cafe hello')
        wrapped = 'notice\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nbody\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nlicense'
        self.assertEqual(strip_gutenberg(wrapped), 'body\n')
        with self.assertRaises(ValueError):
            strip_gutenberg('missing markers')


if __name__ == '__main__':
    unittest.main()
