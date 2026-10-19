import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from corpus_pilot import ingest_pilot,validate_registry,parse_metadata


def fixture():
    docs=[];blobs={'gutenberg-license.html':b'Project Gutenberg license notice'}
    for number in range(1000,1010):
        id=f'pg{number}';title=f'Book {number}';author=f'Writer {number}'
        body=(f'Chapter {number}: Évidence, punctuation! A distinct story.\n'*30)
        raw=(f'Notice\n*** START OF THE PROJECT GUTENBERG EBOOK {title} ***\n'+body+f'*** END OF THE PROJECT GUTENBERG EBOOK {title} ***\nLicense').encode()
        meta=('<table>'+''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k,v in
            [('Title',title),('Author',author),('Copyright','Public domain in the USA.'),('Language','English')])+'</table>').encode()
        blobs[id+'.raw']=raw;blobs[id+'.metadata.html']=meta
        docs.append(dict(id=id,title=title,authors=[author],contributors={},language='en',domain='fiction',topic='story',
            source_family='project-gutenberg',source_url=f'https://www.gutenberg.org/ebooks/{number}',
            download_url=f'https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt',format='gutenberg-utf8',
            revision={'edition':number},work_family=id,author_family=author,split_group=author,
            expected_sha256=hashlib.sha256(raw).hexdigest(),license={'identifier':'LicenseRef-PG-PD-US'},
            attribution=f'{title} by {author}',usage='development-ingestion-pilot-only; not assigned to train/validation/test'))
    registry=dict(corpus_id='test-pilot',documents=docs,evidence=[dict(id='gutenberg-license',url='https://www.gutenberg.org/policy/license',cache_name='gutenberg-license.html')])
    return registry,blobs


class CorpusPilotTests(unittest.TestCase):
    def test_ingestion_retains_provenance_unicode_and_reuses_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);registry,blobs=fixture();source=root/'registry.json';source.write_text(json.dumps(registry));output=root/'pilot'
            report=ingest_pilot(source,output,lambda url,name:blobs[name])
            self.assertEqual(report['extracted_documents'],10)
            self.assertFalse(report['new_test_created']);self.assertFalse(report['model_loaded'])
            text=(output/report['documents'][0]['text_path']).read_text()
            self.assertIn('Évidence, punctuation!',text);self.assertNotIn('*** START',text)
            self.assertEqual((output/report['documents'][0]['raw_path']).read_bytes(),blobs['pg1000.raw'])
            before={p.relative_to(output):p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()}
            def fail(*args):raise AssertionError('must reuse without network')
            self.assertEqual(report,ingest_pilot(source,output,fail))
            self.assertEqual(before,{p.relative_to(output):p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()})
            (output/report['documents'][0]['raw_path']).write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'checksum'):ingest_pilot(source,output,fail)

    def test_wrong_download_or_metadata_never_publishes_partial_snapshot(self):
        for failure in ('hash','rights','author','title'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);registry,blobs=fixture();source=root/'registry.json';source.write_text(json.dumps(registry));output=root/'pilot'
                if failure=='hash':blobs['pg1000.raw']=b'changed'
                elif failure=='rights':blobs['pg1000.metadata.html']=blobs['pg1000.metadata.html'].replace(b'Public domain in the USA.',b'All rights reserved.')
                elif failure=='author':blobs['pg1000.metadata.html']=blobs['pg1000.metadata.html'].replace(b'Writer 1000',b'Another author')
                else:blobs['pg1000.metadata.html']=blobs['pg1000.metadata.html'].replace(b'Book 1000',b'Another title')
                with self.assertRaises(ValueError):ingest_pilot(source,output,lambda url,name:blobs[name])
                self.assertFalse(output.exists())
                self.assertFalse(list(root.glob('pilot.staging-*')))

    def test_unsafe_registry_and_missing_evidence_rejected(self):
        registry,_=fixture()
        for modify in (lambda r:r['documents'][0].update(id='../bad'),
                       lambda r:r['documents'][0].update(id='pg11'),
                       lambda r:r['documents'][0].update(download_url='https://example.com/file'),
                       lambda r:r['documents'][0].update(usage='test'),
                       lambda r:r.update(evidence=[])):
            changed=copy.deepcopy(registry);modify(changed)
            with self.assertRaises(ValueError):validate_registry(changed)
        fields=parse_metadata(b'<table><tr><th>Title</th><td>First<br>Second</td></tr></table>')
        self.assertEqual(fields['Title'],['First Second'])

    def test_latex_is_preserved_and_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            registry,blobs=fixture();revision='a'*40;path='content/sets/example.tex';raw=b'\\begin{document} A $x \\in A$ example. \\end{document}'
            row=registry['documents'][-1]
            row.update(id='olp-example',authors=['Open Logic Project contributors'],source_family='open-logic-project',
                format='latex-source',revision=dict(commit=revision,path=path),
                download_url=f'https://raw.githubusercontent.com/OpenLogicProject/OpenLogic/{revision}/{path}',
                expected_sha256=hashlib.sha256(raw).hexdigest(),license=dict(identifier='CC-BY-4.0'))
            blobs['olp-example.raw']=raw
            for id in ('openlogic-license-page','openlogic-license-text','openlogic-repository-grant'):
                name=id+'.txt';blobs[name]=b'https://creativecommons.org/licenses/by/4.0/'
                registry['evidence'].append(dict(id=id,url='https://openlogicproject.org/olp-license/',cache_name=name))
            root=Path(directory);source=root/'registry.json';source.write_text(json.dumps(registry));output=root/'pilot'
            report=ingest_pilot(source,output,lambda url,name:blobs[name])
            self.assertEqual(report['quarantined_documents'],1)
            row=report['documents'][-1]
            self.assertEqual(row['status'],'quarantined-source-format');self.assertIsNone(row['text_path'])
            self.assertEqual((output/row['raw_path']).read_bytes(),raw)
