"""Versioned, development-only corpus ingestion with retained rights evidence."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import urlparse

from prepare_corpus import normalize,strip_gutenberg


def digest(raw):return hashlib.sha256(raw).hexdigest()


class MetadataRows(HTMLParser):
    def __init__(self):
        super().__init__();self.rows=[];self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='br' and self.cell is not None:self.cell.append(' ')
        if tag=='tr':self.row=[]
        if tag in ('th','td'):self.cell=[]
    def handle_data(self,data):
        if self.cell is not None:self.cell.append(data)
    def handle_endtag(self,tag):
        if tag in ('th','td') and self.cell is not None:
            if self.row is not None:self.row.append(' '.join(''.join(self.cell).split()))
            self.cell=None
        if tag=='tr' and self.row:self.rows.append(self.row);self.row=None


def parse_metadata(raw):
    parser=MetadataRows();parser.feed(raw.decode('utf-8'));fields={}
    for row in parser.rows:
        if len(row)==2:fields.setdefault(row[0],[]).append(row[1])
    return fields


def validate_registry(registry):
    docs=registry['documents']
    if not 10<=len(docs)<=20:raise ValueError('pilot requires 10 to 20 documents')
    ids=set();legacy={'pg11','pg84','pg1342'}
    required=('title','authors','language','domain','topic','source_family','source_url','download_url',
              'format','revision','work_family','author_family','split_group','expected_sha256','license','attribution','usage')
    for row in docs:
        id=row['id']
        if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',id) or id in ids or id in legacy:
            raise ValueError('unsafe, duplicate or excluded document ID')
        ids.add(id)
        for key in required:
            if not row.get(key):raise ValueError(f'missing provenance: {id}/{key}')
        if not re.fullmatch('[0-9a-f]{64}',row['expected_sha256']):raise ValueError('invalid expected hash')
        if row['usage']!='development-ingestion-pilot-only; not assigned to train/validation/test':
            raise ValueError('pilot cannot assign held-out/model splits')
        if any(author.startswith(('Carroll, Lewis','Shelley, Mary Wollstonecraft','Austen, Jane')) for author in row['authors']):
            raise ValueError('excluded legacy author family')
        if row['format']=='gutenberg-utf8':
            if not re.fullmatch('pg[1-9][0-9]*',id):raise ValueError('invalid Gutenberg ID')
            number=id[2:]
            if row['source_url']!=f'https://www.gutenberg.org/ebooks/{number}' or row['download_url']!=f'https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt':
                raise ValueError('unexpected Gutenberg URL')
            if row['license']['identifier']!='LicenseRef-PG-PD-US':raise ValueError('unexpected edition rights basis')
        elif row['format']=='latex-source':
            revision=row['revision']['commit'];path=row['revision']['path']
            if not re.fullmatch('[0-9a-f]{40}',revision) or '..' in Path(path).parts or not path.startswith('content/') or not path.endswith('.tex'):
                raise ValueError('invalid pinned source')
            if row['download_url']!=f'https://raw.githubusercontent.com/OpenLogicProject/OpenLogic/{revision}/{path}':
                raise ValueError('unexpected pinned source URL')
            if row['license']['identifier']!='CC-BY-4.0':raise ValueError('unexpected source grant')
        else:raise ValueError('unsupported format')
    required_evidence=set()
    if any(d['format']=='gutenberg-utf8' for d in docs):required_evidence.add('gutenberg-license')
    if any(d['format']=='latex-source' for d in docs):
        required_evidence.update(('openlogic-license-page','openlogic-license-text','openlogic-repository-grant'))
    if not required_evidence <= {e['id'] for e in registry['evidence']}:
        raise ValueError('missing retained rights evidence')
    for evidence in registry['evidence']:
        url=urlparse(evidence['url'])
        if url.scheme!='https' or url.netloc not in ('www.gutenberg.org','openlogicproject.org','raw.githubusercontent.com'):
            raise ValueError('unexpected evidence URL')
        if Path(evidence['cache_name']).name!=evidence['cache_name']:raise ValueError('unsafe evidence cache name')


def verify_metadata(row,raw):
    fields=parse_metadata(raw)
    if fields.get('Copyright')!=['Public domain in the USA.'] or fields.get('Language')!=['English']:
        raise ValueError('edition rights statement or language differs')
    if fields.get('Title')!=[row['title']] or fields.get('Author')!=row['authors']:
        raise ValueError('edition title or authors differ from reviewed registry')
    for key in ('Translator','Editor','Illustrator','Contributor'):
        if fields.get(key,[])!=row['contributors'].get(key,[]):raise ValueError('edition contributors differ')
    return fields


def safe_file(root,relative):
    path=root/relative
    if not path.resolve().is_relative_to(root.resolve()):raise ValueError('unsafe snapshot path')
    return path


def verify_snapshot(output,registry_bytes):
    registry=json.loads(registry_bytes);validate_registry(registry)
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['registry_sha256']!=digest(registry_bytes):raise ValueError('registry differs from existing snapshot')
    expected=set(manifest['files'])|{'manifest.json'}
    actual={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if actual!=expected:raise ValueError('snapshot file inventory differs')
    for relative,expected_hash in manifest['files'].items():
        if digest(safe_file(output,relative).read_bytes())!=expected_hash:raise ValueError(f'snapshot checksum differs: {relative}')
    by_id={r['id']:r for r in manifest['report']['documents']}
    for row in registry['documents']:
        raw=safe_file(output,by_id[row['id']]['raw_path']).read_bytes()
        if digest(raw)!=row['expected_sha256']:raise ValueError('raw differs from reviewed hash')
    return manifest['report']


def ingest_pilot(registry_path,output,fetch):
    """fetch(url,cache_name) returns bytes; no training/test files are referenced."""
    registry_path,output=Path(registry_path),Path(output)
    registry_bytes=registry_path.read_bytes();registry=json.loads(registry_bytes);validate_registry(registry)
    if output.exists():return verify_snapshot(output,registry_bytes)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name+'.staging-',dir=output.parent) as directory:
        stage=Path(directory);files={};documents=[];seen_normalized={};seen_raw={};attribution=[]
        def save(relative,raw):
            path=stage/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            files[relative]=digest(raw);return relative
        evidence=[]
        for item in registry['evidence']:
            raw=fetch(item['url'],item['cache_name'])
            if item.get('expected_sha256') and digest(raw)!=item['expected_sha256']:raise ValueError('pinned rights evidence changed')
            decoded=raw.decode('utf-8')
            if item['id']=='openlogic-repository-grant' and 'https://creativecommons.org/licenses/by/4.0/' not in decoded:
                raise ValueError('repository license grant missing')
            evidence.append(dict(**item,path=save('evidence/'+item['cache_name'],raw),sha256=digest(raw)))
        for row in registry['documents']:
            raw=fetch(row['download_url'],row['id']+'.raw')
            if digest(raw)!=row['expected_sha256']:raise ValueError(f'reviewed source bytes differ: {row["id"]}')
            if digest(raw) in seen_raw:raise ValueError('duplicate raw documents in pilot')
            seen_raw[digest(raw)]=row['id']
            text=raw.decode('utf-8-sig')
            if '\ufffd' in text:raise ValueError('replacement character in source; inspect encoding')
            suffix='.txt' if row['format']=='gutenberg-utf8' else '.tex'
            raw_path=save('raw/'+row['id']+suffix,raw)
            record=dict(**row,ingested_at_utc=datetime.now(timezone.utc).isoformat(),raw_path=raw_path,raw_sha256=digest(raw),raw_bytes=len(raw),raw_characters=len(text),
                partition='pilot-development-only',eligible_for_future_test=False)
            if row['format']=='gutenberg-utf8':
                metadata=fetch(row['source_url'],row['id']+'.metadata.html')
                fields=verify_metadata(row,metadata)
                record['metadata_path']=save('evidence/'+row['id']+'.metadata.html',metadata)
                record['metadata_sha256']=digest(metadata);record['verified_metadata']=fields
                body=strip_gutenberg(text.replace('\r\n','\n').replace('\r','\n'))
                normalized=normalize(body,'unicode')
                if len(normalized)<1000 or any(ord(c)<32 and c!='\n' for c in normalized):
                    raise ValueError('short or control-character text after extraction')
                if '*** START OF' in normalized or '*** END OF' in normalized:raise ValueError('boundary marker leaked into body')
                encoded=normalized.encode('utf-8');sha=digest(encoded)
                if sha in seen_normalized:raise ValueError('exact normalized duplicate; resolve lineage before ingestion')
                seen_normalized[sha]=row['id']
                record.update(text_path=save('text/'+row['id']+'.txt',encoded),normalized_sha256=sha,
                    normalized_bytes=len(encoded),normalized_characters=len(normalized),status='unicode-text-extracted-pilot-only',
                    transformation='gutenberg boundaries removed; NFC; LF; whitespace normalization; case and punctuation preserved',
                    quality=dict(uppercase_characters=sum(c.isupper() for c in normalized),
                        non_ascii_characters=sum(ord(c)>127 for c in normalized),
                        illustration_markers=normalized.lower().count('[illustration'),
                        in_body_gutenberg_mentions=normalized.lower().count('project gutenberg')))
            else:
                record.update(status='quarantined-source-format',text_path=None,normalized_sha256=None,
                    transformation='none; raw LaTeX retained without compilation or generic stripping',
                    quarantine_reason='Formula/notation macros and document dependencies require a reviewed renderer before prose training.')
            documents.append(record)
            attribution.append(dict(id=row['id'],title=row['title'],attribution=row['attribution'],license=row['license'],
                contributors=row['contributors'],changes=record['transformation'],source_url=row['source_url'],revision=row['revision']))
        save('ATTRIBUTION.json',(json.dumps(attribution,indent=2)+'\n').encode())
        extracted=[d for d in documents if d['text_path']]
        domains=Counter(d['domain'] for d in documents);domain_chars=Counter()
        for d in extracted:domain_chars[d['domain']]+=d['normalized_characters']
        report=dict(corpus_id=registry['corpus_id'],created_at_utc=datetime.now(timezone.utc).isoformat(),
            registry_path=str(registry_path),registry_sha256=digest(registry_bytes),document_count=len(documents),
            source_families=dict(Counter(d['source_family'] for d in documents)),domains=dict(domains),
            independent_split_groups=len({d['split_group'] for d in documents}),
            extracted_documents=len(extracted),quarantined_documents=len(documents)-len(extracted),
            extracted_characters=sum(d['normalized_characters'] for d in extracted),
            extracted_utf8_bytes=sum(d['normalized_bytes'] for d in extracted),extracted_characters_by_domain=dict(domain_chars),
            largest_document_fraction=max(d['normalized_characters'] for d in extracted)/sum(d['normalized_characters'] for d in extracted),
            raw_exact_duplicates=0,normalized_exact_duplicates=0,
            near_duplicate_screen='not completed; exact hashes only; no leakage-clearance claim',
            split_status='No train/validation/test assignments. All inspected pilot families are ineligible for future untouched test.',
            new_test_created=False,new_test_read=False,legacy_test_read=False,model_loaded=False,training_performed=False,
            rights_scope='Edition-specific source rights statements/grants verified and retained; PG public-domain scope is USA only; no worldwide rights determination or corpus redistribution approval.',
            evidence=evidence,documents=documents,rejected_candidates=registry.get('rejected_candidates',[]))
        manifest=dict(registry_sha256=digest(registry_bytes),files=files,report=report)
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        # Publish only a complete, verified staging snapshot.
        verify_snapshot(stage,registry_bytes)
        stage.rename(output)
    return report
