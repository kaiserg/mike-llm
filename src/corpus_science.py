"""Pinned, development-only ingestion for the PLOS science expansion wave."""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
import unicodedata
import xml.etree.ElementTree as ET

from corpus_cleaning import deduplicate_report
from corpus_pilot import digest, safe_file


ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
DOI_RE = re.compile(r"10\.1371/journal\.pcbi\.\d{7}")
HASH_RE = re.compile(r"[0-9a-f]{64}")
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
EXCLUDED = {"fig", "table-wrap", "supplementary-material", "media", "graphic", "inline-graphic"}


def _text(element):
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _visible_text(element):
    pieces = []

    def visit(node):
        if node.tag.split("}")[-1] in EXCLUDED:
            return
        if node.text:
            pieces.append(node.text)
        for child in node:
            visit(child)
            if child.tail:
                pieces.append(child.tail)

    visit(element)
    return " ".join("".join(pieces).split())


def parse_plos_xml(raw):
    """Validate article metadata and render abstract/body prose from PLOS JATS XML."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise ValueError("invalid PLOS XML") from error
    article_meta = root.find("./front/article-meta")
    body = root.find("./body")
    if article_meta is None or body is None:
        raise ValueError("PLOS article metadata or body missing")

    doi = _text(article_meta.find("./article-id[@pub-id-type='doi']"))
    title = _text(article_meta.find("./title-group/article-title"))
    authors = []
    for contributor in article_meta.findall("./contrib-group/contrib[@contrib-type='author']"):
        name = contributor.find("./name")
        if name is not None:
            author = " ".join(filter(None, (_text(name.find("./given-names")), _text(name.find("./surname")))))
        else:
            author = _text(contributor.find("./collab"))
        if author:
            authors.append(author)
    published = article_meta.find("./pub-date[@pub-type='epub']")
    if published is None:
        raise ValueError("electronic publication date missing")
    year, month, day = (_text(published.find(f"./{part}")) for part in ("year", "month", "day"))
    if not (year and month and day):
        raise ValueError("complete electronic publication date missing")
    publication_date = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    permissions = article_meta.find("./permissions")
    license_element = permissions.find("./license") if permissions is not None else None
    license_href = license_element.get(XLINK_HREF, "") if license_element is not None else ""
    license_text = _text(license_element)
    copyright_holder = _text(permissions.find("./copyright-holder")) if permissions is not None else ""
    copyright_year = _text(permissions.find("./copyright-year")) if permissions is not None else ""
    if license_href not in {
        "http://creativecommons.org/licenses/by/4.0/",
        "https://creativecommons.org/licenses/by/4.0/",
    } or "Creative Commons Attribution License" not in license_text:
        raise ValueError("article-specific CC BY 4.0 grant missing")

    blocks = [title]
    abstract = article_meta.find("./abstract")
    abstract_paragraphs = abstract.findall(".//p") if abstract is not None else []
    if abstract_paragraphs:
        blocks.append("Abstract")
        blocks.extend(filter(None, (_visible_text(paragraph) for paragraph in abstract_paragraphs)))

    skipped_sections = 0

    def render_container(element):
        nonlocal skipped_sections
        tag = element.tag.split("}")[-1]
        if tag == "sec" and element.get("sec-type") == "supplementary-material":
            skipped_sections += 1
            return
        if tag in EXCLUDED:
            if tag == "supplementary-material":
                skipped_sections += 1
            return
        if tag == "title":
            value = _visible_text(element)
            if value:
                blocks.append(value)
            return
        if tag == "p":
            value = _visible_text(element)
            if value:
                blocks.append(value)
            return
        for child in element:
            render_container(child)

    render_container(body)
    rendered = unicodedata.normalize("NFC", "\n\n".join(blocks).strip() + "\n")
    audit = {
        "abstract_paragraphs_retained": len(abstract_paragraphs),
        "body_sections_retained": sum(
            1 for section in body.iter("sec") if section.get("sec-type") != "supplementary-material"
        ),
        "figures_excluded": sum(1 for _ in body.iter("fig")),
        "tables_excluded": sum(1 for _ in body.iter("table-wrap")),
        "supplementary_sections_excluded": skipped_sections,
        "references_excluded": root.find("./back/ref-list") is not None,
        "replacement_characters": rendered.count("\ufffd"),
    }
    metadata = {
        "doi": doi,
        "title": title,
        "authors": authors,
        "publication_date": publication_date,
        "copyright_holder": copyright_holder,
        "copyright_year": copyright_year,
        "license_href": license_href,
        "license_text": license_text,
    }
    return metadata, rendered, audit


def validate_registry(registry):
    documents = registry["documents"]
    if not 10 <= len(documents) <= 30:
        raise ValueError("science expansion wave requires 10 to 30 documents")
    if registry["usage"] != "development-only; not assigned to train/validation/test":
        raise ValueError("science expansion may not assign model splits")
    if any(registry["test_policy"].values()):
        raise ValueError("science expansion sources may not enter or read a test")
    seen_ids, seen_dois, seen_authors = set(), set(), set()
    for row in documents:
        if not ID_RE.fullmatch(row["id"]) or row["id"] in seen_ids:
            raise ValueError("unsafe or duplicate document ID")
        seen_ids.add(row["id"])
        required = (
            "doi", "title", "authors", "publication_date", "journal", "language", "domain", "topic",
            "source_url", "download_url", "work_family", "author_family", "split_group",
            "expected_sha256", "license", "attribution", "usage",
        )
        for key in required:
            if not row.get(key):
                raise ValueError(f"missing provenance: {row['id']}/{key}")
        doi = row["doi"]
        if not DOI_RE.fullmatch(doi) or doi in seen_dois:
            raise ValueError("invalid or duplicate DOI")
        seen_dois.add(doi)
        if row["language"] != "en" or row["usage"] != registry["usage"]:
            raise ValueError("unexpected language or usage")
        if not HASH_RE.fullmatch(row["expected_sha256"]):
            raise ValueError("invalid content hash")
        page = f"https://journals.plos.org/ploscompbiol/article?id={doi}"
        download = f"https://journals.plos.org/ploscompbiol/article/file?id={doi}&type=manuscript"
        if row["source_url"] != page or row["download_url"] != download:
            raise ValueError("unexpected PLOS source URL")
        if row["work_family"] != f"doi:{doi}" or row["author_family"] != f"authors:{doi}":
            raise ValueError("unexpected family identity")
        if row["split_group"] != f"author-work:{doi}":
            raise ValueError("unexpected split group")
        license_record = row["license"]
        if license_record.get("identifier") != "CC-BY-4.0":
            raise ValueError("unexpected license")
        for key in ("holder", "terms_url", "evidence", "verified_on"):
            if not license_record.get(key):
                raise ValueError(f"missing license provenance: {row['id']}/{key}")
        if license_record["terms_url"] != "https://creativecommons.org/licenses/by/4.0/":
            raise ValueError("unexpected license terms URL")
        normalized_authors = {author.casefold() for author in row["authors"]}
        if seen_authors & normalized_authors:
            raise ValueError("authors overlap across independent split groups")
        seen_authors.update(normalized_authors)


def verify_science_snapshot(output, registry_bytes):
    registry = json.loads(registry_bytes)
    validate_registry(registry)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["registry_sha256"] != digest(registry_bytes):
        raise ValueError("registry differs from science snapshot")
    expected = set(manifest["files"]) | {"manifest.json"}
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if expected != actual:
        raise ValueError("science snapshot inventory differs")
    for relative, expected_hash in manifest["files"].items():
        if digest(safe_file(output, relative).read_bytes()) != expected_hash:
            raise ValueError(f"science checksum differs: {relative}")
    report_documents = {row["id"]: row for row in manifest["report"]["documents"]}
    if set(report_documents) != {row["id"] for row in registry["documents"]}:
        raise ValueError("science document inventory differs from registry")
    for row in registry["documents"]:
        record = report_documents[row["id"]]
        raw = safe_file(output, record["raw_path"]).read_bytes()
        if digest(raw) != row["expected_sha256"] or digest(raw) != record["raw_sha256"]:
            raise ValueError(f"reviewed XML differs: {row['id']}")
        metadata, rendered, _ = parse_plos_xml(raw)
        if metadata["doi"] != row["doi"] or metadata["title"] != row["title"] or metadata["authors"] != row["authors"]:
            raise ValueError(f"snapshot metadata differs: {row['id']}")
        if metadata["publication_date"] != row["publication_date"] or metadata["copyright_holder"] != row["license"]["holder"]:
            raise ValueError(f"snapshot date or rights holder differs: {row['id']}")
        rendered_raw = safe_file(output, record["text_path"]).read_bytes()
        if rendered_raw != rendered.encode("utf-8") or digest(rendered_raw) != record["rendered_sha256"]:
            raise ValueError(f"rendered science snapshot differs: {row['id']}")
    return manifest["report"]


def ingest_science(registry_path, output, fetch):
    """Fetch pinned PLOS XML; fetch(url, cache_name) returns bytes."""
    registry_path, output = Path(registry_path), Path(output)
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    validate_registry(registry)
    if output.exists():
        return verify_science_snapshot(output, registry_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".staging-", dir=output.parent) as directory:
        stage = Path(directory)
        files = {}

        def save(relative, raw):
            target = safe_file(stage, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            files[relative] = digest(raw)
            return relative

        records, attribution, analysis_documents = [], [], []
        normalized_hashes = set()
        for row in registry["documents"]:
            raw = fetch(row["download_url"], row["id"] + ".xml")
            if digest(raw) != row["expected_sha256"]:
                raise ValueError(f"reviewed PLOS source differs: {row['id']}")
            metadata, rendered, audit = parse_plos_xml(raw)
            for key in ("doi", "title", "authors", "publication_date"):
                if metadata[key] != row[key]:
                    raise ValueError(f"PLOS {key} differs: {row['id']}")
            if metadata["copyright_holder"] != row["license"]["holder"]:
                raise ValueError(f"PLOS rights holder differs: {row['id']}")
            if len(rendered) < 1000 or audit["replacement_characters"]:
                raise ValueError(f"rendered PLOS article requires review: {row['id']}")
            encoded = rendered.encode("utf-8")
            rendered_hash = digest(encoded)
            if rendered_hash in normalized_hashes:
                raise ValueError("exact rendered duplicate")
            normalized_hashes.add(rendered_hash)
            raw_path = save("raw/" + row["id"] + ".xml", raw)
            text_path = save("text/" + row["id"] + ".txt", encoded)
            transformation = (
                "PLOS JATS XML v1: title, abstract, section headings, paragraphs and lists retained; "
                "figures, tables, supplementary material, references and front-matter administration excluded; NFC/LF"
            )
            record = dict(
                row,
                ingested_at_utc=datetime.now(timezone.utc).isoformat(),
                raw_path=raw_path,
                raw_sha256=digest(raw),
                raw_bytes=len(raw),
                text_path=text_path,
                rendered_sha256=rendered_hash,
                rendered_characters=len(rendered),
                rendered_utf8_bytes=len(encoded),
                transformation=transformation,
                transformation_audit=audit,
                xml_license_evidence={
                    "copyright_holder": metadata["copyright_holder"],
                    "copyright_year": metadata["copyright_year"],
                    "license_href": metadata["license_href"],
                    "license_text": metadata["license_text"],
                },
                partition="expansion-development-only",
                eligible_for_future_test=False,
            )
            records.append(record)
            analysis_documents.append({"id": row["id"], "text": rendered})
            attribution.append({
                "id": row["id"], "doi": row["doi"], "title": row["title"], "authors": row["authors"],
                "attribution": row["attribution"], "license": row["license"], "source_url": row["source_url"],
                "raw_sha256": record["raw_sha256"], "changes": transformation,
            })
        attribution_raw = (json.dumps(attribution, indent=2, ensure_ascii=False) + "\n").encode()
        save("ATTRIBUTION.json", attribution_raw)
        deduplication = deduplicate_report(analysis_documents, registry["deduplication"])
        report = {
            "corpus_id": registry["corpus_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "registry_sha256": digest(registry_bytes),
            "document_count": len(records),
            "journal_count": len({row["journal"] for row in records}),
            "independent_split_groups": len({row["split_group"] for row in records}),
            "unique_authors": len({author for row in records for author in row["authors"]}),
            "domains": dict(Counter(row["domain"] for row in records)),
            "topics": dict(Counter(row["topic"] for row in records)),
            "rendered_characters": sum(row["rendered_characters"] for row in records),
            "rendered_utf8_bytes": sum(row["rendered_utf8_bytes"] for row in records),
            "raw_exact_duplicates": len(records) - len({row["raw_sha256"] for row in records}),
            "rendered_exact_duplicates": len(records) - len({row["rendered_sha256"] for row in records}),
            "deduplication": deduplication,
            "documents": records,
            "attribution_path": "ATTRIBUTION.json",
            "attribution_sha256": digest(attribution_raw),
            "scope_limit": "Second source-expansion wave only; all articles come from one journal series and do not complete the target corpus.",
            "split_status": "No train/validation/test assignments. Every inspected article is excluded from the future untouched test.",
            "new_test_created": False,
            "new_test_read": False,
            "legacy_test_read": False,
            "model_loaded": False,
            "training_performed": False,
        }
        manifest = {"registry_sha256": digest(registry_bytes), "files": files, "report": report}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        verify_science_snapshot(stage, registry_bytes)
        stage.rename(output)
    return report
