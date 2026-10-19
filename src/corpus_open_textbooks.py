"""Pinned ingestion for a development-only open-textbook expansion wave."""

from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
import html
import json
from pathlib import Path
import re
import tempfile
import unicodedata
from urllib.parse import urlparse

from corpus_cleaning import deduplicate_report
from corpus_pilot import digest, safe_file


ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
HASH_RE = re.compile(r"[0-9a-f]{64}")
REFERENCE_HEADING_RE = re.compile(
    r"^(?:(?:chapter\s+)?\d+(?:\.\d+)?\s+)?(?:references|bibliography|works cited|sources|further readings?)$",
    re.IGNORECASE,
)
HTML_TAG_RE = re.compile(
    r"</?(?:html|body|head|div|span|p|h[1-6]|a|img|figure|figcaption|table|tr|td|th|ul|ol|li|pre|code|"
    r"section|article|iframe|video|audio|script|style)\b[^>\n]*>",
    re.IGNORECASE,
)


class _PlainHTML(HTMLParser):
    BLOCKS = {
        "address", "article", "aside", "blockquote", "dd", "div", "dl", "dt", "footer", "header",
        "h1", "h2", "h3", "h4", "h5", "h6", "li", "main", "ol", "p", "pre", "section", "table",
        "tbody", "thead", "tfoot", "tr", "ul",
    }
    SUPPRESSED = {"audio", "figure", "iframe", "noscript", "object", "script", "style", "svg", "video"}
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.suppressed_depth = 0
        self.images_excluded = 0
        self.figures_excluded = 0
        self.embeds_excluded = 0

    def _break(self):
        if self.parts and self.parts[-1] != "\n\n":
            self.parts.append("\n\n")

    def handle_starttag(self, tag, attrs):
        tag = tag.casefold()
        attributes = dict(attrs)
        if self.suppressed_depth:
            if tag not in self.VOID:
                self.suppressed_depth += 1
            return
        classes = set(attributes.get("class", "").casefold().split())
        suppressed = tag in self.SUPPRESSED or (tag == "div" and "footnotes" in classes)
        if suppressed:
            self.suppressed_depth = 1
            if tag == "figure":
                self.figures_excluded += 1
            elif tag in {"audio", "iframe", "object", "video"}:
                self.embeds_excluded += 1
            return
        if tag == "img":
            self.images_excluded += 1
        elif tag == "br":
            self.parts.append("\n")
        elif tag in self.BLOCKS:
            self._break()
        elif tag in {"td", "th"}:
            self.parts.append("\t")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag.casefold() not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        tag = tag.casefold()
        if self.suppressed_depth:
            self.suppressed_depth -= 1
            return
        if not self.suppressed_depth and tag in self.BLOCKS:
            self._break()

    def handle_data(self, data):
        if self.suppressed_depth:
            return
        value = " ".join(data.split())
        if value:
            self.parts.append(value + " ")


def render_reviewed_html(title, source_html):
    """Render reviewed Pressbooks/Manifold chapter HTML to NFC plain text."""
    parser = _PlainHTML()
    parser.feed(source_html)
    parser.close()
    body = "".join(parser.parts)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n[ \t]+", "\n", body)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    blocks = [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]
    references_removed = False
    for index, block in enumerate(blocks):
        if REFERENCE_HEADING_RE.fullmatch(block.rstrip(":")):
            blocks = blocks[:index]
            references_removed = True
            break
    if blocks and blocks[0].casefold() == title.casefold():
        blocks = blocks[1:]
    rendered = unicodedata.normalize("NFC", title.strip() + "\n\n" + "\n\n".join(blocks) + "\n")
    audit = {
        "images_excluded": parser.images_excluded,
        "figures_excluded_with_captions": parser.figures_excluded,
        "media_embeds_excluded": parser.embeds_excluded,
        "references_section_excluded": references_removed,
        "html_tags_remaining": len(HTML_TAG_RE.findall(rendered)),
        "replacement_characters": rendered.count("\ufffd"),
    }
    return rendered, audit


def parse_pressbooks(license_raw, content_raw):
    license_record = json.loads(license_raw)
    content_record = json.loads(content_raw)
    license_info = license_record.get("license") or {}
    authors = [row["name"] for row in license_record.get("author", []) if row.get("name")]
    editors = [row["name"] for row in license_record.get("editor", []) if row.get("name")]
    title = html.unescape(content_record["title"]["rendered"])
    rendered, audit = render_reviewed_html(title, content_record["content"]["rendered"])
    return {
        "work_title": license_record.get("name", ""),
        "document_title": title,
        "authors": authors,
        "editors": editors,
        "publisher": (license_record.get("publisher") or {}).get("name", ""),
        "publication_date": license_record.get("datePublished", ""),
        "copyright_year": str(license_record.get("copyrightYear", "")),
        "license_url": license_info.get("url", ""),
        "license_name": license_info.get("name", ""),
        "license_exceptions": license_info.get("description", ""),
        "revision": content_record.get("modified_gmt", ""),
        "source_url": content_record.get("link", ""),
        "rendered": rendered,
        "audit": audit,
    }


def parse_manifold(license_raw, content_raw):
    project = json.loads(license_raw)["data"]["attributes"]
    section = json.loads(content_raw)["data"]["attributes"]
    metadata = project.get("metadata") or {}
    title = html.unescape(section["name"])
    rendered, audit = render_reviewed_html(title, section["body"])
    return {
        "work_title": html.unescape(project.get("titlePlaintext") or project.get("title", "")),
        "document_title": title,
        "publisher": re.sub(r"<[^>]+>", " ", metadata.get("publisher", "")).strip(),
        "rights_holder": re.sub(r"<[^>]+>", " ", metadata.get("rightsHolder", "")).strip(),
        "rights": re.sub(r"<[^>]+>", " ", metadata.get("rights", "")).strip(),
        "creator_names": project.get("creatorNames", ""),
        "revision": project.get("updatedAt", ""),
        "rendered": rendered,
        "audit": audit,
    }


def validate_registry(registry):
    documents = registry["documents"]
    if not 10 <= len(documents) <= 30:
        raise ValueError("open-textbook wave requires 10 to 30 documents")
    if registry["usage"] != "development-only; not assigned to train/validation/test":
        raise ValueError("open-textbook wave may not assign model splits")
    if any(registry["test_policy"].values()):
        raise ValueError("open-textbook sources may not enter or read a test")
    seen_ids, seen_work_ids, seen_authors = set(), set(), set()
    for row in documents:
        if not ID_RE.fullmatch(row["id"]) or row["id"] in seen_ids:
            raise ValueError("unsafe or duplicate document ID")
        seen_ids.add(row["id"])
        required = (
            "provider", "work_id", "work_title", "document_title", "authors", "publisher", "language",
            "domain", "topic", "source_url", "download_url", "license_evidence_url", "revision",
            "work_family", "author_family", "split_group", "expected_sha256", "license_expected_sha256",
            "license", "attribution", "usage",
        )
        for key in required:
            if not row.get(key):
                raise ValueError(f"missing provenance: {row['id']}/{key}")
        if row["work_id"] in seen_work_ids:
            raise ValueError("duplicate open-textbook work")
        seen_work_ids.add(row["work_id"])
        if row["provider"] not in {"pressbooks-v2", "manifold-v1"}:
            raise ValueError("unexpected open-textbook provider")
        if row["language"] != "en" or row["usage"] != registry["usage"]:
            raise ValueError("unexpected language or usage")
        if not HASH_RE.fullmatch(row["expected_sha256"]) or not HASH_RE.fullmatch(row["license_expected_sha256"]):
            raise ValueError("invalid content or license hash")
        for key in ("source_url", "download_url", "license_evidence_url"):
            parsed = urlparse(row[key])
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("source URLs must use HTTPS")
        license_record = row["license"]
        if license_record.get("identifier") != "CC-BY-4.0" or license_record.get("terms_url") != "https://creativecommons.org/licenses/by/4.0/":
            raise ValueError("unexpected open-textbook license")
        normalized_authors = {author.casefold() for author in row["authors"]}
        if seen_authors & normalized_authors:
            raise ValueError("authors overlap across independent split groups")
        seen_authors.update(normalized_authors)
        expected_family = f"work:{row['provider']}:{row['work_id']}"
        if row["work_family"] != expected_family or row["split_group"] != expected_family:
            raise ValueError("unexpected work or split family")


def _parse_row(row, license_raw, content_raw):
    if row["provider"] == "pressbooks-v2":
        parsed = parse_pressbooks(license_raw, content_raw)
        for key in ("work_title", "document_title", "authors", "publisher", "revision", "source_url"):
            if parsed[key] != row[key]:
                raise ValueError(f"Pressbooks {key} differs: {row['id']}")
        if parsed["license_url"] != row["license"]["terms_url"] or "CC BY" not in parsed["license_name"]:
            raise ValueError(f"Pressbooks CC BY 4.0 evidence differs: {row['id']}")
    else:
        parsed = parse_manifold(license_raw, content_raw)
        for key in ("work_title", "document_title", "publisher", "revision"):
            if parsed[key] != row[key]:
                raise ValueError(f"Manifold {key} differs: {row['id']}")
        if not re.search(r"(?:CC\s*BY\s*4\.0|Creative Commons Attribution 4\.0|Attribution 4\.0)", parsed["rights"], re.IGNORECASE):
            raise ValueError(f"Manifold CC BY 4.0 evidence differs: {row['id']}")
        author_evidence = parsed["rendered"] if row.get("author_evidence") == "content" else " ".join(
            (parsed["rights"], parsed["rights_holder"], parsed["creator_names"])
        )
        for author in row["authors"]:
            if author.split()[-1].casefold() not in author_evidence.casefold():
                raise ValueError(f"Manifold rights holder differs: {row['id']}")
    return parsed


def verify_open_textbook_snapshot(output, registry_bytes):
    registry = json.loads(registry_bytes)
    validate_registry(registry)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["registry_sha256"] != digest(registry_bytes):
        raise ValueError("registry differs from open-textbook snapshot")
    expected = set(manifest["files"]) | {"manifest.json"}
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if expected != actual:
        raise ValueError("open-textbook snapshot inventory differs")
    for relative, expected_hash in manifest["files"].items():
        if digest(safe_file(output, relative).read_bytes()) != expected_hash:
            raise ValueError(f"open-textbook checksum differs: {relative}")
    by_id = {row["id"]: row for row in manifest["report"]["documents"]}
    for row in registry["documents"]:
        record = by_id[row["id"]]
        license_raw = safe_file(output, record["license_path"]).read_bytes()
        content_raw = safe_file(output, record["raw_path"]).read_bytes()
        if digest(license_raw) != row["license_expected_sha256"] or digest(content_raw) != row["expected_sha256"]:
            raise ValueError(f"reviewed source differs: {row['id']}")
        parsed = _parse_row(row, license_raw, content_raw)
        rendered_raw = safe_file(output, record["text_path"]).read_bytes()
        if rendered_raw != parsed["rendered"].encode("utf-8") or digest(rendered_raw) != record["rendered_sha256"]:
            raise ValueError(f"rendered open-textbook snapshot differs: {row['id']}")
    return manifest["report"]


def ingest_open_textbooks(registry_path, output, fetch):
    """Fetch pinned API records; fetch(url, cache_name) returns bytes."""
    registry_path, output = Path(registry_path), Path(output)
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    validate_registry(registry)
    if output.exists():
        return verify_open_textbook_snapshot(output, registry_bytes)
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
            license_raw = fetch(row["license_evidence_url"], row["id"] + "-license.json")
            content_raw = fetch(row["download_url"], row["id"] + "-content.json")
            if digest(license_raw) != row["license_expected_sha256"]:
                raise ValueError(f"reviewed license record differs: {row['id']}")
            if digest(content_raw) != row["expected_sha256"]:
                raise ValueError(f"reviewed content record differs: {row['id']}")
            parsed = _parse_row(row, license_raw, content_raw)
            rendered, audit = parsed["rendered"], parsed["audit"]
            if len(rendered) < 1000 or audit["html_tags_remaining"] or audit["replacement_characters"]:
                raise ValueError(f"rendered open-textbook document requires review: {row['id']}")
            encoded = rendered.encode("utf-8")
            rendered_hash = digest(encoded)
            if rendered_hash in normalized_hashes:
                raise ValueError("exact rendered duplicate")
            normalized_hashes.add(rendered_hash)
            license_path = save("evidence/" + row["id"] + ".json", license_raw)
            raw_path = save("raw/" + row["id"] + ".json", content_raw)
            text_path = save("text/" + row["id"] + ".txt", encoded)
            transformation = (
                "Reviewed textbook HTML v1: headings, prose, lists, code and tables retained; images with captions, "
                "media embeds and trailing references excluded; links rendered as labels; NFC/LF"
            )
            record = dict(
                row,
                ingested_at_utc=datetime.now(timezone.utc).isoformat(),
                license_path=license_path,
                license_sha256=digest(license_raw),
                raw_path=raw_path,
                raw_sha256=digest(content_raw),
                raw_bytes=len(content_raw),
                text_path=text_path,
                rendered_sha256=rendered_hash,
                rendered_characters=len(rendered),
                rendered_utf8_bytes=len(encoded),
                transformation=transformation,
                transformation_audit=audit,
                partition="expansion-development-only",
                eligible_for_future_test=False,
            )
            records.append(record)
            analysis_documents.append({"id": row["id"], "text": rendered})
            attribution.append({
                "id": row["id"], "work_title": row["work_title"], "document_title": row["document_title"],
                "authors": row["authors"], "publisher": row["publisher"], "attribution": row["attribution"],
                "license": row["license"], "source_url": row["source_url"], "raw_sha256": record["raw_sha256"],
                "license_sha256": record["license_sha256"], "changes": transformation,
            })
        attribution_raw = (json.dumps(attribution, indent=2, ensure_ascii=False) + "\n").encode()
        save("ATTRIBUTION.json", attribution_raw)
        deduplication = deduplicate_report(analysis_documents, registry["deduplication"])
        report = {
            "corpus_id": registry["corpus_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "registry_sha256": digest(registry_bytes),
            "document_count": len(records),
            "provider_count": len({row["provider"] for row in records}),
            "publisher_count": len({row["publisher"] for row in records}),
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
            "scope_limit": registry.get(
                "scope_limit",
                "Third source-expansion wave only; institutional open textbooks remain overrepresented.",
            ),
            "split_status": "No train/validation/test assignments. Every inspected work is excluded from the future untouched test.",
            "new_test_created": False,
            "new_test_read": False,
            "legacy_test_read": False,
            "model_loaded": False,
            "training_performed": False,
        }
        manifest = {"registry_sha256": digest(registry_bytes), "files": files, "report": report}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        verify_open_textbook_snapshot(stage, registry_bytes)
        stage.rename(output)
    return report
