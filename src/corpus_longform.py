"""Pinned ingestion for modern, repository-hosted long-form prose."""

from collections import Counter
from datetime import datetime, timezone
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
SHA_RE = re.compile(r"[0-9a-f]{40}")
HASH_RE = re.compile(r"[0-9a-f]{64}")
ALLOWED_LICENSES = {"CC-BY-4.0", "CC0-1.0", "MIT"}


def _front_matter(text):
    if not text.startswith("---\n"):
        return text, False
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("unclosed Markdown front matter")
    return text[end + 5 :], True


def render_markdown_collection(title, sources):
    """Convert a reviewed sequence of Markdown files to one NFC text."""
    blocks = [title.strip()]
    audit = Counter()
    for path, source in sources:
        source = source.replace("\r\n", "\n").replace("\r", "\n")
        source, removed = _front_matter(source)
        audit["front_matter_blocks_removed"] += int(removed)
        source, count = re.subn(r"<!--.*?-->", "", source, flags=re.DOTALL)
        audit["html_comments_removed"] += count
        source, count = re.subn(r"\{%.*?%\}|\{\{.*?\}\}", "", source, flags=re.DOTALL)
        audit["template_expressions_removed"] += count
        lines = []
        in_fence = False
        for line in source.splitlines():
            if re.match(r"^\s*(```|~~~)", line):
                in_fence = not in_fence
                audit["code_fence_lines_removed_code_retained"] += 1
                continue
            if re.match(r"^\s*:{3,}(?:\s*\{.*\})?\s*$", line):
                audit["directive_delimiters_removed"] += 1
                continue
            if re.match(r"^\s*(?:\([A-Za-z0-9_-]+\)=|\{:[^}]+\})\s*$", line):
                audit["document_directives_removed"] += 1
                continue
            if re.match(r"^\s*\[[^]]+\]:\s*\S+", line):
                audit["reference_definitions_removed"] += 1
                continue
            lines.append(line)
        source = "\n".join(lines)
        source, count = re.subn(r"!\[([^]]*)\]\([^)]+\)", lambda m: "Image: " + (m.group(1) or "illustration"), source)
        audit["images_replaced_by_alt_text"] += count
        source = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", source)
        source = re.sub(r"\[([^]]+)\]\[[^]]*\]", r"\1", source)
        source = re.sub(r"(?i)<br\s*/?>", "\n", source)
        source = re.sub(r"(?i)</(?:p|div|tr|table|ul|ol|h[1-6])>", "\n", source)
        source = re.sub(r"(?i)</?(?:td|th|li)(?:\s+[^>]*)?>", " ", source)
        source = re.sub(r"<[^>]+>", "", source)
        source = html.unescape(source)
        source = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", source)
        source = re.sub(r"\s*\{#[A-Za-z0-9_-]+\}\s*$", "", source, flags=re.MULTILINE)
        source = re.sub(r"`([^`\n]+)`", r"\1", source)
        source = re.sub(r"\*\*([^*\n]+)\*\*|__([^_\n]+)__", lambda m: m.group(1) or m.group(2), source)
        source = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", source)
        source = re.sub(r"(?m)^\s*>\s?", "", source)
        source = re.sub(r"[ \t]+\n", "\n", source)
        source = re.sub(r"\n{3,}", "\n\n", source).strip()
        if source:
            blocks.append(source)
        audit["source_files"] += 1
    rendered = unicodedata.normalize("NFC", "\n\n".join(blocks).strip() + "\n")
    audit["replacement_characters"] = rendered.count("\ufffd")
    audit["remaining_html_tags"] = len(re.findall(r"<[^>]+>", rendered))
    audit["remaining_template_markers"] = len(re.findall(r"\{%|\{\{", rendered))
    return rendered, dict(audit)


def validate_registry(registry):
    documents = registry["documents"]
    if not 10 <= len(documents) <= 30:
        raise ValueError("long-form wave requires 10 to 30 documents")
    if registry["usage"] != "development-only; not assigned to train/validation/test":
        raise ValueError("long-form wave may not assign model splits")
    if any(registry["test_policy"].values()):
        raise ValueError("long-form sources may not enter or read a test")
    evidence = {row["id"]: row for row in registry["evidence"]}
    if len(evidence) != len(registry["evidence"]):
        raise ValueError("duplicate license evidence ID")
    seen_ids = set()
    for row in documents:
        if not ID_RE.fullmatch(row["id"]) or row["id"] in seen_ids:
            raise ValueError("unsafe or duplicate document ID")
        seen_ids.add(row["id"])
        required = ("title", "authors", "publisher", "language", "domain", "topic", "repository", "source_url", "commit",
                    "commit_date", "work_family", "author_family", "split_group", "license_evidence_id", "license",
                    "attribution", "usage", "sources")
        for key in required:
            if not row.get(key):
                raise ValueError(f"missing provenance: {row['id']}/{key}")
        if row["language"] != "en" or row["usage"] != registry["usage"] or not SHA_RE.fullmatch(row["commit"]):
            raise ValueError("unexpected language, usage, or revision")
        if row["license"]["identifier"] not in ALLOWED_LICENSES:
            raise ValueError("unexpected long-form license")
        if row["source_url"] != f"https://github.com/{row['repository']}/tree/{row['commit']}":
            raise ValueError("collection source URL is not pinned to the repository revision")
        item = evidence.get(row["license_evidence_id"])
        if not item or item["repository"] != row["repository"] or item["commit"] != row["commit"]:
            raise ValueError("license evidence does not match source revision")
        if item["license_identifier"] != row["license"]["identifier"]:
            raise ValueError("license identifier differs from evidence")
        if not row["sources"]:
            raise ValueError("document has no source files")
        for source in row["sources"]:
            path = Path(source["path"])
            if path.is_absolute() or ".." in path.parts or path.suffix != ".md" or not HASH_RE.fullmatch(source["expected_sha256"]):
                raise ValueError("unsafe or unpinned Markdown source")
            expected = f"https://raw.githubusercontent.com/{row['repository']}/{row['commit']}/{source['path']}"
            if source["download_url"] != expected:
                raise ValueError("source URL is not pinned to the repository revision")
    for item in registry["evidence"]:
        if not ID_RE.fullmatch(item["id"]) or not SHA_RE.fullmatch(item["commit"]) or not HASH_RE.fullmatch(item["expected_sha256"]):
            raise ValueError("invalid license evidence identity, revision, or hash")
        parsed = urlparse(item["url"])
        if parsed.scheme != "https" or parsed.netloc != "raw.githubusercontent.com" or Path(item["cache_name"]).name != item["cache_name"]:
            raise ValueError("unsafe license evidence location")


def _validate_license(item, raw):
    text = raw.decode("utf-8", errors="strict")
    identifier = item["license_identifier"]
    if identifier == "CC-BY-4.0" and not ("Attribution 4.0" in text or "CC-BY license" in text or "CC BY 4.0" in text):
        raise ValueError("CC BY 4.0 grant missing from evidence")
    if identifier == "CC0-1.0" and not ("CC0 1.0" in text and "public domain" in text.lower()):
        raise ValueError("CC0 grant missing from evidence")
    if identifier == "MIT" and not ("MIT License" in text and "Permission is hereby granted" in text):
        raise ValueError("MIT grant missing from evidence")


def verify_longform_snapshot(output, registry_bytes):
    registry = json.loads(registry_bytes)
    validate_registry(registry)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["registry_sha256"] != digest(registry_bytes):
        raise ValueError("registry differs from long-form snapshot")
    expected = set(manifest["files"]) | {"manifest.json"}
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if expected != actual:
        raise ValueError("long-form snapshot inventory differs")
    for relative, expected_hash in manifest["files"].items():
        if digest(safe_file(output, relative).read_bytes()) != expected_hash:
            raise ValueError(f"long-form checksum differs: {relative}")
    report = manifest["report"]
    evidence_by_id = {row["id"]: row for row in report["evidence"]}
    if set(evidence_by_id) != {row["id"] for row in registry["evidence"]}:
        raise ValueError("long-form license inventory differs")
    for item in registry["evidence"]:
        raw = safe_file(output, evidence_by_id[item["id"]]["path"]).read_bytes()
        if digest(raw) != item["expected_sha256"]:
            raise ValueError(f"license snapshot differs: {item['id']}")
        _validate_license(item, raw)
    documents_by_id = {row["id"]: row for row in report["documents"]}
    if set(documents_by_id) != {row["id"] for row in registry["documents"]}:
        raise ValueError("long-form document inventory differs")
    for row in registry["documents"]:
        record = documents_by_id[row["id"]]
        for key in ("title", "authors", "publisher", "domain", "topic", "repository", "commit", "split_group"):
            if record[key] != row[key]:
                raise ValueError(f"long-form provenance differs: {row['id']}/{key}")
        if len(record["sources"]) != len(row["sources"]):
            raise ValueError(f"long-form source inventory differs: {row['id']}")
        decoded_sources = []
        for expected_source, saved_source in zip(row["sources"], record["sources"]):
            if saved_source["path"] != expected_source["path"]:
                raise ValueError(f"long-form source order differs: {row['id']}")
            raw = safe_file(output, saved_source["raw_path"]).read_bytes()
            if digest(raw) != expected_source["expected_sha256"]:
                raise ValueError(f"long-form raw source differs: {row['id']}/{expected_source['path']}")
            decoded_sources.append((expected_source["path"], raw.decode("utf-8-sig")))
        rendered, _ = render_markdown_collection(row["title"], decoded_sources)
        rendered_raw = safe_file(output, record["text_path"]).read_bytes()
        if rendered_raw != rendered.encode("utf-8") or digest(rendered_raw) != record["rendered_sha256"]:
            raise ValueError(f"long-form rendered text differs: {row['id']}")
    attribution_raw = safe_file(output, report["attribution_path"]).read_bytes()
    if digest(attribution_raw) != report["attribution_sha256"]:
        raise ValueError("long-form attribution differs")
    return report


def ingest_longform(registry_path, output, fetch):
    registry_path, output = Path(registry_path), Path(output)
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    validate_registry(registry)
    if output.exists():
        return verify_longform_snapshot(output, registry_bytes)
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

        evidence_records = []
        for item in registry["evidence"]:
            raw = fetch(item["url"], item["cache_name"])
            if digest(raw) != item["expected_sha256"]:
                raise ValueError(f"license evidence differs: {item['id']}")
            _validate_license(item, raw)
            evidence_records.append(dict(item, path=save("evidence/" + item["cache_name"], raw), sha256=digest(raw)))

        records, attributions, analysis_documents = [], [], []
        rendered_hashes = set()
        for row in registry["documents"]:
            decoded_sources, source_records = [], []
            for index, source in enumerate(row["sources"]):
                cache_name = f"{row['id']}-{index:02d}.md"
                raw = fetch(source["download_url"], cache_name)
                if digest(raw) != source["expected_sha256"]:
                    raise ValueError(f"reviewed source differs: {row['id']}/{source['path']}")
                text = raw.decode("utf-8-sig")
                if "\ufffd" in text:
                    raise ValueError("replacement character in Markdown source")
                decoded_sources.append((source["path"], text))
                source_records.append(dict(source, raw_path=save("raw/" + cache_name, raw), raw_bytes=len(raw)))
            rendered, audit = render_markdown_collection(row["title"], decoded_sources)
            if len(rendered) < 5000 or audit["replacement_characters"] or audit["remaining_template_markers"]:
                raise ValueError(f"rendered long-form document requires review: {row['id']}")
            encoded = rendered.encode("utf-8")
            rendered_hash = digest(encoded)
            if rendered_hash in rendered_hashes:
                raise ValueError("exact rendered duplicate")
            rendered_hashes.add(rendered_hash)
            text_path = save("text/" + row["id"] + ".txt", encoded)
            record = dict(row, sources=source_records, ingested_at_utc=datetime.now(timezone.utc).isoformat(),
                          text_path=text_path, rendered_sha256=rendered_hash, rendered_characters=len(rendered),
                          rendered_utf8_bytes=len(encoded), transformation="Repository Markdown collection v1: front matter, template syntax, fence markers, HTML and links normalized; prose and code retained; NFC/LF",
                          transformation_audit=audit, partition="expansion-development-only", eligible_for_future_test=False)
            records.append(record)
            analysis_documents.append({"id": row["id"], "text": rendered})
            attributions.append({key: row[key] for key in ("id", "title", "authors", "publisher", "attribution", "license")})
        attribution_raw = (json.dumps(attributions, indent=2, ensure_ascii=False) + "\n").encode()
        save("ATTRIBUTION.json", attribution_raw)
        deduplication = deduplicate_report(analysis_documents, registry["deduplication"])
        report = {
            "corpus_id": registry["corpus_id"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "registry_sha256": digest(registry_bytes), "document_count": len(records),
            "repository_count": len({row["repository"] for row in records}),
            "publisher_count": len({row["publisher"] for row in records}),
            "independent_split_groups": len({row["split_group"] for row in records}),
            "domains": dict(Counter(row["domain"] for row in records)),
            "topics": dict(Counter(row["topic"] for row in records)),
            "rendered_characters": sum(row["rendered_characters"] for row in records),
            "rendered_utf8_bytes": sum(row["rendered_utf8_bytes"] for row in records),
            "rendered_exact_duplicates": len(records) - len(rendered_hashes),
            "deduplication": deduplication, "evidence": evidence_records, "documents": records,
            "attribution_path": "ATTRIBUTION.json", "attribution_sha256": digest(attribution_raw),
            "scope_limit": "Modern repository-hosted long-form wave only; source families remain grouped and splits are not frozen.",
            "split_status": "No train/validation/test assignments. Every inspected source family is excluded from the future untouched test.",
            "new_test_created": False, "new_test_read": False, "legacy_test_read": False,
            "model_loaded": False, "training_performed": False,
        }
        manifest = {"registry_sha256": digest(registry_bytes), "files": files, "report": report}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        verify_longform_snapshot(stage, registry_bytes)
        stage.rename(output)
    return report
