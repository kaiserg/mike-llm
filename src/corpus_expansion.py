"""Pinned, development-only ingestion for the first corpus-v2 expansion wave."""

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


def parse_front_matter(text):
    if not text.startswith("---\n"):
        raise ValueError("missing YAML front matter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("unclosed YAML front matter")
    fields = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip('"\'')
    return fields, text[end + 5:]


def render_carpentries_markdown(text):
    """Render reviewed Carpentries Markdown to plain educational text."""
    metadata, body = parse_front_matter(text)
    body = re.sub(r"\{%\s*comment\s*%\}.*?\{%\s*endcomment\s*%\}", "", body, flags=re.DOTALL)
    body = re.sub(r"\{%.*?%\}", "", body, flags=re.DOTALL)
    body = re.sub(r"\{\{.*?\}\}", "", body, flags=re.DOTALL)
    lines = []
    directive_lines = 0
    fence_lines = 0
    reference_lines = 0
    for line in body.splitlines():
        if re.fullmatch(r":{5,}(?:\s+[A-Za-z_-]+)?\s*", line):
            directive_lines += 1
            continue
        if re.match(r"^\s*(```|~~~)", line):
            fence_lines += 1
            continue
        if re.match(r"^\s*\{:\s*[^}]+\}\s*$", line):
            directive_lines += 1
            continue
        if re.match(r"^\s*\{alt=['\"].*\}\s*$", line):
            directive_lines += 1
            continue
        if re.match(r"^\s*\[[^]]+\]:\s*\S+", line):
            reference_lines += 1
            continue
        lines.append(line)
    body = "\n".join(lines)
    def image_label(match):
        label = match.group(2) or match.group(1) or "illustration"
        embedded = re.search(r"alt=['\"](.*?)['\"]", label)
        return "Image: " + (embedded.group(1) if embedded else label)

    body = re.sub(
        r"!\[([^]]*)\]\([^)]+\)(?:\{[^}]*alt=['\"]([^'\"]*)['\"][^}]*\})?",
        image_label,
        body,
    )
    body = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", body)
    body = re.sub(r"\[([^]]+)\]\[[^]]*\]", r"\1", body)
    body = re.sub(r"<(https?://[^>]+)>", r"\1", body)
    body = re.sub(r"(?i)<br\s*/?>", "\n", body)
    body = re.sub(r"(?i)</(?:p|div|tr|table|ul|ol|h[1-6])>", "\n", body)
    body = re.sub(r"(?i)</?(?:td|th|li)(?:\s+[^>]*)?>", " ", body)
    body = re.sub(r"<[^>]+>", "", body)
    body = html.unescape(body)
    body = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", body)
    body = re.sub(r"\s*\{#[A-Za-z0-9_-]+\}\s*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"`([^`\n]+)`", r"\1", body)
    body = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", body)
    body = re.sub(r"__([^_\n]+)__", r"\1", body)
    body = re.sub(r"(?m)^\s*>\s?", "", body)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"
    body = unicodedata.normalize("NFC", body)
    unresolved = {
        "liquid_tags": len(re.findall(r"\{%|\{\{", body)),
        "directive_delimiters": len(re.findall(r"(?m)^:{5,}", body)),
        "fence_delimiters": len(re.findall(r"(?m)^\s*(?:```|~~~)", body)),
        "html_tags": len(re.findall(r"<[^>]+>", body)),
    }
    audit = {
        "yaml_front_matter_removed": True,
        "directive_lines_removed": directive_lines,
        "code_fence_lines_removed_code_retained": fence_lines,
        "reference_definition_lines_removed": reference_lines,
        "image_links_replaced_by_alt_text": len(re.findall(r"(?m)^Image: ", body)),
        "unresolved_markup": unresolved,
    }
    return metadata, body, audit


def validate_registry(registry):
    documents = registry["documents"]
    if not 10 <= len(documents) <= 30:
        raise ValueError("expansion wave requires 10 to 30 documents")
    if registry["usage"] != "development-only; not assigned to train/validation/test":
        raise ValueError("expansion may not assign model splits")
    if any(registry["test_policy"].values()):
        raise ValueError("expansion sources may not enter or read a test")
    evidence_by_id = {item["id"]: item for item in registry["evidence"]}
    if len(evidence_by_id) != len(registry["evidence"]):
        raise ValueError("duplicate evidence ID")
    seen = set()
    for row in documents:
        if not ID_RE.fullmatch(row["id"]) or row["id"] in seen:
            raise ValueError("unsafe or duplicate document ID")
        seen.add(row["id"])
        required = (
            "title", "organization", "language", "domain", "topic", "repository", "path",
            "commit", "commit_date", "source_url", "download_url", "history_url", "work_family",
            "author_family", "split_group", "expected_sha256", "license_evidence_id", "license",
            "attribution", "usage",
        )
        for key in required:
            if not row.get(key):
                raise ValueError(f"missing provenance: {row['id']}/{key}")
        if row["usage"] != registry["usage"] or row["language"] != "en":
            raise ValueError("unexpected usage or language")
        if not SHA_RE.fullmatch(row["commit"]) or not HASH_RE.fullmatch(row["expected_sha256"]):
            raise ValueError("invalid revision or content hash")
        if not row["path"].startswith("episodes/") or not row["path"].endswith(".md") or ".." in Path(row["path"]).parts:
            raise ValueError("invalid lesson path")
        raw = f"https://raw.githubusercontent.com/{row['repository']}/{row['commit']}/{row['path']}"
        blob = f"https://github.com/{row['repository']}/blob/{row['commit']}/{row['path']}"
        history = f"https://github.com/{row['repository']}/commits/{row['commit']}/{row['path']}"
        if row["download_url"] != raw or row["source_url"] != blob or row["history_url"] != history:
            raise ValueError("unexpected pinned source URL")
        if row["license_evidence_id"] not in evidence_by_id or row["license"]["identifier"] != "CC-BY-4.0":
            raise ValueError("missing or unexpected license evidence")
        license_evidence = evidence_by_id[row["license_evidence_id"]]
        if license_evidence["repository"] != row["repository"] or license_evidence["commit"] != row["commit"]:
            raise ValueError("license evidence does not match source revision")
    for item in registry["evidence"]:
        if not ID_RE.fullmatch(item["id"]) or not HASH_RE.fullmatch(item["expected_sha256"]):
            raise ValueError("invalid evidence identity or hash")
        parsed = urlparse(item["url"])
        if parsed.scheme != "https" or parsed.netloc != "raw.githubusercontent.com":
            raise ValueError("unexpected evidence host")
        expected_url = f"https://raw.githubusercontent.com/{item['repository']}/{item['commit']}/LICENSE.md"
        if item["url"] != expected_url or not SHA_RE.fullmatch(item["commit"]):
            raise ValueError("license evidence is not pinned to the repository revision")
        if Path(item["cache_name"]).name != item["cache_name"]:
            raise ValueError("unsafe evidence cache name")


def verify_expansion_snapshot(output, registry_bytes):
    registry = json.loads(registry_bytes)
    validate_registry(registry)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["registry_sha256"] != digest(registry_bytes):
        raise ValueError("registry differs from expansion snapshot")
    expected = set(manifest["files"]) | {"manifest.json"}
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if expected != actual:
        raise ValueError("expansion snapshot inventory differs")
    for relative, expected_hash in manifest["files"].items():
        if digest(safe_file(output, relative).read_bytes()) != expected_hash:
            raise ValueError(f"expansion checksum differs: {relative}")
    report_documents = {row["id"]: row for row in manifest["report"]["documents"]}
    if set(report_documents) != {row["id"] for row in registry["documents"]}:
        raise ValueError("expansion document inventory differs from registry")
    for row in registry["documents"]:
        record = report_documents[row["id"]]
        raw = safe_file(output, record["raw_path"]).read_bytes()
        if digest(raw) != row["expected_sha256"] or digest(raw) != record["raw_sha256"]:
            raise ValueError(f"reviewed raw source differs: {row['id']}")
        metadata, rendered, _ = render_carpentries_markdown(raw.decode("utf-8-sig"))
        if metadata.get("title") != row["title"]:
            raise ValueError(f"snapshot title differs: {row['id']}")
        rendered_raw = safe_file(output, record["text_path"]).read_bytes()
        if rendered_raw != rendered.encode("utf-8") or digest(rendered_raw) != record["rendered_sha256"]:
            raise ValueError(f"rendered snapshot differs: {row['id']}")
    report_evidence = {row["id"]: row for row in manifest["report"]["evidence"]}
    for row in registry["evidence"]:
        record = report_evidence[row["id"]]
        if digest(safe_file(output, record["path"]).read_bytes()) != row["expected_sha256"]:
            raise ValueError(f"license snapshot differs: {row['id']}")
    return manifest["report"]


def ingest_expansion(registry_path, output, fetch):
    """Fetch pinned lesson sources; fetch(url, cache_name) returns bytes."""
    registry_path, output = Path(registry_path), Path(output)
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    validate_registry(registry)
    if output.exists():
        return verify_expansion_snapshot(output, registry_bytes)
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

        evidence = []
        for item in registry["evidence"]:
            raw = fetch(item["url"], item["cache_name"])
            if digest(raw) != item["expected_sha256"]:
                raise ValueError(f"license evidence differs: {item['id']}")
            decoded = raw.decode("utf-8")
            if "Creative Commons" not in decoded or "https://creativecommons.org/licenses/by/4.0/" not in decoded:
                raise ValueError("CC BY 4.0 grant missing from evidence")
            evidence.append(dict(item, path=save("evidence/" + item["cache_name"], raw), sha256=digest(raw)))

        records = []
        attribution = []
        normalized_hashes = set()
        analysis_documents = []
        for row in registry["documents"]:
            raw = fetch(row["download_url"], row["id"] + ".md")
            if digest(raw) != row["expected_sha256"]:
                raise ValueError(f"reviewed source differs: {row['id']}")
            text = raw.decode("utf-8-sig")
            if "\ufffd" in text:
                raise ValueError("replacement character in lesson source")
            metadata, rendered, audit = render_carpentries_markdown(text)
            if metadata.get("title") != row["title"]:
                raise ValueError(f"lesson title differs: {row['id']}")
            if len(rendered) < 1000 or any(audit["unresolved_markup"].values()):
                raise ValueError(f"rendered lesson requires review: {row['id']}")
            encoded = rendered.encode("utf-8")
            rendered_hash = digest(encoded)
            if rendered_hash in normalized_hashes:
                raise ValueError("exact rendered duplicate")
            normalized_hashes.add(rendered_hash)
            raw_path = save("raw/" + row["id"] + ".md", raw)
            text_path = save("text/" + row["id"] + ".txt", encoded)
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
                transformation="Carpentries Markdown v1: YAML/directives/fences removed; code and prose retained; links rendered as labels; images rendered as alt text; NFC/LF",
                transformation_audit=audit,
                partition="expansion-development-only",
                eligible_for_future_test=False,
            )
            records.append(record)
            analysis_documents.append({"id": row["id"], "text": rendered})
            attribution.append({
                "id": row["id"], "title": row["title"], "organization": row["organization"],
                "attribution": row["attribution"], "license": row["license"], "source_url": row["source_url"],
                "history_url": row["history_url"], "revision": {"commit": row["commit"], "path": row["path"]},
                "changes": record["transformation"],
            })
        attribution_raw = (json.dumps(attribution, indent=2) + "\n").encode()
        save("ATTRIBUTION.json", attribution_raw)
        deduplication = deduplicate_report(analysis_documents, registry["deduplication"])
        report = {
            "corpus_id": registry["corpus_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "registry_sha256": digest(registry_bytes),
            "document_count": len(records),
            "repository_count": len({row["repository"] for row in records}),
            "independent_split_groups": len({row["split_group"] for row in records}),
            "domains": dict(Counter(row["domain"] for row in records)),
            "topics": dict(Counter(row["topic"] for row in records)),
            "rendered_characters": sum(row["rendered_characters"] for row in records),
            "rendered_utf8_bytes": sum(row["rendered_utf8_bytes"] for row in records),
            "raw_exact_duplicates": len(records) - len({row["raw_sha256"] for row in records}),
            "rendered_exact_duplicates": len(records) - len({row["rendered_sha256"] for row in records}),
            "deduplication": deduplication,
            "evidence": evidence,
            "documents": records,
            "attribution_path": "ATTRIBUTION.json",
            "attribution_sha256": digest(attribution_raw),
            "scope_limit": "First modern educational expansion wave only; not the complete target corpus and not sufficient to freeze final splits.",
            "split_status": "No train/validation/test assignments. Every inspected source family is excluded from the future untouched test.",
            "new_test_created": False,
            "new_test_read": False,
            "legacy_test_read": False,
            "model_loaded": False,
            "training_performed": False,
        }
        manifest = {"registry_sha256": digest(registry_bytes), "files": files, "report": report}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_expansion_snapshot(stage, registry_bytes)
        stage.rename(output)
    return report
