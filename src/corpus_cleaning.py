"""Deterministic editorial cleanup and near-duplicate screening for corpus v2."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata

from corpus_pilot import digest, safe_file, verify_snapshot


WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def word_tokens(text):
    """Return comparison-only Unicode tokens without changing saved prose."""
    return [match.group(0).casefold() for match in WORD_RE.finditer(text)]


def stable_hash(value):
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def shingles(tokens, width):
    if len(tokens) < width:
        return set()
    return {stable_hash("\x1f".join(tokens[index:index + width])) for index in range(len(tokens) - width + 1)}


def jaccard(left, right):
    union = len(left | right)
    return len(left & right) / union if union else 1.0


def source_fingerprint(report):
    """Hash stable source identities/content while excluding acquisition timestamps."""
    payload = {
        "corpus_id": report["corpus_id"],
        "registry_sha256": report["registry_sha256"],
        "documents": [
            {
                "id": document["id"],
                "raw_sha256": document["raw_sha256"],
                "normalized_sha256": document["normalized_sha256"],
                "status": document["status"],
                "text_path": document["text_path"],
            }
            for document in report["documents"]
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return digest(canonical)


def remove_illustration_blocks(text):
    """Remove reviewed bracketed blocks that begin with [Illustration."""
    output = []
    cursor = 0
    count = 0
    removed = 0
    pattern = re.compile(r"(?m)^\[Illustration(?:s)?\b", re.IGNORECASE)
    while match := pattern.search(text, cursor):
        end = text.find("]", match.end())
        if end < 0:
            raise ValueError("unclosed illustration block")
        end += 1
        output.append(text[cursor:match.start()])
        removed += end - match.start()
        count += 1
        cursor = end
    output.append(text[cursor:])
    return "".join(output), count, removed


def clean_document(text, actions):
    """Apply explicit ordered actions and return cleaned text plus an audit log."""
    current = text
    audit = []
    for rule in actions:
        before = len(current)
        action = rule["action"]
        count = 0
        if action == "drop_leading_paragraphs":
            paragraphs = current.split("\n\n")
            requested = rule["count"]
            if requested <= 0 or len(paragraphs) <= requested:
                raise ValueError("invalid leading paragraph count")
            current = "\n\n".join(paragraphs[requested:])
            count = requested
        elif action == "drop_leading_production_credit":
            paragraphs = current.split("\n\n")
            if not paragraphs or not paragraphs[0].casefold().startswith("produced by "):
                raise ValueError("expected leading production credit")
            current = "\n\n".join(paragraphs[1:])
            count = 1
        elif action == "drop_bracketed_illustration_blocks":
            current, count, _ = remove_illustration_blocks(current)
            if count == 0:
                raise ValueError("expected at least one illustration block")
        else:
            raise ValueError(f"unsupported cleaning action: {action}")
        current = re.sub(r"[ \t]+\n", "\n", current)
        current = re.sub(r"\n{3,}", "\n\n", current).strip() + "\n"
        audit.append({
            "action": action,
            "reason": rule["reason"],
            "matches_removed": count,
            "characters_removed": before - len(current),
        })
    if unicodedata.normalize("NFC", current) != current:
        raise ValueError("cleaning produced non-NFC text")
    return current, audit


def paragraph_records(documents, min_words, shingle_width):
    records = []
    for document in documents:
        for index, paragraph in enumerate(document["text"].split("\n\n")):
            tokens = word_tokens(paragraph)
            if len(tokens) < min_words:
                continue
            records.append({
                "document_id": document["id"],
                "paragraph_index": index,
                "word_count": len(tokens),
                "shingles": shingles(tokens, shingle_width),
                "sha256": digest(" ".join(tokens).encode("utf-8")),
            })
    return records


def paragraph_near_duplicates(documents, policy):
    records = paragraph_records(documents, policy["paragraph_min_words"], policy["paragraph_shingle_words"])
    postings = defaultdict(list)
    intersections = Counter()
    for index, record in enumerate(records):
        for shingle in record["shingles"]:
            for other in postings[shingle]:
                if records[other]["document_id"] == record["document_id"]:
                    continue
                left_size = len(records[other]["shingles"])
                right_size = len(record["shingles"])
                if min(left_size, right_size) / max(left_size, right_size) >= policy["paragraph_near_duplicate_jaccard"]:
                    intersections[(other, index)] += 1
            postings[shingle].append(index)
    matches = []
    for (left_index, right_index), intersection in sorted(intersections.items()):
        left, right = records[left_index], records[right_index]
        similarity = intersection / (len(left["shingles"]) + len(right["shingles"]) - intersection)
        if similarity < policy["paragraph_near_duplicate_jaccard"]:
            continue
        matches.append({
            "left_document_id": left["document_id"],
            "left_paragraph_index": left["paragraph_index"],
            "right_document_id": right["document_id"],
            "right_paragraph_index": right["paragraph_index"],
            "left_words": left["word_count"],
            "right_words": right["word_count"],
            "canonical_exact_match": left["sha256"] == right["sha256"],
            "jaccard": similarity,
        })
    return records, matches


def deduplicate_report(documents, policy):
    width = policy["document_shingle_words"]
    document_sets = {document["id"]: shingles(word_tokens(document["text"]), width) for document in documents}
    comparisons = []
    ids = sorted(document_sets)
    for offset, left in enumerate(ids):
        for right in ids[offset + 1:]:
            similarity = jaccard(document_sets[left], document_sets[right])
            comparisons.append({"left": left, "right": right, "jaccard": similarity})
    comparisons.sort(key=lambda row: (-row["jaccard"], row["left"], row["right"]))
    paragraph_rows, paragraph_matches = paragraph_near_duplicates(documents, policy)
    return {
        "comparison_tokenization": policy["tokenization"],
        "document_shingle_words": width,
        "document_pair_count": len(comparisons),
        "document_near_duplicate_threshold": policy["document_near_duplicate_jaccard"],
        "document_near_duplicates": [row for row in comparisons if row["jaccard"] >= policy["document_near_duplicate_jaccard"]],
        "top_document_pairs": comparisons[:policy["report_top_document_pairs"]],
        "paragraphs_screened": len(paragraph_rows),
        "paragraph_min_words": policy["paragraph_min_words"],
        "paragraph_candidate_method": "exhaustive shared-shingle inverted index with a necessary length-ratio bound",
        "paragraph_near_duplicate_threshold": policy["paragraph_near_duplicate_jaccard"],
        "cross_document_paragraph_near_duplicates": paragraph_matches,
        "limitations": [
            "This is a deterministic lexical screen, not proof that semantic duplicates or contamination are absent.",
            "Every cross-document paragraph pair that shares a shingle and can satisfy the threshold by length is checked with exact hashed-shingle Jaccard.",
            "Only cross-document paragraphs with the configured minimum word count are screened.",
        ],
    }


def verify_clean_snapshot(output, input_manifest_bytes, rules_bytes):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["input_manifest_sha256"] != digest(input_manifest_bytes):
        raise ValueError("input pilot manifest differs")
    if manifest["rules_sha256"] != digest(rules_bytes):
        raise ValueError("cleaning rules differ")
    expected = set(manifest["files"]) | {"manifest.json"}
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if expected != actual:
        raise ValueError("clean snapshot file inventory differs")
    for relative, expected_hash in manifest["files"].items():
        if digest(safe_file(output, relative).read_bytes()) != expected_hash:
            raise ValueError(f"clean snapshot checksum differs: {relative}")
    return manifest["report"]


def clean_pilot(registry_path, input_dir, rules_path, output_dir):
    """Clean an immutable pilot snapshot without opening any test split."""
    registry_path = Path(registry_path)
    input_dir = Path(input_dir)
    rules_path = Path(rules_path)
    output_dir = Path(output_dir)
    registry_bytes = registry_path.read_bytes()
    input_report = verify_snapshot(input_dir, registry_bytes)
    input_manifest_bytes = (input_dir / "manifest.json").read_bytes()
    rules_bytes = rules_path.read_bytes()
    rules = json.loads(rules_bytes)
    if rules["input_corpus_id"] != input_report["corpus_id"]:
        raise ValueError("rules target a different corpus")
    input_content_fingerprint = source_fingerprint(input_report)
    if rules["input_content_fingerprint"] != input_content_fingerprint:
        raise ValueError("rules target different input content")
    if rules["scope"] != "pilot-development-only; no train/validation/test assignment":
        raise ValueError("cleaning rules may not assign model splits")
    if any(rules["test_policy"].values()):
        raise ValueError("pilot cleaning may not create/read a test or make pilot families eligible")
    if output_dir.exists():
        return verify_clean_snapshot(output_dir, input_manifest_bytes, rules_bytes)

    extracted = [document for document in input_report["documents"] if document["text_path"]]
    known_ids = {document["id"] for document in extracted}
    if not set(rules["document_actions"]) <= known_ids:
        raise ValueError("rules reference a missing or quarantined document")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output_dir.name + ".staging-", dir=output_dir.parent) as directory:
        stage = Path(directory)
        files = {}

        def save(relative, raw):
            path = safe_file(stage, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            files[relative] = digest(raw)

        cleaned_documents = []
        analysis_documents = []
        for source in extracted:
            input_path = safe_file(input_dir, source["text_path"])
            raw = input_path.read_bytes()
            if digest(raw) != source["normalized_sha256"]:
                raise ValueError(f"input text checksum differs: {source['id']}")
            text = raw.decode("utf-8")
            actions = rules["document_actions"].get(source["id"], [])
            cleaned, audit = clean_document(text, actions)
            encoded = cleaned.encode("utf-8")
            relative = f"text/{source['id']}.txt"
            save(relative, encoded)
            cleaned_documents.append({
                "id": source["id"],
                "title": source["title"],
                "authors": source["authors"],
                "contributors": source["contributors"],
                "domain": source["domain"],
                "topic": source["topic"],
                "source_family": source["source_family"],
                "work_family": source["work_family"],
                "author_family": source["author_family"],
                "split_group": source["split_group"],
                "source_url": source["source_url"],
                "license": source["license"],
                "attribution": source["attribution"],
                "input_text_path": source["text_path"],
                "input_sha256": source["normalized_sha256"],
                "input_characters": len(text),
                "output_text_path": relative,
                "output_sha256": digest(encoded),
                "output_characters": len(cleaned),
                "characters_removed": len(text) - len(cleaned),
                "actions": audit,
                "partition": "pilot-development-only",
                "eligible_for_future_test": False,
            })
            analysis_documents.append({"id": source["id"], "text": cleaned})

        attribution = (input_dir / "ATTRIBUTION.json").read_bytes()
        save("ATTRIBUTION.json", attribution)
        deduplication = deduplicate_report(analysis_documents, rules["deduplication"])
        output_characters = sum(document["output_characters"] for document in cleaned_documents)
        report = {
            "corpus_id": rules["input_corpus_id"] + "-cleaned-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": rules["scope"],
            "input_manifest_sha256": digest(input_manifest_bytes),
            "input_content_fingerprint": input_content_fingerprint,
            "rules_id": rules["rules_id"],
            "rules_sha256": digest(rules_bytes),
            "document_count": len(cleaned_documents),
            "input_characters": sum(document["input_characters"] for document in cleaned_documents),
            "output_characters": output_characters,
            "characters_removed": sum(document["characters_removed"] for document in cleaned_documents),
            "documents_changed": sum(bool(document["actions"]) for document in cleaned_documents),
            "documents_unchanged": sum(not document["actions"] for document in cleaned_documents),
            "documents": cleaned_documents,
            "deduplication": deduplication,
            "attribution_path": "ATTRIBUTION.json",
            "attribution_sha256": digest(attribution),
            "quarantined_sources_carried_forward": [
                {"id": document["id"], "status": document["status"], "raw_sha256": document["raw_sha256"]}
                for document in input_report["documents"] if not document["text_path"]
            ],
            "split_status": "No train/validation/test assignments. Pilot families remain ineligible for the future untouched test.",
            "new_test_created": False,
            "new_test_read": False,
            "legacy_test_read": False,
            "model_loaded": False,
            "training_performed": False,
        }
        manifest = {
            "input_manifest_sha256": digest(input_manifest_bytes),
            "rules_sha256": digest(rules_bytes),
            "files": files,
            "report": report,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_clean_snapshot(stage, input_manifest_bytes, rules_bytes)
        stage.rename(output_dir)
    return report
