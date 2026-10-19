"""Cross-wave identity, balance, and lexical-overlap audit for corpus v2."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlparse

from corpus_cleaning import jaccard, paragraph_near_duplicates, shingles, word_tokens
from corpus_pilot import digest, safe_file


DEFAULT_POLICY = {
    "tokenization": "Unicode word tokens, casefolded; punctuation and whitespace ignored for comparison only",
    "document_shingle_words": 5,
    "document_near_duplicate_jaccard": 0.8,
    "paragraph_min_words": 30,
    "paragraph_shingle_words": 5,
    "paragraph_near_duplicate_jaccard": 0.8,
}


def normalized_identity(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).strip()


def canonical_publisher(value):
    value = re.sub(r"\s+", " ", value).strip()
    if value.startswith("Rod Library, University of Northern Iowa Funding for this project"):
        return "Rod Library, University of Northern Iowa"
    aliases = {
        "the open education network": "Open Education Network",
    }
    return aliases.get(normalized_identity(value), value)


def _load_stage(spec):
    report_path = Path(spec["report"])
    snapshot = Path(spec["snapshot"])
    report_raw = report_path.read_bytes()
    report = json.loads(report_raw)
    for key in ("new_test_read", "legacy_test_read", "model_loaded", "training_performed"):
        if report.get(key):
            raise ValueError(f"audit input used test or model state: {spec['id']}/{key}")
    documents = []
    for row in report["documents"]:
        relative = row[spec["text_path_key"]]
        raw = safe_file(snapshot, relative).read_bytes()
        if digest(raw) != row[spec["text_hash_key"]]:
            raise ValueError(f"audit text checksum differs: {spec['id']}/{row['id']}")
        text = raw.decode("utf-8")
        if len(text) != row[spec["character_key"]]:
            raise ValueError(f"audit character count differs: {spec['id']}/{row['id']}")
        authors = row.get("authors") or [row.get("organization") or spec["publisher_fallback"]]
        publisher = canonical_publisher(
            row.get("publisher") or row.get("journal") or row.get("organization") or spec["publisher_fallback"]
        )
        source_url = row["source_url"]
        documents.append({
            "id": f"{spec['id']}:{row['id']}",
            "local_id": row["id"],
            "stage": spec["id"],
            "title": row.get("document_title") or row.get("title"),
            "authors": authors,
            "publisher": publisher,
            "domain": row["domain"],
            "topic": row["topic"],
            "source_host": urlparse(source_url).netloc.casefold(),
            "work_family": row["work_family"],
            "split_group": row["split_group"],
            "characters": len(text),
            "sha256": digest(raw),
            "text": text,
        })
    return {
        "id": spec["id"],
        "report_path": spec["report"],
        "report_sha256": digest(report_raw),
        "document_count": len(documents),
        "characters": sum(row["characters"] for row in documents),
        "documents": documents,
    }


def _distribution(documents, key):
    counts, characters = Counter(), Counter()
    for row in documents:
        value = row[key]
        counts[value] += 1
        characters[value] += row["characters"]
    total = sum(characters.values())
    return [
        {
            key: value,
            "documents": counts[value],
            "characters": characters[value],
            "character_share": characters[value] / total,
        }
        for value in sorted(counts, key=lambda value: (-characters[value], value))
    ]


def audit_corpus(specs, policy=None):
    policy = dict(DEFAULT_POLICY if policy is None else policy)
    stages = [_load_stage(spec) for spec in specs]
    documents = [document for stage in stages for document in stage["documents"]]
    if len({row["id"] for row in documents}) != len(documents):
        raise ValueError("qualified document IDs are not unique")

    exact_groups = defaultdict(list)
    work_groups = defaultdict(list)
    split_groups = defaultdict(list)
    author_groups = defaultdict(list)
    for row in documents:
        exact_groups[row["sha256"]].append(row["id"])
        work_groups[normalized_identity(row["work_family"])].append(row["id"])
        split_groups[normalized_identity(row["split_group"])].append(row["id"])
        for author in row["authors"]:
            author_groups[normalized_identity(author)].append(row["id"])

    width = policy["document_shingle_words"]
    document_sets = {row["id"]: shingles(word_tokens(row["text"]), width) for row in documents}
    cross_pairs = []
    for left_index, left in enumerate(documents):
        for right in documents[left_index + 1:]:
            if left["stage"] == right["stage"]:
                continue
            cross_pairs.append({
                "left": left["id"],
                "right": right["id"],
                "jaccard": jaccard(document_sets[left["id"]], document_sets[right["id"]]),
            })
    cross_pairs.sort(key=lambda row: (-row["jaccard"], row["left"], row["right"]))

    paragraph_rows, paragraph_matches = paragraph_near_duplicates(documents, policy)
    stage_by_id = {row["id"]: row["stage"] for row in documents}
    cross_paragraph_matches = [
        row for row in paragraph_matches
        if stage_by_id[row["left_document_id"]] != stage_by_id[row["right_document_id"]]
    ]
    total_characters = sum(row["characters"] for row in documents)
    top_documents = sorted(documents, key=lambda row: (-row["characters"], row["id"]))[:10]
    publisher_distribution = _distribution(documents, "publisher")
    domain_distribution = _distribution(documents, "domain")
    host_distribution = _distribution(documents, "source_host")

    publisher_hhi = sum(row["character_share"] ** 2 for row in publisher_distribution)
    domain_hhi = sum(row["character_share"] ** 2 for row in domain_distribution)
    historical_characters = sum(row["characters"] for row in documents if row["stage"] == "pilot-cleaned")
    modern_characters = total_characters - historical_characters
    return {
        "schema_version": 1,
        "scope": "development-only corpus-v2 acquisition audit; no split assignment or test access",
        "inputs": [{key: stage[key] for key in ("id", "report_path", "report_sha256", "document_count", "characters")} for stage in stages],
        "totals": {
            "documents": len(documents),
            "characters": total_characters,
            "stages": len(stages),
            "normalized_authors_or_organizations": len(author_groups),
            "work_families": len(work_groups),
            "split_groups": len(split_groups),
            "publishers": len(publisher_distribution),
            "source_hosts": len(host_distribution),
            "domains": len(domain_distribution),
        },
        "era_balance": {
            "historical_pilot_characters": historical_characters,
            "modern_expansion_characters": modern_characters,
            "historical_pilot_share": historical_characters / total_characters,
            "modern_expansion_share": modern_characters / total_characters,
        },
        "stage_distribution": _distribution(documents, "stage"),
        "domain_distribution": domain_distribution,
        "publisher_distribution": publisher_distribution,
        "source_host_distribution": host_distribution,
        "concentration": {
            "publisher_character_hhi": publisher_hhi,
            "domain_character_hhi": domain_hhi,
            "largest_document_share": top_documents[0]["characters"] / total_characters,
            "largest_ten_document_share": sum(row["characters"] for row in top_documents) / total_characters,
            "top_documents": [{key: row[key] for key in ("id", "title", "domain", "publisher", "characters")} for row in top_documents],
        },
        "identity_audit": {
            "exact_content_duplicate_groups": [ids for ids in exact_groups.values() if len(ids) > 1],
            "cross_stage_work_family_collisions": [ids for ids in work_groups.values() if len({item.split(':', 1)[0] for item in ids}) > 1],
            "cross_stage_split_group_collisions": [ids for ids in split_groups.values() if len({item.split(':', 1)[0] for item in ids}) > 1],
            "cross_stage_author_collisions": [
                {"normalized_author": author, "documents": ids}
                for author, ids in sorted(author_groups.items())
                if len({item.split(':', 1)[0] for item in ids}) > 1
            ],
        },
        "cross_stage_lexical_screen": {
            "comparison_tokenization": policy["tokenization"],
            "document_shingle_words": width,
            "document_pairs_screened": len(cross_pairs),
            "document_near_duplicate_threshold": policy["document_near_duplicate_jaccard"],
            "document_near_duplicates": [row for row in cross_pairs if row["jaccard"] >= policy["document_near_duplicate_jaccard"]],
            "top_document_pairs": cross_pairs[:25],
            "paragraphs_screened_all_stages": len(paragraph_rows),
            "paragraph_min_words": policy["paragraph_min_words"],
            "paragraph_near_duplicate_threshold": policy["paragraph_near_duplicate_jaccard"],
            "cross_stage_paragraph_near_duplicates": cross_paragraph_matches,
        },
        "gaps": [
            "Historical public-domain books still dominate character volume.",
            "English-language material dominates every accepted stage.",
            "Modern institutional education and research-guidance prose is overrepresented.",
            "Current counts do not yet establish balanced train, validation, or test splits.",
        ],
        "decision": "Do not freeze splits. Acquire against measured era, genre, publisher, and language gaps, then rerun this audit before reserving a new test.",
        "new_test_created": False,
        "new_test_read": False,
        "legacy_test_read": False,
        "model_loaded": False,
        "training_performed": False,
    }

