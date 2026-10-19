import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from corpus_cleaning import clean_document, clean_pilot, deduplicate_report, source_fingerprint
from corpus_pilot import ingest_pilot


def pilot_fixture():
    documents = []
    blobs = {"gutenberg-license.html": b"Project Gutenberg license notice"}
    for number in range(3000, 3010):
        document_id = f"pg{number}"
        title = f"Book {number}"
        author = f"Writer {number}"
        prefix = "Produced by Example Volunteer\n\n" if number == 3000 else ""
        body = (f"Chapter {number}: Évidence, punctuation! A distinct story with enough words for testing.\n" * 30)
        raw = (f"Notice\n*** START OF THE PROJECT GUTENBERG EBOOK {title} ***\n" + prefix + body
               + f"*** END OF THE PROJECT GUTENBERG EBOOK {title} ***\nLicense").encode()
        metadata = ("<table>" + "".join(
            f"<tr><th>{key}</th><td>{value}</td></tr>"
            for key, value in [
                ("Title", title), ("Author", author),
                ("Copyright", "Public domain in the USA."), ("Language", "English"),
            ]
        ) + "</table>").encode()
        blobs[document_id + ".raw"] = raw
        blobs[document_id + ".metadata.html"] = metadata
        documents.append({
            "id": document_id,
            "title": title,
            "authors": [author],
            "contributors": {},
            "language": "en",
            "domain": "fiction",
            "topic": "story",
            "source_family": "project-gutenberg",
            "source_url": f"https://www.gutenberg.org/ebooks/{number}",
            "download_url": f"https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt",
            "format": "gutenberg-utf8",
            "revision": {"edition": number},
            "work_family": document_id,
            "author_family": author,
            "split_group": author,
            "expected_sha256": hashlib.sha256(raw).hexdigest(),
            "license": {"identifier": "LicenseRef-PG-PD-US"},
            "attribution": f"{title} by {author}",
            "usage": "development-ingestion-pilot-only; not assigned to train/validation/test",
        })
    registry = {
        "corpus_id": "test-pilot",
        "documents": documents,
        "evidence": [{
            "id": "gutenberg-license",
            "url": "https://www.gutenberg.org/policy/license",
            "cache_name": "gutenberg-license.html",
        }],
    }
    return registry, blobs


def rules(input_content_fingerprint):
    return {
        "rules_id": "test-cleaning-v1",
        "input_corpus_id": "test-pilot",
        "input_content_fingerprint": input_content_fingerprint,
        "scope": "pilot-development-only; no train/validation/test assignment",
        "document_actions": {
            "pg3000": [{
                "action": "drop_leading_production_credit",
                "reason": "credit remains in provenance",
            }],
        },
        "deduplication": {
            "tokenization": "test tokens",
            "document_shingle_words": 5,
            "document_near_duplicate_jaccard": 0.8,
            "paragraph_min_words": 30,
            "paragraph_shingle_words": 5,
            "paragraph_near_duplicate_jaccard": 0.8,
            "report_top_document_pairs": 10,
        },
        "test_policy": {
            "new_test_created": False,
            "new_test_read": False,
            "legacy_test_read": False,
            "pilot_families_eligible_for_future_test": False,
        },
    }


class CorpusCleaningTests(unittest.TestCase):
    def test_explicit_cleaning_actions_preserve_unicode_and_structure(self):
        source = ("Produced by Example\n\nTitle\n\n[Illustration: A missing image\nwith a caption.]\n\n"
                  "Évidence remains.\n")
        actions = [
            {"action": "drop_leading_production_credit", "reason": "credit"},
            {"action": "drop_bracketed_illustration_blocks", "reason": "missing image"},
        ]
        cleaned, audit = clean_document(source, actions)
        self.assertEqual(cleaned, "Title\n\nÉvidence remains.\n")
        self.assertEqual([row["matches_removed"] for row in audit], [1, 1])

    def test_near_duplicate_screen_confirms_similar_cross_document_paragraphs(self):
        base = " ".join(f"word{index}" for index in range(80))
        changed = base.replace("word40", "replacement")
        documents = [{"id": "a", "text": base}, {"id": "b", "text": changed}]
        policy = rules("0" * 64)["deduplication"]
        report = deduplicate_report(documents, policy)
        self.assertEqual(len(report["document_near_duplicates"]), 1)
        self.assertEqual(len(report["cross_document_paragraph_near_duplicates"]), 1)
        self.assertFalse(report["cross_document_paragraph_near_duplicates"][0]["canonical_exact_match"])

    def test_clean_snapshot_is_reproducible_provenance_complete_and_test_free(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, blobs = pilot_fixture()
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry))
            input_dir = root / "raw"
            input_report = ingest_pilot(registry_path, input_dir, lambda url, name: blobs[name])
            changed_timestamps = copy.deepcopy(input_report)
            changed_timestamps["created_at_utc"] = "different"
            for document in changed_timestamps["documents"]:
                document["ingested_at_utc"] = "different"
            self.assertEqual(source_fingerprint(input_report), source_fingerprint(changed_timestamps))
            rules_path = root / "rules.json"
            rules_path.write_text(json.dumps(rules(source_fingerprint(input_report))))
            output_dir = root / "clean"
            report = clean_pilot(registry_path, input_dir, rules_path, output_dir)
            self.assertEqual(report["document_count"], 10)
            self.assertFalse(report["new_test_read"])
            self.assertFalse(report["model_loaded"])
            changed = next(row for row in report["documents"] if row["id"] == "pg3000")
            self.assertNotIn("Produced by", (output_dir / changed["output_text_path"]).read_text())
            self.assertEqual((output_dir / "ATTRIBUTION.json").read_bytes(), (input_dir / "ATTRIBUTION.json").read_bytes())
            before = {path.relative_to(output_dir): path.stat().st_mtime_ns for path in output_dir.rglob("*") if path.is_file()}
            self.assertEqual(report, clean_pilot(registry_path, input_dir, rules_path, output_dir))
            self.assertEqual(before, {path.relative_to(output_dir): path.stat().st_mtime_ns for path in output_dir.rglob("*") if path.is_file()})
            (output_dir / changed["output_text_path"]).write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                clean_pilot(registry_path, input_dir, rules_path, output_dir)


if __name__ == "__main__":
    unittest.main()
