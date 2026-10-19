import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from corpus_longform import ingest_longform, render_markdown_collection, validate_registry


def fixture():
    commit = "a" * 40
    repository = "example/longform"
    license_raw = b"Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
    blobs = {"license.md": license_raw}
    evidence = [{
        "id": "example-license", "repository": repository, "commit": commit,
        "url": f"https://raw.githubusercontent.com/{repository}/{commit}/LICENSE.md",
        "cache_name": "license.md", "expected_sha256": hashlib.sha256(license_raw).hexdigest(),
        "license_identifier": "CC-BY-4.0",
    }]
    documents = []
    for index in range(10):
        source = (f"---\ntitle: Collection {index}\n---\n\n# Section\n\n"
                  + (f"Distinct long-form paragraph {index} explains a practical decision with evidence and context. " * 70)
                  + "\n").encode()
        cache_name = f"collection-{index}-00.md"
        blobs[cache_name] = source
        path = f"guides/{index}.md"
        documents.append({
            "id": f"collection-{index}", "title": f"Collection {index}", "authors": [f"Author {index}"],
            "publisher": f"Publisher {index}", "language": "en", "domain": f"domain-{index}",
            "topic": f"topic-{index}", "repository": repository,
            "source_url": f"https://github.com/{repository}/tree/{commit}", "commit": commit,
            "commit_date": "2026-01-01T00:00:00Z", "work_family": f"work:{index}",
            "author_family": f"author:{index}", "split_group": f"work:{index}",
            "license_evidence_id": "example-license",
            "license": {"identifier": "CC-BY-4.0", "terms_url": "https://creativecommons.org/licenses/by/4.0/"},
            "attribution": f"Collection {index}, Author {index}, CC BY 4.0",
            "usage": "development-only; not assigned to train/validation/test",
            "sources": [{
                "path": path,
                "download_url": f"https://raw.githubusercontent.com/{repository}/{commit}/{path}",
                "expected_sha256": hashlib.sha256(source).hexdigest(),
            }],
        })
    registry = {
        "corpus_id": "test-longform", "usage": "development-only; not assigned to train/validation/test",
        "test_policy": {"new_test_created": False, "new_test_read": False, "legacy_test_read": False, "sources_eligible_for_future_test": False},
        "evidence": evidence, "documents": documents,
        "deduplication": {"tokenization": "test words", "document_shingle_words": 5,
                          "document_near_duplicate_jaccard": 0.8, "paragraph_min_words": 30,
                          "paragraph_shingle_words": 5, "paragraph_near_duplicate_jaccard": 0.8,
                          "report_top_document_pairs": 10},
    }
    return registry, blobs


class CorpusLongformTests(unittest.TestCase):
    def test_renderer_combines_sources_and_removes_repository_markup(self):
        rendered, audit = render_markdown_collection("A collection", [
            ("one.md", "---\ntitle: One\n---\n# Heading\n\n[Useful prose](https://example.com).\n\n```python\nprint('kept')\n```\n"),
            ("two.md", "<!-- note -->\n(er)=\n{:toc}\n![Diagram](image.png)\n\n{% include thing %}\n*More text.*\n"),
        ])
        self.assertTrue(rendered.startswith("A collection\n\nHeading"))
        self.assertIn("Useful prose", rendered)
        self.assertIn("print('kept')", rendered)
        self.assertIn("Image: Diagram", rendered)
        self.assertIn("More text.", rendered)
        self.assertNotIn("{:toc}", rendered)
        self.assertNotIn("(er)=", rendered)
        self.assertNotIn("https://example.com", rendered)
        self.assertEqual(audit["source_files"], 2)
        self.assertEqual(audit["remaining_template_markers"], 0)

    def test_ingestion_is_pinned_idempotent_and_test_free(self):
        registry, blobs = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry))
            output = root / "snapshot"
            report = ingest_longform(registry_path, output, lambda _url, name: blobs[name])
            self.assertEqual(report["document_count"], 10)
            self.assertEqual(report["publisher_count"], 10)
            self.assertFalse(report["new_test_read"])
            self.assertFalse(report["model_loaded"])
            before = {path.relative_to(output): path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}
            self.assertEqual(report, ingest_longform(registry_path, output, lambda *_: self.fail("must reuse offline")))
            self.assertEqual(before, {path.relative_to(output): path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()})
            first = report["documents"][0]
            (output / first["text_path"]).write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                ingest_longform(registry_path, output, lambda *_: self.fail("must reuse offline"))

    def test_registry_rejects_test_access_unpinned_paths_and_license_changes(self):
        registry, _ = fixture()
        changed = copy.deepcopy(registry)
        changed["test_policy"]["new_test_read"] = True
        with self.assertRaisesRegex(ValueError, "test"):
            validate_registry(changed)
        changed = copy.deepcopy(registry)
        changed["documents"][0]["sources"][0]["path"] = "../secret.md"
        with self.assertRaisesRegex(ValueError, "source"):
            validate_registry(changed)
        changed = copy.deepcopy(registry)
        changed["documents"][0]["license"]["identifier"] = "unknown"
        with self.assertRaisesRegex(ValueError, "license"):
            validate_registry(changed)


if __name__ == "__main__":
    unittest.main()
