import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from corpus_expansion import ingest_expansion, render_carpentries_markdown, validate_registry


def fixture():
    revision = "a" * 40
    repository = "example/lesson"
    license_raw = (b"All instructional material is available under the Creative Commons Attribution license.\n"
                   b"https://creativecommons.org/licenses/by/4.0/\n")
    evidence = [{
        "id": "example-license",
        "repository": repository,
        "commit": revision,
        "url": f"https://raw.githubusercontent.com/{repository}/{revision}/LICENSE.md",
        "cache_name": "license.md",
        "expected_sha256": hashlib.sha256(license_raw).hexdigest(),
    }]
    blobs = {"license.md": license_raw}
    documents = []
    for index in range(10):
        document_id = f"lesson-{index}"
        title = f"Lesson {index}"
        body = (f"This is educational prose for lesson {index}, with a distinct repeated explanation and Évidence.\n\n" * 20)
        raw = (f"---\ntitle: {title}\nteaching: 10\n---\n\n"
               "::::::::::::::::::::::::::::::::::::::: objectives\n\n- Learn carefully.\n\n"
               "::::::::::::::::::::::::::::::::::::::::::::::::::\n\n" + body).encode()
        blobs[document_id + ".md"] = raw
        path = f"episodes/{index}.md"
        documents.append({
            "id": document_id,
            "title": title,
            "organization": "Example contributors",
            "language": "en",
            "domain": "education",
            "topic": f"topic-{index}",
            "repository": repository,
            "path": path,
            "commit": revision,
            "commit_date": "2026-01-01T00:00:00Z",
            "source_url": f"https://github.com/{repository}/blob/{revision}/{path}",
            "download_url": f"https://raw.githubusercontent.com/{repository}/{revision}/{path}",
            "history_url": f"https://github.com/{repository}/commits/{revision}/{path}",
            "work_family": f"repository:{repository}",
            "author_family": "organization:example",
            "split_group": "source-family:example",
            "expected_sha256": hashlib.sha256(raw).hexdigest(),
            "license_evidence_id": "example-license",
            "license": {"identifier": "CC-BY-4.0"},
            "attribution": f"{title}, Example contributors, CC BY 4.0",
            "usage": "development-only; not assigned to train/validation/test",
        })
    registry = {
        "corpus_id": "test-expansion",
        "usage": "development-only; not assigned to train/validation/test",
        "test_policy": {
            "new_test_created": False,
            "new_test_read": False,
            "legacy_test_read": False,
            "sources_eligible_for_future_test": False,
        },
        "evidence": evidence,
        "documents": documents,
        "deduplication": {
            "tokenization": "test words",
            "document_shingle_words": 5,
            "document_near_duplicate_jaccard": 0.8,
            "paragraph_min_words": 30,
            "paragraph_shingle_words": 5,
            "paragraph_near_duplicate_jaccard": 0.8,
            "report_top_document_pairs": 10,
        },
    }
    return registry, blobs


class CorpusExpansionTests(unittest.TestCase):
    def test_renderer_preserves_code_and_prose_while_resolving_reviewed_markup(self):
        source = """---
title: Example
teaching: 10
---

::::::::::::::::::::::::::::::::::::::: objectives

## A [linked heading](https://example.com) {#anchor}

```python
print("Évidence")
```

![](figure.svg){alt='A useful diagram'}

![alt='A second diagram' .output](second.svg)
{alt='duplicate image metadata' .output}

<table><tr><td>Value</td><td>Meaning</td></tr></table>

::::::::::::::::::::::::::::::::::::::::::::::::::
"""
        metadata, rendered, audit = render_carpentries_markdown(source)
        self.assertEqual(metadata["title"], "Example")
        self.assertIn("A linked heading", rendered)
        self.assertIn('print("Évidence")', rendered)
        self.assertIn("Image: A useful diagram", rendered)
        self.assertIn("Image: A second diagram", rendered)
        self.assertNotIn("duplicate image metadata", rendered)
        self.assertIn("Value", rendered)
        self.assertFalse(any(audit["unresolved_markup"].values()))

    def test_ingestion_is_pinned_attributed_idempotent_and_test_free(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, blobs = fixture()
            source = root / "registry.json"
            source.write_text(json.dumps(registry))
            output = root / "expansion"
            report = ingest_expansion(source, output, lambda url, name: blobs[name])
            self.assertEqual(report["document_count"], 10)
            self.assertEqual(report["independent_split_groups"], 1)
            self.assertFalse(report["new_test_read"])
            self.assertFalse(report["model_loaded"])
            self.assertTrue((output / "ATTRIBUTION.json").is_file())
            before = {path.relative_to(output): path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}
            self.assertEqual(report, ingest_expansion(source, output, lambda *args: self.fail("must reuse offline")))
            self.assertEqual(before, {path.relative_to(output): path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()})
            first = report["documents"][0]
            (output / first["text_path"]).write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                ingest_expansion(source, output, lambda *args: self.fail("must reuse offline"))

    def test_registry_rejects_test_access_and_changed_license(self):
        registry, _ = fixture()
        changed = copy.deepcopy(registry)
        changed["test_policy"]["new_test_read"] = True
        with self.assertRaisesRegex(ValueError, "test"):
            validate_registry(changed)
        changed = copy.deepcopy(registry)
        changed["documents"][0]["license"]["identifier"] = "unknown"
        with self.assertRaisesRegex(ValueError, "license"):
            validate_registry(changed)


if __name__ == "__main__":
    unittest.main()
