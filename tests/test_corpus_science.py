import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from corpus_science import ingest_science, parse_plos_xml, validate_registry


def article_xml(index):
    doi = f"10.1371/journal.pcbi.{index:07d}"
    prose = " ".join(f"scientific term {index} explanation evidence method result" for _ in range(45))
    return f'''<?xml version="1.0" encoding="utf-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta>
<article-id pub-id-type="doi">{doi}</article-id>
<title-group><article-title>Article {index}</article-title></title-group>
<contrib-group><contrib contrib-type="author"><name><surname>Author{index}</surname><given-names>A.</given-names></name></contrib></contrib-group>
<pub-date pub-type="epub"><day>{index + 1}</day><month>2</month><year>2026</year></pub-date>
<permissions><copyright-year>2026</copyright-year><copyright-holder>Author{index}</copyright-holder>
<license xlink:href="http://creativecommons.org/licenses/by/4.0/"><license-p>This article uses the Creative Commons Attribution License.</license-p></license></permissions>
<abstract><p>Abstract {prose}</p></abstract></article-meta></front>
<body><sec><title>Rule {index}</title><p>{prose}</p>
<fig><caption><p>Figure text must be excluded.</p></caption></fig>
<table-wrap><table><tr><td>Table text must be excluded.</td></tr></table></table-wrap></sec>
<sec sec-type="supplementary-material"><title>Supporting information</title><p>Supplement must be excluded.</p></sec></body>
<back><ref-list><ref><mixed-citation>Reference must be excluded.</mixed-citation></ref></ref-list></back></article>'''.encode()


def fixture():
    documents, blobs = [], {}
    for index in range(10):
        raw = article_xml(index)
        doi = f"10.1371/journal.pcbi.{index:07d}"
        document_id = f"article-{index}"
        blobs[document_id + ".xml"] = raw
        documents.append({
            "id": document_id,
            "doi": doi,
            "title": f"Article {index}",
            "authors": [f"A. Author{index}"],
            "publication_date": f"2026-02-{index + 1:02d}",
            "journal": "PLOS Computational Biology",
            "language": "en",
            "domain": "science",
            "topic": f"topic-{index}",
            "source_url": f"https://journals.plos.org/ploscompbiol/article?id={doi}",
            "download_url": f"https://journals.plos.org/ploscompbiol/article/file?id={doi}&type=manuscript",
            "work_family": f"doi:{doi}",
            "author_family": f"authors:{doi}",
            "split_group": f"author-work:{doi}",
            "expected_sha256": hashlib.sha256(raw).hexdigest(),
            "license": {
                "identifier": "CC-BY-4.0",
                "holder": f"Author{index}",
                "terms_url": "https://creativecommons.org/licenses/by/4.0/",
                "evidence": "Article permissions element.",
                "verified_on": "2026-10-19",
            },
            "attribution": f"Article {index}, CC BY 4.0",
            "usage": "development-only; not assigned to train/validation/test",
        })
    registry = {
        "corpus_id": "test-science",
        "usage": "development-only; not assigned to train/validation/test",
        "test_policy": {
            "new_test_created": False,
            "new_test_read": False,
            "legacy_test_read": False,
            "sources_eligible_for_future_test": False,
        },
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


class CorpusScienceTests(unittest.TestCase):
    def test_jats_renderer_keeps_prose_and_verifies_article_license(self):
        metadata, rendered, audit = parse_plos_xml(article_xml(0))
        self.assertEqual(metadata["doi"], "10.1371/journal.pcbi.0000000")
        self.assertEqual(metadata["authors"], ["A. Author0"])
        self.assertEqual(metadata["publication_date"], "2026-02-01")
        self.assertIn("Abstract", rendered)
        self.assertIn("Rule 0", rendered)
        self.assertNotIn("Figure text", rendered)
        self.assertNotIn("Table text", rendered)
        self.assertNotIn("Supplement must", rendered)
        self.assertNotIn("Reference must", rendered)
        self.assertEqual(audit["figures_excluded"], 1)
        self.assertEqual(audit["tables_excluded"], 1)
        self.assertTrue(audit["references_excluded"])

    def test_ingestion_is_attributed_idempotent_and_test_free(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, blobs = fixture()
            source = root / "registry.json"
            source.write_text(json.dumps(registry))
            output = root / "science"
            report = ingest_science(source, output, lambda url, name: blobs[name])
            self.assertEqual(report["document_count"], 10)
            self.assertEqual(report["independent_split_groups"], 10)
            self.assertEqual(report["unique_authors"], 10)
            self.assertFalse(report["new_test_read"])
            self.assertFalse(report["model_loaded"])
            self.assertTrue((output / "ATTRIBUTION.json").is_file())
            before = {path.relative_to(output): path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}
            self.assertEqual(report, ingest_science(source, output, lambda *args: self.fail("must reuse offline")))
            self.assertEqual(before, {path.relative_to(output): path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()})
            first = report["documents"][0]
            (output / first["text_path"]).write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                ingest_science(source, output, lambda *args: self.fail("must reuse offline"))

    def test_registry_rejects_test_access_author_overlap_and_changed_license(self):
        registry, _ = fixture()
        changed = copy.deepcopy(registry)
        changed["test_policy"]["new_test_read"] = True
        with self.assertRaisesRegex(ValueError, "test"):
            validate_registry(changed)
        changed = copy.deepcopy(registry)
        changed["documents"][1]["authors"] = changed["documents"][0]["authors"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_registry(changed)
        changed = copy.deepcopy(registry)
        changed["documents"][0]["license"]["identifier"] = "unknown"
        with self.assertRaisesRegex(ValueError, "license"):
            validate_registry(changed)


if __name__ == "__main__":
    unittest.main()
