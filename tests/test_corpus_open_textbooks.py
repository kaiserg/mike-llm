import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from corpus_open_textbooks import (
    _parse_row,
    ingest_open_textbooks,
    parse_manifold,
    render_reviewed_html,
    validate_registry,
)


def fixture():
    documents, blobs = [], {}
    for index in range(10):
        document_id = f"textbook-{index}"
        title = f"Reviewed chapter {index}"
        author = f"Author {index}"
        prose = " ".join(
            f"topic{index} evidence explanation method example conclusion" for _ in range(45)
        )
        license_record = {
            "name": f"Open textbook {index}",
            "author": [{"name": author}],
            "editor": [],
            "publisher": {"name": f"Publisher {index}"},
            "datePublished": "2026-01-01",
            "copyrightYear": 2026,
            "license": {
                "name": "CC BY 4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "description": "",
            },
        }
        content_record = {
            "title": {"rendered": title},
            "content": {
                "rendered": (
                    f"<h1>{title}</h1><p>{prose}</p>"
                    "<figure><img src='x'><figcaption>Excluded caption</figcaption></figure>"
                    "<h2>References</h2><p>Excluded citation</p>"
                )
            },
            "modified_gmt": "2026-01-02T03:04:05",
            "link": f"https://books.example/book-{index}/chapter-{index}/",
        }
        license_raw = json.dumps(license_record).encode()
        content_raw = json.dumps(content_record).encode()
        blobs[document_id + "-license.json"] = license_raw
        blobs[document_id + "-content.json"] = content_raw
        work_id = f"books.example/book-{index}"
        documents.append({
            "id": document_id,
            "provider": "pressbooks-v2",
            "work_id": work_id,
            "work_title": f"Open textbook {index}",
            "document_title": title,
            "authors": [author],
            "publisher": f"Publisher {index}",
            "language": "en",
            "domain": f"domain-{index}",
            "topic": f"topic-{index}",
            "source_url": content_record["link"],
            "download_url": f"https://books.example/api/chapters/{index}",
            "license_evidence_url": f"https://books.example/api/books/{index}",
            "revision": content_record["modified_gmt"],
            "work_family": f"work:pressbooks-v2:{work_id}",
            "author_family": f"authors:author-{index}",
            "split_group": f"work:pressbooks-v2:{work_id}",
            "expected_sha256": hashlib.sha256(content_raw).hexdigest(),
            "license_expected_sha256": hashlib.sha256(license_raw).hexdigest(),
            "license": {
                "identifier": "CC-BY-4.0",
                "terms_url": "https://creativecommons.org/licenses/by/4.0/",
                "evidence": "Edition metadata API record.",
            },
            "attribution": f"{title}, {author}, CC BY 4.0",
            "usage": "development-only; not assigned to train/validation/test",
        })
    registry = {
        "corpus_id": "test-open-textbooks",
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


class CorpusOpenTextbookTests(unittest.TestCase):
    def test_html_renderer_keeps_text_code_tables_and_inequalities(self):
        source = """
        <h1>Example</h1>
        <p>Keep this explanation where 0 &lt; 1, 2 &gt; 1, and &lt;v 1, v 2&gt; is a tuple.</p>
        <pre>for item in values: print(item)</pre>
        <table><tr><th>Term</th><th>Meaning</th></tr><tr><td>Q</td><td>query</td></tr></table>
        <img src="standalone.png" alt="excluded">
        <figure><img src="figure.png"/><figcaption>Excluded figure caption.</figcaption></figure>
        <iframe src="video.example"></iframe>
        <div class="footnotes"><p>Excluded footnote.</p></div>
        <h2>Works Cited</h2><p>Excluded reference.</p>
        """
        rendered, audit = render_reviewed_html("Example", source)
        self.assertIn("0 < 1, 2 > 1", rendered)
        self.assertIn("<v 1, v 2>", rendered)
        self.assertIn("for item in values", rendered)
        self.assertIn("Term", rendered)
        self.assertIn("query", rendered)
        self.assertNotIn("Excluded figure", rendered)
        self.assertNotIn("Excluded footnote", rendered)
        self.assertNotIn("Excluded reference", rendered)
        self.assertEqual(audit["images_excluded"], 1)
        self.assertEqual(audit["figures_excluded_with_captions"], 1)
        self.assertEqual(audit["media_embeds_excluded"], 1)
        self.assertTrue(audit["references_section_excluded"])
        self.assertEqual(audit["html_tags_remaining"], 0)

    def test_manifold_parser_reads_rights_and_excludes_media(self):
        project = {"data": {"attributes": {
            "titlePlaintext": "Open work",
            "updatedAt": "2026-01-02T03:04:05Z",
            "metadata": {
                "publisher": "<p>Open Publisher</p>",
                "rightsHolder": "<p>Ada Author</p>",
                "rights": "<p>Open work by Ada Author is licensed CC BY 4.0.</p>",
            },
        }}}
        section = {"data": {"attributes": {
            "name": "Open chapter",
            "body": "<h1>Open chapter</h1><p>Useful prose.</p><video>Excluded media.</video>",
        }}}
        parsed = parse_manifold(json.dumps(project).encode(), json.dumps(section).encode())
        self.assertEqual(parsed["work_title"], "Open work")
        self.assertEqual(parsed["publisher"], " Open Publisher ".strip())
        self.assertEqual(parsed["creator_names"], "")
        self.assertIn("Useful prose", parsed["rendered"])
        self.assertNotIn("Excluded media", parsed["rendered"])
        self.assertEqual(parsed["audit"]["media_embeds_excluded"], 1)

    def test_manifold_accepts_creator_or_content_author_evidence(self):
        project = {"data": {"attributes": {
            "titlePlaintext": "Open work",
            "updatedAt": "2026-01-02T03:04:05Z",
            "creatorNames": "Ada Author",
            "metadata": {
                "publisher": "Open Publisher",
                "rightsHolder": "",
                "rights": "Creative Commons Attribution 4.0 International License",
            },
        }}}
        section = {"data": {"attributes": {
            "name": "Open chapter",
            "body": "<h1>Open chapter</h1><p>Written by Bea Researcher.</p>",
        }}}
        base = {
            "id": "open-chapter", "provider": "manifold-v1", "work_title": "Open work",
            "document_title": "Open chapter", "publisher": "Open Publisher",
            "revision": "2026-01-02T03:04:05Z", "authors": ["Ada Author"],
        }
        _parse_row(base, json.dumps(project).encode(), json.dumps(section).encode())
        content_author = dict(base, authors=["Bea Researcher"], author_evidence="content")
        _parse_row(content_author, json.dumps(project).encode(), json.dumps(section).encode())

    def test_ingestion_is_pinned_attributed_idempotent_and_test_free(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, blobs = fixture()
            source = root / "registry.json"
            source.write_text(json.dumps(registry))
            output = root / "open-textbooks"
            report = ingest_open_textbooks(source, output, lambda url, name: blobs[name])
            self.assertEqual(report["document_count"], 10)
            self.assertEqual(report["independent_split_groups"], 10)
            self.assertEqual(report["unique_authors"], 10)
            self.assertFalse(report["new_test_read"])
            self.assertFalse(report["model_loaded"])
            self.assertTrue((output / "ATTRIBUTION.json").is_file())
            before = {path.relative_to(output): path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}
            self.assertEqual(report, ingest_open_textbooks(source, output, lambda *args: self.fail("must reuse offline")))
            self.assertEqual(before, {path.relative_to(output): path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()})
            first = report["documents"][0]
            (output / first["text_path"]).write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                ingest_open_textbooks(source, output, lambda *args: self.fail("must reuse offline"))

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
