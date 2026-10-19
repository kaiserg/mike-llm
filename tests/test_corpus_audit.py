import json
from pathlib import Path
import tempfile
import unittest

from corpus_audit import audit_corpus, canonical_publisher
from corpus_pilot import digest


def make_stage(root, stage_id, text, author, work_family):
    snapshot = root / stage_id
    (snapshot / "text").mkdir(parents=True)
    raw = text.encode()
    (snapshot / "text" / "doc.txt").write_bytes(raw)
    report = {
        "new_test_read": False,
        "legacy_test_read": False,
        "model_loaded": False,
        "training_performed": False,
        "documents": [{
            "id": "doc", "title": f"Document {stage_id}", "authors": [author],
            "publisher": f"Publisher {stage_id}", "domain": "example", "topic": stage_id,
            "source_url": f"https://{stage_id}.example/document", "work_family": work_family,
            "split_group": work_family, "text_path": "text/doc.txt",
            "text_sha256": digest(raw), "characters": len(text),
        }],
    }
    report_path = root / f"{stage_id}.json"
    report_path.write_text(json.dumps(report))
    return {
        "id": stage_id, "report": str(report_path), "snapshot": str(snapshot),
        "text_path_key": "text_path", "text_hash_key": "text_sha256",
        "character_key": "characters", "publisher_fallback": "Fallback",
    }


class CorpusAuditTests(unittest.TestCase):
    def test_cross_stage_identity_balance_and_overlap_are_reported(self):
        text = " ".join(f"shared word sequence number {index}" for index in range(40)) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = [
                make_stage(root, "stage-a", text, "Ada Author", "work:shared"),
                make_stage(root, "stage-b", text, "Ada Author", "work:shared"),
            ]
            report = audit_corpus(specs)
            self.assertEqual(report["totals"]["documents"], 2)
            self.assertEqual(len(report["identity_audit"]["exact_content_duplicate_groups"]), 1)
            self.assertEqual(len(report["identity_audit"]["cross_stage_work_family_collisions"]), 1)
            self.assertEqual(len(report["identity_audit"]["cross_stage_author_collisions"]), 1)
            self.assertEqual(len(report["cross_stage_lexical_screen"]["document_near_duplicates"]), 1)
            self.assertEqual(len(report["cross_stage_lexical_screen"]["cross_stage_paragraph_near_duplicates"]), 1)
            self.assertFalse(report["new_test_read"])

    def test_changed_snapshot_fails_closed(self):
        text = "enough original text for a checksum"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = make_stage(root, "stage-a", text, "Ada Author", "work:a")
            (Path(spec["snapshot"]) / "text" / "doc.txt").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                audit_corpus([spec])

    def test_publisher_aliases_are_canonicalized(self):
        self.assertEqual(canonical_publisher("The Open Education Network"), "Open Education Network")
        self.assertEqual(
            canonical_publisher("Rod Library, University of Northern Iowa  Funding for this project was provided"),
            "Rod Library, University of Northern Iowa",
        )


if __name__ == "__main__":
    unittest.main()
