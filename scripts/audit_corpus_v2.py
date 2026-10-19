"""Audit identities, balance, and cross-wave overlap in corpus v2."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus_audit import audit_corpus
from corpus_pilot import digest


SPECS = [
    {"id": "pilot-cleaned", "report": "docs/results/corpus-v2-cleaning-and-dedup.json", "snapshot": "data/processed/corpus-v2-pilot-cleaned", "text_path_key": "output_text_path", "text_hash_key": "output_sha256", "character_key": "output_characters", "publisher_fallback": "Project Gutenberg"},
    {"id": "wave1-education", "report": "docs/results/corpus-v2-expansion-wave1.json", "snapshot": "data/raw/corpus-v2-expansion-wave1", "text_path_key": "text_path", "text_hash_key": "rendered_sha256", "character_key": "rendered_characters", "publisher_fallback": "The Carpentries"},
    {"id": "wave2-science", "report": "docs/results/corpus-v2-science-wave2.json", "snapshot": "data/raw/corpus-v2-science-wave2", "text_path_key": "text_path", "text_hash_key": "rendered_sha256", "character_key": "rendered_characters", "publisher_fallback": "PLOS Computational Biology"},
    {"id": "wave3-textbooks", "report": "docs/results/corpus-v2-open-textbooks-wave3.json", "snapshot": "data/raw/corpus-v2-open-textbooks-wave3", "text_path_key": "text_path", "text_hash_key": "rendered_sha256", "character_key": "rendered_characters", "publisher_fallback": "Unspecified open-textbook publisher"},
    {"id": "wave4-independent", "report": "docs/results/corpus-v2-independent-publishers-wave4.json", "snapshot": "data/raw/corpus-v2-independent-publishers-wave4", "text_path_key": "text_path", "text_hash_key": "rendered_sha256", "character_key": "rendered_characters", "publisher_fallback": "Unspecified independent publisher"},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/results/corpus-v2-cross-wave-audit.json"))
    args = parser.parse_args()
    report = audit_corpus(SPECS)
    report["implementation_sha256"] = {
        path: digest(Path(path).read_bytes())
        for path in ("scripts/audit_corpus_v2.py", "src/corpus_audit.py", "src/corpus_cleaning.py")
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output.exists() and args.output.read_text() != serialized:
        raise ValueError("audit report differs; inspect inputs or choose a new --output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    print(json.dumps({
        "documents": report["totals"]["documents"],
        "characters": report["totals"]["characters"],
        "publishers": report["totals"]["publishers"],
        "domains": report["totals"]["domains"],
        "cross_stage_document_near_duplicates": len(report["cross_stage_lexical_screen"]["document_near_duplicates"]),
        "cross_stage_paragraph_near_duplicates": len(report["cross_stage_lexical_screen"]["cross_stage_paragraph_near_duplicates"]),
        "new_test_read": report["new_test_read"],
        "model_loaded": report["model_loaded"],
    }, indent=2))


if __name__ == "__main__":
    main()
