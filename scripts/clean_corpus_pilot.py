"""Clean and screen the reviewed development-only corpus-v2 pilot."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus_cleaning import clean_pilot
from corpus_pilot import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/corpus_v2_sources.json"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/corpus-v2-pilot"))
    parser.add_argument("--rules", type=Path, default=Path("configs/corpus_v2_cleaning_rules.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/corpus-v2-pilot-cleaned"))
    parser.add_argument("--report", type=Path, default=Path("docs/results/corpus-v2-cleaning-and-dedup.json"))
    args = parser.parse_args()

    report = clean_pilot(args.registry, args.input_dir, args.rules, args.output_dir)
    report = dict(report, implementation_sha256={
        path: digest(Path(path).read_bytes())
        for path in ("scripts/clean_corpus_pilot.py", "src/corpus_cleaning.py")
    })
    serialized = json.dumps(report, indent=2) + "\n"
    if args.report.exists():
        if args.report.read_text() != serialized:
            raise ValueError("report differs; choose a new --report path")
    else:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    summary = {
        key: report[key]
        for key in ("document_count", "input_characters", "output_characters", "characters_removed", "documents_changed")
    }
    summary.update({
        "document_near_duplicates": len(report["deduplication"]["document_near_duplicates"]),
        "paragraph_near_duplicates": len(report["deduplication"]["cross_document_paragraph_near_duplicates"]),
        "new_test_read": report["new_test_read"],
        "model_loaded": report["model_loaded"],
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
