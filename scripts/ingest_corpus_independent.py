"""Ingest the pinned development-only independent-publisher expansion wave."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus_open_textbooks import ingest_open_textbooks
from corpus_pilot import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("configs/corpus_v2_independent_sources.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/corpus-v2-independent-publishers-wave4"))
    parser.add_argument("--report", type=Path, default=Path("docs/results/corpus-v2-independent-publishers-wave4.json"))
    parser.add_argument("--cache-dir", type=Path, help="optional reviewed local JSON cache")
    args = parser.parse_args()

    def fetch(url, name):
        if args.cache_dir and (args.cache_dir / name).is_file():
            return (args.cache_dir / name).read_bytes()
        print("Retrieving:", url, flush=True)
        response = subprocess.run([
            "curl", "--silent", "--show-error", "--fail", "--location",
            "--user-agent", "mike-llm-corpus-ingestion", "--header", "Accept: application/json",
            "--proto", "=https", "--proto-redir", "=https", "--max-time", "60",
            "--max-filesize", "2000000", url,
        ], check=True, capture_output=True)
        time.sleep(0.25)
        return response.stdout

    report = ingest_open_textbooks(args.sources, args.output_dir, fetch)
    report = dict(report, implementation_sha256={
        path: digest(Path(path).read_bytes())
        for path in (
            "scripts/ingest_corpus_independent.py", "src/corpus_open_textbooks.py", "src/corpus_cleaning.py"
        )
    })
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report.exists():
        if args.report.read_text() != serialized:
            raise ValueError("report differs; choose a new --report path")
    else:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    print(json.dumps({
        "documents": report["document_count"],
        "independent_groups": report["independent_split_groups"],
        "publishers": report["publisher_count"],
        "characters": report["rendered_characters"],
        "document_near_duplicates": len(report["deduplication"]["document_near_duplicates"]),
        "paragraph_near_duplicates": len(report["deduplication"]["cross_document_paragraph_near_duplicates"]),
        "new_test_read": report["new_test_read"],
        "model_loaded": report["model_loaded"],
    }, indent=2))


if __name__ == "__main__":
    main()
