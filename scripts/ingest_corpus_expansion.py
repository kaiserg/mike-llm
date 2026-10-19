"""Ingest the pinned, development-only first corpus-v2 expansion wave."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus_expansion import ingest_expansion
from corpus_pilot import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("configs/corpus_v2_expansion_sources.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/corpus-v2-expansion-wave1"))
    parser.add_argument("--report", type=Path, default=Path("docs/results/corpus-v2-expansion-wave1.json"))
    parser.add_argument("--cache-dir", type=Path, help="optional reviewed local cache")
    args = parser.parse_args()

    def fetch(url, name):
        if args.cache_dir and (args.cache_dir / name).is_file():
            return (args.cache_dir / name).read_bytes()
        print("Retrieving:", url, flush=True)
        response = subprocess.run([
            "curl", "--silent", "--show-error", "--fail", "--location",
            "--user-agent", "mike-llm-corpus-ingestion",
            "--proto", "=https", "--proto-redir", "=https", "--max-time", "60",
            "--max-filesize", "2000000", url,
        ], check=True, capture_output=True)
        time.sleep(0.25)
        return response.stdout

    report = ingest_expansion(args.sources, args.output_dir, fetch)
    report = dict(report, implementation_sha256={
        path: digest(Path(path).read_bytes())
        for path in ("scripts/ingest_corpus_expansion.py", "src/corpus_expansion.py", "src/corpus_cleaning.py")
    })
    serialized = json.dumps(report, indent=2) + "\n"
    if args.report.exists():
        if args.report.read_text() != serialized:
            raise ValueError("report differs; choose a new --report path")
    else:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    print(json.dumps({
        "documents": report["document_count"],
        "source_groups": report["independent_split_groups"],
        "characters": report["rendered_characters"],
        "document_near_duplicates": len(report["deduplication"]["document_near_duplicates"]),
        "paragraph_near_duplicates": len(report["deduplication"]["cross_document_paragraph_near_duplicates"]),
        "new_test_read": report["new_test_read"],
        "model_loaded": report["model_loaded"],
    }, indent=2))


if __name__ == "__main__":
    main()
