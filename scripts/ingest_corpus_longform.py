"""Ingest the pinned modern long-form corpus expansion wave."""

import argparse
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus_longform import ingest_longform
from corpus_pilot import digest


def fetch(url, _cache_name):
    request = Request(url, headers={"User-Agent": "mike-llm-corpus-ingestion/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/corpus_v2_longform_sources.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/corpus-v2-longform-wave5"))
    parser.add_argument("--report", type=Path, default=Path("docs/results/corpus-v2-longform-wave5.json"))
    args = parser.parse_args()
    report = ingest_longform(args.registry, args.output_dir, fetch)
    report["implementation_sha256"] = {
        path: digest(Path(path).read_bytes())
        for path in ("scripts/ingest_corpus_longform.py", "src/corpus_longform.py", "src/corpus_cleaning.py")
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report.exists() and args.report.read_text() != serialized:
        raise ValueError("long-form report differs; inspect inputs or choose a new --report path")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(serialized)
    print(json.dumps({key: report[key] for key in ("document_count", "repository_count", "publisher_count", "rendered_characters", "new_test_read", "model_loaded")}, indent=2))


if __name__ == "__main__":
    main()
