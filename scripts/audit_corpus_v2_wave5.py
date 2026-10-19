"""Rerun the corpus-v2 cross-wave audit after the modern long-form wave."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_corpus_v2 import SPECS
from corpus_audit import audit_corpus
from corpus_pilot import digest


LONGFORM_SPEC = {
    "id": "wave5-longform",
    "report": "docs/results/corpus-v2-longform-wave5.json",
    "snapshot": "data/raw/corpus-v2-longform-wave5",
    "text_path_key": "text_path",
    "text_hash_key": "rendered_sha256",
    "character_key": "rendered_characters",
    "publisher_fallback": "Unspecified modern long-form publisher",
}
OUTPUT = Path("docs/results/corpus-v2-cross-wave-audit-after-wave5.json")


def main():
    report = audit_corpus([*SPECS, LONGFORM_SPEC])
    report["implementation_sha256"] = {
        path: digest(Path(path).read_bytes())
        for path in ("scripts/audit_corpus_v2_wave5.py", "scripts/audit_corpus_v2.py", "src/corpus_audit.py", "src/corpus_cleaning.py")
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if OUTPUT.exists() and OUTPUT.read_text() != serialized:
        raise ValueError("wave-5 audit report differs; inspect inputs before replacing it")
    OUTPUT.write_text(serialized)
    print(json.dumps({
        "documents": report["totals"]["documents"],
        "characters": report["totals"]["characters"],
        "publishers": report["totals"]["publishers"],
        "historical_share": report["era_balance"]["historical_pilot_share"],
        "largest_ten_share": report["concentration"]["largest_ten_document_share"],
        "cross_stage_document_near_duplicates": len(report["cross_stage_lexical_screen"]["document_near_duplicates"]),
        "cross_stage_paragraph_near_duplicates": len(report["cross_stage_lexical_screen"]["cross_stage_paragraph_near_duplicates"]),
    }, indent=2))


if __name__ == "__main__":
    main()
