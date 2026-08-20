from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.corpus_acquisition.contracts import CandidateAssessment
from dac_her.corpus_acquisition.m3_rebase import rebase_downloaded_m3_snapshot
from dac_her.corpus_acquisition.profile import load_acquisition_profile
from dac_her.corpus_acquisition.quality_contracts import CorpusQualityAssessment
from pipeline_core.literature.catalog_contracts import LiteratureCatalogPacket


def _read_jsonl(path: Path, model):
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebase already-verified downloaded M3 papers onto an append-expanded "
            "catalog without network access. Produces a new M3-compatible starting "
            "snapshot for knowledge-aware backfill."
        )
    )
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--assessments", required=True, type=Path)
    parser.add_argument("--quality-assessments", required=True, type=Path)
    parser.add_argument("--source-m3-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rebase-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = load_acquisition_profile(args.profile)
    packet = LiteratureCatalogPacket.model_validate_json(
        args.catalog.read_text(encoding="utf-8")
    )
    report = rebase_downloaded_m3_snapshot(
        profile=profile,
        packet=packet,
        assessments=_read_jsonl(args.assessments, CandidateAssessment),
        quality_assessments=_read_jsonl(
            args.quality_assessments,
            CorpusQualityAssessment,
        ),
        source_m3_dir=args.source_m3_dir,
        output_dir=args.output_dir,
        rebase_id=args.rebase_id,
    )
    print("Expanded-catalog M3 rebase complete")
    print("Retained verified PDFs:", report["retained_downloaded_count"])
    print("Dropped:", report["dropped_count"])
    print("Network acquisition performed:", report["network_acquisition_performed"])
    print("M3-compatible snapshot:", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
