from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dac_her.hypothesis_trend_grounding import (
    GroundingSourceArtifact,
    build_hypothesis_trend_grounding_bundle,
    sha256_file,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"Expected JSONL object: {path}:{line_no}"
            )
        rows.append(value)
    return rows


def artifact(role: str, path: Path) -> GroundingSourceArtifact:
    if not path.exists():
        raise ValueError(f"Required source artifact missing: {path}")
    return GroundingSourceArtifact(
        role=role,
        path=str(path),
        sha256=sha256_file(path),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic Trend/CrossContext -> hypothesis grounding "
            "sidecar without modifying HypothesisContext."
        )
    )
    parser.add_argument("--trend-dir", required=True, type=Path)
    parser.add_argument("--precision-dir", required=True, type=Path)
    parser.add_argument("--context-dir", type=Path, default=None)
    parser.add_argument("--assessment-dir", type=Path, default=None)
    parser.add_argument("--domain-profile", default="sers_au_ag")
    parser.add_argument(
        "--grounding-semantics-id",
        default="sers_au_ag_hypothesis_trend_grounding_v1_alpha4c5a",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    trend_summary_path = args.trend_dir / "summary.json"
    evidence_path = args.trend_dir / "evidence.jsonl"
    precision_summary_path = args.precision_dir / "summary.json"
    local_results_path = args.precision_dir / "local_results.jsonl"

    sources = [
        artifact("trend_summary", trend_summary_path),
        artifact("trend_evidence", evidence_path),
        artifact("precision_summary", precision_summary_path),
        artifact("paper_local_trend_results", local_results_path),
    ]

    context_summary = None
    profiles = []
    assessment_summary = None
    assessments = []
    contrasts = []

    if args.context_dir is not None:
        context_summary_path = args.context_dir / "summary.json"
        profiles_path = args.context_dir / "context_profiles.jsonl"
        sources.extend([
            artifact("cross_context_summary", context_summary_path),
            artifact("trend_context_profiles", profiles_path),
        ])
        context_summary = read_json(context_summary_path)
        profiles = read_jsonl(profiles_path)

    if args.assessment_dir is not None:
        assessment_summary_path = args.assessment_dir / "summary.json"
        assessments_path = args.assessment_dir / "assessments.jsonl"
        contrasts_path = args.assessment_dir / "pairwise_contrasts.jsonl"
        sources.extend([
            artifact(
                "cross_context_assessment_summary",
                assessment_summary_path,
            ),
            artifact(
                "cross_context_assessments",
                assessments_path,
            ),
            artifact(
                "pairwise_trend_contrasts",
                contrasts_path,
            ),
        ])
        assessment_summary = read_json(assessment_summary_path)
        assessments = read_jsonl(assessments_path)
        contrasts = read_jsonl(contrasts_path)

    bundle = build_hypothesis_trend_grounding_bundle(
        domain_profile_id=args.domain_profile,
        grounding_semantics_id=args.grounding_semantics_id,
        trend_summary=read_json(trend_summary_path),
        precision_summary=read_json(precision_summary_path),
        evidence_rows=read_jsonl(evidence_path),
        local_result_rows=read_jsonl(local_results_path),
        context_summary=context_summary,
        context_profile_rows=profiles,
        assessment_summary=assessment_summary,
        assessment_rows=assessments,
        contrast_rows=contrasts,
        source_artifacts=sources,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        bundle.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Hypothesis Trend grounding built")
    print("Bundle ID:", bundle.bundle_id)
    print("Bundle SHA256:", bundle.bundle_sha256)
    print("Contract semantics:", bundle.contract_semantics_id)
    print("Grounding semantics:", bundle.grounding_semantics_id)
    print("Relations:", bundle.relation_count)
    print("Local Trend results:", bundle.local_result_count)
    print(
        "Cross-context statuses:",
        json.dumps(
            bundle.cross_context_status_counts,
            sort_keys=True,
        ),
    )
    print(
        "Support roles:",
        json.dumps(bundle.support_role_counts, sort_keys=True),
    )
    print(
        "Local / replicated / context-dependency / reversal / gap:",
        bundle.local_empirical_premise_count,
        bundle.cross_context_replicated_premise_count,
        bundle.context_dependency_signal_count,
        bundle.reversal_counterevidence_count,
        bundle.replication_gap_signal_count,
    )
    print("Zero yield:", bundle.zero_yield)
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
