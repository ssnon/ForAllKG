from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from dac_her.cross_context_trend import (
    CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
    TrendContextDimension,
    TrendContextProfile,
)
from dac_her.cross_context_trend_assessment import (
    CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID,
    audit_deterministic_cross_context_assessments,
    build_deterministic_cross_context_assessments,
    classify_pair_role,
)
from dac_her.domains.cross_context_trend_registry import (
    get_cross_context_trend_adapter,
)
from dac_her.domains.registry import get_domain_profile
from dac_her.domains.trend_precision_registry import (
    get_trend_precision_adapter,
)
from dac_her.domains.trend_registry import get_trend_adapter
from dac_her.trend_precision import PaperLocalTrendResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL row must be an object at "
                    f"{path}:{line_number}."
                )
            rows.append(row)
    return rows


def _write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def _paper_local_result_from_row(
    row: Mapping[str, Any],
) -> PaperLocalTrendResult:
    values = dict(row)
    for key in (
        "member_trend_ids",
        "evidence_kinds",
        "trend_subject_ids",
        "reference_subject_ids",
        "source_claim_ids",
        "source_measurement_ids",
        "source_measurement_result_ids",
        "source_calculation_ids",
        "source_node_ids",
    ):
        values[key] = tuple(
            str(value)
            for value in values.get(key, [])
        )
    return PaperLocalTrendResult(**values)


def _context_profile_from_row(
    row: Mapping[str, Any],
) -> TrendContextProfile:
    values = dict(row)
    raw_dimensions = values.get("dimensions", [])
    if not isinstance(raw_dimensions, list):
        raise ValueError(
            "TrendContextProfile dimensions must be a list."
        )
    values["dimensions"] = tuple(
        TrendContextDimension(
            name=str(item["name"]),
            status=str(item["status"]),
            normalized_value=str(
                item.get("normalized_value", "")
            ),
            source_values=tuple(
                str(value)
                for value in item.get(
                    "source_values",
                    [],
                )
            ),
            source_node_ids=tuple(
                str(value)
                for value in item.get(
                    "source_node_ids",
                    [],
                )
            ),
            provenance_scopes=tuple(
                str(value)
                for value in item.get(
                    "provenance_scopes",
                    [],
                )
            ),
        )
        for item in raw_dimensions
    )
    for key in (
        "evidence_kinds",
        "member_trend_ids",
        "source_comparison_context_ids",
        "source_method_context_ids",
        "source_claim_ids",
        "source_measurement_ids",
        "source_measurement_result_ids",
        "source_calculation_ids",
        "source_node_ids",
    ):
        values[key] = tuple(
            str(value)
            for value in values.get(key, [])
        )
    return TrendContextProfile(**values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic cross-paper PairwiseTrendContrast "
            "and CrossContextTrendAssessment sidecars from frozen "
            "TrendContextProfile rows. No extraction, context "
            "projection, ranking policy, or LLM calls are performed."
        )
    )
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--data-root", default="data_sers")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument(
        "--mode",
        choices=("evidence", "mechanism", "exploratory"),
        default="exploratory",
    )
    parser.add_argument("--trend-id", required=True)
    parser.add_argument("--precision-id", required=True)
    parser.add_argument("--context-id", required=True)
    parser.add_argument("--assessment-id", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    profile = get_domain_profile(args.domain_profile)
    trend_adapter = get_trend_adapter(profile)
    precision_adapter = get_trend_precision_adapter(profile)
    context_adapter = get_cross_context_trend_adapter(profile)

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    corpus_root = (
        data_root
        / "corpus"
        / args.corpus_id
        / args.mode
    )
    trend_root = corpus_root / "trend" / args.trend_id
    precision_root = (
        trend_root
        / "precision"
        / args.precision_id
    )
    context_root = (
        precision_root
        / "cross_context"
        / args.context_id
    )

    precision_summary_path = (
        precision_root / "summary.json"
    )
    local_results_path = (
        precision_root / "local_results.jsonl"
    )
    context_summary_path = (
        context_root / "summary.json"
    )
    context_audit_path = context_root / "audit.json"
    context_profiles_path = (
        context_root / "context_profiles.jsonl"
    )

    for path in (
        precision_summary_path,
        local_results_path,
        context_summary_path,
        context_audit_path,
        context_profiles_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    precision_summary = json.loads(
        precision_summary_path.read_text(
            encoding="utf-8"
        )
    )
    context_summary = json.loads(
        context_summary_path.read_text(
            encoding="utf-8"
        )
    )
    context_audit = json.loads(
        context_audit_path.read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(
        precision_summary,
        dict,
    ):
        raise ValueError(
            "Precision summary must be an object."
        )
    if not isinstance(context_summary, dict):
        raise ValueError(
            "Context summary must be an object."
        )
    if not isinstance(context_audit, dict):
        raise ValueError(
            "Context audit must be an object."
        )

    if not bool(
        precision_summary.get(
            "structural_gate",
            False,
        )
    ):
        raise ValueError(
            "Parent precision structural gate is false."
        )
    if not bool(
        context_summary.get(
            "structural_gate",
            False,
        )
    ):
        raise ValueError(
            "Parent context-projection structural gate is false."
        )
    if not bool(
        context_audit.get(
            "structural_gate",
            False,
        )
    ):
        raise ValueError(
            "Parent context-projection audit structural gate "
            "is false."
        )

    expected_precision = {
        "domain_profile_id": profile.profile_id,
        "corpus_id": args.corpus_id,
        "corpus_mode": args.mode,
        "trend_id": args.trend_id,
        "trend_semantics_id":
            trend_adapter.semantics_id,
        "precision_semantics_id":
            precision_adapter.precision_semantics_id,
    }
    for key, expected in expected_precision.items():
        observed = str(
            precision_summary.get(key, "")
        )
        if observed != str(expected):
            raise ValueError(
                "Precision binding mismatch for "
                f"{key}: {observed!r} != {expected!r}."
            )

    expected_context = {
        "context_id": args.context_id,
        "domain_profile_id": profile.profile_id,
        "contract_semantics_id":
            CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
        "context_semantics_id":
            context_adapter.context_semantics_id,
        "corpus_id": args.corpus_id,
        "corpus_mode": args.mode,
        "trend_id": args.trend_id,
        "trend_semantics_id":
            trend_adapter.semantics_id,
        "precision_id": args.precision_id,
        "precision_semantics_id":
            precision_adapter.precision_semantics_id,
    }
    for key, expected in expected_context.items():
        observed = str(
            context_summary.get(key, "")
        )
        if observed != str(expected):
            raise ValueError(
                "Context-projection binding mismatch for "
                f"{key}: {observed!r} != {expected!r}."
            )

    if (
        context_summary.get(
            "pairwise_contrasts_built"
        )
        is not False
    ):
        raise ValueError(
            "alpha4c.3c requires a pure alpha4c.3b context "
            "projection source."
        )
    if (
        context_summary.get(
            "cross_context_assessments_built"
        )
        is not False
    ):
        raise ValueError(
            "Parent context source already reports assessments."
        )
    if (
        context_summary.get(
            "paper_global_context_fallback_used"
        )
        is not False
    ):
        raise ValueError(
            "Context source used forbidden paper-global fallback."
        )
    if (
        context_summary.get(
            "numeric_ranking_reused_as_trend_policy"
        )
        is not False
    ):
        raise ValueError(
            "Context source reused numeric-ranking policy."
        )

    local_results = [
        _paper_local_result_from_row(row)
        for row in _read_jsonl(local_results_path)
    ]
    context_profiles = [
        _context_profile_from_row(row)
        for row in _read_jsonl(
            context_profiles_path
        )
    ]

    if (
        len(local_results)
        != int(
            context_summary.get(
                "local_result_count",
                -1,
            )
        )
    ):
        raise ValueError(
            "Context summary local-result count mismatch."
        )
    if (
        len(context_profiles)
        != int(
            context_summary.get(
                "profile_count",
                -1,
            )
        )
    ):
        raise ValueError(
            "Context summary profile count mismatch."
        )

    contrasts, assessments = (
        build_deterministic_cross_context_assessments(
            context_profiles
        )
    )
    audit = (
        audit_deterministic_cross_context_assessments(
            local_results=local_results,
            profiles=context_profiles,
            contrasts=contrasts,
            assessments=assessments,
        )
    )

    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else (
            context_root
            / "assessment"
            / args.assessment_id
        )
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    contrasts_path = _write_jsonl(
        output_root / "pairwise_contrasts.jsonl",
        (
            contrast.to_row()
            for contrast in contrasts
        ),
    )
    assessments_path = _write_jsonl(
        output_root / "assessments.jsonl",
        (
            assessment.to_row()
            for assessment in assessments
        ),
    )
    audit_path = output_root / "audit.json"
    audit_path.write_text(
        json.dumps(
            audit.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    context_profile_sha256 = _sha256_file(
        context_profiles_path
    )
    status_counts = Counter(
        assessment.status
        for assessment in assessments
    )
    direction_relation_counts = Counter(
        contrast.direction_relation
        for contrast in contrasts
    )
    context_relation_counts = Counter(
        contrast.context_relation
        for contrast in contrasts
    )
    evidence_kind_relation_counts = Counter(
        contrast.evidence_kind_relation
        for contrast in contrasts
    )
    pair_role_counts = Counter(
        classify_pair_role(contrast)
        for contrast in contrasts
    )

    summary = {
        "assessment_id": args.assessment_id,
        "domain_profile_id": profile.profile_id,
        "contract_semantics_id":
            CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
        "assessment_semantics_id":
            CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID,
        "corpus_id": args.corpus_id,
        "corpus_mode": args.mode,
        "trend_id": args.trend_id,
        "trend_semantics_id":
            trend_adapter.semantics_id,
        "precision_id": args.precision_id,
        "precision_semantics_id":
            precision_adapter.precision_semantics_id,
        "context_id": args.context_id,
        "context_semantics_id":
            context_adapter.context_semantics_id,
        "context_profile_sha256":
            context_profile_sha256,
        "local_result_count":
            audit.local_result_count,
        "context_profile_count":
            audit.context_profile_count,
        "relation_count":
            audit.relation_count,
        "expected_cross_paper_pair_count":
            audit.expected_cross_paper_pair_count,
        "pairwise_contrast_count":
            audit.pairwise_contrast_count,
        "assessment_count":
            audit.assessment_count,
        "status_counts": dict(
            sorted(status_counts.items())
        ),
        "pair_role_counts": dict(
            sorted(pair_role_counts.items())
        ),
        "direction_relation_counts": dict(
            sorted(direction_relation_counts.items())
        ),
        "context_relation_counts": dict(
            sorted(context_relation_counts.items())
        ),
        "evidence_kind_relation_counts": dict(
            sorted(
                evidence_kind_relation_counts.items()
            )
        ),
        "majority_vote_used": False,
        "same_paper_pairs_allowed": False,
        "numeric_ranking_reused_as_trend_policy": False,
        "context_reprojected": False,
        "causal_status_promoted": False,
        "structural_gate":
            audit.structural_gate,
        "sources": {
            "precision_summary": str(
                precision_summary_path
            ),
            "context_summary": str(
                context_summary_path
            ),
            "context_profiles": str(
                context_profiles_path
            ),
        },
        "outputs": {
            "pairwise_contrasts": str(
                contrasts_path
            ),
            "assessments": str(
                assessments_path
            ),
            "audit": str(audit_path),
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Cross-context trend assessments built")
    print("Assessment ID:", args.assessment_id)
    print(
        "Assessment semantics:",
        CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID,
    )
    print(
        "Relations / assessments:",
        audit.relation_count,
        "/",
        audit.assessment_count,
    )
    print(
        "Expected / actual cross-paper pairs:",
        audit.expected_cross_paper_pair_count,
        "/",
        audit.pairwise_contrast_count,
    )
    print(
        "Statuses:",
        json.dumps(
            summary["status_counts"],
            sort_keys=True,
        ),
    )
    print(
        "Pair roles:",
        json.dumps(
            summary["pair_role_counts"],
            sort_keys=True,
        ),
    )
    print("Structural gate:", audit.structural_gate)
    print("Saved:", output_root)
    return 0 if audit.structural_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
