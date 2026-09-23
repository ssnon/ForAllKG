from __future__ import annotations

import argparse
import json
from pathlib import Path

from domains.registry import get_domain_profile
from pipeline_core.discovery.counterevidence_projection_retrieval import (
    CounterevidenceProjectionQueryPlan,
)
from pipeline_core.discovery.external_novelty_contracts import (
    PriorArtPacket,
)
from pipeline_core.discovery.grounded_factor_projection import (
    GroundedFactorProjectionReport,
)
from pipeline_core.discovery.projection_relation_adjudication import (
    InstructorProjectionRelationAdjudicationBackend,
    build_projection_relation_candidate_report,
    run_projection_relation_adjudication,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificRelationIRReport,
)
from pipeline_core.discovery.supporting_projection_retrieval import (
    SupportingProjectionQueryPlan,
)
from pipeline_core.discovery.scientific_relation_second_pass import (
    ScientificRelationSecondPassPlan,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        value.model_dump(mode="json")
        if hasattr(value, "model_dump")
        else value
    )
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a bounded typed review set from supporting and "
            "counterevidence retrieval, then adjudicate prior-art relation "
            "signals without creating novelty or non-obviousness authority."
        )
    )
    parser.add_argument("--relation-ir", required=True, type=Path)
    parser.add_argument("--projection", required=True, type=Path)
    parser.add_argument("--supporting-plan", required=True, type=Path)
    parser.add_argument("--supporting-prior-art", required=True, type=Path)
    parser.add_argument("--counterevidence-plan", required=True, type=Path)
    parser.add_argument("--counterevidence-prior-art", required=True, type=Path)
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--second-pass-plan", default=None, type=Path)
    parser.add_argument("--second-pass-prior-art", default=None, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--max-review-works-per-claim",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-exhaustive-rounds",
        type=int,
        default=3,
        help=(
            "Bounded total adjudication rounds per claim. Later rounds "
            "contain only presented work IDs omitted by earlier model output."
        ),
    )
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument(
        "--candidates-only",
        action="store_true",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
    )
    args = parser.parse_args()

    relation_ir = ScientificRelationIRReport.model_validate_json(
        args.relation_ir.read_text(encoding="utf-8")
    )
    projection = GroundedFactorProjectionReport.model_validate_json(
        args.projection.read_text(encoding="utf-8")
    )
    supporting_plan = SupportingProjectionQueryPlan.model_validate_json(
        args.supporting_plan.read_text(encoding="utf-8")
    )
    supporting_packet = PriorArtPacket.model_validate_json(
        args.supporting_prior_art.read_text(encoding="utf-8")
    )
    counter_plan = CounterevidenceProjectionQueryPlan.model_validate_json(
        args.counterevidence_plan.read_text(encoding="utf-8")
    )
    counter_packet = PriorArtPacket.model_validate_json(
        args.counterevidence_prior_art.read_text(encoding="utf-8")
    )
    domain_profile = get_domain_profile(
        args.domain_profile
    )

    if (args.second_pass_plan is None) != (
        args.second_pass_prior_art is None
    ):
        raise ValueError(
            "--second-pass-plan and --second-pass-prior-art "
            "must be supplied together"
        )

    second_pass_plan = (
        ScientificRelationSecondPassPlan.model_validate_json(
            args.second_pass_plan.read_text(encoding="utf-8")
        )
        if args.second_pass_plan is not None
        else None
    )
    second_pass_packet = (
        PriorArtPacket.model_validate_json(
            args.second_pass_prior_art.read_text(encoding="utf-8")
        )
        if args.second_pass_prior_art is not None
        else None
    )

    candidate_report = build_projection_relation_candidate_report(
        relation_ir_report=relation_ir,
        projection_report=projection,
        supporting_plan=supporting_plan,
        supporting_packet=supporting_packet,
        counterevidence_plan=counter_plan,
        counterevidence_packet=counter_packet,
        domain_profile=domain_profile,
        max_review_works_per_claim=(
            args.max_review_works_per_claim
        ),
        second_pass_plan=second_pass_plan,
        second_pass_packet=second_pass_packet,
    )

    prefix = args.output_prefix
    candidate_path = prefix.with_suffix(
        ".candidates.json"
    )
    report_path = prefix.with_suffix(
        ".review.json"
    )
    prompt_path = prefix.with_suffix(
        ".prompts.json"
    )

    _write(candidate_path, candidate_report)

    print("Projection relation candidate set complete")
    print("Claims:", candidate_report.claim_count)
    print(
        "Review candidates:",
        candidate_report.total_review_candidate_count,
    )
    for row in candidate_report.claims:
        print()
        print("Claim:", row.claim_id)
        print(
            "Candidates:",
            row.candidate_count,
            "abstracts=",
            row.abstract_candidate_count,
            "source_unique=",
            row.source_unique_work_count,
            "typed_excluded=",
            row.typed_identity_excluded_work_count,
        )
        for candidate in row.candidates:
            print(
                " ",
                candidate.review_work_id,
                "| score=",
                candidate.selection_score,
                "| typed=",
                candidate.typed_compatibility_state,
                "| endpoint_cov=",
                round(candidate.endpoint_lexical_coverage, 3),
                "| endpoint_disc=",
                [
                    round(value, 3)
                    for value in candidate.endpoint_discriminative_coverages
                ],
                "| endpoint_match=",
                candidate.endpoint_discriminative_match_counts,
                "| endpoint_need=",
                candidate.endpoint_required_match_counts,
                "| endpoint_support=",
                candidate.endpoint_supported_count,
                "| anchor=",
                candidate.relation_anchor_tier,
                "| identity_cov=",
                round(candidate.identity_lexical_coverage, 3),
                "| doc_domains=",
                candidate.document_domain_labels,
                "| doc_scope=",
                candidate.document_scope_features,
                "| lanes=",
                candidate.selected_lanes,
                "| second_pass=",
                candidate.second_pass_roles,
                "| support=",
                candidate.supporting_projection_kinds,
                "| counter=",
                candidate.counterevidence_modes,
                "| title=",
                candidate.title,
            )

    print()
    print("Candidate report:", candidate_path)

    if args.candidates_only:
        print("CANDIDATES_ONLY=True")
        print("RELATION_ADJUDICATION_PERFORMED=False")
        print("NOVELTY_AUTHORITY_CREATED=False")
        return 0

    backend = InstructorProjectionRelationAdjudicationBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        capture_prompts=args.save_prompts,
    )
    report = run_projection_relation_adjudication(
        candidate_report=candidate_report,
        projection_report=projection,
        backend=backend,
        max_exhaustive_rounds=args.max_exhaustive_rounds,
    )
    _write(report_path, report)

    if args.save_prompts:
        _write(
            prompt_path,
            backend.prompt_records,
        )

    print()
    print("Projection relation adjudication shadow complete")
    print(
        "Reviewed claims:",
        report.reviewed_claim_count,
    )
    print(
        "Presented works:",
        report.presented_work_count,
    )
    print(
        "Classified match records:",
        report.classified_work_count,
    )
    print(
        "Unclassified presented works:",
        report.unclassified_work_count,
    )
    print("LLM calls:", report.llm_calls_performed)
    print(
        "Max exhaustive rounds per claim:",
        args.max_exhaustive_rounds,
    )
    print(
        "Direct signals:",
        report.direct_signal_work_count,
    )
    print(
        "Lower-order signals:",
        report.lower_order_signal_work_count,
    )
    print(
        "Counterevidence signals:",
        report.counterevidence_signal_work_count,
    )
    print(
        "Conflicting signals:",
        report.conflicting_signal_work_count,
    )

    for row in report.reviews:
        print()
        print("Claim:", row.claim_id)
        print("Relation state:", row.relation_state)
        print(
            "Coverage:",
            "presented=",
            row.presented_work_count,
            "classified=",
            row.classified_work_count,
            "unclassified=",
            len(row.unclassified_work_ids),
        )
        print(
            "Direct:",
            row.direct_prior_art_work_ids,
        )
        print(
            "Partial:",
            row.partial_prior_art_work_ids,
        )
        print(
            "Lower-order:",
            row.lower_order_prior_art_work_ids,
        )
        print(
            "Directional counterevidence:",
            row.directional_counterevidence_work_ids,
        )
        print(
            "Contextual conflict:",
            row.contextual_conflict_work_ids,
        )
        print(
            "Conflicting:",
            row.conflicting_prior_art_work_ids,
        )
        for match in row.matches:
            print(
                " ",
                match.relationship,
                "(model=" + match.original_relationship + ")",
                match.work_id,
                "confidence=",
                match.confidence,
                "typed=",
                match.typed_compatibility_state,
                "reasons=",
                match.deterministic_reason_codes,
            )
            if match.evidence_span:
                print(
                    "   evidence:",
                    match.evidence_span,
                )

    print()
    print("Diagnostic only: true")
    print("Absence-based novelty authorized: false")
    print("Positive non-obviousness authority created: false")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Production selection changed: false")
    print("Review report:", report_path)
    if args.save_prompts:
        print("Prompts:", prompt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
