from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from domains.registry import get_domain_profile

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
    PriorArtPacket,
)
from pipeline_core.discovery.node_mapping import (
    DEFAULT_EMBED_MODEL,
    SentenceTransformerEncoder,
)
from pipeline_core.discovery.nonobviousness_full_shadow import (
    compile_forced_adjudication_if_determined,
    derive_conservative_nonobviousness_inputs,
)
from pipeline_core.discovery.novelty_adjudication import (
    EstablishedPriorArtRelation,
    NonObviousnessEvidencePacket,
    assess_adjudication_readiness,
)
from pipeline_core.discovery.novelty_adjudication_llm import (
    InstructorOpenAICompatibleNonObviousnessAdjudicationBackend,
    review_and_compile_nonobviousness_adjudication,
)
from pipeline_core.discovery.novelty_closure_compiler import (
    compile_nonobviousness_evidence_closure,
)
from pipeline_core.discovery.novelty_closure_execution import (
    build_closure_execution_plan,
    expand_closure_query_plan_source_preserving,
    rank_closure_candidates,
)
from pipeline_core.discovery.novelty_closure_llm import (
    InstructorOpenAICompatibleClosureReviewBackend,
    review_and_compile_closure_target,
)
from pipeline_core.discovery.novelty_closure_relationships_llm import (
    InstructorOpenAICompatibleClosureRelationshipBackend,
    review_and_compile_closure_relationships,
)
from pipeline_core.discovery.novelty_closure_planner import (
    build_closure_retrieval_plan,
)
from pipeline_core.discovery.novelty_nonobviousness import (
    assess_structural_nonobviousness,
)
from pipeline_core.discovery.novelty_residue import (
    extract_novelty_residue,
)
from pipeline_core.discovery.prior_art_matching import (
    PriorArtRanker,
)
from pipeline_core.discovery.prior_art_provider_plan import (
    build_literature_providers,
    load_literature_provider_plan,
    require_standard_or_full_auto_plan,
)
from pipeline_core.discovery.prior_art_retrieval import (
    LiteratureRetriever,
    canonicalize_prior_art_packet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the N9 full non-obviousness shadow after intake: "
            "targeted closure, evidence compilation, structural gate, "
            "readiness, independent adjudication for READY candidates, "
            "and deterministic final compilation."
        )
    )

    parser.add_argument(
        "--query-plan",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--external-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--external-prior-art",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--intake-shadow",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--provider-plan",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--domain-profile",
        required=True,
    )
    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--base-url",
        default=None,
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--max-ranked-works",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-ready-claims",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def dumpable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return dumpable(
            value.model_dump(mode="json")
        )

    if is_dataclass(value):
        return dumpable(
            asdict(value)
        )

    if isinstance(value, dict):
        return {
            str(key): dumpable(row)
            for key, row in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            dumpable(row)
            for row in value
        ]

    return value


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            dumpable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def safe_id(value: str) -> str:
    return "".join(
        char
        if char.isalnum() or char in "-_"
        else "_"
        for char in value
    )


def main() -> int:
    args = parse_args()

    plan = LiteratureQueryPlan.model_validate_json(
        args.query_plan.read_text(
            encoding="utf-8"
        )
    )

    report = ExternalNoveltyReport.model_validate_json(
        args.external_report.read_text(
            encoding="utf-8"
        )
    )

    external_prior = PriorArtPacket.model_validate_json(
        args.external_prior_art.read_text(
            encoding="utf-8"
        )
    )

    intake = json.loads(
        args.intake_shadow.read_text(
            encoding="utf-8"
        )
    )

    if (
        plan.source_portfolio_id
        != report.source_portfolio_id
    ):
        raise ValueError(
            "Full N9 shadow provenance mismatch: "
            "query plan and external report."
        )

    if (
        report.source_prior_art_packet_id
        != external_prior.packet_id
    ):
        raise ValueError(
            "Full N9 shadow provenance mismatch: "
            "external report and prior-art packet."
        )

    if (
        intake.get("source_query_plan_id")
        != plan.plan_id
    ):
        raise ValueError(
            "Full N9 shadow provenance mismatch: "
            "intake and query plan."
        )

    if (
        intake.get("source_external_report_id")
        != report.report_id
    ):
        raise ValueError(
            "Full N9 shadow provenance mismatch: "
            "intake and external report."
        )

    if (
        intake.get("source_prior_art_packet_id")
        != external_prior.packet_id
    ):
        raise ValueError(
            "Full N9 shadow provenance mismatch: "
            "intake and prior-art packet."
        )

    ready_ids: list[str] = []

    for hypothesis in intake.get(
        "hypotheses",
        [],
    ):
        for claim_id in hypothesis.get(
            "ready_for_closure_claim_ids",
            [],
        ):
            claim_id = str(
                claim_id
            )

            if claim_id not in ready_ids:
                ready_ids.append(
                    claim_id
                )

    residues = extract_novelty_residue(
        plan,
        report,
    )

    claims_by_id = {
        claim.claim_id: claim
        for residue in residues
        for claim in residue.claims
    }

    missing = [
        claim_id
        for claim_id in ready_ids
        if claim_id not in claims_by_id
    ]

    if missing:
        raise ValueError(
            "Intake READY claim IDs are absent from "
            "current residue extraction: "
            + ", ".join(missing)
        )

    selected_ids = ready_ids[
        : max(
            0,
            int(args.max_ready_claims),
        )
    ]

    deferred_ids = ready_ids[
        len(selected_ids):
    ]

    # No closure-eligible claims means there is nothing scientific to
    # retrieve or review. Fail cheaply and deterministically before
    # provider, embedding-model, or LLM initialization.
    if not selected_ids:
        result = {
            "schema_version":
                "nonobviousness-full-shadow-v1",

            "shadow_only":
                True,

            "scientific_selection_changed":
                False,

            "source_portfolio_id":
                report.source_portfolio_id,

            "source_query_plan_id":
                plan.plan_id,

            "source_external_report_id":
                report.report_id,

            "source_prior_art_packet_id":
                external_prior.packet_id,

            "source_intake_shadow":
                str(args.intake_shadow),

            "provider_plan":
                str(args.provider_plan),

            "ready_claim_count":
                len(ready_ids),

            "processed_ready_claim_count":
                0,

            "deferred_ready_claim_count":
                len(deferred_ids),

            "full_state_counts": (
                {
                    "DEFERRED_READY_CLAIM_BUDGET":
                        len(deferred_ids)
                }
                if deferred_ids
                else {}
            ),

            "claims": [
                {
                    "hypothesis_id":
                        claims_by_id[claim_id].hypothesis_id,
                    "claim_id":
                        claim_id,
                    "claim_text":
                        claims_by_id[claim_id].claim_text,
                    "intake_state":
                        "READY_FOR_CLOSURE",
                    "full_shadow_state":
                        "DEFERRED_READY_CLAIM_BUDGET",
                }
                for claim_id in deferred_ids
            ],

            "execution": {
                "closure_retrieval_performed":
                    False,
                "ranking_model_initialized":
                    False,
                "closure_reviewer_initialized":
                    False,
                "reason":
                    "no_ready_claims_selected_for_closure",
            },

            "epistemic_policy": {
                "existing_intake_artifact_semantics_preserved":
                    True,
                "targeted_closure_is_search_bounded":
                    True,
                "query_variants_are_not_evidence":
                    True,
                "identity_anchored_negative_coverage_required":
                    True,
                "historical_or_external_absence_is_not_novelty":
                    True,
                "higher_order_structure_not_inferred_from_free_text":
                    True,
                "scope_compatibility_unassessed_fails_closed":
                    True,
                "ready_candidate_requires_independent_adjudicator":
                    True,
                "production_selection_unchanged":
                    True,
            },
        }

        write_json(
            args.output,
            result,
        )

        print(
            "N9 full non-obviousness shadow built"
        )
        print(
            "Ready claims:",
            len(ready_ids),
        )
        print("Processed:", 0)
        print(
            "Deferred:",
            len(deferred_ids),
        )
        print(
            "States:",
            result["full_state_counts"],
        )
        print(
            "Closure retrieval performed:",
            False,
        )
        print(
            "Scientific selection changed:",
            False,
        )

        return 0

    profile = get_domain_profile(
        args.domain_profile
    )

    provider_plan = (
        load_literature_provider_plan(
            args.provider_plan
        )
    )

    require_standard_or_full_auto_plan(
        provider_plan
    )

    providers = build_literature_providers(
        provider_plan
    )

    retriever = LiteratureRetriever(
        providers,
        results_per_query=(
            args.results_per_query
        ),
    )

    encoder = SentenceTransformerEncoder(
        args.embed_model,
        device=args.device,
    )

    ranker = PriorArtRanker(
        encoder,
        max_ranked_works_per_claim=(
            args.max_ranked_works
        ),
        domain_profile=profile,
    )

    backend = (
        InstructorOpenAICompatibleClosureReviewBackend(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            temperature=0.0,
            parse_retries=1,
            capture_prompts=True,
        )
    )

    relationship_backend = (
        InstructorOpenAICompatibleClosureRelationshipBackend(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            temperature=0.0,
            parse_retries=1,
            capture_prompts=True,
        )
    )

    adjudication_backend = (
        InstructorOpenAICompatibleNonObviousnessAdjudicationBackend(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            temperature=0.0,
            parse_retries=1,
            capture_prompts=True,
        )
    )

    detail_dir = (
        args.output.parent
        / (
            args.output.stem
            + ".details"
        )
    )

    rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()

    for claim_id in selected_ids:
        claim = claims_by_id[
            claim_id
        ]

        inputs = (
            derive_conservative_nonobviousness_inputs(
                claim
            )
        )

        closure_plan = (
            build_closure_retrieval_plan(
                claim
            )
        )

        execution_plan = (
            build_closure_execution_plan(
                source_portfolio_id=(
                    report.source_portfolio_id
                ),
                closure_plan=closure_plan,
            )
        )

        expanded_plan = (
            expand_closure_query_plan_source_preserving(
                plan=execution_plan,
                max_queries_per_target=3,
            )
        )

        packet = (
            retriever.retrieve(
                expanded_plan
            ).packet
        )

        packet = (
            canonicalize_prior_art_packet(
                packet
            )
        )

        candidates = rank_closure_candidates(
            plan=expanded_plan,
            packet=packet,
            ranker=ranker,
        )

        reviews = []

        for target in expanded_plan.targets:
            reviews.append(
                review_and_compile_closure_target(
                    backend=backend,
                    target=target,
                    candidates=candidates[
                        target.target_id
                    ],
                    packet=packet,
                    plan=expanded_plan,
                )
            )

        targets_by_slot = {
            target.slot: target
            for target
            in expanded_plan.targets
        }

        relationship_outcome = (
            review_and_compile_closure_relationships(
                backend=relationship_backend,
                reviews=reviews,
                packet=packet,
                targets_by_slot=targets_by_slot,
            )
        )

        closure_compilation = (
            compile_nonobviousness_evidence_closure(
                reviews=reviews,
                bridge_kind=(
                    relationship_outcome
                    .compiled
                    .bridge_kind
                ),
                scope_compatible=(
                    relationship_outcome
                    .compiled
                    .scope_compatible
                ),
            )
        )

        closure = (
            closure_compilation.closure
        )

        structural = (
            assess_structural_nonobviousness(
                closure,
                inputs.structure,
            )
        )

        readiness = (
            assess_adjudication_readiness(
                structural_status=(
                    structural.status
                ),
                vector=inputs.vector,
            )
        )

        established_relations = tuple(
            EstablishedPriorArtRelation(
                relation_statement=(
                    targets_by_slot[
                        review.slot
                    ].source_text
                ),
                relationship_status=(
                    "ESTABLISHED"
                ),
                work_ids=tuple(
                    review.positive_work_ids
                ),
                scope_note=(
                    "N10 closure slot "
                    + review.slot
                    + "; compiled cross-slot scope_compatible="
                    + str(
                        relationship_outcome
                        .compiled
                        .scope_compatible
                    )
                    + "; bridge_kind="
                    + str(
                        relationship_outcome
                        .compiled
                        .bridge_kind
                    )
                    + "."
                ),
            )
            for review in reviews
            if (
                review.evidence_state
                == "ESTABLISHED"
                and review.positive_work_ids
            )
        )

        evidence_packet = (
            NonObviousnessEvidencePacket(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                structural_status=(
                    structural.status
                ),
                vector=inputs.vector,
                established_relations=(
                    established_relations
                ),
                direct_full_claim_prior_art=(
                    closure.full_relation
                    == "ESTABLISHED"
                ),
                evidence_closure_sufficient=(
                    structural.status
                    != "INSUFFICIENT_CLOSURE"
                ),
            )
        )

        if (
            readiness.readiness
            == "READY_FOR_NONOBVIOUSNESS_REVIEW"
        ):
            independent_adjudication = (
                review_and_compile_nonobviousness_adjudication(
                    backend=adjudication_backend,
                    readiness=readiness,
                    packet=evidence_packet,
                    prior_art=packet,
                )
            )

            adjudication_status = (
                "INDEPENDENT_ADJUDICATION_COMPILED"
            )

            final_adjudication = (
                independent_adjudication.compiled
            )
        else:
            independent_adjudication = None

            (
                adjudication_status,
                final_adjudication,
            ) = (
                compile_forced_adjudication_if_determined(
                    readiness=readiness,
                    packet=evidence_packet,
                )
            )

        if final_adjudication is not None:
            full_state = (
                final_adjudication.verdict
            )
        else:
            full_state = adjudication_status

        state_counts[
            full_state
        ] += 1

        detail_root = (
            detail_dir
            / safe_id(
                claim.claim_id
            )
        )

        write_json(
            detail_root
            / "closure_plan.json",
            closure_plan,
        )

        write_json(
            detail_root
            / "execution_plan.json",
            expanded_plan,
        )

        write_json(
            detail_root
            / "prior_art.json",
            packet,
        )

        write_json(
            detail_root
            / "candidates.json",
            candidates,
        )

        write_json(
            detail_root
            / "slot_reviews.json",
            reviews,
        )

        write_json(
            detail_root
            / "closure_relationships.json",
            relationship_outcome,
        )

        write_json(
            detail_root
            / "structural_and_adjudication.json",
            {
                "conservative_inputs":
                    inputs,
                "closure_relationships":
                    relationship_outcome,
                "closure_compilation":
                    closure_compilation,
                "structural_assessment":
                    structural,
                "readiness":
                    readiness,
                "evidence_packet":
                    evidence_packet,
                "independent_adjudication":
                    independent_adjudication,
                "adjudication_status":
                    adjudication_status,
                "final_adjudication":
                    final_adjudication,
            },
        )

        rows.append(
            {
                "hypothesis_id":
                    claim.hypothesis_id,

                "claim_id":
                    claim.claim_id,

                "claim_text":
                    claim.claim_text,

                "intake_state":
                    "READY_FOR_CLOSURE",

                "closure_states": {
                    "BASE_RELATION":
                        closure.base_relation,
                    "DISTINGUISHING_FACTOR_EFFECT":
                        closure.distinguishing_factor_effect,
                    "BRIDGE_RELATION":
                        closure.bridge_relation,
                    "FULL_RELATION":
                        closure.full_relation,
                },

                "relationship_review_performed":
                    relationship_outcome.review_performed,

                "bridge_kind":
                    relationship_outcome.compiled.bridge_kind,

                "scope_compatible":
                    relationship_outcome.compiled.scope_compatible,

                "relationship_reason_codes":
                    list(
                        relationship_outcome
                        .compiled
                        .reason_codes
                    ),

                "structural_status":
                    structural.status,

                "structural_reason_codes":
                    list(
                        structural.reason_codes
                    ),

                "readiness":
                    readiness.readiness,

                "readiness_reason_codes":
                    list(
                        readiness.reason_codes
                    ),

                "independent_adjudication_performed": (
                    independent_adjudication.review_performed
                    if independent_adjudication
                    is not None
                    else False
                ),

                "adjudication_sanitizer_reason_codes": (
                    list(
                        independent_adjudication
                        .sanitizer_reason_codes
                    )
                    if independent_adjudication
                    is not None
                    else []
                ),

                "adjudication_status":
                    adjudication_status,

                "final_verdict": (
                    final_adjudication.verdict
                    if final_adjudication
                    is not None
                    else None
                ),

                "final_reason_codes": (
                    list(
                        final_adjudication.reason_codes
                    )
                    if final_adjudication
                    is not None
                    else []
                ),

                "conservative_input_reason_codes":
                    list(
                        inputs.reason_codes
                    ),

                "detail_dir":
                    str(detail_root),
            }
        )

    for claim_id in deferred_ids:
        state_counts[
            "DEFERRED_READY_CLAIM_BUDGET"
        ] += 1

        claim = claims_by_id[
            claim_id
        ]

        rows.append(
            {
                "hypothesis_id":
                    claim.hypothesis_id,
                "claim_id":
                    claim.claim_id,
                "claim_text":
                    claim.claim_text,
                "intake_state":
                    "READY_FOR_CLOSURE",
                "full_shadow_state":
                    "DEFERRED_READY_CLAIM_BUDGET",
            }
        )

    result = {
        "schema_version":
            "nonobviousness-full-shadow-v1",

        "shadow_only":
            True,

        "scientific_selection_changed":
            False,

        "source_portfolio_id":
            report.source_portfolio_id,

        "source_query_plan_id":
            plan.plan_id,

        "source_external_report_id":
            report.report_id,

        "source_prior_art_packet_id":
            external_prior.packet_id,

        "source_intake_shadow":
            str(args.intake_shadow),

        "provider_plan":
            str(args.provider_plan),

        "ready_claim_count":
            len(ready_ids),

        "processed_ready_claim_count":
            len(selected_ids),

        "deferred_ready_claim_count":
            len(deferred_ids),

        "full_state_counts":
            dict(
                sorted(
                    state_counts.items()
                )
            ),

        "claims":
            rows,

        "epistemic_policy": {
            "existing_intake_artifact_semantics_preserved":
                True,

            "targeted_closure_is_search_bounded":
                True,

            "query_variants_are_not_evidence":
                True,

            "identity_anchored_negative_coverage_required":
                True,

            "historical_or_external_absence_is_not_novelty":
                True,

            "higher_order_structure_not_inferred_from_free_text":
                True,

            "scope_compatibility_unassessed_fails_closed":
                True,

            "cross_slot_relationship_review_uses_established_positive_evidence_only":
                True,

            "cross_slot_relationship_review_skipped_until_lower_order_closure_complete":
                True,

            "ready_candidate_requires_independent_adjudicator":
                True,

            "independent_adjudicator_uses_established_positive_evidence_only":
                True,

            "adjudicator_cannot_invent_additional_scientific_assumptions":
                True,

            "production_selection_unchanged":
                True,
        },
    }

    write_json(
        args.output,
        result,
    )

    write_json(
        detail_dir
        / "closure_review_prompts.json",
        backend.prompt_records,
    )

    write_json(
        detail_dir
        / "closure_relationship_review_prompts.json",
        relationship_backend.prompt_records,
    )

    write_json(
        detail_dir
        / "nonobviousness_adjudication_prompts.json",
        adjudication_backend.prompt_records,
    )

    print(
        "N9 full non-obviousness shadow built"
    )
    print(
        "Ready claims:",
        len(ready_ids),
    )
    print(
        "Processed:",
        len(selected_ids),
    )
    print(
        "Deferred:",
        len(deferred_ids),
    )
    print(
        "States:",
        result["full_state_counts"],
    )
    print(
        "Scientific selection changed:",
        result[
            "scientific_selection_changed"
        ],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
