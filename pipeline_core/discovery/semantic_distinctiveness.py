from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.scientific_distinctiveness_contracts import (
    ScientificDistinctivenessReport,
    ScientificDistinctivenessReview,
)
from pipeline_core.discovery.semantic_distinctiveness_contracts import (
    SemanticDistinctivenessDraft,
    SemanticDistinctivenessReview,
    SemanticDistinctivenessTier,
)
from pipeline_core.discovery.semantic_distinctiveness_prompt import (
    SemanticDistinctivenessPrompt,
)


SEMANTIC_DISTINCTIVENESS_AGGREGATION_VERSION = (
    "semantic-distinctiveness-aggregation-v2.1"
)


_DIMENSION_FIELDS = (
    "conceptual_prior_art_density",
    "straightforward_reconstruction",
    "mechanism_switch",
    "ranking_or_regime_change",
    "counterfactual_distinctiveness",
    "evidence_role_complementarity",
)


_STRONG_STRUCTURE_FIELDS = (
    "mechanism_switch",
    "ranking_or_regime_change",
    "counterfactual_distinctiveness",
)


def _canonical_json(
    value: object,
) -> str:
    if hasattr(
        value,
        "model_dump",
    ):
        value = value.model_dump(
            mode="json"
        )

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(
    value: object,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            value
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(
            part
        )
        for part in parts
    ).encode(
        "utf-8"
    )

    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def _ordered_unique(
    values,
) -> list[str]:
    result = []
    seen = set()

    for value in values:
        value = str(
            value
        )

        if (
            not value
            or value in seen
        ):
            continue

        seen.add(
            value
        )

        result.append(
            value
        )

    return result


def derive_semantic_distinctiveness_tier(
    *,
    scientific_review: ScientificDistinctivenessReview,
    draft: SemanticDistinctivenessDraft,
) -> tuple[
    SemanticDistinctivenessTier,
    list[str],
]:
    """Deterministic semantic-distinctiveness aggregation v2.1.

    Semantic roles:

    Primary structural signals:
      - mechanism_switch
      - ranking_or_regime_change
      - counterfactual_distinctiveness

    Negative reconstructability signal:
      - straightforward_reconstruction

    Context/support diagnostics:
      - conceptual_prior_art_density
      - evidence_role_complementarity

    Aggregation principles:

    1. Direct prior-art saturation is LOW.
    2. Search-coverage-limited evidence is INDETERMINATE.
    3. HIGH straightforward reconstruction with no primary structural
       HIGH is LOW.
    4. HIGH straightforward reconstruction caps a primary structural
       HIGH at MODERATE.
    5. One or more primary structural HIGH dimensions establish HIGH
       when straightforward reconstruction is LOW or MODERATE.
    6. Prior-art density and evidence-role complementarity remain
       diagnostic context; neither is a mandatory HIGH gate or veto.
    7. Otherwise the hypothesis remains MODERATE.
    """

    reconstruction = (
        draft
        .straightforward_reconstruction
        .level
    )

    primary_high = [
        field_name
        for field_name
        in _STRONG_STRUCTURE_FIELDS
        if (
            getattr(
                draft,
                field_name,
            ).level
            == "HIGH"
        )
    ]


    # ------------------------------------------------------------
    # Positive direct prior-art evidence.
    # ------------------------------------------------------------

    if (
        scientific_review.evidence_pattern
        == "DIRECT_PRIOR_ART_SATURATED"
    ):
        return (
            "LOW",
            [
                "DIRECT_PRIOR_ART_SATURATED"
            ],
        )


    # ------------------------------------------------------------
    # Insufficient search coverage cannot establish positive
    # scientific distinctiveness from missing prior art.
    # ------------------------------------------------------------

    if (
        scientific_review.evidence_pattern
        == "SEARCH_COVERAGE_LIMITED"
    ):
        return (
            "INDETERMINATE",
            [
                (
                    "SEARCH_COVERAGE_LIMITED_"
                    "NO_POSITIVE_LOW_EVIDENCE"
                )
            ],
        )


    # ------------------------------------------------------------
    # Critical semantic structure itself unavailable.
    # ------------------------------------------------------------

    if (
        reconstruction
        == "INDETERMINATE"
        and not primary_high
    ):
        return (
            "INDETERMINATE",
            [
                "CRITICAL_SEMANTIC_STRUCTURE_INDETERMINATE"
            ],
        )


    # ------------------------------------------------------------
    # Readily reconstructed and no genuinely strong structural
    # distinction.
    # ------------------------------------------------------------

    if (
        reconstruction
        == "HIGH"
        and not primary_high
    ):
        return (
            "LOW",
            [
                "HIGH_STRAIGHTFORWARD_RECONSTRUCTION",
                "NO_HIGH_PRIMARY_STRUCTURAL_FEATURE",
            ],
        )


    # ------------------------------------------------------------
    # Strong structural content exists, but the proposed relation is
    # already nearly reconstructed from supplied evidence.
    # ------------------------------------------------------------

    if (
        reconstruction
        == "HIGH"
        and primary_high
    ):
        return (
            "MODERATE",
            [
                "HIGH_PRIMARY_STRUCTURAL_FEATURE",
                "HIGH_STRAIGHTFORWARD_RECONSTRUCTION_CAP",
                *[
                    f"PRIMARY_HIGH:{field_name}"
                    for field_name
                    in primary_high
                ],
            ],
        )


    # ------------------------------------------------------------
    # Central v2.1 rule:
    #
    # a genuine mechanism switch, ranking/regime change, or strong
    # counterfactual isolation may establish HIGH when it is not
    # already straightforwardly reconstructed.
    #
    # Density and evidence-role complementarity remain recorded
    # dimensions but do not gate HIGH.
    # ------------------------------------------------------------

    if (
        reconstruction
        in {
            "LOW",
            "MODERATE",
        }
        and primary_high
    ):
        return (
            "HIGH",
            [
                "HIGH_PRIMARY_STRUCTURAL_FEATURE",
                (
                    "NOT_HIGH_STRAIGHTFORWARD_"
                    "RECONSTRUCTION"
                ),
                *[
                    f"PRIMARY_HIGH:{field_name}"
                    for field_name
                    in primary_high
                ],
            ],
        )


    return (
        "MODERATE",
        [
            "NO_DECISIVE_LOW_OR_HIGH_RULE"
        ],
    )

def compile_semantic_distinctiveness_review(
    *,
    scientific_report: ScientificDistinctivenessReport,
    scientific_review: ScientificDistinctivenessReview,
    prompt: SemanticDistinctivenessPrompt,
    draft: SemanticDistinctivenessDraft,
    backend_name: str,
    requested_model: str,
    served_model: str,
    review_pass_index: int,
    reference_contract_repair_count: int = 0,
    reference_contract_repair_issues: list[str] | None = None,
) -> SemanticDistinctivenessReview:

    if (
        draft.hypothesis_id
        != scientific_review.hypothesis_id
    ):
        raise ValueError(
            "semantic draft hypothesis_id mismatch"
        )

    if (
        prompt.hypothesis_id
        != scientific_review.hypothesis_id
    ):
        raise ValueError(
            "semantic prompt hypothesis_id mismatch"
        )

    allowed_claim_ids = set(
        prompt.allowed_claim_ids
    )

    allowed_work_ids = set(
        prompt.allowed_work_ids
    )

    referenced_claim_ids = []
    referenced_work_ids = []

    for field_name in _DIMENSION_FIELDS:
        assessment = getattr(
            draft,
            field_name,
        )

        unknown_claim_ids = (
            set(
                assessment.claim_ids
            )
            - allowed_claim_ids
        )

        if unknown_claim_ids:
            raise ValueError(
                "semantic dimension references unknown claim IDs: "
                f"{sorted(unknown_claim_ids)}"
            )

        unknown_work_ids = (
            set(
                assessment.work_ids
            )
            - allowed_work_ids
        )

        if unknown_work_ids:
            raise ValueError(
                "semantic dimension references unknown work IDs: "
                f"{sorted(unknown_work_ids)}"
            )

        referenced_claim_ids.extend(
            assessment.claim_ids
        )

        referenced_work_ids.extend(
            assessment.work_ids
        )


    overall_tier, tier_reason_codes = (
        derive_semantic_distinctiveness_tier(
            scientific_review=
                scientific_review,

            draft=
                draft,
        )
    )


    scientific_review_sha256 = (
        _sha256_json(
            scientific_review
        )
    )

    review_id = _stable_id(
        "semantic_distinctiveness_review",
        scientific_report.report_id,
        scientific_review.hypothesis_id,
        prompt.prompt_sha256,
        requested_model,
        review_pass_index,
    )


    body = {
        "schema_version":
            "semantic-distinctiveness-review-v2",

        "review_id":
            review_id,

        "hypothesis_id":
            scientific_review.hypothesis_id,

        "source_scientific_report_id":
            scientific_report.report_id,

        "source_scientific_report_sha256":
            scientific_report.report_sha256,

        "source_scientific_review_sha256":
            scientific_review_sha256,

        "source_external_novelty_report_id":
            (
                scientific_report
                .source_external_novelty_report_id
            ),

        "source_prior_art_packet_id":
            (
                scientific_report
                .source_prior_art_packet_id
            ),

        "source_external_novelty_status":
            (
                scientific_review
                .external_novelty_status
            ),

        "source_evidence_pattern":
            scientific_review.evidence_pattern,

        "prompt_version":
            prompt.prompt_version,

        "prompt_sha256":
            prompt.prompt_sha256,

        "backend_name":
            str(
                backend_name
            ),

        "requested_model":
            str(
                requested_model
            ),

        "served_model":
            str(
                served_model
            ),

        "review_pass_index":
            int(
                review_pass_index
            ),

        "reference_contract_repair_count":
            int(
                reference_contract_repair_count
            ),

        "reference_contract_repair_issues":
            list(
                reference_contract_repair_issues
                or []
            ),

        "conceptual_prior_art_density":
            (
                draft
                .conceptual_prior_art_density
                .model_dump(
                    mode="json"
                )
            ),

        "straightforward_reconstruction":
            (
                draft
                .straightforward_reconstruction
                .model_dump(
                    mode="json"
                )
            ),

        "mechanism_switch":
            (
                draft
                .mechanism_switch
                .model_dump(
                    mode="json"
                )
            ),

        "ranking_or_regime_change":
            (
                draft
                .ranking_or_regime_change
                .model_dump(
                    mode="json"
                )
            ),

        "counterfactual_distinctiveness":
            (
                draft
                .counterfactual_distinctiveness
                .model_dump(
                    mode="json"
                )
            ),

        "evidence_role_complementarity":
            (
                draft
                .evidence_role_complementarity
                .model_dump(
                    mode="json"
                )
            ),

        "overall_tier":
            overall_tier,

        "overall_tier_aggregation_version":
            (
                SEMANTIC_DISTINCTIVENESS_AGGREGATION_VERSION
            ),

        "overall_tier_reason_codes":
            tier_reason_codes,

        "confidence":
            draft.confidence,

        "referenced_claim_ids":
            _ordered_unique(
                referenced_claim_ids
            ),

        "referenced_prior_art_work_ids":
            _ordered_unique(
                referenced_work_ids
            ),

        "diagnostic_only":
            True,

        "retrieval_performed":
            False,

        "action_policy_applied":
            False,

        "scientific_selection_changed":
            False,

        "planner_changed":
            False,

        "novelty_status_changed":
            False,

        "epistemic_scope": (
            "semantic_reasoning_over_supplied_"
            "frozen_prior_art_only"
        ),
    }


    return (
        SemanticDistinctivenessReview(
            **body,
            review_sha256=_sha256_json(
                body
            ),
        )
    )
