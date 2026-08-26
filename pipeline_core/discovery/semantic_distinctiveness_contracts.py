from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyStatus,
)
from pipeline_core.discovery.scientific_distinctiveness_contracts import (
    DistinctivenessEvidencePattern,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


SemanticDimensionLevel = Literal[
    "LOW",
    "MODERATE",
    "HIGH",
    "INDETERMINATE",
]

SemanticDistinctivenessTier = Literal[
    "LOW",
    "MODERATE",
    "HIGH",
    "INDETERMINATE",
]

SemanticReviewConfidence = Literal[
    "LOW",
    "MODERATE",
    "HIGH",
]


class SemanticDimensionAssessment(StrictModel):
    """One evidence-bounded semantic assessment.

    `level` is dimension-specific:

    - conceptual_prior_art_density:
        HIGH means the supplied reviewed evidence densely covers the
        conceptual neighborhood.
    - straightforward_reconstruction:
        HIGH means the proposed claim is largely a direct conjunction or
        recombination of already represented relations.
    - mechanism_switch:
        HIGH means a genuine switch/competition/change in governing
        mechanism is explicitly supported by the supplied claim structure.
    - ranking_or_regime_change:
        HIGH means the hypothesis makes a qualitative ordering, optimum,
        sign, threshold, or regime-change claim.
    - counterfactual_distinctiveness:
        HIGH means the hypothesis isolates a residual effect while holding
        an important alternative explanation/comparator approximately fixed.
    - evidence_role_complementarity:
        HIGH means multiple substantively different evidence roles are
        required to reconstruct the proposed higher-order claim.
    """

    level: SemanticDimensionLevel

    rationale: str = Field(
        min_length=1
    )

    claim_ids: list[str] = Field(
        max_length=8,
    )

    work_ids: list[str] = Field(
        max_length=12,
    )

    @model_validator(mode="after")
    def _canonicalize_exact_duplicate_references(
        self,
    ) -> "SemanticDimensionAssessment":
        # Exact repeated references carry no additional semantic
        # information. Canonicalize them deterministically while
        # preserving first-occurrence order.
        #
        # This deliberately does NOT perform fuzzy matching,
        # prefix matching, typo correction, or ID substitution.
        # Unknown references remain the strict compiler's job and
        # still fail closed.
        self.claim_ids = list(
            dict.fromkeys(
                self.claim_ids
            )
        )

        self.work_ids = list(
            dict.fromkeys(
                self.work_ids
            )
        )

        return self


class SemanticDistinctivenessDraft(StrictModel):
    hypothesis_id: str

    conceptual_prior_art_density: (
        SemanticDimensionAssessment
    )

    straightforward_reconstruction: (
        SemanticDimensionAssessment
    )

    mechanism_switch: (
        SemanticDimensionAssessment
    )

    ranking_or_regime_change: (
        SemanticDimensionAssessment
    )

    counterfactual_distinctiveness: (
        SemanticDimensionAssessment
    )

    evidence_role_complementarity: (
        SemanticDimensionAssessment
    )

    confidence: SemanticReviewConfidence


class SemanticDistinctivenessReview(StrictModel):
    schema_version: Literal[
        "semantic-distinctiveness-review-v2"
    ] = "semantic-distinctiveness-review-v2"

    review_id: str
    review_sha256: str

    hypothesis_id: str

    source_scientific_report_id: str
    source_scientific_report_sha256: str
    source_scientific_review_sha256: str

    source_external_novelty_report_id: str
    source_prior_art_packet_id: str

    source_external_novelty_status: (
        ExternalNoveltyStatus
    )

    source_evidence_pattern: (
        DistinctivenessEvidencePattern
    )

    prompt_version: str
    prompt_sha256: str

    backend_name: str
    requested_model: str
    served_model: str

    review_pass_index: int = Field(
        ge=1
    )

    reference_contract_repair_count: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    reference_contract_repair_issues: list[str] = Field(
        default_factory=list,
        max_length=8,
    )

    conceptual_prior_art_density: (
        SemanticDimensionAssessment
    )

    straightforward_reconstruction: (
        SemanticDimensionAssessment
    )

    mechanism_switch: (
        SemanticDimensionAssessment
    )

    ranking_or_regime_change: (
        SemanticDimensionAssessment
    )

    counterfactual_distinctiveness: (
        SemanticDimensionAssessment
    )

    evidence_role_complementarity: (
        SemanticDimensionAssessment
    )

    overall_tier: SemanticDistinctivenessTier

    overall_tier_aggregation_version: Literal[
        "semantic-distinctiveness-aggregation-v2.1"
    ] = "semantic-distinctiveness-aggregation-v2.1"

    overall_tier_reason_codes: list[str] = Field(
        default_factory=list,
        max_length=8,
    )

    confidence: SemanticReviewConfidence

    referenced_claim_ids: list[str] = Field(
        default_factory=list
    )

    referenced_prior_art_work_ids: list[str] = Field(
        default_factory=list
    )

    diagnostic_only: Literal[True] = True
    retrieval_performed: Literal[False] = False
    action_policy_applied: Literal[False] = False
    scientific_selection_changed: Literal[False] = False
    planner_changed: Literal[False] = False
    novelty_status_changed: Literal[False] = False

    epistemic_scope: Literal[
        "semantic_reasoning_over_supplied_frozen_prior_art_only"
    ] = (
        "semantic_reasoning_over_supplied_frozen_prior_art_only"
    )
