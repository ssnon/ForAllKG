from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.evidence_family_selection import (
    FamilyPremiseSelectionReport,
)
from pipeline_core.discovery.hypothesis_evidence_diversity import (
    HypothesisEvidenceDiversityReport,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyGapPlan,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


EvidencePortfolioRecommendation = Literal[
    "NO_ACTION",
    "REDUNDANCY_REVIEW",
    "ALTERNATIVE_CAPACITY_PRESENT",
]

OperatorPortfolioRecommendation = Literal[
    "NO_OPERATOR_SIGNAL",
    "SINGLE_OPERATOR_OPPORTUNITY",
    "MULTI_OPERATOR_OPPORTUNITY",
]


class PortfolioDiversityShadowReport(StrictModel):
    schema_version: Literal[
        "portfolio-diversity-shadow-report-v1"
    ] = "portfolio-diversity-shadow-report-v1"

    source_evidence_diversity_report_id: str
    source_family_selection_report_id: str | None = None
    source_novelty_gap_plan_id: str | None = None

    hypothesis_count: int
    eligible_statement_count: int
    used_statement_count: int
    unused_eligible_statement_count: int

    exact_premise_set_duplicate_group_count: int
    max_pairwise_statement_jaccard: float
    shared_core_statement_count: int

    family_pseudo_diversity_guard_available: bool
    potential_parent_all_children_redundancy_count: int

    operator_opportunities: list[str] = Field(default_factory=list)
    operator_opportunity_count: int = 0

    evidence_recommendation: EvidencePortfolioRecommendation
    operator_recommendation: OperatorPortfolioRecommendation

    reason_codes: list[str] = Field(default_factory=list)

    evidence_alternative_relevance_established: Literal[False] = False
    operator_usage_inferred_from_hypothesis_text: Literal[False] = False

    diagnostic_only: Literal[True] = True
    scientific_selection_changed: Literal[False] = False
    production_selection_authority: Literal[False] = False
    alpha6_trigger_authority: Literal[False] = False


def build_portfolio_diversity_shadow_report(
    *,
    evidence: HypothesisEvidenceDiversityReport,
    family_selection: FamilyPremiseSelectionReport | None = None,
    novelty_gap_plan: NoveltyGapPlan | None = None,
    high_overlap_threshold: float = 0.80,
) -> PortfolioDiversityShadowReport:
    if not 0.0 <= float(high_overlap_threshold) <= 1.0:
        raise ValueError("high_overlap_threshold must be within [0,1]")

    family_redundancy_count = 0

    if family_selection is not None:
        if (
            family_selection.source_portfolio_id
            != evidence.source_portfolio_id
        ):
            raise ValueError(
                "family-selection/evidence-diversity portfolio mismatch"
            )

        if (
            family_selection.source_context_id
            != evidence.source_context_id
        ):
            raise ValueError(
                "family-selection/evidence-diversity context mismatch"
            )

        family_redundancy_count = (
            family_selection
            .potential_parent_all_children_redundancy_count
        )

    operator_opportunities: list[str] = []

    if novelty_gap_plan is not None:
        if (
            novelty_gap_plan.source_portfolio_id
            != evidence.source_portfolio_id
        ):
            raise ValueError(
                "novelty-gap/evidence-diversity portfolio mismatch"
            )

        for gap in novelty_gap_plan.gaps:
            for operator in gap.sharpening_operators:
                value = str(operator)
                if value not in operator_opportunities:
                    operator_opportunities.append(value)

    duplicate_pressure = (
        evidence.exact_premise_set_duplicate_group_count > 0
    )

    high_overlap_pressure = (
        evidence.hypothesis_count >= 2
        and evidence.max_pairwise_statement_jaccard
        >= float(high_overlap_threshold)
    )

    family_redundancy_pressure = (
        family_redundancy_count > 0
    )

    candidate_alternative_capacity = bool(
        evidence.unused_eligible_statement_ids
    )

    reasons: list[str] = []

    if duplicate_pressure:
        reasons.append(
            "EXACT_PREMISE_SET_DUPLICATION_PRESENT"
        )

    if high_overlap_pressure:
        reasons.append(
            "HIGH_PAIRWISE_PREMISE_OVERLAP_PRESENT"
        )

    if family_redundancy_pressure:
        reasons.append(
            "PARENT_CHILD_PSEUDO_DIVERSITY_RISK_PRESENT"
        )

    if candidate_alternative_capacity:
        reasons.append(
            "UNUSED_ELIGIBLE_EVIDENCE_PRESENT"
        )

    redundancy_pressure = bool(
        duplicate_pressure
        or high_overlap_pressure
        or family_redundancy_pressure
    )

    if redundancy_pressure and candidate_alternative_capacity:
        evidence_recommendation = (
            "ALTERNATIVE_CAPACITY_PRESENT"
        )
        reasons.extend(
            [
                "REDUNDANCY_AND_UNUSED_EVIDENCE_COEXIST",
                "ALTERNATIVE_RELEVANCE_NOT_ESTABLISHED",
            ]
        )

    elif redundancy_pressure:
        evidence_recommendation = (
            "REDUNDANCY_REVIEW"
        )
        reasons.append(
            "NO_UNUSED_ELIGIBLE_EVIDENCE_CAPACITY"
        )

    else:
        evidence_recommendation = "NO_ACTION"
        reasons.append(
            "NO_PORTFOLIO_REDUNDANCY_TRIGGER"
        )

    if not operator_opportunities:
        operator_recommendation = (
            "NO_OPERATOR_SIGNAL"
        )
    elif len(operator_opportunities) == 1:
        operator_recommendation = (
            "SINGLE_OPERATOR_OPPORTUNITY"
        )
        reasons.append(
            "EXPLICIT_SINGLE_SHARPENING_OPERATOR_AVAILABLE"
        )
    else:
        operator_recommendation = (
            "MULTI_OPERATOR_OPPORTUNITY"
        )
        reasons.append(
            "EXPLICIT_MULTIPLE_SHARPENING_OPERATORS_AVAILABLE"
        )

    return PortfolioDiversityShadowReport(
        source_evidence_diversity_report_id=(
            evidence.report_id
        ),
        source_family_selection_report_id=(
            None
            if family_selection is None
            else family_selection.report_id
        ),
        source_novelty_gap_plan_id=(
            None
            if novelty_gap_plan is None
            else novelty_gap_plan.plan_id
        ),
        hypothesis_count=evidence.hypothesis_count,
        eligible_statement_count=(
            evidence.eligible_statement_count
        ),
        used_statement_count=(
            evidence.used_statement_count
        ),
        unused_eligible_statement_count=len(
            evidence.unused_eligible_statement_ids
        ),
        exact_premise_set_duplicate_group_count=(
            evidence.exact_premise_set_duplicate_group_count
        ),
        max_pairwise_statement_jaccard=(
            evidence.max_pairwise_statement_jaccard
        ),
        shared_core_statement_count=(
            evidence.shared_core_statement_count
        ),
        family_pseudo_diversity_guard_available=(
            family_selection is not None
        ),
        potential_parent_all_children_redundancy_count=(
            family_redundancy_count
        ),
        operator_opportunities=(
            operator_opportunities
        ),
        operator_opportunity_count=len(
            operator_opportunities
        ),
        evidence_recommendation=(
            evidence_recommendation
        ),
        operator_recommendation=(
            operator_recommendation
        ),
        reason_codes=list(
            dict.fromkeys(reasons)
        ),
    )


__all__ = [
    "PortfolioDiversityShadowReport",
    "build_portfolio_diversity_shadow_report",
]
