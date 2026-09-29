from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.feasibility.experimental_contracts import (
    ExperimentalRealizabilityReport,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisPortfolio,
)
from pipeline_core.runtime.validation_contracts import (
    ValidationSpecification,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ResearchValueSignal = Literal[
    "STRONG",
    "PARTIAL",
    "WEAK",
    "UNRESOLVED",
]


class ResearchValueDimension(StrictModel):
    signal: ResearchValueSignal
    rationale: str
    source_fields: list[str] = Field(default_factory=list)
    proxy_only: bool = True


class ResearchValueShadowCard(StrictModel):
    schema_version: Literal[
        "research-value-shadow-card-v1"
    ] = "research-value-shadow-card-v1"

    hypothesis_id: str
    hypothesis_type: str

    mechanistic_discrimination: ResearchValueDimension
    two_sided_outcome_informativeness: ResearchValueDimension
    observable_decisiveness: ResearchValueDimension
    information_gain_proxy: ResearchValueDimension
    experimental_resolvability: ResearchValueDimension

    experimental_disposition: str
    relative_cost_burden: str
    relative_effort_burden: str

    value_argument_class: Literal[
        "VALUE_ARGUMENT_SUPPORTED",
        "VALUE_ARGUMENT_PARTIAL",
        "VALUE_ARGUMENT_UNRESOLVED",
        "LOW_INFORMATION_EXPERIMENT_RISK",
    ]

    reason_codes: list[str] = Field(default_factory=list)

    novelty_signal_consumed: Literal[False] = False
    external_novelty_used_as_value_evidence: Literal[False] = False
    conceptual_knownness_used_as_value_evidence: Literal[False] = False

    diagnostic_only: Literal[True] = True
    research_value_selection_authority: Literal[False] = False
    scientific_selection_changed: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ResearchValueShadowReport(StrictModel):
    schema_version: Literal[
        "research-value-shadow-report-v1"
    ] = "research-value-shadow-report-v1"

    source_portfolio_id: str
    hypothesis_count: int
    assessed_count: int
    cards: list[ResearchValueShadowCard] = Field(default_factory=list)

    novelty_signal_consumed: Literal[False] = False
    diagnostic_only: Literal[True] = True
    research_value_selection_authority: Literal[False] = False
    scientific_selection_changed: Literal[False] = False
    production_selection_authority: Literal[False] = False


def _norm(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def _matching_observable_count(card: HypothesisCard) -> tuple[int, int]:
    prediction_observables = {
        _norm(row.observable)
        for row in card.predicted_observations
        if _norm(row.observable)
    }
    falsifier_observables = {
        _norm(row.observable)
        for row in card.falsification_criteria
        if _norm(row.observable)
    }
    return (
        len(prediction_observables & falsifier_observables),
        len(prediction_observables | falsifier_observables),
    )


def _mechanistic_discrimination(
    card: HypothesisCard,
    spec: ValidationSpecification,
) -> ResearchValueDimension:
    comparisons = list(spec.required_comparisons)
    strategy = str(spec.validation_strategy)

    alternative_markers = (
        "alternative",
        "compare",
        "across",
        "while varying",
        "matched",
    )

    explicit_contrast = any(
        any(marker in _norm(value) for marker in alternative_markers)
        for value in comparisons
    )

    if (
        strategy == "mechanism_validation"
        and explicit_contrast
    ):
        return ResearchValueDimension(
            signal="STRONG",
            rationale=(
                "The validation design explicitly compares the proposed "
                "mechanism against an alternative account."
            ),
            source_fields=[
                "validation_strategy",
                "required_comparisons",
            ],
        )

    if comparisons and (
        card.hypothesis_type
        in {
            "mechanistic_extension",
            "context_dependency",
            "design_lever_interaction",
            "descriptor_mediation",
        }
    ):
        return ResearchValueDimension(
            signal="PARTIAL",
            rationale=(
                "The hypothesis has an explicit comparative validation design, "
                "but the comparison is not necessarily a direct competing-"
                "mechanism test."
            ),
            source_fields=[
                "hypothesis_type",
                "required_comparisons",
            ],
        )

    if comparisons:
        return ResearchValueDimension(
            signal="PARTIAL",
            rationale=(
                "A comparative design is present, providing some discriminative "
                "capacity between scientific alternatives."
            ),
            source_fields=[
                "required_comparisons",
            ],
        )

    return ResearchValueDimension(
        signal="UNRESOLVED",
        rationale=(
            "No explicit comparative validation structure establishes that the "
            "experiment distinguishes competing scientific explanations."
        ),
        source_fields=[
            "required_comparisons",
        ],
    )


def _two_sided_outcome_informativeness(
    card: HypothesisCard,
    spec: ValidationSpecification,
) -> ResearchValueDimension:
    matched, union_count = _matching_observable_count(card)

    has_success = bool(spec.success_patterns)
    has_falsification = bool(spec.falsification_patterns)

    if (
        matched > 0
        and has_success
        and has_falsification
    ):
        return ResearchValueDimension(
            signal="STRONG",
            rationale=(
                "At least one observable is explicitly shared by prediction and "
                "falsification criteria, and the validation specification records "
                "both success and falsification patterns."
            ),
            source_fields=[
                "predicted_observations",
                "falsification_criteria",
                "success_patterns",
                "falsification_patterns",
            ],
        )

    if (
        card.predicted_observations
        and card.falsification_criteria
    ):
        return ResearchValueDimension(
            signal="PARTIAL",
            rationale=(
                "Both supporting and falsifying outcomes are represented, but "
                "their observable linkage is incomplete."
            ),
            source_fields=[
                "predicted_observations",
                "falsification_criteria",
            ],
        )

    return ResearchValueDimension(
        signal="WEAK",
        rationale=(
            "The hypothesis lacks a complete two-sided prediction/falsification "
            "structure."
        ),
        source_fields=[
            "predicted_observations",
            "falsification_criteria",
        ],
    )


def _observable_decisiveness(
    card: HypothesisCard,
    spec: ValidationSpecification,
) -> ResearchValueDimension:
    matched, union_count = _matching_observable_count(card)

    if matched > 0 and spec.primary_observables:
        return ResearchValueDimension(
            signal="STRONG",
            rationale=(
                "Prediction and falsification share explicit observables and the "
                "validation plan identifies primary observables for measurement."
            ),
            source_fields=[
                "predicted_observations",
                "falsification_criteria",
                "primary_observables",
            ],
        )

    if matched > 0:
        return ResearchValueDimension(
            signal="PARTIAL",
            rationale=(
                "Prediction and falsification share an observable, but the "
                "downstream validation plan does not yet identify a primary "
                "observable set."
            ),
            source_fields=[
                "predicted_observations",
                "falsification_criteria",
            ],
        )

    if union_count > 0:
        return ResearchValueDimension(
            signal="WEAK",
            rationale=(
                "Observable names exist but do not form a shared decisive "
                "prediction/falsification target."
            ),
            source_fields=[
                "predicted_observations",
                "falsification_criteria",
            ],
        )

    return ResearchValueDimension(
        signal="UNRESOLVED",
        rationale="No explicit observable structure is available.",
        source_fields=[],
    )


def _information_gain_proxy(
    *,
    discrimination: ResearchValueDimension,
    outcome: ResearchValueDimension,
    spec: ValidationSpecification,
) -> ResearchValueDimension:
    if (
        discrimination.signal == "STRONG"
        and outcome.signal == "STRONG"
    ):
        return ResearchValueDimension(
            signal="STRONG",
            rationale=(
                "Proxy only: the design combines explicit scientific "
                "discrimination with interpretable supporting and falsifying "
                "outcomes, so multiple experimental outcomes can update the "
                "scientific model."
            ),
            source_fields=[
                "required_comparisons",
                "success_patterns",
                "falsification_patterns",
            ],
            proxy_only=True,
        )

    if (
        discrimination.signal
        in {"STRONG", "PARTIAL"}
        and outcome.signal
        in {"STRONG", "PARTIAL"}
    ):
        return ResearchValueDimension(
            signal="PARTIAL",
            rationale=(
                "Proxy only: some discriminative and two-sided outcome structure "
                "is present, but expected information gain is not quantified."
            ),
            source_fields=[
                "required_comparisons",
                "success_patterns",
                "falsification_patterns",
            ],
            proxy_only=True,
        )

    return ResearchValueDimension(
        signal="UNRESOLVED",
        rationale=(
            "Expected information gain cannot be supported from the current "
            "structural validation artifacts alone."
        ),
        source_fields=[],
        proxy_only=True,
    )


def _experimental_resolvability(
    report: ExperimentalRealizabilityReport,
) -> ResearchValueDimension:
    disposition = str(report.disposition)

    if disposition == "experimentally_plausible":
        return ResearchValueDimension(
            signal="STRONG",
            rationale=(
                "Existing experimental feasibility assessment considers the "
                "validation system experimentally plausible."
            ),
            source_fields=[
                "experimental.disposition",
            ],
        )

    if disposition in {
        "conditionally_plausible",
        "high_complexity",
    }:
        return ResearchValueDimension(
            signal="PARTIAL",
            rationale=(
                "The experiment appears addressable but carries conditional or "
                "high-complexity execution burden."
            ),
            source_fields=[
                "experimental.disposition",
                "relative_cost_burden",
                "relative_effort_burden",
            ],
        )

    if disposition == "experimentally_implausible":
        return ResearchValueDimension(
            signal="WEAK",
            rationale=(
                "The existing feasibility layer considers the validation system "
                "experimentally implausible."
            ),
            source_fields=[
                "experimental.disposition",
            ],
        )

    return ResearchValueDimension(
        signal="UNRESOLVED",
        rationale=(
            "Experimental resolvability remains unresolved in the feasibility "
            "layer."
        ),
        source_fields=[
            "experimental.disposition",
        ],
    )


def assess_research_value_card(
    *,
    card: HypothesisCard,
    specification: ValidationSpecification,
    experimental: ExperimentalRealizabilityReport,
) -> ResearchValueShadowCard:
    if specification.hypothesis_id != card.hypothesis_id:
        raise ValueError("validation specification / hypothesis mismatch")
    if experimental.hypothesis_id != card.hypothesis_id:
        raise ValueError("experimental report / hypothesis mismatch")

    discrimination = _mechanistic_discrimination(
        card,
        specification,
    )
    outcome = _two_sided_outcome_informativeness(
        card,
        specification,
    )
    decisiveness = _observable_decisiveness(
        card,
        specification,
    )
    information_gain = _information_gain_proxy(
        discrimination=discrimination,
        outcome=outcome,
        spec=specification,
    )
    resolvability = _experimental_resolvability(
        experimental
    )

    reasons: list[str] = []

    strong_or_partial = {
        "STRONG",
        "PARTIAL",
    }

    if (
        discrimination.signal in strong_or_partial
        and outcome.signal == "STRONG"
        and decisiveness.signal in strong_or_partial
        and resolvability.signal in strong_or_partial
    ):
        value_class = "VALUE_ARGUMENT_SUPPORTED"
        reasons.append(
            "DISCRIMINATIVE_TWO_SIDED_RESOLVABLE_STRUCTURE"
        )

    elif (
        outcome.signal == "WEAK"
        or decisiveness.signal == "WEAK"
    ):
        value_class = "LOW_INFORMATION_EXPERIMENT_RISK"
        reasons.append(
            "WEAK_DECISIVE_INFORMATION_STRUCTURE"
        )

    elif (
        discrimination.signal == "UNRESOLVED"
        or resolvability.signal == "UNRESOLVED"
    ):
        value_class = "VALUE_ARGUMENT_UNRESOLVED"
        reasons.append(
            "VALUE_CRITICAL_DIMENSION_UNRESOLVED"
        )

    else:
        value_class = "VALUE_ARGUMENT_PARTIAL"
        reasons.append(
            "PARTIAL_RESEARCH_VALUE_STRUCTURE"
        )

    if (
        str(experimental.relative_cost_burden)
        in {"high", "very_high"}
    ):
        reasons.append(
            "HIGH_RESOURCE_COST_BURDEN"
        )

    if (
        str(experimental.relative_effort_burden)
        in {"high", "very_high"}
    ):
        reasons.append(
            "HIGH_RESOURCE_EFFORT_BURDEN"
        )

    return ResearchValueShadowCard(
        hypothesis_id=card.hypothesis_id,
        hypothesis_type=card.hypothesis_type,
        mechanistic_discrimination=discrimination,
        two_sided_outcome_informativeness=outcome,
        observable_decisiveness=decisiveness,
        information_gain_proxy=information_gain,
        experimental_resolvability=resolvability,
        experimental_disposition=str(
            experimental.disposition
        ),
        relative_cost_burden=str(
            experimental.relative_cost_burden
        ),
        relative_effort_burden=str(
            experimental.relative_effort_burden
        ),
        value_argument_class=value_class,
        reason_codes=reasons,
    )


def assess_research_value_portfolio(
    *,
    portfolio: HypothesisPortfolio,
    specifications: Sequence[ValidationSpecification],
    experimental_reports: Sequence[ExperimentalRealizabilityReport],
) -> ResearchValueShadowReport:
    spec_by_id = {
        row.hypothesis_id: row
        for row in specifications
    }
    exp_by_id = {
        row.hypothesis_id: row
        for row in experimental_reports
    }

    cards = []

    for card in portfolio.hypotheses:
        spec = spec_by_id.get(
            card.hypothesis_id
        )
        exp = exp_by_id.get(
            card.hypothesis_id
        )

        if spec is None or exp is None:
            continue

        cards.append(
            assess_research_value_card(
                card=card,
                specification=spec,
                experimental=exp,
            )
        )

    return ResearchValueShadowReport(
        source_portfolio_id=portfolio.portfolio_id,
        hypothesis_count=len(
            portfolio.hypotheses
        ),
        assessed_count=len(cards),
        cards=cards,
    )


__all__ = [
    "ResearchValueDimension",
    "ResearchValueShadowCard",
    "ResearchValueShadowReport",
    "assess_research_value_card",
    "assess_research_value_portfolio",
]
