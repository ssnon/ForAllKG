from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_contracts import HypothesisCard, HypothesisPortfolio
from pipeline_core.discovery.reframing.contradiction_resolution import (
    ContradictionResolutionRunReport,
    ScientificContradictionResolutionCandidate,
)
from pipeline_core.discovery.reframing.cross_lane_portfolio import (
    CrossLaneScientificReasoningShadowPortfolio,
)
from pipeline_core.discovery.reframing.proxy_challenge import (
    ProxyChallengeRunReport,
    ScientificProxyChallengeCandidate,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframeCandidate,
    ScientificReframingShadowReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ProductionSourceLane = Literal["RELATIONAL_DISCOVERY", "SCIENTIFIC_REFRAMING"]
ProductionSourceKind = Literal[
    "relational_hypothesis",
    "scientific_reframe",
    "scientific_proxy_challenge",
    "scientific_contradiction_resolution",
]


class ProductionCandidatePrediction(StrictModel):
    observable: str = Field(min_length=1)
    expected_direction: str | None = None
    rationale: str | None = None
    baseline_expectation: str | None = None
    alternative_expectation: str | None = None
    discriminating_outcome: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "ProductionCandidatePrediction":
        relational = self.expected_direction is not None or self.rationale is not None
        contrast = any(
            value is not None
            for value in (
                self.baseline_expectation,
                self.alternative_expectation,
                self.discriminating_outcome,
            )
        )
        if relational == contrast:
            raise ValueError(
                "prediction must use exactly one of relational or model-contrast shape"
            )
        if relational and (self.expected_direction is None or self.rationale is None):
            raise ValueError("relational prediction requires direction and rationale")
        if contrast and (
            self.baseline_expectation is None
            or self.alternative_expectation is None
            or self.discriminating_outcome is None
        ):
            raise ValueError("contrast prediction requires all contrast fields")
        return self


class ProductionCandidateFalsifier(StrictModel):
    observable: str | None = None
    falsifying_outcome: str = Field(min_length=1)


class ProductionCandidateDiscriminatingTest(StrictModel):
    test_design: str = Field(min_length=1)
    primary_observables: list[str] = Field(min_length=1)
    baseline_favoring_outcome: str = Field(min_length=1)
    alternative_favoring_outcome: str = Field(min_length=1)


class ProductionScientificCandidate(StrictModel):
    schema_version: Literal[
        "production-facing-scientific-candidate-v1"
    ] = "production-facing-scientific-candidate-v1"

    candidate_id: str = Field(min_length=1)
    source_entry_id: str = Field(min_length=1)
    source_lane: ProductionSourceLane
    source_object_kind: ProductionSourceKind
    source_schema_version: str = Field(min_length=1)
    source_object_id: str = Field(min_length=1)
    source_scientific_neighborhood_cluster_id: str | None = None
    source_epistemic_status: str = Field(min_length=1)

    title: str = Field(min_length=1)
    reasoning_label: str = Field(min_length=1)
    scientific_proposal: str = Field(min_length=1)
    reasoning_rationale: str = Field(min_length=1)
    premise_statement_ids: list[str] = Field(default_factory=list)
    gap_statement_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    predictions: list[ProductionCandidatePrediction] = Field(default_factory=list)
    falsifiers: list[ProductionCandidateFalsifier] = Field(default_factory=list)
    discriminating_test: ProductionCandidateDiscriminatingTest | None = None
    unresolved_questions: list[str] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    source_object_preserved: Literal[True] = True
    source_authority_preserved: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    redundancy_pruning_performed: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_source_lane(self) -> "ProductionScientificCandidate":
        if self.source_object_kind == "relational_hypothesis":
            if self.source_lane != "RELATIONAL_DISCOVERY":
                raise ValueError("relational hypothesis must remain in relational lane")
        elif self.source_lane != "SCIENTIFIC_REFRAMING":
            raise ValueError("reframing candidate must remain in reframing lane")
        return self


class ProductionCandidateLineageValidation(StrictModel):
    cross_lane_entries_covered_exactly: Literal[True] = True
    relational_source_ids_matched: Literal[True] = True
    reframing_source_ids_matched: Literal[True] = True
    source_task_identity_preserved: Literal[True] = True
    source_context_identity_preserved: Literal[True] = True


class ProductionFacingScientificCandidatePortfolio(StrictModel):
    schema_version: Literal[
        "production-facing-scientific-candidate-portfolio-v1"
    ] = "production-facing-scientific-candidate-portfolio-v1"

    portfolio_id: str = Field(min_length=1)
    source_cross_lane_portfolio_id: str = Field(min_length=1)
    source_cross_lane_portfolio_file_sha256: str = Field(min_length=64, max_length=64)
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    question: str = Field(min_length=1)
    domain_profile_id: str = Field(min_length=1)

    candidates: list[ProductionScientificCandidate]
    lineage_validation: ProductionCandidateLineageValidation
    candidate_count: int = Field(ge=0)
    relational_candidate_count: int = Field(ge=0)
    reframing_candidate_count: int = Field(ge=0)
    source_kind_counts: dict[str, int] = Field(default_factory=dict)

    llm_calls_performed: Literal[0] = 0
    shadow_only: Literal[True] = True
    common_downstream_contract_materialized: Literal[True] = True
    source_lane_schemas_replaced: Literal[False] = False
    source_objects_mutated: Literal[False] = False
    cross_lane_synthesis_performed: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    redundancy_pruning_performed: Literal[False] = False
    final_hypothesis_selection_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ProductionFacingScientificCandidatePortfolio":
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must match candidates")
        relational = [c for c in self.candidates if c.source_lane == "RELATIONAL_DISCOVERY"]
        reframing = [c for c in self.candidates if c.source_lane == "SCIENTIFIC_REFRAMING"]
        if self.relational_candidate_count != len(relational):
            raise ValueError("relational_candidate_count mismatch")
        if self.reframing_candidate_count != len(reframing):
            raise ValueError("reframing_candidate_count mismatch")
        counts = Counter(c.source_object_kind for c in self.candidates)
        if dict(sorted(counts.items())) != dict(sorted(self.source_kind_counts.items())):
            raise ValueError("source_kind_counts mismatch")
        ids = [c.candidate_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("production-facing candidate IDs must be unique")
        source_ids = [c.source_object_id for c in self.candidates]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source object IDs must be unique")
        return self


def _stable_id(prefix: str, *parts: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _test(row) -> ProductionCandidateDiscriminatingTest:
    return ProductionCandidateDiscriminatingTest(
        test_design=row.test_design,
        primary_observables=list(row.primary_observables),
        baseline_favoring_outcome=row.baseline_favoring_outcome,
        alternative_favoring_outcome=row.alternative_favoring_outcome,
    )


def _relational_candidate(entry, card: HypothesisCard) -> ProductionScientificCandidate:
    return ProductionScientificCandidate(
        candidate_id=_stable_id("production_candidate", entry.entry_id, card.hypothesis_id),
        source_entry_id=entry.entry_id,
        source_lane=entry.lane_id,
        source_object_kind="relational_hypothesis",
        source_schema_version=card.schema_version,
        source_object_id=card.hypothesis_id,
        source_scientific_neighborhood_cluster_id=entry.source_scientific_neighborhood_cluster_id,
        source_epistemic_status=entry.source_epistemic_status,
        title=card.title,
        reasoning_label=entry.reasoning_label,
        scientific_proposal=card.hypothesis_statement,
        reasoning_rationale=card.inferential_bridge,
        premise_statement_ids=list(card.premise_statement_ids),
        gap_statement_ids=list(card.gap_statement_ids),
        assumptions=list(card.assumptions),
        predictions=[
            ProductionCandidatePrediction(
                observable=row.observable,
                expected_direction=row.expected_direction,
                rationale=row.rationale,
            )
            for row in card.predicted_observations
        ],
        falsifiers=[
            ProductionCandidateFalsifier(
                observable=row.observable,
                falsifying_outcome=row.falsifying_outcome,
            )
            for row in card.falsification_criteria
        ],
    )


def _reframe_candidate(entry, candidate: ScientificReframeCandidate) -> ProductionScientificCandidate:
    return ProductionScientificCandidate(
        candidate_id=_stable_id("production_candidate", entry.entry_id, candidate.reframe_id),
        source_entry_id=entry.entry_id,
        source_lane=entry.lane_id,
        source_object_kind="scientific_reframe",
        source_schema_version=candidate.schema_version,
        source_object_id=candidate.reframe_id,
        source_scientific_neighborhood_cluster_id=entry.source_scientific_neighborhood_cluster_id,
        source_epistemic_status=entry.source_epistemic_status,
        title=candidate.title,
        reasoning_label=entry.reasoning_label,
        scientific_proposal=candidate.alternative_model.summary,
        reasoning_rationale=candidate.challenged_assumption,
        premise_statement_ids=list(candidate.premise_statement_ids),
        gap_statement_ids=list(candidate.gap_statement_ids),
        assumptions=list(candidate.alternative_model.assumptions),
        predictions=[
            ProductionCandidatePrediction(
                observable=row.observable,
                baseline_expectation=row.baseline_expectation,
                alternative_expectation=row.alternative_expectation,
                discriminating_outcome=row.discriminating_outcome,
            )
            for row in candidate.differential_predictions
        ],
        falsifiers=[
            ProductionCandidateFalsifier(falsifying_outcome=row.falsifying_outcome)
            for row in candidate.falsifiers
        ],
        discriminating_test=_test(candidate.discriminating_test),
        unresolved_questions=list(candidate.unresolved_questions),
    )


def _proxy_candidate(entry, candidate: ScientificProxyChallengeCandidate) -> ProductionScientificCandidate:
    rationale = (
        f"Challenge the assumption that {candidate.challenged_proxy_assumption} "
        f"The observable under challenge is {candidate.challenged_observable}; "
        f"the intended construct is {candidate.target_construct}."
    )
    return ProductionScientificCandidate(
        candidate_id=_stable_id("production_candidate", entry.entry_id, candidate.candidate_id),
        source_entry_id=entry.entry_id,
        source_lane=entry.lane_id,
        source_object_kind="scientific_proxy_challenge",
        source_schema_version=candidate.schema_version,
        source_object_id=candidate.candidate_id,
        source_scientific_neighborhood_cluster_id=entry.source_scientific_neighborhood_cluster_id,
        source_epistemic_status=entry.source_epistemic_status,
        title=candidate.title,
        reasoning_label=entry.reasoning_label,
        scientific_proposal=candidate.alternative_model.summary,
        reasoning_rationale=rationale,
        premise_statement_ids=list(candidate.premise_statement_ids),
        gap_statement_ids=list(candidate.gap_statement_ids),
        assumptions=list(candidate.alternative_model.assumptions),
        predictions=[
            ProductionCandidatePrediction(
                observable=row.observable,
                baseline_expectation=row.baseline_expectation,
                alternative_expectation=row.alternative_expectation,
                discriminating_outcome=row.discriminating_outcome,
            )
            for row in candidate.differential_predictions
        ],
        falsifiers=[
            ProductionCandidateFalsifier(falsifying_outcome=row.falsifying_outcome)
            for row in candidate.falsifiers
        ],
        discriminating_test=_test(candidate.discriminating_test),
        unresolved_questions=list(candidate.unresolved_questions),
    )


def _contradiction_candidate(
    entry, candidate: ScientificContradictionResolutionCandidate
) -> ProductionScientificCandidate:
    rationale = (
        f"Apparent tension: {candidate.apparent_contradiction} "
        f"Proposed reconciliation principle: {candidate.resolution_principle}"
    )
    return ProductionScientificCandidate(
        candidate_id=_stable_id("production_candidate", entry.entry_id, candidate.candidate_id),
        source_entry_id=entry.entry_id,
        source_lane=entry.lane_id,
        source_object_kind="scientific_contradiction_resolution",
        source_schema_version=candidate.schema_version,
        source_object_id=candidate.candidate_id,
        source_scientific_neighborhood_cluster_id=entry.source_scientific_neighborhood_cluster_id,
        source_epistemic_status=entry.source_epistemic_status,
        title=candidate.title,
        reasoning_label=entry.reasoning_label,
        scientific_proposal=candidate.resolution_model.summary,
        reasoning_rationale=rationale,
        premise_statement_ids=list(candidate.premise_statement_ids),
        gap_statement_ids=list(candidate.gap_statement_ids),
        assumptions=list(candidate.resolution_model.assumptions),
        predictions=[
            ProductionCandidatePrediction(
                observable=row.observable,
                baseline_expectation=row.baseline_expectation,
                alternative_expectation=row.alternative_expectation,
                discriminating_outcome=row.discriminating_outcome,
            )
            for row in candidate.differential_predictions
        ],
        falsifiers=[
            ProductionCandidateFalsifier(falsifying_outcome=row.falsifying_outcome)
            for row in candidate.falsifiers
        ],
        discriminating_test=_test(candidate.discriminating_test),
        unresolved_questions=list(candidate.unresolved_questions),
    )


def _source_map(
    *,
    relational_portfolio: HypothesisPortfolio,
    reframe_shadow: ScientificReframingShadowReport,
    proxy_shadow: ProxyChallengeRunReport | None,
    contradiction_shadow: ContradictionResolutionRunReport | None,
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for card in relational_portfolio.hypotheses:
        rows[card.hypothesis_id] = card
    for candidate in reframe_shadow.candidates:
        rows[candidate.reframe_id] = candidate
    if proxy_shadow is not None:
        for candidate in proxy_shadow.candidates:
            rows[candidate.candidate_id] = candidate
    if contradiction_shadow is not None:
        for candidate in contradiction_shadow.candidates:
            rows[candidate.candidate_id] = candidate
    return rows


def build_production_facing_candidate_portfolio(
    *,
    cross_lane_portfolio: CrossLaneScientificReasoningShadowPortfolio,
    relational_portfolio: HypothesisPortfolio,
    reframe_shadow: ScientificReframingShadowReport,
    proxy_shadow: ProxyChallengeRunReport | None,
    contradiction_shadow: ContradictionResolutionRunReport | None,
    source_cross_lane_portfolio_file_sha256: str,
) -> ProductionFacingScientificCandidatePortfolio:
    if relational_portfolio.source_context_id != cross_lane_portfolio.source_context_id:
        raise ValueError("relational portfolio context mismatch")
    if reframe_shadow.source_task_id != cross_lane_portfolio.source_task_id:
        raise ValueError("reframe shadow task mismatch")
    if reframe_shadow.source_context_id != cross_lane_portfolio.source_context_id:
        raise ValueError("reframe shadow context mismatch")
    for name, report in (("proxy", proxy_shadow), ("contradiction", contradiction_shadow)):
        if report is None:
            continue
        if report.source_task_id != cross_lane_portfolio.source_task_id:
            raise ValueError(f"{name} shadow task mismatch")
        if report.source_context_id != cross_lane_portfolio.source_context_id:
            raise ValueError(f"{name} shadow context mismatch")

    sources = _source_map(
        relational_portfolio=relational_portfolio,
        reframe_shadow=reframe_shadow,
        proxy_shadow=proxy_shadow,
        contradiction_shadow=contradiction_shadow,
    )
    entry_ids = {entry.source_object_id for entry in cross_lane_portfolio.entries}
    if entry_ids != set(sources):
        missing = sorted(entry_ids - set(sources))
        extra = sorted(set(sources) - entry_ids)
        raise ValueError(
            "cross-lane/source candidate mismatch: "
            f"missing_source_objects={missing}; extra_source_objects={extra}"
        )

    candidates: list[ProductionScientificCandidate] = []
    for entry in cross_lane_portfolio.entries:
        source = sources[entry.source_object_id]
        if isinstance(source, HypothesisCard):
            candidate = _relational_candidate(entry, source)
        elif isinstance(source, ScientificReframeCandidate):
            candidate = _reframe_candidate(entry, source)
        elif isinstance(source, ScientificProxyChallengeCandidate):
            candidate = _proxy_candidate(entry, source)
        elif isinstance(source, ScientificContradictionResolutionCandidate):
            candidate = _contradiction_candidate(entry, source)
        else:
            raise TypeError(f"unsupported source candidate type: {type(source)!r}")
        candidates.append(candidate)

    relational_count = sum(c.source_lane == "RELATIONAL_DISCOVERY" for c in candidates)
    reframing_count = len(candidates) - relational_count
    source_kind_counts = Counter(c.source_object_kind for c in candidates)
    portfolio_id = _stable_id(
        "production_facing_scientific_candidate_portfolio",
        cross_lane_portfolio.portfolio_id,
        source_cross_lane_portfolio_file_sha256,
        *(candidate.candidate_id for candidate in candidates),
    )
    return ProductionFacingScientificCandidatePortfolio(
        portfolio_id=portfolio_id,
        source_cross_lane_portfolio_id=cross_lane_portfolio.portfolio_id,
        source_cross_lane_portfolio_file_sha256=source_cross_lane_portfolio_file_sha256,
        source_task_id=cross_lane_portfolio.source_task_id,
        source_context_id=cross_lane_portfolio.source_context_id,
        source_context_sha256=cross_lane_portfolio.source_context_sha256,
        question=cross_lane_portfolio.question,
        domain_profile_id=cross_lane_portfolio.domain_profile_id,
        candidates=candidates,
        lineage_validation=ProductionCandidateLineageValidation(),
        candidate_count=len(candidates),
        relational_candidate_count=relational_count,
        reframing_candidate_count=reframing_count,
        source_kind_counts=dict(sorted(source_kind_counts.items())),
    )


__all__ = [
    "ProductionCandidateDiscriminatingTest",
    "ProductionCandidateFalsifier",
    "ProductionCandidatePrediction",
    "ProductionScientificCandidate",
    "ProductionFacingScientificCandidatePortfolio",
    "build_production_facing_candidate_portfolio",
]
