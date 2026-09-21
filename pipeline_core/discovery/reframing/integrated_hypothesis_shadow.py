from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.cross_lane_synthesis import (
    CrossLaneSynthesisShadowReport,
    CrossLaneSynthesizedHypothesis,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionFacingScientificCandidatePortfolio,
    ProductionScientificCandidate,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


IntegratedHypothesisOrigin = Literal["LEGACY_RELATIONAL", "CROSS_LANE_SYNTHESIS"]


class IntegratedHypothesisPrediction(StrictModel):
    observable: str = Field(min_length=1)
    expected_result: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    source_candidate_ids: list[str] = Field(min_length=1)


class IntegratedHypothesisFalsifier(StrictModel):
    observable: str | None = None
    falsifying_outcome: str = Field(min_length=1)
    source_candidate_ids: list[str] = Field(min_length=1)


class IntegratedHypothesisDiscriminatingTest(StrictModel):
    test_design: str = Field(min_length=1)
    primary_observables: list[str] = Field(min_length=1)
    source_candidate_ids: list[str] = Field(min_length=1)
    favoring_outcomes: list[str] = Field(min_length=1)


class IntegratedShadowHypothesis(StrictModel):
    schema_version: Literal[
        "integrated-scientific-hypothesis-shadow-entry-v1"
    ] = "integrated-scientific-hypothesis-shadow-entry-v1"

    entry_id: str = Field(min_length=1)
    origin: IntegratedHypothesisOrigin
    title: str = Field(min_length=1)
    hypothesis_statement: str = Field(min_length=1)
    reasoning_rationale: str = Field(min_length=1)
    premise_statement_ids: list[str] = Field(default_factory=list)
    gap_statement_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    predictions: list[IntegratedHypothesisPrediction] = Field(default_factory=list)
    falsifiers: list[IntegratedHypothesisFalsifier] = Field(default_factory=list)
    discriminating_test: IntegratedHypothesisDiscriminatingTest | None = None
    unresolved_questions: list[str] = Field(default_factory=list)

    source_candidate_ids: list[str] = Field(min_length=1)
    source_object_ids: list[str] = Field(min_length=1)
    source_lanes: list[Literal["RELATIONAL_DISCOVERY", "SCIENTIFIC_REFRAMING"]] = Field(
        min_length=1
    )
    synthesis_kind: str | None = None

    shadow_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_origin(self) -> "IntegratedShadowHypothesis":
        if len(self.source_candidate_ids) != len(set(self.source_candidate_ids)):
            raise ValueError("source_candidate_ids must be unique")
        if len(self.source_object_ids) != len(set(self.source_object_ids)):
            raise ValueError("source_object_ids must be unique")
        if self.origin == "LEGACY_RELATIONAL":
            if self.source_lanes != ["RELATIONAL_DISCOVERY"]:
                raise ValueError("legacy relational entry must remain in relational lane")
            if self.synthesis_kind is not None:
                raise ValueError("legacy relational entry must not carry synthesis_kind")
        else:
            if set(self.source_lanes) != {"RELATIONAL_DISCOVERY", "SCIENTIFIC_REFRAMING"}:
                raise ValueError("synthesized entry must integrate both source lanes")
            if not self.synthesis_kind:
                raise ValueError("synthesized entry requires synthesis_kind")
        return self


class IntegratedShadowLineage(StrictModel):
    source_candidate_portfolio_id_matched: Literal[True] = True
    source_candidate_portfolio_sha256_matched: Literal[True] = True
    source_task_identity_preserved: Literal[True] = True
    source_context_identity_preserved: Literal[True] = True
    synthesis_source_candidate_ids_resolved: Literal[True] = True
    legacy_relational_candidates_preserved_exactly: Literal[True] = True
    raw_reframing_candidates_preserved_in_source_portfolio: Literal[True] = True
    synthesized_hypotheses_preserved_exactly: Literal[True] = True


class IntegratedScientificHypothesisShadowPortfolio(StrictModel):
    schema_version: Literal[
        "integrated-scientific-hypothesis-shadow-portfolio-v1"
    ] = "integrated-scientific-hypothesis-shadow-portfolio-v1"

    portfolio_id: str = Field(min_length=1)
    source_candidate_portfolio_id: str = Field(min_length=1)
    source_candidate_portfolio_sha256: str = Field(min_length=64, max_length=64)
    source_synthesis_report_id: str = Field(min_length=1)
    source_synthesis_report_sha256: str = Field(min_length=64, max_length=64)
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    question: str = Field(min_length=1)

    legacy_hypotheses: list[IntegratedShadowHypothesis]
    integrated_hypotheses: list[IntegratedShadowHypothesis]
    synthesized_additions: list[IntegratedShadowHypothesis]
    lineage_validation: IntegratedShadowLineage

    legacy_hypothesis_count: int = Field(ge=0)
    integrated_hypothesis_count: int = Field(ge=0)
    synthesized_addition_count: int = Field(ge=0)
    raw_reframing_source_candidate_count: int = Field(ge=0)
    source_candidate_count: int = Field(ge=0)
    source_kind_counts: dict[str, int] = Field(default_factory=dict)

    synthesis_covered_relational_candidate_ids: list[str] = Field(default_factory=list)
    synthesis_covered_reframing_candidate_ids: list[str] = Field(default_factory=list)
    synthesis_uncovered_relational_candidate_ids: list[str] = Field(default_factory=list)
    synthesis_uncovered_reframing_candidate_ids: list[str] = Field(default_factory=list)

    llm_calls_performed: Literal[0] = 0
    shadow_only: Literal[True] = True
    presentation_policy: Literal[
        "legacy_relational_plus_cross_lane_syntheses"
    ] = "legacy_relational_plus_cross_lane_syntheses"
    raw_reframing_candidates_presented_as_final_hypotheses: Literal[False] = False
    raw_reframing_candidates_discarded: Literal[False] = False
    synthesized_hypotheses_selected_or_ranked: Literal[False] = False
    source_candidates_ranked: Literal[False] = False
    source_candidates_pruned: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    final_production_selection_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    shadow_output_differs_from_legacy: bool

    @model_validator(mode="after")
    def validate_counts(self) -> "IntegratedScientificHypothesisShadowPortfolio":
        if self.legacy_hypothesis_count != len(self.legacy_hypotheses):
            raise ValueError("legacy_hypothesis_count mismatch")
        if self.integrated_hypothesis_count != len(self.integrated_hypotheses):
            raise ValueError("integrated_hypothesis_count mismatch")
        if self.synthesized_addition_count != len(self.synthesized_additions):
            raise ValueError("synthesized_addition_count mismatch")
        if self.integrated_hypotheses != self.legacy_hypotheses + self.synthesized_additions:
            raise ValueError("integrated view must equal legacy view plus synthesized additions")
        if self.shadow_output_differs_from_legacy != bool(self.synthesized_additions):
            raise ValueError("shadow_output_differs_from_legacy mismatch")
        ids = [row.entry_id for row in self.integrated_hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("integrated hypothesis entry IDs must be unique")
        return self


def _canonical_json(value: object) -> str:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _legacy_prediction(candidate: ProductionScientificCandidate, row) -> IntegratedHypothesisPrediction:
    if row.expected_direction is None or row.rationale is None:
        raise ValueError("legacy relational prediction must use relational prediction shape")
    return IntegratedHypothesisPrediction(
        observable=row.observable,
        expected_result=row.expected_direction,
        rationale=row.rationale,
        source_candidate_ids=[candidate.candidate_id],
    )


def _legacy_entry(candidate: ProductionScientificCandidate) -> IntegratedShadowHypothesis:
    if candidate.source_lane != "RELATIONAL_DISCOVERY":
        raise ValueError("legacy view can contain relational candidates only")
    return IntegratedShadowHypothesis(
        entry_id=_stable_id("integrated_shadow_hypothesis", "legacy", candidate.candidate_id),
        origin="LEGACY_RELATIONAL",
        title=candidate.title,
        hypothesis_statement=candidate.scientific_proposal,
        reasoning_rationale=candidate.reasoning_rationale,
        premise_statement_ids=list(candidate.premise_statement_ids),
        gap_statement_ids=list(candidate.gap_statement_ids),
        assumptions=list(candidate.assumptions),
        predictions=[_legacy_prediction(candidate, row) for row in candidate.predictions],
        falsifiers=[
            IntegratedHypothesisFalsifier(
                observable=row.observable,
                falsifying_outcome=row.falsifying_outcome,
                source_candidate_ids=[candidate.candidate_id],
            )
            for row in candidate.falsifiers
        ],
        discriminating_test=None,
        unresolved_questions=list(candidate.unresolved_questions),
        source_candidate_ids=[candidate.candidate_id],
        source_object_ids=[candidate.source_object_id],
        source_lanes=["RELATIONAL_DISCOVERY"],
    )


def _synthesis_entry(row: CrossLaneSynthesizedHypothesis) -> IntegratedShadowHypothesis:
    return IntegratedShadowHypothesis(
        entry_id=_stable_id("integrated_shadow_hypothesis", "synthesis", row.hypothesis_id),
        origin="CROSS_LANE_SYNTHESIS",
        title=row.title,
        hypothesis_statement=row.hypothesis_statement,
        reasoning_rationale=row.synthesis_rationale,
        premise_statement_ids=list(row.premise_statement_ids),
        gap_statement_ids=list(row.gap_statement_ids),
        assumptions=list(row.assumptions),
        predictions=[
            IntegratedHypothesisPrediction(
                observable=item.observable,
                expected_result=item.expected_result,
                rationale=item.rationale,
                source_candidate_ids=list(item.source_candidate_ids),
            )
            for item in row.predictions
        ],
        falsifiers=[
            IntegratedHypothesisFalsifier(
                observable=item.observable,
                falsifying_outcome=item.falsifying_outcome,
                source_candidate_ids=list(item.source_candidate_ids),
            )
            for item in row.falsifiers
        ],
        discriminating_test=IntegratedHypothesisDiscriminatingTest(
            test_design=row.discriminating_test.test_design,
            primary_observables=list(row.discriminating_test.primary_observables),
            source_candidate_ids=list(row.discriminating_test.source_candidate_ids),
            favoring_outcomes=list(row.discriminating_test.favoring_outcomes),
        ),
        unresolved_questions=list(row.unresolved_questions),
        source_candidate_ids=list(row.source_candidate_ids),
        source_object_ids=list(row.source_object_ids),
        source_lanes=list(row.source_lanes),
        synthesis_kind=row.synthesis_kind,
    )


def build_integrated_scientific_hypothesis_shadow(
    *,
    candidate_portfolio: ProductionFacingScientificCandidatePortfolio,
    synthesis_report: CrossLaneSynthesisShadowReport,
) -> IntegratedScientificHypothesisShadowPortfolio:
    if synthesis_report.source_candidate_portfolio_id != candidate_portfolio.portfolio_id:
        raise ValueError("synthesis report source candidate portfolio ID mismatch")
    candidate_sha = _sha256_json(candidate_portfolio)
    if synthesis_report.source_candidate_portfolio_sha256 != candidate_sha:
        raise ValueError("synthesis report source candidate portfolio SHA mismatch")
    if synthesis_report.source_task_id != candidate_portfolio.source_task_id:
        raise ValueError("synthesis report task mismatch")
    if synthesis_report.source_context_id != candidate_portfolio.source_context_id:
        raise ValueError("synthesis report context mismatch")
    if synthesis_report.question != candidate_portfolio.question:
        raise ValueError("synthesis report question mismatch")

    candidate_map = {row.candidate_id: row for row in candidate_portfolio.candidates}
    if set(synthesis_report.source_candidate_ids) != set(candidate_map):
        raise ValueError("synthesis report source candidate inventory mismatch")

    relational = [
        row for row in candidate_portfolio.candidates if row.source_lane == "RELATIONAL_DISCOVERY"
    ]
    reframing = [
        row for row in candidate_portfolio.candidates if row.source_lane == "SCIENTIFIC_REFRAMING"
    ]
    legacy_entries = [_legacy_entry(row) for row in relational]
    synthesis_entries = [_synthesis_entry(row) for row in synthesis_report.hypotheses]

    covered_rel: set[str] = set()
    covered_ref: set[str] = set()
    for hypothesis in synthesis_report.hypotheses:
        for candidate_id in hypothesis.source_candidate_ids:
            source = candidate_map.get(candidate_id)
            if source is None:
                raise ValueError(f"synthesis references unknown source candidate: {candidate_id}")
            if source.source_lane == "RELATIONAL_DISCOVERY":
                covered_rel.add(candidate_id)
            else:
                covered_ref.add(candidate_id)

    relational_ids = {row.candidate_id for row in relational}
    reframing_ids = {row.candidate_id for row in reframing}
    source_kind_counts = Counter(row.source_object_kind for row in candidate_portfolio.candidates)
    synthesis_sha = _sha256_json(synthesis_report)
    portfolio_id = _stable_id(
        "integrated_scientific_hypothesis_shadow",
        candidate_portfolio.portfolio_id,
        synthesis_report.report_id,
        synthesis_sha,
    )
    return IntegratedScientificHypothesisShadowPortfolio(
        portfolio_id=portfolio_id,
        source_candidate_portfolio_id=candidate_portfolio.portfolio_id,
        source_candidate_portfolio_sha256=candidate_sha,
        source_synthesis_report_id=synthesis_report.report_id,
        source_synthesis_report_sha256=synthesis_sha,
        source_task_id=candidate_portfolio.source_task_id,
        source_context_id=candidate_portfolio.source_context_id,
        question=candidate_portfolio.question,
        legacy_hypotheses=legacy_entries,
        integrated_hypotheses=[*legacy_entries, *synthesis_entries],
        synthesized_additions=synthesis_entries,
        lineage_validation=IntegratedShadowLineage(),
        legacy_hypothesis_count=len(legacy_entries),
        integrated_hypothesis_count=len(legacy_entries) + len(synthesis_entries),
        synthesized_addition_count=len(synthesis_entries),
        raw_reframing_source_candidate_count=len(reframing),
        source_candidate_count=candidate_portfolio.candidate_count,
        source_kind_counts=dict(sorted(source_kind_counts.items())),
        synthesis_covered_relational_candidate_ids=sorted(covered_rel),
        synthesis_covered_reframing_candidate_ids=sorted(covered_ref),
        synthesis_uncovered_relational_candidate_ids=sorted(relational_ids - covered_rel),
        synthesis_uncovered_reframing_candidate_ids=sorted(reframing_ids - covered_ref),
        shadow_output_differs_from_legacy=bool(synthesis_entries),
    )


__all__ = [
    "IntegratedHypothesisDiscriminatingTest",
    "IntegratedHypothesisFalsifier",
    "IntegratedHypothesisPrediction",
    "IntegratedScientificHypothesisShadowPortfolio",
    "IntegratedShadowHypothesis",
    "IntegratedShadowLineage",
    "build_integrated_scientific_hypothesis_shadow",
]
