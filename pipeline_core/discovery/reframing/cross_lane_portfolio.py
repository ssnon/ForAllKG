from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.reframing.contradiction_resolution import (
    ContradictionResolutionRunReport,
)
from pipeline_core.discovery.reframing.proxy_challenge import ProxyChallengeRunReport
from pipeline_core.discovery.reframing.reasoning_portfolio import (
    UnifiedScientificReasoningShadowPortfolio,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframingShadowReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ReasoningLaneId = Literal[
    "RELATIONAL_DISCOVERY",
    "SCIENTIFIC_REFRAMING",
]

SourceObjectKind = Literal[
    "relational_hypothesis",
    "reframing_candidate",
]


class CrossLaneEntryEnvelope(StrictModel):
    entry_id: str = Field(min_length=1)
    lane_id: ReasoningLaneId
    source_object_kind: SourceObjectKind
    source_schema_version: str = Field(min_length=1)
    source_object_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    reasoning_label: str = Field(min_length=1)
    premise_count: int = Field(ge=0)
    gap_count: int = Field(ge=0)
    prediction_count: int = Field(ge=0)
    falsifier_or_test_count: int = Field(ge=0)
    source_scientific_neighborhood_cluster_id: str | None = None
    source_epistemic_status: str = Field(min_length=1)

    source_object_preserved: Literal[True] = True
    retyped_into_other_lane_schema: Literal[False] = False
    cross_lane_quality_ranking_performed: Literal[False] = False
    cross_lane_redundancy_pruning_performed: Literal[False] = False


class RelationalDiscoveryLaneSnapshot(StrictModel):
    lane_id: Literal["RELATIONAL_DISCOVERY"] = "RELATIONAL_DISCOVERY"
    source_portfolio_id: str = Field(min_length=1)
    source_portfolio_file_sha256: str = Field(min_length=64, max_length=64)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    source_report_id: str = Field(min_length=1)
    source_report_sha256: str = Field(min_length=1)
    domain_profile_id: str = Field(min_length=1)
    hypothesis_ids: list[str] = Field(default_factory=list)
    hypothesis_count: int = Field(ge=0)
    abstention_reason: str | None = None
    novelty_status_counts: dict[str, int] = Field(default_factory=dict)

    source_portfolio_reused_unchanged: Literal[True] = True
    relational_selection_recomputed: Literal[False] = False
    relational_hypotheses_retyped_as_reframes: Literal[False] = False
    source_novelty_state_preserved: Literal[True] = True

    @model_validator(mode="after")
    def validate_counts(self) -> "RelationalDiscoveryLaneSnapshot":
        if self.hypothesis_count != len(self.hypothesis_ids):
            raise ValueError("relational hypothesis_count must match hypothesis_ids")
        if len(self.hypothesis_ids) != len(set(self.hypothesis_ids)):
            raise ValueError("relational hypothesis IDs must be unique")
        return self


class ScientificReframingLaneSnapshot(StrictModel):
    lane_id: Literal["SCIENTIFIC_REFRAMING"] = "SCIENTIFIC_REFRAMING"
    source_portfolio_id: str = Field(min_length=1)
    source_portfolio_file_sha256: str = Field(min_length=64, max_length=64)
    source_mode_contrast_report_id: str = Field(min_length=1)
    candidate_ids: list[str] = Field(default_factory=list)
    candidate_count: int = Field(ge=0)
    occupied_reasoning_mode_count: int = Field(ge=0)
    scientific_neighborhood_cluster_count: int = Field(ge=0)

    shadow_only: Literal[True] = True
    source_portfolio_reused_unchanged: Literal[True] = True
    reframing_candidates_retyped_as_relational_hypotheses: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificReframingLaneSnapshot":
        if self.candidate_count != len(self.candidate_ids):
            raise ValueError("reframing candidate_count must match candidate_ids")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("reframing candidate IDs must be unique")
        return self


class CrossLaneLineageValidation(StrictModel):
    task_identity_matched: Literal[True] = True
    relational_context_identity_matched: Literal[True] = True
    reframing_context_identity_matched: Literal[True] = True
    reframing_candidate_lineage_complete: Literal[True] = True
    source_file_digests_recorded: Literal[True] = True


class CrossLaneScientificReasoningShadowPortfolio(StrictModel):
    schema_version: Literal[
        "cross-lane-scientific-reasoning-shadow-portfolio-v1"
    ] = "cross-lane-scientific-reasoning-shadow-portfolio-v1"

    portfolio_id: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    source_context_file_sha256: str = Field(min_length=64, max_length=64)
    question: str = Field(min_length=1)
    domain_profile_id: str = Field(min_length=1)

    relational_lane: RelationalDiscoveryLaneSnapshot
    reframing_lane: ScientificReframingLaneSnapshot
    entries: list[CrossLaneEntryEnvelope]
    lineage_validation: CrossLaneLineageValidation

    lane_count: Literal[2] = 2
    entry_count: int = Field(ge=0)
    relational_entry_count: int = Field(ge=0)
    reframing_entry_count: int = Field(ge=0)

    llm_calls_performed: Literal[0] = 0
    shadow_only: Literal[True] = True
    cross_lane_envelope_assembled: Literal[True] = True
    source_lane_schemas_preserved: Literal[True] = True
    source_lane_authorities_preserved: Literal[True] = True
    cross_lane_pairwise_comparison_performed: Literal[False] = False
    cross_lane_scientific_equivalence_evaluated: Literal[False] = False
    cross_lane_scientific_neighborhood_assignment_performed: Literal[False] = False
    cross_lane_quality_ranking_performed: Literal[False] = False
    cross_lane_redundancy_pruning_performed: Literal[False] = False
    cross_lane_winner_selected: Literal[False] = False
    integrated_portfolio_is_production_selection: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    cross_lane_external_novelty_evaluated: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_portfolio(self) -> "CrossLaneScientificReasoningShadowPortfolio":
        if self.entry_count != len(self.entries):
            raise ValueError("entry_count must match entries")
        relational = [row for row in self.entries if row.lane_id == "RELATIONAL_DISCOVERY"]
        reframing = [row for row in self.entries if row.lane_id == "SCIENTIFIC_REFRAMING"]
        if self.relational_entry_count != len(relational):
            raise ValueError("relational_entry_count must match relational entries")
        if self.reframing_entry_count != len(reframing):
            raise ValueError("reframing_entry_count must match reframing entries")
        ids = [row.entry_id for row in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("cross-lane entry IDs must be unique")
        relational_ids = {row.source_object_id for row in relational}
        if relational_ids != set(self.relational_lane.hypothesis_ids):
            raise ValueError("relational entries must cover relational lane hypotheses")
        reframing_ids = {row.source_object_id for row in reframing}
        if reframing_ids != set(self.reframing_lane.candidate_ids):
            raise ValueError("reframing entries must cover reframing lane candidates")
        return self


def _stable_id(prefix: str, *parts: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _relational_entry(card) -> CrossLaneEntryEnvelope:
    return CrossLaneEntryEnvelope(
        entry_id=_stable_id("cross_lane_entry", "RELATIONAL_DISCOVERY", card.hypothesis_id),
        lane_id="RELATIONAL_DISCOVERY",
        source_object_kind="relational_hypothesis",
        source_schema_version=card.schema_version,
        source_object_id=card.hypothesis_id,
        title=card.title,
        reasoning_label=card.hypothesis_type,
        premise_count=len(card.premise_statement_ids),
        gap_count=len(card.gap_statement_ids),
        prediction_count=len(card.predicted_observations),
        falsifier_or_test_count=len(card.falsification_criteria),
        source_scientific_neighborhood_cluster_id=None,
        source_epistemic_status=card.status,
    )


def _reframing_entry(portfolio, entry) -> CrossLaneEntryEnvelope:
    source_profile = next(
        row for row in portfolio.entries if row.candidate_id == entry.candidate_id
    )
    return CrossLaneEntryEnvelope(
        entry_id=_stable_id("cross_lane_entry", "SCIENTIFIC_REFRAMING", entry.candidate_id),
        lane_id="SCIENTIFIC_REFRAMING",
        source_object_kind="reframing_candidate",
        source_schema_version="unified-scientific-reasoning-shadow-portfolio-v1",
        source_object_id=entry.candidate_id,
        title=f"{entry.operator_id} reframing candidate",
        reasoning_label=entry.representation_transform,
        premise_count=entry.premise_count,
        gap_count=entry.gap_count,
        prediction_count=entry.prediction_observable_count,
        falsifier_or_test_count=entry.test_observable_count,
        source_scientific_neighborhood_cluster_id=(
            source_profile.scientific_neighborhood_cluster_id
        ),
        source_epistemic_status="hypothesis_only",
    )


def _validate_relational_context(
    *,
    context: HypothesisContext,
    portfolio: HypothesisPortfolio,
) -> None:
    if portfolio.source_context_id != context.context_id:
        raise ValueError("relational portfolio source_context_id mismatch")
    if portfolio.source_context_sha256 != context.context_sha256:
        raise ValueError("relational portfolio source_context_sha256 mismatch")
    if portfolio.source_report_id != context.source_report_id:
        raise ValueError("relational portfolio source_report_id mismatch")
    if portfolio.source_report_sha256 != context.source_report_sha256:
        raise ValueError("relational portfolio source_report_sha256 mismatch")
    if portfolio.domain_profile_id != context.domain_profile_id:
        raise ValueError("relational portfolio domain_profile_id mismatch")
    for card in portfolio.hypotheses:
        if card.source_context_id != context.context_id:
            raise ValueError("relational hypothesis source_context_id mismatch")
        if card.source_context_sha256 != context.context_sha256:
            raise ValueError("relational hypothesis source_context_sha256 mismatch")
        if card.source_report_id != context.source_report_id:
            raise ValueError("relational hypothesis source_report_id mismatch")
        if card.source_report_sha256 != context.source_report_sha256:
            raise ValueError("relational hypothesis source_report_sha256 mismatch")


def _validate_reframing_lineage(
    *,
    context: HypothesisContext,
    portfolio: UnifiedScientificReasoningShadowPortfolio,
    reframe_shadow: ScientificReframingShadowReport,
    proxy_shadow: ProxyChallengeRunReport | None,
    contradiction_shadow: ContradictionResolutionRunReport | None,
) -> None:
    reports = [reframe_shadow]
    if proxy_shadow is not None:
        reports.append(proxy_shadow)
    if contradiction_shadow is not None:
        reports.append(contradiction_shadow)
    for report in reports:
        if report.source_task_id != context.task_id:
            raise ValueError("reframing lineage source_task_id mismatch")
        if report.source_context_id != context.context_id:
            raise ValueError("reframing lineage source_context_id mismatch")
        if report.source_context_sha256 != context.context_sha256:
            raise ValueError("reframing lineage source_context_sha256 mismatch")
    if portfolio.source_task_id != context.task_id:
        raise ValueError("reframing portfolio source_task_id mismatch")

    source_candidate_ids = {row.reframe_id for row in reframe_shadow.candidates}
    if proxy_shadow is not None:
        source_candidate_ids.update(row.candidate_id for row in proxy_shadow.candidates)
    if contradiction_shadow is not None:
        source_candidate_ids.update(
            row.candidate_id for row in contradiction_shadow.candidates
        )
    portfolio_candidate_ids = {row.candidate_id for row in portfolio.entries}
    if source_candidate_ids != portfolio_candidate_ids:
        missing = sorted(portfolio_candidate_ids - source_candidate_ids)
        extra = sorted(source_candidate_ids - portfolio_candidate_ids)
        raise ValueError(
            "reframing candidate lineage mismatch: "
            f"missing_from_source_reports={missing}; extra_source_candidates={extra}"
        )


def build_cross_lane_scientific_reasoning_portfolio(
    *,
    context: HypothesisContext,
    relational_portfolio: HypothesisPortfolio,
    reframing_portfolio: UnifiedScientificReasoningShadowPortfolio,
    reframe_shadow: ScientificReframingShadowReport,
    proxy_shadow: ProxyChallengeRunReport | None,
    contradiction_shadow: ContradictionResolutionRunReport | None,
    source_context_file_sha256: str,
    relational_portfolio_file_sha256: str,
    reframing_portfolio_file_sha256: str,
) -> CrossLaneScientificReasoningShadowPortfolio:
    _validate_relational_context(context=context, portfolio=relational_portfolio)
    _validate_reframing_lineage(
        context=context,
        portfolio=reframing_portfolio,
        reframe_shadow=reframe_shadow,
        proxy_shadow=proxy_shadow,
        contradiction_shadow=contradiction_shadow,
    )

    relational_ids = [row.hypothesis_id for row in relational_portfolio.hypotheses]
    reframing_ids = [row.candidate_id for row in reframing_portfolio.entries]
    novelty_counts = Counter(row.novelty_status for row in relational_portfolio.hypotheses)

    relational_lane = RelationalDiscoveryLaneSnapshot(
        source_portfolio_id=relational_portfolio.portfolio_id,
        source_portfolio_file_sha256=relational_portfolio_file_sha256,
        source_context_id=relational_portfolio.source_context_id,
        source_context_sha256=relational_portfolio.source_context_sha256,
        source_report_id=relational_portfolio.source_report_id,
        source_report_sha256=relational_portfolio.source_report_sha256,
        domain_profile_id=relational_portfolio.domain_profile_id,
        hypothesis_ids=relational_ids,
        hypothesis_count=len(relational_ids),
        abstention_reason=relational_portfolio.abstention_reason,
        novelty_status_counts=dict(sorted(novelty_counts.items())),
    )
    reframing_lane = ScientificReframingLaneSnapshot(
        source_portfolio_id=reframing_portfolio.portfolio_id,
        source_portfolio_file_sha256=reframing_portfolio_file_sha256,
        source_mode_contrast_report_id=reframing_portfolio.source_mode_contrast_report_id,
        candidate_ids=reframing_ids,
        candidate_count=len(reframing_ids),
        occupied_reasoning_mode_count=reframing_portfolio.occupied_reasoning_mode_count,
        scientific_neighborhood_cluster_count=(
            reframing_portfolio.scientific_neighborhood_cluster_count
        ),
    )

    entries = [_relational_entry(row) for row in relational_portfolio.hypotheses]
    entries.extend(
        _reframing_entry(reframing_portfolio, row)
        for row in reframing_portfolio.entries
    )

    portfolio_id = _stable_id(
        "cross_lane_scientific_reasoning_portfolio",
        context.task_id,
        context.context_id,
        relational_portfolio.portfolio_id,
        reframing_portfolio.portfolio_id,
        relational_portfolio_file_sha256,
        reframing_portfolio_file_sha256,
    )
    return CrossLaneScientificReasoningShadowPortfolio(
        portfolio_id=portfolio_id,
        source_task_id=context.task_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_context_file_sha256=source_context_file_sha256,
        question=context.question,
        domain_profile_id=context.domain_profile_id,
        relational_lane=relational_lane,
        reframing_lane=reframing_lane,
        entries=entries,
        lineage_validation=CrossLaneLineageValidation(),
        entry_count=len(entries),
        relational_entry_count=len(relational_ids),
        reframing_entry_count=len(reframing_ids),
    )
