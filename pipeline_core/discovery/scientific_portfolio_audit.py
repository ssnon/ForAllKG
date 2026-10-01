from __future__ import annotations

import hashlib
import json
from collections import Counter
from statistics import median
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.scientific_portfolio_materialization import (
    ScientificPortfolioMaterializationReport,
)
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidatePool,
    ScientificPortfolioEvaluationReport,
    ScientificPortfolioSelectionReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScientificPortfolioAudit(StrictModel):
    schema_version: Literal["scientific-portfolio-audit-v1"] = (
        "scientific-portfolio-audit-v1"
    )
    audit_id: str
    audit_sha256: str
    source_pool_id: str
    source_evaluation_report_id: str
    source_selection_id: str
    source_materialization_report_id: str

    raw_frontier_idea_count: int = Field(ge=0)
    raw_evolution_idea_count: int = Field(ge=0)
    projected_candidate_count: int = Field(ge=0)
    evaluated_candidate_count: int = Field(ge=0)
    retained_candidate_count: int = Field(ge=0)
    retained_unique_family_count: int = Field(ge=0)
    materialized_hypothesis_count: int = Field(ge=0)
    materialization_success_fraction: float = Field(ge=0.0, le=1.0)

    exploration_retained_candidate_count: int = Field(ge=0)
    verification_ready_hypothesis_count: int = Field(ge=0)
    verification_ready_fraction_of_exploration: float = Field(ge=0.0, le=1.0)
    normalized_scientific_sketch_count: int = Field(ge=0)

    retained_count_by_profile: dict[str, int] = Field(default_factory=dict)
    retained_count_by_origin: dict[str, int] = Field(default_factory=dict)
    verification_burden_counts: dict[str, int] = Field(default_factory=dict)
    retained_verification_burden_counts: dict[str, int] = Field(default_factory=dict)

    frontier_candidate_retained: bool
    evolution_candidate_retained: bool
    exploratory_bridge_retained: bool
    reframe_retained: bool
    downstream_verification_ready: bool

    structural_projection_only: Literal[True] = True
    selection_view_origin_blind: Literal[True] = True
    common_scientific_sketch_normalization: Literal[True] = True
    exploration_retention_independent_of_materialization: Literal[True] = True
    materialization_failure_is_not_exploration_rejection: Literal[True] = True
    single_scalar_score_used: Literal[False] = False
    overall_winner_selected: Literal[False] = False
    literature_novelty_certified: Literal[False] = False
    scientific_truth_certified: Literal[False] = False
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False


class ScientificPortfolioCohortCase(StrictModel):
    case_id: str
    audit_id: str
    projected_candidate_count: int = Field(ge=0)
    retained_candidate_count: int = Field(ge=0)
    retained_unique_family_count: int = Field(ge=0)
    materialized_hypothesis_count: int = Field(ge=0)
    materialization_success_fraction: float = Field(ge=0.0, le=1.0)
    exploration_retained_candidate_count: int = Field(ge=0)
    verification_ready_hypothesis_count: int = Field(ge=0)
    verification_ready_fraction_of_exploration: float = Field(ge=0.0, le=1.0)
    frontier_candidate_retained: bool
    evolution_candidate_retained: bool
    exploratory_bridge_retained: bool
    reframe_retained: bool
    downstream_verification_ready: bool


class ScientificPortfolioCohortAudit(StrictModel):
    schema_version: Literal["scientific-portfolio-cohort-audit-v1"] = (
        "scientific-portfolio-cohort-audit-v1"
    )
    cohort_id: str
    cohort_sha256: str
    case_count: int = Field(ge=1)
    cases: list[ScientificPortfolioCohortCase] = Field(min_length=1)
    total_retained_candidate_count: int = Field(ge=0)
    total_materialized_hypothesis_count: int = Field(ge=0)
    total_exploration_retained_candidate_count: int = Field(ge=0)
    total_verification_ready_hypothesis_count: int = Field(ge=0)
    case_count_with_frontier_retention: int = Field(ge=0)
    case_count_with_evolution_retention: int = Field(ge=0)
    case_count_with_exploratory_bridge_retention: int = Field(ge=0)
    case_count_with_reframe_retention: int = Field(ge=0)
    case_count_ready_for_downstream_verification: int = Field(ge=0)
    median_projected_candidate_count: float = Field(ge=0.0)
    median_retained_candidate_count: float = Field(ge=0.0)
    median_retained_unique_family_count: float = Field(ge=0.0)
    median_materialized_hypothesis_count: float = Field(ge=0.0)
    median_materialization_success_fraction: float = Field(ge=0.0, le=1.0)
    median_verification_ready_fraction_of_exploration: float = Field(ge=0.0, le=1.0)

    cross_case_winner_selected: Literal[False] = False
    scientific_quality_ranking_across_cases: Literal[False] = False
    production_selection_authority: Literal[False] = False


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def build_scientific_portfolio_audit(
    *,
    pool: ScientificPortfolioCandidatePool,
    evaluation: ScientificPortfolioEvaluationReport,
    selection: ScientificPortfolioSelectionReport,
    materialization: ScientificPortfolioMaterializationReport,
) -> ScientificPortfolioAudit:
    if evaluation.source_pool_id != pool.pool_id:
        raise ValueError("evaluation/pool lineage mismatch")
    if selection.source_pool_id != pool.pool_id:
        raise ValueError("selection/pool lineage mismatch")
    if selection.source_evaluation_report_id != evaluation.report_id:
        raise ValueError("selection/evaluation lineage mismatch")
    if materialization.source_selection_id != selection.selection_id:
        raise ValueError("materialization/selection lineage mismatch")

    eval_by_id = {x.candidate_id: x for x in evaluation.evaluations}
    retained_burdens = Counter(
        eval_by_id[x].verification_burden for x in selection.retained_candidate_ids
    )
    retained_profiles = set(selection.retained_count_by_profile)
    fraction = (
        materialization.materialized_hypothesis_count / selection.retained_count
        if selection.retained_count
        else 0.0
    )
    provisional = ScientificPortfolioAudit(
        audit_id="pending",
        audit_sha256="pending",
        source_pool_id=pool.pool_id,
        source_evaluation_report_id=evaluation.report_id,
        source_selection_id=selection.selection_id,
        source_materialization_report_id=materialization.report_id,
        raw_frontier_idea_count=pool.raw_frontier_idea_count,
        raw_evolution_idea_count=pool.raw_evolution_idea_count,
        projected_candidate_count=pool.projected_candidate_count,
        evaluated_candidate_count=evaluation.evaluation_count,
        retained_candidate_count=selection.retained_count,
        retained_unique_family_count=selection.retained_unique_family_count,
        materialized_hypothesis_count=materialization.materialized_hypothesis_count,
        materialization_success_fraction=fraction,
        exploration_retained_candidate_count=selection.retained_count,
        verification_ready_hypothesis_count=materialization.materialized_hypothesis_count,
        verification_ready_fraction_of_exploration=fraction,
        normalized_scientific_sketch_count=evaluation.evaluation_count,
        retained_count_by_profile=dict(selection.retained_count_by_profile),
        retained_count_by_origin=dict(selection.retained_count_by_origin),
        verification_burden_counts=dict(evaluation.verification_burden_counts),
        retained_verification_burden_counts=dict(sorted(retained_burdens.items())),
        frontier_candidate_retained=selection.retained_count_by_origin.get("FRONTIER", 0) > 0,
        evolution_candidate_retained=selection.retained_count_by_origin.get("EVOLUTION", 0) > 0,
        exploratory_bridge_retained="EXPLORATORY_BRIDGE" in retained_profiles,
        reframe_retained="REFRAME" in retained_profiles,
        downstream_verification_ready=materialization.materialized_hypothesis_count > 0,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("audit_id", None)
    payload.pop("audit_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "audit_id": f"scientific_portfolio_audit:{digest[:20]}",
            "audit_sha256": digest,
        }
    )


def build_scientific_portfolio_cohort_audit(
    cases: Sequence[tuple[str, ScientificPortfolioAudit]],
) -> ScientificPortfolioCohortAudit:
    if not cases:
        raise ValueError("cohort requires at least one case")
    rows = [
        ScientificPortfolioCohortCase(
            case_id=case_id,
            audit_id=audit.audit_id,
            projected_candidate_count=audit.projected_candidate_count,
            retained_candidate_count=audit.retained_candidate_count,
            retained_unique_family_count=audit.retained_unique_family_count,
            materialized_hypothesis_count=audit.materialized_hypothesis_count,
            materialization_success_fraction=audit.materialization_success_fraction,
            exploration_retained_candidate_count=audit.exploration_retained_candidate_count,
            verification_ready_hypothesis_count=audit.verification_ready_hypothesis_count,
            verification_ready_fraction_of_exploration=(
                audit.verification_ready_fraction_of_exploration
            ),
            frontier_candidate_retained=audit.frontier_candidate_retained,
            evolution_candidate_retained=audit.evolution_candidate_retained,
            exploratory_bridge_retained=audit.exploratory_bridge_retained,
            reframe_retained=audit.reframe_retained,
            downstream_verification_ready=audit.downstream_verification_ready,
        )
        for case_id, audit in cases
    ]
    provisional = ScientificPortfolioCohortAudit(
        cohort_id="pending",
        cohort_sha256="pending",
        case_count=len(rows),
        cases=rows,
        total_retained_candidate_count=sum(x.retained_candidate_count for x in rows),
        total_materialized_hypothesis_count=sum(x.materialized_hypothesis_count for x in rows),
        total_exploration_retained_candidate_count=sum(
            x.exploration_retained_candidate_count for x in rows
        ),
        total_verification_ready_hypothesis_count=sum(
            x.verification_ready_hypothesis_count for x in rows
        ),
        case_count_with_frontier_retention=sum(x.frontier_candidate_retained for x in rows),
        case_count_with_evolution_retention=sum(x.evolution_candidate_retained for x in rows),
        case_count_with_exploratory_bridge_retention=sum(x.exploratory_bridge_retained for x in rows),
        case_count_with_reframe_retention=sum(x.reframe_retained for x in rows),
        case_count_ready_for_downstream_verification=sum(x.downstream_verification_ready for x in rows),
        median_projected_candidate_count=float(median(x.projected_candidate_count for x in rows)),
        median_retained_candidate_count=float(median(x.retained_candidate_count for x in rows)),
        median_retained_unique_family_count=float(median(x.retained_unique_family_count for x in rows)),
        median_materialized_hypothesis_count=float(median(x.materialized_hypothesis_count for x in rows)),
        median_materialization_success_fraction=float(median(x.materialization_success_fraction for x in rows)),
        median_verification_ready_fraction_of_exploration=float(
            median(x.verification_ready_fraction_of_exploration for x in rows)
        ),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("cohort_id", None)
    payload.pop("cohort_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "cohort_id": f"scientific_portfolio_cohort:{digest[:20]}",
            "cohort_sha256": digest,
        }
    )


__all__ = [
    "ScientificPortfolioAudit",
    "ScientificPortfolioCohortAudit",
    "build_scientific_portfolio_audit",
    "build_scientific_portfolio_cohort_audit",
]
