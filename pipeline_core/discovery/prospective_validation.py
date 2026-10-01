from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from statistics import median
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.hypothesis_evidence_diversity import (
    HypothesisEvidenceDiversityAssessor,
)
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidate,
    ScientificPortfolioCandidatePool,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ProspectiveArm = Literal[
    "LEGACY",
    "FRONTIER_BALANCED",
    "EVOLUTION_BALANCED",
    "PORTFOLIO_SELECTED",
]


ARM_ORDER: tuple[ProspectiveArm, ...] = (
    "LEGACY",
    "FRONTIER_BALANCED",
    "EVOLUTION_BALANCED",
    "PORTFOLIO_SELECTED",
)


class ProspectiveArmSelection(StrictModel):
    schema_version: Literal[
        "prospective-arm-selection-v1"
    ] = "prospective-arm-selection-v1"
    selection_id: str
    selection_sha256: str
    arm: ProspectiveArm
    source_pool_id: str
    source_pool_sha256: str
    source_context_id: str
    source_context_sha256: str
    max_hypotheses: int = Field(ge=1)
    eligible_candidate_count: int = Field(ge=0)
    selected_candidate_count: int = Field(ge=0)
    selected_unique_family_count: int = Field(ge=0)
    selected_count_by_origin: dict[str, int] = Field(default_factory=dict)
    selected_count_by_form: dict[str, int] = Field(default_factory=dict)
    retained_candidate_ids: list[str] = Field(default_factory=list)
    selected_family_signatures: list[str] = Field(default_factory=list)

    selection_policy: Literal[
        "DETERMINISTIC_FAMILY_BALANCED_NO_QUALITY_RANKING"
    ] = "DETERMINISTIC_FAMILY_BALANCED_NO_QUALITY_RANKING"
    scientific_quality_ranking_performed: Literal[False] = False
    overall_winner_selected: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @property
    def retained_count(self) -> int:
        return self.selected_candidate_count


class ProspectiveArmMetrics(StrictModel):
    schema_version: Literal[
        "prospective-arm-metrics-v2"
    ] = "prospective-arm-metrics-v2"
    case_id: str
    arm: ProspectiveArm
    source_portfolio_path: str
    hypothesis_count: int = Field(ge=0)

    selected_candidate_count: int | None = Field(default=None, ge=0)
    selected_unique_family_count: int | None = Field(default=None, ge=0)
    selected_count_by_origin: dict[str, int] = Field(default_factory=dict)
    selected_count_by_form: dict[str, int] = Field(default_factory=dict)

    # Selection-space vs grounded-promotion yield are separate axes.
    # A low materialization yield is data, not an execution failure.
    materialized_hypothesis_count: int = Field(default=0, ge=0)
    materialization_yield_fraction: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    conceptual_selected_family_fraction: float | None = Field(
        default=None, ge=0.0, le=1.0
    )

    hypothesis_type_counts: dict[str, int] = Field(default_factory=dict)
    evidence_used_statement_count: int = Field(ge=0)
    evidence_eligible_statement_coverage: float = Field(ge=0.0, le=1.0)
    evidence_distinct_premise_set_count: int = Field(ge=0)
    evidence_exact_duplicate_premise_group_count: int = Field(ge=0)
    evidence_mean_pairwise_statement_jaccard: float = Field(ge=0.0, le=1.0)
    evidence_max_pairwise_statement_jaccard: float = Field(ge=0.0, le=1.0)
    premise_distinct_set_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    semantic_status: str
    external_novelty_status: str
    external_status_counts: dict[str, int] = Field(default_factory=dict)
    n9_status: str
    feasibility_status: str
    feasibility_disposition_counts: dict[str, int] = Field(default_factory=dict)

    # Operational completion is deliberately separate from scientific
    # acceptance/rejection and from provider-budget pauses.
    operational_status: str = "UNKNOWN_LEGACY"
    provider_budget_pause_stage: str | None = None
    downstream_verified_hypothesis_count: int = Field(default=0, ge=0)
    downstream_verification_yield_fraction: float | None = Field(
        default=None, ge=0.0, le=1.0
    )

    verification_completed: bool
    authority_ok: bool
    n10_run: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ProspectiveCaseAudit(StrictModel):
    schema_version: Literal[
        "prospective-case-ablation-audit-v2"
    ] = "prospective-case-ablation-audit-v2"
    case_id: str
    run_dir: str
    context_id: str
    context_sha256: str
    arm_count: int = Field(ge=0)
    arms: list[ProspectiveArmMetrics] = Field(default_factory=list)
    all_four_arms_present: bool
    selected_arm_verification_complete: bool
    execution_status: Literal[
        "COMPLETE",
        "PARTIAL",
        "PAUSED_PROVIDER_BUDGET_EXHAUSTED",
    ] = "COMPLETE"
    paused_arm: ProspectiveArm | None = None
    production_selection_authority: Literal[False] = False


class ProspectiveCohortAudit(StrictModel):
    schema_version: Literal[
        "prospective-cohort-ablation-audit-v2",
        "prospective-cohort-ablation-audit-v3",
    ] = "prospective-cohort-ablation-audit-v3"
    cohort_id: str
    cohort_sha256: str
    cohort_kind: Literal["DEVELOPMENT", "HELD_OUT"]
    case_count: int = Field(ge=0)
    planned_case_count: int = Field(default=0, ge=0)
    complete_case_count: int = Field(default=0, ge=0)
    partial_case_count: int = Field(default=0, ge=0)
    paused_provider_budget_case_count: int = Field(default=0, ge=0)
    four_arm_record_present_case_count: int = Field(default=0, ge=0)
    complete_four_arm_case_count: int = Field(ge=0)
    cases: list[ProspectiveCaseAudit] = Field(default_factory=list)

    arm_case_counts: dict[str, int] = Field(default_factory=dict)
    arm_semantic_accepted_counts: dict[str, int] = Field(default_factory=dict)
    arm_external_complete_counts: dict[str, int] = Field(default_factory=dict)
    arm_n9_complete_counts: dict[str, int] = Field(default_factory=dict)
    arm_supported_feasibility_complete_counts: dict[str, int] = Field(default_factory=dict)
    arm_hypothesis_count_medians: dict[str, float] = Field(default_factory=dict)
    arm_materialization_yield_medians: dict[str, float | None] = Field(default_factory=dict)
    arm_downstream_verification_yield_medians: dict[str, float] = Field(default_factory=dict)
    arm_conceptual_family_fraction_medians: dict[str, float | None] = Field(default_factory=dict)
    arm_premise_distinct_set_fraction_medians: dict[str, float] = Field(default_factory=dict)
    arm_evidence_coverage_medians: dict[str, float] = Field(default_factory=dict)
    arm_mean_pairwise_premise_jaccard_medians: dict[str, float] = Field(default_factory=dict)
    arm_external_status_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    arm_operational_status_counts: dict[str, dict[str, int]] = Field(default_factory=dict)

    portfolio_selected_frontier_retention_case_count: int = Field(ge=0)
    portfolio_selected_evolution_retention_case_count: int = Field(ge=0)
    all_authority_invariants_hold: bool

    scientific_superiority_established: Literal[False] = False
    cross_case_winner_selected: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ProductionIntegrationReadinessShadow(StrictModel):
    schema_version: Literal[
        "production-integration-readiness-shadow-v1"
    ] = "production-integration-readiness-shadow-v1"
    readiness_id: str
    readiness_sha256: str
    status: Literal[
        "DEVELOPMENT_ABLATION_INCOMPLETE",
        "HELD_OUT_PROSPECTIVE_REQUIRED",
        "HELD_OUT_PROSPECTIVE_INCOMPLETE",
        "READY_FOR_HUMAN_PRODUCTION_REVIEW",
    ]
    development_case_count: int = Field(ge=0)
    held_out_case_count: int = Field(ge=0)
    operational_gate_checks: dict[str, bool] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)

    scientific_superiority_established: Literal[False] = False
    automatic_production_promotion_allowed: Literal[False] = False
    human_scientific_review_required: Literal[True] = True
    n10_required_before_any_production_novelty_claim: Literal[True] = True
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    return (
        f"{prefix}:"
        + hashlib.sha256(
            _canonical_json(parts).encode("utf-8")
        ).hexdigest()[:20]
    )


def provider_budget_exhausted_from_text(*texts: str) -> bool:
    """Conservatively classify an external-provider budget exhaustion.

    This is operational classification only. It never changes scientific
    status, prior-art interpretation, novelty authority, or selection.
    """

    combined = "\n".join(str(value or "") for value in texts).lower()
    if not combined.strip():
        return False

    explicit_budget_markers = (
        "daily_remaining_usd\": 0",
        "credits_remaining\": 0",
        "x-ratelimit-remaining: 0",
        "x-ratelimit-remaining-usd: 0",
        "daily budget",
        "budget exhausted",
    )
    if any(marker in combined for marker in explicit_budget_markers):
        return True

    has_429 = (
        "http/2 429" in combined
        or " 429 " in combined
        or "status code: 429" in combined
        or "too many requests" in combined
    )
    provider_hint = (
        "openalex" in combined
        or "api.openalex.org" in combined
        or "x-ratelimit" in combined
    )
    return bool(has_429 and provider_hint)


def verification_summary_needs_resume(summary: Mapping[str, Any]) -> bool:
    """Return True only for operationally incomplete verification.

    Scientific terminal states (empty portfolio or semantic rejection) are
    terminal. Provider-budget pauses and operational failures are resumable.
    Old v1 summaries are interpreted from their top-level status so existing
    successful artifacts remain reusable.
    """

    operational = str(summary.get("operational_status") or "").strip()
    if operational in {"COMPLETE", "SCIENTIFIC_TERMINAL"}:
        return False
    if operational in {
        "PAUSED_PROVIDER_BUDGET_EXHAUSTED",
        "FAILED_OPERATIONAL",
        "IN_PROGRESS",
    }:
        return True

    legacy_status = str(summary.get("status") or "").strip()
    if legacy_status in {
        "COMPLETE_SHADOW_VERIFICATION",
        "SKIPPED_EMPTY_PORTFOLIO",
        "SCIENTIFIC_TERMINAL_SEMANTIC_REJECTED",
    }:
        return False
    return True


def _origin_rank(value: str) -> int:
    return {
        "FRONTIER": 0,
        "EVOLUTION": 1,
    }.get(str(value), 99)


def _eligible_candidates(
    pool: ScientificPortfolioCandidatePool,
    arm: ProspectiveArm,
) -> list[ScientificPortfolioCandidate]:
    if arm == "FRONTIER_BALANCED":
        return [
            row for row in pool.candidates
            if row.origin == "FRONTIER"
        ]
    if arm == "EVOLUTION_BALANCED":
        return list(pool.candidates)
    raise ValueError(
        "candidate-pool selection is defined only for "
        "FRONTIER_BALANCED and EVOLUTION_BALANCED"
    )


def build_family_balanced_arm_selection(
    *,
    pool: ScientificPortfolioCandidatePool,
    arm: ProspectiveArm,
    max_hypotheses: int = 8,
) -> ProspectiveArmSelection:
    if max_hypotheses < 1:
        raise ValueError("max_hypotheses must be >= 1")

    eligible = _eligible_candidates(pool, arm)

    groups: dict[
        tuple[str, str],
        list[ScientificPortfolioCandidate],
    ] = defaultdict(list)
    for row in eligible:
        groups[(row.origin, row.idea_form)].append(row)

    for rows in groups.values():
        rows.sort(
            key=lambda x: (
                x.conceptual_family_signature,
                x.candidate_id,
            )
        )

    keys = sorted(
        groups,
        key=lambda key: (
            _origin_rank(key[0]),
            key[1],
        ),
    )

    chosen: list[ScientificPortfolioCandidate] = []
    seen_families: set[str] = set()

    while len(chosen) < max_hypotheses:
        added = False
        for key in keys:
            rows = groups[key]
            while rows:
                row = rows.pop(0)
                family = str(
                    row.conceptual_family_signature
                    or row.candidate_id
                )
                if family in seen_families:
                    continue
                seen_families.add(family)
                chosen.append(row)
                added = True
                break
            if len(chosen) >= max_hypotheses:
                break
        if not added:
            break

    provisional = ProspectiveArmSelection(
        selection_id="pending",
        selection_sha256="pending",
        arm=arm,
        source_pool_id=pool.pool_id,
        source_pool_sha256=pool.pool_sha256,
        source_context_id=pool.source_context_id,
        source_context_sha256=pool.source_context_sha256,
        max_hypotheses=max_hypotheses,
        eligible_candidate_count=len(eligible),
        selected_candidate_count=len(chosen),
        selected_unique_family_count=len(
            {
                row.conceptual_family_signature
                for row in chosen
            }
        ),
        selected_count_by_origin=dict(
            sorted(
                Counter(
                    row.origin for row in chosen
                ).items()
            )
        ),
        selected_count_by_form=dict(
            sorted(
                Counter(
                    row.idea_form for row in chosen
                ).items()
            )
        ),
        retained_candidate_ids=[
            row.candidate_id for row in chosen
        ],
        selected_family_signatures=[
            row.conceptual_family_signature
            for row in chosen
        ],
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("selection_id", None)
    payload.pop("selection_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "selection_id":
                f"prospective_arm_selection:{digest[:20]}",
            "selection_sha256": digest,
        }
    )


def cap_legacy_portfolio(
    portfolio: HypothesisPortfolio,
    *,
    max_hypotheses: int,
) -> HypothesisPortfolio:
    if max_hypotheses < 1:
        raise ValueError("max_hypotheses must be >= 1")
    if len(portfolio.hypotheses) <= max_hypotheses:
        return portfolio

    cards = list(portfolio.hypotheses[:max_hypotheses])
    digest = _sha(
        {
            "source_portfolio_id": portfolio.portfolio_id,
            "max_hypotheses": max_hypotheses,
            "hypothesis_ids": [
                row.hypothesis_id for row in cards
            ],
        }
    )
    return portfolio.model_copy(
        update={
            "portfolio_id":
                f"prospective_legacy_cap:{digest[:20]}",
            "hypotheses": cards,
            "abstention_reason": None,
        }
    )


def _infer_legacy_operational_status(
    *,
    hypothesis_count: int,
    semantic_status: str,
    external_status: str,
    n9_status: str,
    feasibility_status: str,
    verification_completed: bool,
) -> str:
    """Normalize pre-hardening verification summaries conservatively.

    Empty grounded portfolios are a scientific terminal state: there is
    simply nothing to send downstream. They are not an operational
    failure and should not appear as UNKNOWN_LEGACY.

    Other incomplete historical summaries remain UNKNOWN_LEGACY so the
    resume classifier can retry them rather than guessing why they ended.
    """
    if verification_completed:
        return "COMPLETE"

    if (
        hypothesis_count == 0
        and semantic_status == "NOT_RUN"
        and external_status == "NOT_RUN"
        and n9_status == "NOT_RUN"
        and feasibility_status == "NOT_RUN"
    ):
        return "SCIENTIFIC_TERMINAL"

    return "UNKNOWN_LEGACY"


def build_arm_metrics(
    *,
    case_id: str,
    arm: ProspectiveArm,
    context: HypothesisContext,
    portfolio: HypothesisPortfolio,
    portfolio_path: str,
    verification_summary: Mapping[str, Any],
    arm_selection: ProspectiveArmSelection | None = None,
) -> ProspectiveArmMetrics:
    diversity = HypothesisEvidenceDiversityAssessor().assess(
        context,
        portfolio,
    )

    semantic_status = str(
        (
            verification_summary.get("semantic")
            or {}
        ).get("status")
        or "NOT_RUN"
    )
    external = (
        verification_summary.get("external_novelty")
        or {}
    )
    external_status = str(
        external.get("status") or "NOT_RUN"
    )
    n9_status = str(
        (
            verification_summary.get("n9")
            or {}
        ).get("status")
        or "NOT_RUN"
    )
    feasibility = (
        verification_summary.get("feasibility")
        or {}
    )
    feasibility_status = str(
        feasibility.get("status")
        or "NOT_RUN"
    )

    verification_completed = (
        semantic_status == "ACCEPTED"
        and external_status == "COMPLETE"
        and n9_status in {"COMPLETE", "INTAKE_ONLY"}
        and feasibility_status in {
            "COMPLETE",
            "SKIPPED_UNSUPPORTED_DOMAIN",
        }
    )

    pause_stage = verification_summary.get("resume_stage")

    selected_count = (
        arm_selection.selected_candidate_count
        if arm_selection is not None
        else None
    )
    selected_family_count = (
        arm_selection.selected_unique_family_count
        if arm_selection is not None
        else None
    )
    hypothesis_count = len(portfolio.hypotheses)

    operational_status = str(
        verification_summary.get("operational_status")
        or _infer_legacy_operational_status(
            hypothesis_count=hypothesis_count,
            semantic_status=semantic_status,
            external_status=external_status,
            n9_status=n9_status,
            feasibility_status=feasibility_status,
            verification_completed=verification_completed,
        )
    )

    materialization_yield = (
        hypothesis_count / selected_count
        if selected_count not in {None, 0}
        else None
    )
    conceptual_family_fraction = (
        selected_family_count / selected_count
        if selected_count not in {None, 0}
        and selected_family_count is not None
        else None
    )
    premise_distinct_fraction = (
        diversity.distinct_premise_set_count / hypothesis_count
        if hypothesis_count
        else 0.0
    )
    verified_count = hypothesis_count if verification_completed else 0
    verification_yield = (
        verified_count / hypothesis_count
        if hypothesis_count
        else None
    )

    authority_ok = (
        bool(
            verification_summary.get(
                "production_selection_authority",
                False,
            )
        )
        is False
        and bool(
            verification_summary.get(
                "n10_run",
                False,
            )
        )
        is False
    )

    return ProspectiveArmMetrics(
        case_id=case_id,
        arm=arm,
        source_portfolio_path=str(portfolio_path),
        hypothesis_count=hypothesis_count,
        selected_candidate_count=selected_count,
        selected_unique_family_count=selected_family_count,
        selected_count_by_origin=(
            dict(arm_selection.selected_count_by_origin)
            if arm_selection is not None
            else {}
        ),
        selected_count_by_form=(
            dict(arm_selection.selected_count_by_form)
            if arm_selection is not None
            else {}
        ),
        materialized_hypothesis_count=hypothesis_count,
        materialization_yield_fraction=materialization_yield,
        conceptual_selected_family_fraction=conceptual_family_fraction,
        hypothesis_type_counts=dict(
            sorted(
                Counter(
                    row.hypothesis_type
                    for row in portfolio.hypotheses
                ).items()
            )
        ),
        evidence_used_statement_count=(
            diversity.used_statement_count
        ),
        evidence_eligible_statement_coverage=(
            diversity.eligible_statement_coverage
        ),
        evidence_distinct_premise_set_count=(
            diversity.distinct_premise_set_count
        ),
        evidence_exact_duplicate_premise_group_count=(
            diversity.exact_premise_set_duplicate_group_count
        ),
        evidence_mean_pairwise_statement_jaccard=(
            diversity.mean_pairwise_statement_jaccard
        ),
        evidence_max_pairwise_statement_jaccard=(
            diversity.max_pairwise_statement_jaccard
        ),
        premise_distinct_set_fraction=premise_distinct_fraction,
        semantic_status=semantic_status,
        external_novelty_status=external_status,
        external_status_counts=dict(
            external.get("status_counts") or {}
        ),
        n9_status=n9_status,
        feasibility_status=feasibility_status,
        feasibility_disposition_counts=dict(
            feasibility.get(
                "final_disposition_counts"
            )
            or {}
        ),
        operational_status=operational_status,
        provider_budget_pause_stage=(
            str(pause_stage) if pause_stage else None
        ),
        downstream_verified_hypothesis_count=verified_count,
        downstream_verification_yield_fraction=verification_yield,
        verification_completed=verification_completed,
        authority_ok=authority_ok,
    )


def build_case_audit(
    *,
    case_id: str,
    run_dir: str,
    context: HypothesisContext,
    arms: Sequence[ProspectiveArmMetrics],
    execution_status: Literal[
        "COMPLETE",
        "PARTIAL",
        "PAUSED_PROVIDER_BUDGET_EXHAUSTED",
    ] | None = None,
    paused_arm: ProspectiveArm | None = None,
) -> ProspectiveCaseAudit:
    by_arm = {row.arm: row for row in arms}
    if execution_status is None:
        execution_status = (
            "COMPLETE"
            if len(arms) == len(ARM_ORDER)
            and all(
                row.operational_status in {
                    "COMPLETE",
                    "SCIENTIFIC_TERMINAL",
                    "UNKNOWN_LEGACY",
                }
                for row in arms
            )
            else "PARTIAL"
        )
    return ProspectiveCaseAudit(
        case_id=case_id,
        run_dir=str(run_dir),
        context_id=context.context_id,
        context_sha256=context.context_sha256,
        arm_count=len(arms),
        arms=list(arms),
        all_four_arms_present=all(
            arm in by_arm for arm in ARM_ORDER
        ),
        selected_arm_verification_complete=(
            by_arm.get("PORTFOLIO_SELECTED")
            is not None
            and by_arm[
                "PORTFOLIO_SELECTED"
            ].verification_completed
        ),
        execution_status=execution_status,
        paused_arm=paused_arm,
    )


def _median(
    values: Sequence[float | int],
) -> float:
    return float(median(values)) if values else 0.0


def _median_optional(
    values: Sequence[float | int],
) -> float | None:
    return float(median(values)) if values else None


def build_cohort_audit(
    *,
    cohort_kind: Literal["DEVELOPMENT", "HELD_OUT"],
    cases: Sequence[ProspectiveCaseAudit],
    planned_case_count: int | None = None,
) -> ProspectiveCohortAudit:
    by_arm: dict[
        str,
        list[ProspectiveArmMetrics],
    ] = defaultdict(list)
    for case in cases:
        for row in case.arms:
            by_arm[row.arm].append(row)

    arm_case_counts = {
        arm: len(by_arm.get(arm, []))
        for arm in ARM_ORDER
    }
    semantic_counts = {
        arm: sum(
            row.semantic_status == "ACCEPTED"
            for row in by_arm.get(arm, [])
        )
        for arm in ARM_ORDER
    }
    external_counts = {
        arm: sum(
            row.external_novelty_status == "COMPLETE"
            for row in by_arm.get(arm, [])
        )
        for arm in ARM_ORDER
    }
    n9_counts = {
        arm: sum(
            row.n9_status in {"COMPLETE", "INTAKE_ONLY"}
            for row in by_arm.get(arm, [])
        )
        for arm in ARM_ORDER
    }
    feasibility_counts = {
        arm: sum(
            row.feasibility_status == "COMPLETE"
            for row in by_arm.get(arm, [])
        )
        for arm in ARM_ORDER
    }

    external_status_counts: dict[
        str,
        dict[str, int],
    ] = {}
    for arm in ARM_ORDER:
        counter: Counter[str] = Counter()
        for row in by_arm.get(arm, []):
            counter.update(row.external_status_counts)
        external_status_counts[arm] = dict(
            sorted(counter.items())
        )

    operational_status_counts: dict[
        str, dict[str, int]
    ] = {}
    for arm in ARM_ORDER:
        operational_status_counts[arm] = dict(
            sorted(
                Counter(
                    row.operational_status
                    for row in by_arm.get(arm, [])
                ).items()
            )
        )

    selected_rows = by_arm.get(
        "PORTFOLIO_SELECTED",
        [],
    )
    frontier_retention = sum(
        int(
            row.selected_count_by_origin.get(
                "FRONTIER",
                0,
            )
            > 0
        )
        for row in selected_rows
    )
    evolution_retention = sum(
        int(
            row.selected_count_by_origin.get(
                "EVOLUTION",
                0,
            )
            > 0
        )
        for row in selected_rows
    )

    provisional = ProspectiveCohortAudit(
        cohort_id="pending",
        cohort_sha256="pending",
        cohort_kind=cohort_kind,
        case_count=len(cases),
        planned_case_count=(
            int(planned_case_count)
            if planned_case_count is not None
            else len(cases)
        ),
        complete_case_count=sum(
            case.execution_status == "COMPLETE"
            for case in cases
        ),
        partial_case_count=sum(
            case.execution_status == "PARTIAL"
            for case in cases
        ),
        paused_provider_budget_case_count=sum(
            case.execution_status
            == "PAUSED_PROVIDER_BUDGET_EXHAUSTED"
            for case in cases
        ),
        four_arm_record_present_case_count=sum(
            case.all_four_arms_present
            for case in cases
        ),
        complete_four_arm_case_count=sum(
            case.all_four_arms_present
            and len(case.arms) == len(ARM_ORDER)
            and all(
                row.operational_status
                in {"COMPLETE", "SCIENTIFIC_TERMINAL"}
                for row in case.arms
            )
            for case in cases
        ),
        cases=list(cases),
        arm_case_counts=arm_case_counts,
        arm_semantic_accepted_counts=semantic_counts,
        arm_external_complete_counts=external_counts,
        arm_n9_complete_counts=n9_counts,
        arm_supported_feasibility_complete_counts=(
            feasibility_counts
        ),
        arm_hypothesis_count_medians={
            arm: _median(
                [
                    row.hypothesis_count
                    for row in by_arm.get(arm, [])
                ]
            )
            for arm in ARM_ORDER
        },
        arm_materialization_yield_medians={
            arm: _median_optional(
                [
                    row.materialization_yield_fraction
                    for row in by_arm.get(arm, [])
                    if row.materialization_yield_fraction is not None
                ]
            )
            for arm in ARM_ORDER
        },
        arm_downstream_verification_yield_medians={
            arm: _median(
                [
                    row.downstream_verification_yield_fraction
                    for row in by_arm.get(arm, [])
                    if row.downstream_verification_yield_fraction is not None
                ]
            )
            for arm in ARM_ORDER
        },
        arm_conceptual_family_fraction_medians={
            arm: _median_optional(
                [
                    row.conceptual_selected_family_fraction
                    for row in by_arm.get(arm, [])
                    if row.conceptual_selected_family_fraction is not None
                ]
            )
            for arm in ARM_ORDER
        },
        arm_premise_distinct_set_fraction_medians={
            arm: _median(
                [
                    row.premise_distinct_set_fraction
                    for row in by_arm.get(arm, [])
                ]
            )
            for arm in ARM_ORDER
        },
        arm_evidence_coverage_medians={
            arm: _median(
                [
                    row.evidence_eligible_statement_coverage
                    for row in by_arm.get(arm, [])
                ]
            )
            for arm in ARM_ORDER
        },
        arm_mean_pairwise_premise_jaccard_medians={
            arm: _median(
                [
                    row.evidence_mean_pairwise_statement_jaccard
                    for row in by_arm.get(arm, [])
                ]
            )
            for arm in ARM_ORDER
        },
        arm_external_status_counts=external_status_counts,
        arm_operational_status_counts=operational_status_counts,
        portfolio_selected_frontier_retention_case_count=(
            frontier_retention
        ),
        portfolio_selected_evolution_retention_case_count=(
            evolution_retention
        ),
        all_authority_invariants_hold=all(
            row.authority_ok
            for case in cases
            for row in case.arms
        ),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("cohort_id", None)
    payload.pop("cohort_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "cohort_id":
                f"prospective_cohort:{digest[:20]}",
            "cohort_sha256": digest,
        }
    )


def build_production_integration_readiness_shadow(
    *,
    development: ProspectiveCohortAudit,
    held_out: ProspectiveCohortAudit | None,
    min_held_out_cases: int = 4,
) -> ProductionIntegrationReadinessShadow:
    if min_held_out_cases < 1:
        raise ValueError("min_held_out_cases must be >= 1")

    dev_complete = (
        development.cohort_kind == "DEVELOPMENT"
        and development.planned_case_count > 0
        and development.case_count == development.planned_case_count
        and development.complete_case_count == development.planned_case_count
        and development.complete_four_arm_case_count
        == development.planned_case_count
        and development.paused_provider_budget_case_count == 0
        and development.partial_case_count == 0
        and development.all_authority_invariants_hold
    )

    checks: dict[str, bool] = {
        "development_four_arm_ablation_complete":
            dev_complete,
        "held_out_cohort_present": held_out is not None,
        "held_out_case_count_sufficient": False,
        "held_out_four_arm_ablation_complete": False,
        "held_out_selected_semantic_complete": False,
        "held_out_selected_external_complete": False,
        "held_out_selected_n9_complete": False,
        "held_out_selected_feasibility_capability_respected": False,
        "authority_invariants_hold": (
            development.all_authority_invariants_hold
        ),
    }
    reasons: list[str] = []

    if not dev_complete:
        status = "DEVELOPMENT_ABLATION_INCOMPLETE"
        reasons.append(
            "development_four_arm_ablation_incomplete"
        )
    elif held_out is None:
        status = "HELD_OUT_PROSPECTIVE_REQUIRED"
        reasons.append(
            "development_cohort_was_used_during_architecture_tuning"
        )
    else:
        selected = [
            row
            for case in held_out.cases
            for row in case.arms
            if row.arm == "PORTFOLIO_SELECTED"
        ]
        checks["held_out_case_count_sufficient"] = (
            held_out.planned_case_count >= min_held_out_cases
            and held_out.case_count == held_out.planned_case_count
        )
        checks["held_out_four_arm_ablation_complete"] = (
            held_out.complete_case_count == held_out.planned_case_count
            and held_out.complete_four_arm_case_count
            == held_out.planned_case_count
            and held_out.paused_provider_budget_case_count == 0
            and held_out.partial_case_count == 0
        )
        checks["held_out_selected_semantic_complete"] = (
            len(selected) == held_out.planned_case_count
            and all(
                row.semantic_status == "ACCEPTED"
                for row in selected
            )
        )
        checks["held_out_selected_external_complete"] = (
            len(selected) == held_out.planned_case_count
            and all(
                row.external_novelty_status == "COMPLETE"
                for row in selected
            )
        )
        checks["held_out_selected_n9_complete"] = (
            len(selected) == held_out.planned_case_count
            and all(
                row.n9_status
                in {"COMPLETE", "INTAKE_ONLY"}
                for row in selected
            )
        )
        checks[
            "held_out_selected_feasibility_capability_respected"
        ] = (
            len(selected) == held_out.planned_case_count
            and all(
                row.feasibility_status
                in {
                    "COMPLETE",
                    "SKIPPED_UNSUPPORTED_DOMAIN",
                }
                for row in selected
            )
        )
        checks["authority_invariants_hold"] = (
            development.all_authority_invariants_hold
            and held_out.all_authority_invariants_hold
        )

        if all(checks.values()):
            status = "READY_FOR_HUMAN_PRODUCTION_REVIEW"
            reasons.append(
                "operational_shadow_gate_complete"
            )
            reasons.append(
                "scientific_superiority_not_automatically_inferred"
            )
        else:
            status = "HELD_OUT_PROSPECTIVE_INCOMPLETE"
            reasons.extend(
                key
                for key, passed in checks.items()
                if not passed
            )

    provisional = ProductionIntegrationReadinessShadow(
        readiness_id="pending",
        readiness_sha256="pending",
        status=status,
        development_case_count=development.case_count,
        held_out_case_count=(
            held_out.case_count
            if held_out is not None
            else 0
        ),
        operational_gate_checks=checks,
        reason_codes=list(dict.fromkeys(reasons)),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("readiness_id", None)
    payload.pop("readiness_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "readiness_id":
                f"production_integration_readiness:{digest[:20]}",
            "readiness_sha256": digest,
        }
    )


__all__ = [
    "ARM_ORDER",
    "ProspectiveArm",
    "ProspectiveArmSelection",
    "ProspectiveArmMetrics",
    "ProspectiveCaseAudit",
    "ProspectiveCohortAudit",
    "ProductionIntegrationReadinessShadow",
    "build_family_balanced_arm_selection",
    "cap_legacy_portfolio",
    "build_arm_metrics",
    "build_case_audit",
    "build_cohort_audit",
    "build_production_integration_readiness_shadow",
]
