from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import LiteratureQueryPlan
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    AtomicCrossLaneSynthesisReport,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionFacingScientificCandidatePortfolio,
)
from pipeline_core.discovery.reframing.scientific_synthesis_n10_authority import (
    ScientificSynthesisNoveltyCertificationReport,
)
from pipeline_core.discovery.projection_relation_adjudication import (
    ProjectionRelationCandidateReport,
)
from pipeline_core.discovery.scientific_certification_gate import (
    ScientificCertificationGateReport,
)
from pipeline_core.discovery.scientific_hypothesis_evidence_aggregation import (
    ScientificHypothesisEvidenceAggregationReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class AtomicScientificVerifierProspectiveCohortFreeze(StrictModel):
    schema_version: Literal[
        "atomic-scientific-verifier-prospective-cohort-freeze-v1"
    ] = "atomic-scientific-verifier-prospective-cohort-freeze-v1"

    freeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_context_id: str
    source_context_sha256: str
    source_task_id: str
    source_candidate_portfolio_id: str
    source_atomic_report_id: str
    atomic_portfolio_id: str
    atomic_query_plan_id: str

    hypothesis_ids: list[str]
    claim_ids: list[str]
    hypothesis_count: int = Field(ge=0)
    claim_count: int = Field(ge=0)

    frozen_before_old_n10: Literal[True] = True
    frozen_before_new_verifier: Literal[True] = True
    verifier_results_observed_before_freeze: Literal[False] = False
    old_n10_results_observed_before_freeze: Literal[False] = False
    production_selection_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(self) -> "AtomicScientificVerifierProspectiveCohortFreeze":
        if self.hypothesis_count != len(self.hypothesis_ids):
            raise ValueError("prospective freeze hypothesis_count mismatch")
        if self.claim_count != len(self.claim_ids):
            raise ValueError("prospective freeze claim_count mismatch")
        if len(self.hypothesis_ids) != len(set(self.hypothesis_ids)):
            raise ValueError("prospective freeze duplicate hypothesis IDs")
        if len(self.claim_ids) != len(set(self.claim_ids)):
            raise ValueError("prospective freeze duplicate claim IDs")

        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective freeze SHA mismatch")
        if observed_id != "atomic_scientific_verifier_cohort:" + expected_sha[:20]:
            raise ValueError("prospective freeze ID mismatch")
        return self


def build_atomic_scientific_verifier_prospective_cohort_freeze(
    *,
    context: HypothesisContext,
    candidate_portfolio: ProductionFacingScientificCandidatePortfolio,
    atomic_report: AtomicCrossLaneSynthesisReport,
    atomic_portfolio: HypothesisPortfolio,
    query_plan: LiteratureQueryPlan,
) -> AtomicScientificVerifierProspectiveCohortFreeze:
    if candidate_portfolio.source_context_id != context.context_id:
        raise ValueError("candidate/context ID mismatch")
    if candidate_portfolio.source_context_sha256 != context.context_sha256:
        raise ValueError("candidate/context SHA mismatch")
    if candidate_portfolio.source_task_id != context.task_id:
        raise ValueError("candidate/context task mismatch")
    if candidate_portfolio.domain_profile_id != context.domain_profile_id:
        raise ValueError("candidate/context domain mismatch")

    if atomic_report.source_candidate_portfolio_id != candidate_portfolio.portfolio_id:
        raise ValueError("atomic report/candidate portfolio mismatch")
    if atomic_report.source_context_id != context.context_id:
        raise ValueError("atomic report/context mismatch")
    if atomic_report.source_task_id != context.task_id:
        raise ValueError("atomic report/task mismatch")

    report_hypothesis_ids = [row.hypothesis_id for row in atomic_report.hypotheses]
    portfolio_hypothesis_ids = [row.hypothesis_id for row in atomic_portfolio.hypotheses]
    if set(report_hypothesis_ids) != set(portfolio_hypothesis_ids):
        raise ValueError("atomic report/portfolio hypothesis sets differ")
    if len(report_hypothesis_ids) != len(set(report_hypothesis_ids)):
        raise ValueError("atomic report contains duplicate hypothesis IDs")

    if atomic_portfolio.source_context_id != context.context_id:
        raise ValueError("atomic portfolio/context mismatch")
    if atomic_portfolio.source_context_sha256 != context.context_sha256:
        raise ValueError("atomic portfolio/context SHA mismatch")
    if atomic_portfolio.domain_profile_id != context.domain_profile_id:
        raise ValueError("atomic portfolio/domain mismatch")

    if query_plan.source_portfolio_id != atomic_portfolio.portfolio_id:
        raise ValueError("atomic query plan/portfolio mismatch")

    plan_hypothesis_ids = [row.hypothesis_id for row in query_plan.claims]
    if set(plan_hypothesis_ids) != set(portfolio_hypothesis_ids):
        raise ValueError("atomic query-plan hypothesis set differs from portfolio")

    report_claim_ids = sorted(
        spec.claim_id
        for hypothesis in atomic_report.hypotheses
        for spec in hypothesis.atomic_specifications
    )
    plan_claim_ids = sorted(
        claim.claim_id
        for row in query_plan.claims
        for claim in row.claims
    )
    if report_claim_ids != plan_claim_ids:
        raise ValueError("atomic report/query-plan claim sets differ")

    hypothesis_ids = sorted(portfolio_hypothesis_ids)
    claim_ids = sorted(report_claim_ids)
    body = {
        "schema_version": "atomic-scientific-verifier-prospective-cohort-freeze-v1",
        "source_context_id": context.context_id,
        "source_context_sha256": context.context_sha256,
        "source_task_id": context.task_id,
        "source_candidate_portfolio_id": candidate_portfolio.portfolio_id,
        "source_atomic_report_id": atomic_report.report_id,
        "atomic_portfolio_id": atomic_portfolio.portfolio_id,
        "atomic_query_plan_id": query_plan.plan_id,
        "hypothesis_ids": hypothesis_ids,
        "claim_ids": claim_ids,
        "hypothesis_count": len(hypothesis_ids),
        "claim_count": len(claim_ids),
        "frozen_before_old_n10": True,
        "frozen_before_new_verifier": True,
        "verifier_results_observed_before_freeze": False,
        "old_n10_results_observed_before_freeze": False,
        "production_selection_authority": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return AtomicScientificVerifierProspectiveCohortFreeze(
        **body,
        freeze_id="atomic_scientific_verifier_cohort:" + digest[:20],
        freeze_sha256=digest,
    )


NormalizedDecision = Literal["CERTIFIED", "UNRESOLVED", "REJECTED"]


class AtomicScientificVerifierProspectiveComparisonRow(StrictModel):
    hypothesis_id: str
    old_n10_decision: NormalizedDecision
    old_n10_selection_class: str
    old_n10_positive_nonobviousness_authority: bool
    new_verifier_decision: NormalizedDecision

    decisions_agree: bool
    bounded_closure_state: str
    bounded_external_distinctness_state: str
    positive_nonobviousness_authority_state: str
    fatal_blocker_state: str
    new_verifier_reason_codes: list[str]
    typed_identity_excluded_work_count: int = Field(ge=0)
    diagnostic_factors: list[str]

    production_selection_authority: Literal[False] = False


class AtomicScientificVerifierProspectiveComparisonReport(StrictModel):
    schema_version: Literal[
        "atomic-scientific-verifier-prospective-comparison-report-v1"
    ] = "atomic-scientific-verifier-prospective-comparison-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_cohort_freeze_id: str
    source_old_n10_report_id: str
    source_new_verifier_report_id: str
    source_aggregation_report_id: str
    source_relation_candidate_report_id: str

    rows: list[AtomicScientificVerifierProspectiveComparisonRow]
    hypothesis_count: int = Field(ge=0)
    agreement_count: int = Field(ge=0)
    disagreement_count: int = Field(ge=0)
    comparison_cells: dict[str, int]

    cohort_changed_after_freeze: Literal[False] = False
    old_n10_mutated_by_verifier: Literal[False] = False
    production_selection_changed: Literal[False] = False
    comparison_is_diagnostic_only: Literal[True] = True

    @model_validator(mode="after")
    def validate_report(self) -> "AtomicScientificVerifierProspectiveComparisonReport":
        if self.hypothesis_count != len(self.rows):
            raise ValueError("prospective comparison hypothesis_count mismatch")
        agreements = sum(row.decisions_agree for row in self.rows)
        if self.agreement_count != agreements:
            raise ValueError("prospective comparison agreement_count mismatch")
        if self.disagreement_count != self.hypothesis_count - agreements:
            raise ValueError("prospective comparison disagreement_count mismatch")
        expected = Counter(
            f"OLD_{row.old_n10_decision}__NEW_{row.new_verifier_decision}"
            for row in self.rows
        )
        if dict(sorted(expected.items())) != dict(sorted(self.comparison_cells.items())):
            raise ValueError("prospective comparison cells mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective comparison SHA mismatch")
        if observed_id != "atomic_scientific_verifier_comparison:" + expected_sha[:20]:
            raise ValueError("prospective comparison ID mismatch")
        return self


def _normalize_old_n10(status: str) -> NormalizedDecision:
    mapping = {
        "NOVELTY_CERTIFIED": "CERTIFIED",
        "NOVELTY_UNRESOLVED": "UNRESOLVED",
        "NOVELTY_REJECTED": "REJECTED",
    }
    try:
        return mapping[status]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError("unsupported old N10 certification status: " + status) from exc


def _diagnostic_factors(
    *,
    new_decision,
    aggregation,
    typed_identity_excluded_work_count: int,
) -> list[str]:
    factors: list[str] = []
    if new_decision.bounded_closure_state != "BOUNDED_REVIEW_CLOSED":
        factors.append("coverage_failure")
    if (
        new_decision.bounded_external_distinctness_state
        != "BOUNDED_EXTERNAL_DISTINCTNESS_SUPPORTED"
    ):
        factors.append("external_distinctness_failure")
    if new_decision.positive_nonobviousness_authority_state != "AUTHORIZED":
        factors.append("positive_nonobviousness_absence")
    if "DIRECT_PRIOR_ART" in new_decision.fatal_blocker_state:
        factors.append("direct_prior_art_blocker")
    if "FATAL_CONTRADICTION" in new_decision.fatal_blocker_state:
        factors.append("fatal_contradiction")

    kinds = aggregation.strong_pressure_kind_counts
    if kinds.get("LOWER_ORDER_RELATION_PRIOR_ART", 0):
        factors.append("lower_order_prior_art")
    if kinds.get("DIRECTIONAL_COUNTEREVIDENCE", 0):
        factors.append("directional_counterevidence")
    if kinds.get("CONTEXTUAL_CONFLICT", 0):
        factors.append("contextual_conflict")
    if kinds.get("CONFLICTING_PRIOR_ART", 0):
        factors.append("conflicting_prior_art")
    if kinds.get("DIRECT_PRIOR_ART", 0):
        factors.append("direct_prior_art_pressure")
    if typed_identity_excluded_work_count:
        factors.append("typed_identity_exclusion_present")
    return factors


def build_atomic_scientific_verifier_prospective_comparison(
    *,
    cohort_freeze: AtomicScientificVerifierProspectiveCohortFreeze,
    old_n10: ScientificSynthesisNoveltyCertificationReport,
    new_verifier: ScientificCertificationGateReport,
    aggregation: ScientificHypothesisEvidenceAggregationReport,
    relation_candidates: ProjectionRelationCandidateReport,
) -> AtomicScientificVerifierProspectiveComparisonReport:
    if old_n10.source_portfolio_id != cohort_freeze.atomic_portfolio_id:
        raise ValueError("old N10 report does not derive from frozen atomic portfolio")

    frozen_ids = set(cohort_freeze.hypothesis_ids)
    old_by_id = {row.hypothesis_id: row for row in old_n10.decisions}
    new_by_id = {row.hypothesis_id: row for row in new_verifier.decisions}
    agg_by_id = {row.hypothesis_id: row for row in aggregation.hypothesis_aggregations}
    typed_excluded_by_hypothesis: dict[str, int] = {}
    for claim in relation_candidates.claims:
        typed_excluded_by_hypothesis[claim.hypothesis_id] = (
            typed_excluded_by_hypothesis.get(claim.hypothesis_id, 0)
            + claim.typed_identity_excluded_work_count
        )
    for label, observed in (
        ("old N10", set(old_by_id)),
        ("new verifier", set(new_by_id)),
        ("aggregation", set(agg_by_id)),
        ("relation candidates", set(typed_excluded_by_hypothesis)),
    ):
        if observed != frozen_ids:
            raise ValueError(label + " hypothesis set differs from frozen cohort")

    rows: list[AtomicScientificVerifierProspectiveComparisonRow] = []
    for hypothesis_id in sorted(frozen_ids):
        old = old_by_id[hypothesis_id]
        new = new_by_id[hypothesis_id]
        agg = agg_by_id[hypothesis_id]
        old_decision = _normalize_old_n10(old.certification_status)
        typed_excluded = typed_excluded_by_hypothesis[hypothesis_id]
        rows.append(
            AtomicScientificVerifierProspectiveComparisonRow(
                hypothesis_id=hypothesis_id,
                old_n10_decision=old_decision,
                old_n10_selection_class=old.selection_class,
                old_n10_positive_nonobviousness_authority=(
                    old.positive_nonobviousness_authority
                ),
                new_verifier_decision=new.decision,
                decisions_agree=(old_decision == new.decision),
                bounded_closure_state=new.bounded_closure_state,
                bounded_external_distinctness_state=(
                    new.bounded_external_distinctness_state
                ),
                positive_nonobviousness_authority_state=(
                    new.positive_nonobviousness_authority_state
                ),
                fatal_blocker_state=new.fatal_blocker_state,
                new_verifier_reason_codes=list(new.reason_codes),
                typed_identity_excluded_work_count=typed_excluded,
                diagnostic_factors=_diagnostic_factors(
                    new_decision=new,
                    aggregation=agg,
                    typed_identity_excluded_work_count=typed_excluded,
                ),
            )
        )

    cells = Counter(
        f"OLD_{row.old_n10_decision}__NEW_{row.new_verifier_decision}"
        for row in rows
    )
    body = {
        "schema_version": "atomic-scientific-verifier-prospective-comparison-report-v1",
        "source_cohort_freeze_id": cohort_freeze.freeze_id,
        "source_old_n10_report_id": old_n10.report_id,
        "source_new_verifier_report_id": new_verifier.report_id,
        "source_aggregation_report_id": aggregation.report_id,
        "source_relation_candidate_report_id": relation_candidates.report_id,
        "rows": [row.model_dump(mode="json") for row in rows],
        "hypothesis_count": len(rows),
        "agreement_count": sum(row.decisions_agree for row in rows),
        "disagreement_count": sum(not row.decisions_agree for row in rows),
        "comparison_cells": dict(sorted(cells.items())),
        "cohort_changed_after_freeze": False,
        "old_n10_mutated_by_verifier": False,
        "production_selection_changed": False,
        "comparison_is_diagnostic_only": True,
    }
    digest = _sha256_json(body)
    return AtomicScientificVerifierProspectiveComparisonReport(
        **body,
        report_id="atomic_scientific_verifier_comparison:" + digest[:20],
        report_sha256=digest,
    )


__all__ = [
    "AtomicScientificVerifierProspectiveCohortFreeze",
    "AtomicScientificVerifierProspectiveComparisonReport",
    "AtomicScientificVerifierProspectiveComparisonRow",
    "build_atomic_scientific_verifier_prospective_cohort_freeze",
    "build_atomic_scientific_verifier_prospective_comparison",
]
