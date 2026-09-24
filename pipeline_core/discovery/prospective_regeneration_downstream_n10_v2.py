from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.nonobviousness_post_generation import (
    _validate_n10_gate,
    certified_novelty_portfolio_id,
)
from pipeline_core.discovery.prospective_regeneration_downstream_semantic_v2 import (
    ProspectiveRegenerationDownstreamSemanticReportV2,
)
from pipeline_core.discovery.prospective_regeneration_downstream_v2 import (
    ProspectiveRegenerationDownstreamV2Freeze,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


RegenerationNoveltyCertificationStatus = Literal[
    "NOVELTY_CERTIFIED",
    "NOVELTY_UNRESOLVED",
    "NOVELTY_REJECTED",
]


class RegenerationNoveltyDecisionV2(StrictModel):
    source_final_hypothesis_id: str
    regenerated_hypothesis_id: str
    n10_selection_class: Literal["ELIGIBLE", "CONDITIONAL", "INELIGIBLE"]
    n10_action: str | None = None
    certification_status: RegenerationNoveltyCertificationStatus
    novelty_certified: bool
    candidate_retained: Literal[True] = True
    unresolved_dimensions: list[str] = Field(default_factory=list)
    blocking_claim_ids: list[str] = Field(default_factory=list)
    unresolved_claim_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class RegenerationNoveltyArtifactBundleV2(StrictModel):
    source_final_hypothesis_id: str
    regenerated_hypothesis_id: str
    source_portfolio: str
    query_plan: str
    prior_art: str
    external_report: str
    n9_intake: str
    n9_full_closure: str
    n10_candidate_gate: str
    n10_production_gate: str
    candidate_portfolio: str
    certification_report: str
    certified_portfolio: str
    candidate_equals_final_identity: Literal[True] = True
    scientific_candidate_preserved: Literal[True] = True


class RegenerationNoveltyCertificationReportV2(StrictModel):
    schema_version: Literal[
        "regeneration-post-generation-novelty-certification-v2"
    ] = "regeneration-post-generation-novelty-certification-v2"
    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_final_hypothesis_id: str
    source_regenerated_portfolio_id: str
    candidate_portfolio_id: str
    certified_portfolio_id: str
    decisions: list[RegenerationNoveltyDecisionV2]
    candidate_artifacts: list[RegenerationNoveltyArtifactBundleV2]
    candidate_count: int = Field(ge=0)
    certified_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    authority_mode: Literal["certification_only"] = "certification_only"
    authority_source: Literal[
        "n10_role_aware_nonobviousness_v2"
    ] = "n10_role_aware_nonobviousness_v2"
    authority_scope: Literal[
        "regeneration_v2_novelty_certification_only"
    ] = "regeneration_v2_novelty_certification_only"
    candidate_portfolio_preserved: Literal[True] = True
    candidate_survival_authority: Literal[False] = False
    novelty_certification_authority: Literal[True] = True
    conditional_is_positive: Literal[False] = False
    absence_is_novelty: Literal[False] = False
    ineligible_deletes_scientific_candidate: Literal[False] = False
    policy: dict[str, str] = Field(default_factory=lambda: {
        "ELIGIBLE": "NOVELTY_CERTIFIED",
        "CONDITIONAL": "NOVELTY_UNRESOLVED",
        "INELIGIBLE": "NOVELTY_REJECTED",
    })

    @model_validator(mode="after")
    def validate_report(self):
        if self.candidate_count != len(self.decisions):
            raise ValueError("candidate_count mismatch")
        if self.candidate_count != len(self.candidate_artifacts):
            raise ValueError("candidate artifact count mismatch")
        statuses = [row.certification_status for row in self.decisions]
        if self.certified_count != statuses.count("NOVELTY_CERTIFIED"):
            raise ValueError("certified_count mismatch")
        if self.unresolved_count != statuses.count("NOVELTY_UNRESOLVED"):
            raise ValueError("unresolved_count mismatch")
        if self.rejected_count != statuses.count("NOVELTY_REJECTED"):
            raise ValueError("rejected_count mismatch")
        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("regeneration novelty certification SHA mismatch")
        if observed_id != "regeneration_novelty_certification_v2:" + expected_sha[:20]:
            raise ValueError("regeneration novelty certification ID mismatch")
        return self


class RegenerationExternalN10LineageResultV2(StrictModel):
    source_final_hypothesis_id: str
    regenerated_hypothesis_id: str
    certification_status: RegenerationNoveltyCertificationStatus
    n10_selection_class: str
    external_novelty_stage_invocations: Literal[1] = 1
    n9_intake_stage_invocations: Literal[1] = 1
    n9_full_closure_stage_invocations: Literal[1] = 1
    n10_certification_stage_invocations: Literal[1] = 1
    binding_plan_performed: Literal[False] = False
    preverifier_gate_v2_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False
    second_regeneration_performed: Literal[False] = False


class ProspectiveRegenerationExternalN10ReportV2(StrictModel):
    schema_version: Literal[
        "prospective-regeneration-external-n10-v2"
    ] = "prospective-regeneration-external-n10-v2"
    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: Literal["P16", "P17", "P18", "P19", "P20"]
    source_semantic_report_id: str
    source_semantic_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_downstream_freeze_id: str
    source_downstream_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineages: list[RegenerationExternalN10LineageResultV2]
    lineage_count: int = Field(ge=0)
    certified_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    external_novelty_stage_invocations: int = Field(ge=0)
    n9_intake_stage_invocations: int = Field(ge=0)
    n9_full_closure_stage_invocations: int = Field(ge=0)
    n10_certification_stage_invocations: int = Field(ge=0)
    targeted_novelty_continuation_performed: Literal[False] = False
    novelty_refinement_performed: Literal[False] = False
    binding_plan_performed: Literal[False] = False
    preverifier_gate_v2_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False
    downstream_retry_performed: Literal[False] = False
    second_regeneration_performed: Literal[False] = False
    regenerated_portfolio_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self):
        if self.lineage_count != len(self.lineages):
            raise ValueError("lineage_count mismatch")
        n = len(self.lineages)
        for field in (
            "external_novelty_stage_invocations",
            "n9_intake_stage_invocations",
            "n9_full_closure_stage_invocations",
            "n10_certification_stage_invocations",
        ):
            if getattr(self, field) != n:
                raise ValueError(field + " mismatch")
        statuses = [row.certification_status for row in self.lineages]
        if self.certified_count != statuses.count("NOVELTY_CERTIFIED"):
            raise ValueError("certified_count mismatch")
        if self.unresolved_count != statuses.count("NOVELTY_UNRESOLVED"):
            raise ValueError("unresolved_count mismatch")
        if self.rejected_count != statuses.count("NOVELTY_REJECTED"):
            raise ValueError("rejected_count mismatch")
        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("external-N10-v2 SHA mismatch")
        if observed_id != "prospective_regeneration_external_n10_v2:" + expected_sha[:20]:
            raise ValueError("external-N10-v2 ID mismatch")
        return self


def unresolved_dimensions_from_gate_row(row: dict[str, Any]) -> list[str]:
    dimensions: set[str] = set()
    for requirement in row.get("resolution_requirements") or []:
        role = str(requirement.get("novelty_selection_role") or "")
        action = str(requirement.get("action") or "")
        outcome = str(requirement.get("nonobviousness_outcome") or "")
        reasons = {str(value) for value in (requirement.get("reason_codes") or [])}
        if role == "REQUIRED_ENABLING_RELATION":
            dimensions.add("REQUIRED_ENABLING_RELATION")
        if role != "NOVELTY_BEARING":
            continue
        if "SPECIFICATION" in action or "atomic_specification_incomplete" in reasons:
            dimensions.add("SPECIFICATION")
        if (
            action == "RESOLVE_NOVELTY_BEARING_EVIDENCE"
            or outcome == "INSUFFICIENT_FOR_JUDGMENT"
            or "candidate_not_ready_for_adjudication" in reasons
        ):
            dimensions.add("EVIDENCE_CLOSURE")
        if (
            action == "RESOLVE_NOVELTY_BEARING_PRIOR_ART_RELATION"
            or "partial_prior_art_requires_resolution" in reasons
        ):
            dimensions.add("PRIOR_ART_RELATION")
    return sorted(dimensions)


def certification_status_from_production_gate(
    *,
    regenerated_hypothesis_id: str,
    production_gate: dict[str, Any],
) -> tuple[RegenerationNoveltyCertificationStatus, dict[str, Any]]:
    row = _validate_n10_gate(
        candidate_id=regenerated_hypothesis_id,
        gate=production_gate,
    )
    selection_class = str(row.get("selection_class") or "")
    positive = bool(row.get("positive_nonobviousness_authority"))
    fallback_allowed = bool(row.get("fallback_allowed"))
    if selection_class == "ELIGIBLE":
        if not positive or not fallback_allowed:
            raise ValueError("ELIGIBLE regenerated candidate lacks positive N10 authority")
        return "NOVELTY_CERTIFIED", row
    if selection_class == "CONDITIONAL":
        if positive:
            raise ValueError("CONDITIONAL cannot carry positive N10 authority")
        return "NOVELTY_UNRESOLVED", row
    if selection_class == "INELIGIBLE":
        if positive:
            raise ValueError("INELIGIBLE cannot carry positive N10 authority")
        return "NOVELTY_REJECTED", row
    raise ValueError("unsupported N10 selection class: " + selection_class)


def build_regeneration_novelty_certification_v2(
    *,
    source_final_hypothesis_id: str,
    portfolio: HypothesisPortfolio,
    production_gate: dict[str, Any],
    artifact_bundle: RegenerationNoveltyArtifactBundleV2,
) -> tuple[HypothesisPortfolio, HypothesisPortfolio, RegenerationNoveltyCertificationReportV2]:
    if len(portfolio.hypotheses) != 1:
        raise ValueError("regeneration downstream expects exactly one hypothesis per lineage")
    card = portfolio.hypotheses[0]
    status, gate_row = certification_status_from_production_gate(
        regenerated_hypothesis_id=card.hypothesis_id,
        production_gate=production_gate,
    )
    certified_cards = [card] if status == "NOVELTY_CERTIFIED" else []
    certified_reason = None if certified_cards else (
        "No regenerated scientific candidate currently carries positive novelty-certification authority."
    )
    certified_id = certified_novelty_portfolio_id(
        source_portfolio_id=portfolio.portfolio_id,
        domain_profile_id=portfolio.domain_profile_id,
        source_context_sha256=portfolio.source_context_sha256,
        certified_hypothesis_ids=tuple(row.hypothesis_id for row in certified_cards),
        abstention_reason=certified_reason,
    )
    certified = HypothesisPortfolio(
        portfolio_id=certified_id,
        domain_profile_id=portfolio.domain_profile_id,
        source_context_id=portfolio.source_context_id,
        source_context_sha256=portfolio.source_context_sha256,
        source_report_id=portfolio.source_report_id,
        source_report_sha256=portfolio.source_report_sha256,
        hypotheses=certified_cards,
        abstention_reason=certified_reason,
    )
    decision = RegenerationNoveltyDecisionV2(
        source_final_hypothesis_id=source_final_hypothesis_id,
        regenerated_hypothesis_id=card.hypothesis_id,
        n10_selection_class=str(gate_row.get("selection_class")),
        n10_action=(str(gate_row.get("action")) if gate_row.get("action") is not None else None),
        certification_status=status,
        novelty_certified=(status == "NOVELTY_CERTIFIED"),
        unresolved_dimensions=(
            unresolved_dimensions_from_gate_row(gate_row)
            if status == "NOVELTY_UNRESOLVED" else []
        ),
        blocking_claim_ids=list(gate_row.get("blocking_claim_ids") or []),
        unresolved_claim_ids=list(gate_row.get("unresolved_claim_ids") or []),
        reason_codes=list(gate_row.get("reason_codes") or []),
    )
    body = {
        "schema_version": "regeneration-post-generation-novelty-certification-v2",
        "source_final_hypothesis_id": source_final_hypothesis_id,
        "source_regenerated_portfolio_id": portfolio.portfolio_id,
        "candidate_portfolio_id": portfolio.portfolio_id,
        "certified_portfolio_id": certified.portfolio_id,
        "decisions": [decision.model_dump(mode="json")],
        "candidate_artifacts": [artifact_bundle.model_dump(mode="json")],
        "candidate_count": 1,
        "certified_count": int(status == "NOVELTY_CERTIFIED"),
        "unresolved_count": int(status == "NOVELTY_UNRESOLVED"),
        "rejected_count": int(status == "NOVELTY_REJECTED"),
        "authority_mode": "certification_only",
        "authority_source": "n10_role_aware_nonobviousness_v2",
        "authority_scope": "regeneration_v2_novelty_certification_only",
        "candidate_portfolio_preserved": True,
        "candidate_survival_authority": False,
        "novelty_certification_authority": True,
        "conditional_is_positive": False,
        "absence_is_novelty": False,
        "ineligible_deletes_scientific_candidate": False,
        "policy": {
            "ELIGIBLE": "NOVELTY_CERTIFIED",
            "CONDITIONAL": "NOVELTY_UNRESOLVED",
            "INELIGIBLE": "NOVELTY_REJECTED",
        },
    }
    digest = _sha256_json(body)
    report = RegenerationNoveltyCertificationReportV2(
        **body,
        report_id="regeneration_novelty_certification_v2:" + digest[:20],
        report_sha256=digest,
    )
    return portfolio, certified, report


def build_external_n10_execution_report_v2(
    *,
    semantic_report: ProspectiveRegenerationDownstreamSemanticReportV2,
    downstream_freeze: ProspectiveRegenerationDownstreamV2Freeze,
    lineage_results: list[RegenerationExternalN10LineageResultV2],
) -> ProspectiveRegenerationExternalN10ReportV2:
    budget = downstream_freeze.policy.budget
    if budget.external_novelty_stage_max != 1:
        raise ValueError("external novelty budget drift")
    if budget.n9_shadow_intake_stage_max != 1:
        raise ValueError("N9 intake budget drift")
    if budget.n9_full_closure_stage_max != 1:
        raise ValueError("N9 closure budget drift")
    if budget.n10_certification_stage_max != 1:
        raise ValueError("N10 certification budget drift")
    if downstream_freeze.policy.post_generation_refinement_enabled:
        raise ValueError("novelty refinement must remain disabled")
    if downstream_freeze.policy.bounded_continuation_enabled:
        raise ValueError("bounded continuation must remain disabled")
    eligible_ids = {
        row.source_final_hypothesis_id for row in semantic_report.lineages
        if row.eligible_for_external_novelty
    }
    result_ids = {row.source_final_hypothesis_id for row in lineage_results}
    if eligible_ids != result_ids:
        raise ValueError("external-N10 lineage population differs from semantic eligibility")
    statuses = [row.certification_status for row in lineage_results]
    n = len(lineage_results)
    body = {
        "schema_version": "prospective-regeneration-external-n10-v2",
        "case_id": semantic_report.case_id,
        "source_semantic_report_id": semantic_report.report_id,
        "source_semantic_report_sha256": semantic_report.report_sha256,
        "source_downstream_freeze_id": downstream_freeze.freeze_id,
        "source_downstream_freeze_sha256": downstream_freeze.freeze_sha256,
        "lineages": [row.model_dump(mode="json") for row in lineage_results],
        "lineage_count": n,
        "certified_count": statuses.count("NOVELTY_CERTIFIED"),
        "unresolved_count": statuses.count("NOVELTY_UNRESOLVED"),
        "rejected_count": statuses.count("NOVELTY_REJECTED"),
        "external_novelty_stage_invocations": n,
        "n9_intake_stage_invocations": n,
        "n9_full_closure_stage_invocations": n,
        "n10_certification_stage_invocations": n,
        "targeted_novelty_continuation_performed": False,
        "novelty_refinement_performed": False,
        "binding_plan_performed": False,
        "preverifier_gate_v2_performed": False,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "downstream_retry_performed": False,
        "second_regeneration_performed": False,
        "regenerated_portfolio_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveRegenerationExternalN10ReportV2(
        **body,
        report_id="prospective_regeneration_external_n10_v2:" + digest[:20],
        report_sha256=digest,
    )
