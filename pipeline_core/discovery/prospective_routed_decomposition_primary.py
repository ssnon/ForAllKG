from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import NoveltyClaim
from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    PreVerifierContractGateV2Report,
    build_preverifier_contract_gate_v2,
)
from pipeline_core.discovery.prospective_routed_route_compiler import (
    ProspectiveRoutedDispatchReport,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingClaimPlan,
    RelationalAtomicBindingHypothesisPlan,
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_projection import (
    _ATOMIC_KINDS,
    _candidate_card_and_claim,
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
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


DecompositionStatus = Literal[
    "DETERMINISTIC_DECOMPOSITION_AVAILABLE",
    "DETERMINISTIC_DECOMPOSITION_UNAVAILABLE",
]

HypothesisPrimaryStatus = Literal[
    "PRIMARY_MATERIALIZED",
    "PRIMARY_UNAVAILABLE_REGENERATION_REQUIRED",
]


class DeterministicDecompositionAssessment(StrictModel):
    composite_claim_id: str
    status: DecompositionStatus
    component_claim_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    source_supported_only: Literal[True] = True
    new_claim_generated: Literal[False] = False
    llm_call_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_assessment(
        self,
    ) -> "DeterministicDecompositionAssessment":
        if (
            self.status == "DETERMINISTIC_DECOMPOSITION_AVAILABLE"
            and self.reason_codes
        ):
            raise ValueError(
                "available deterministic decomposition cannot carry blockers"
            )
        if (
            self.status == "DETERMINISTIC_DECOMPOSITION_UNAVAILABLE"
            and not self.reason_codes
        ):
            raise ValueError(
                "unavailable deterministic decomposition requires blockers"
            )
        return self


class RoutedDecompositionHypothesisResult(StrictModel):
    candidate_hypothesis_id: str
    final_hypothesis_id: str
    source_claim_ids: list[str]
    output_claim_ids: list[str]

    primary_status: HypothesisPrimaryStatus
    decompositions: list[DeterministicDecompositionAssessment]

    post_gate_ready_claim_ids: list[str]
    post_gate_novelty_bearing_ready_claim_ids: list[str]

    regeneration_fallback_required: bool
    regeneration_fallback_allowed: Literal[True] = True

    scientific_content_added: Literal[False] = False
    new_claim_generated: Literal[False] = False
    source_alignment_performed: Literal[False] = False
    specification_repair_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> "RoutedDecompositionHypothesisResult":
        if self.primary_status == "PRIMARY_UNAVAILABLE_REGENERATION_REQUIRED":
            if not self.regeneration_fallback_required:
                raise ValueError(
                    "unavailable decomposition must require regeneration"
                )
        if self.post_gate_novelty_bearing_ready_claim_ids:
            if any(
                claim_id not in self.post_gate_ready_claim_ids
                for claim_id in self.post_gate_novelty_bearing_ready_claim_ids
            ):
                raise ValueError(
                    "novelty-ready claims must be post-gate ready"
                )
        return self


class ProspectiveRoutedDecompositionPrimaryReport(StrictModel):
    schema_version: Literal[
        "prospective-routed-decomposition-primary-report-v1"
    ] = "prospective-routed-decomposition-primary-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    case_id: Literal["P11", "P12", "P13", "P14", "P15"]

    source_dispatch_report_id: str
    source_dispatch_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_binding_plan_id: str
    source_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    post_route_binding_plan_id: str
    post_route_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    post_route_gate_report_id: str
    post_route_gate_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    hypotheses: list[RoutedDecompositionHypothesisResult]
    hypothesis_count: int = Field(ge=0)
    primary_materialized_count: int = Field(ge=0)
    primary_unavailable_count: int = Field(ge=0)
    regeneration_fallback_required_count: int = Field(ge=0)

    deterministic_decomposition_count: int = Field(ge=0)
    unavailable_decomposition_count: int = Field(ge=0)

    all_input_hypotheses_on_decomposition_route: Literal[True] = True
    deterministic_topology_only: Literal[True] = True
    decomposition_llm_calls_performed: Literal[0] = 0
    audit_llm_calls_performed: Literal[0] = 0
    source_alignment_performed: Literal[False] = False
    specification_repair_performed: Literal[False] = False
    regeneration_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "ProspectiveRoutedDecompositionPrimaryReport":
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError("hypothesis_count mismatch")
        if self.primary_materialized_count != sum(
            row.primary_status == "PRIMARY_MATERIALIZED"
            for row in self.hypotheses
        ):
            raise ValueError("primary_materialized_count mismatch")
        if self.primary_unavailable_count != sum(
            row.primary_status
            == "PRIMARY_UNAVAILABLE_REGENERATION_REQUIRED"
            for row in self.hypotheses
        ):
            raise ValueError("primary_unavailable_count mismatch")
        if self.regeneration_fallback_required_count != sum(
            row.regeneration_fallback_required
            for row in self.hypotheses
        ):
            raise ValueError(
                "regeneration_fallback_required_count mismatch"
            )

        available = sum(
            item.status == "DETERMINISTIC_DECOMPOSITION_AVAILABLE"
            for row in self.hypotheses
            for item in row.decompositions
        )
        unavailable = sum(
            item.status == "DETERMINISTIC_DECOMPOSITION_UNAVAILABLE"
            for row in self.hypotheses
            for item in row.decompositions
        )
        if self.deterministic_decomposition_count != available:
            raise ValueError("deterministic_decomposition_count mismatch")
        if self.unavailable_decomposition_count != unavailable:
            raise ValueError("unavailable_decomposition_count mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError(
                "routed decomposition primary report SHA mismatch"
            )
        if observed_id != (
            "prospective_routed_decomposition_primary_report:"
            + expected_sha[:20]
        ):
            raise ValueError(
                "routed decomposition primary report ID mismatch"
            )
        return self


def assess_deterministic_decomposition(
    *,
    composite_claim: NoveltyClaim,
    source_claims_by_id: dict[str, NoveltyClaim],
) -> DeterministicDecompositionAssessment:
    reasons: list[str] = []

    if composite_claim.kind != "composite":
        reasons.append(
            "source_claim_not_composite:" + str(composite_claim.kind)
        )

    component_ids = list(
        dict.fromkeys(composite_claim.higher_order_component_claim_ids)
    )

    if len(component_ids) < 2:
        reasons.append(
            "insufficient_explicit_component_cardinality:"
            + str(len(component_ids))
        )

    if not composite_claim.higher_order_relation_basis:
        reasons.append("missing_higher_order_relation_basis")

    if composite_claim.higher_order_relation_reason_codes:
        reasons.append(
            "higher_order_relation_provenance_not_clean"
        )

    resolved: list[NoveltyClaim] = []
    for component_id in component_ids:
        component = source_claims_by_id.get(component_id)
        if component is None:
            reasons.append(
                "missing_component_claim:" + component_id
            )
            continue
        resolved.append(component)

    for component in resolved:
        if component.claim_id == composite_claim.claim_id:
            reasons.append("composite_self_component")
        if component.kind not in _ATOMIC_KINDS:
            reasons.append(
                "unsupported_component_atomic_kind:"
                + component.claim_id
                + ":"
                + component.kind
            )

    reasons = list(dict.fromkeys(reasons))
    return DeterministicDecompositionAssessment(
        composite_claim_id=composite_claim.claim_id,
        status=(
            "DETERMINISTIC_DECOMPOSITION_AVAILABLE"
            if not reasons
            else "DETERMINISTIC_DECOMPOSITION_UNAVAILABLE"
        ),
        component_claim_ids=component_ids,
        reason_codes=reasons,
    )


def _hypothesis_binding_status(
    claims: list[RelationalAtomicBindingClaimPlan],
) -> str:
    ready = [
        row
        for row in claims
        if row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
    ]
    novelty_ready = [
        row
        for row in ready
        if row.novelty_selection_role == "NOVELTY_BEARING"
    ]
    if novelty_ready:
        return "READY_FOR_LITERAL_ENDPOINT_BINDING"
    if ready:
        return "NO_NOVELTY_BEARING_BINDABLE_CLAIM"
    return "NO_BINDABLE_CLAIMS"


def _rebuild_hypothesis(
    *,
    source: RelationalAtomicBindingHypothesisPlan,
    removed_claim_ids: set[str],
) -> RelationalAtomicBindingHypothesisPlan:
    claims = [
        row
        for row in source.claims
        if row.claim_id not in removed_claim_ids
    ]
    ready = sum(
        row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
        for row in claims
    )
    novelty_ready = sum(
        row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
        and row.novelty_selection_role == "NOVELTY_BEARING"
        for row in claims
    )

    payload = source.model_dump(mode="json")
    payload["claims"] = [
        row.model_dump(mode="json")
        for row in claims
    ]
    payload["claim_count"] = len(claims)
    payload["binding_ready_claim_count"] = ready
    payload["novelty_bearing_binding_ready_claim_count"] = novelty_ready
    payload["binding_status"] = _hypothesis_binding_status(claims)
    return RelationalAtomicBindingHypothesisPlan.model_validate(payload)


def rebuild_binding_plan_after_decomposition(
    *,
    source_plan: RelationalAtomicBindingPlan,
    removed_claim_ids_by_final_hypothesis: dict[str, set[str]],
) -> RelationalAtomicBindingPlan:
    hypotheses = [
        _rebuild_hypothesis(
            source=hypothesis,
            removed_claim_ids=removed_claim_ids_by_final_hypothesis.get(
                hypothesis.final_hypothesis_id,
                set(),
            ),
        )
        for hypothesis in source_plan.hypotheses
    ]

    ready_h = sum(
        row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
        for row in hypotheses
    )
    binding_ready = sum(
        row.binding_ready_claim_count
        for row in hypotheses
    )
    novelty_ready = sum(
        row.novelty_bearing_binding_ready_claim_count
        for row in hypotheses
    )

    body = source_plan.model_dump(mode="json")
    body.pop("plan_id")
    body.pop("plan_sha256")
    body["hypotheses"] = [
        row.model_dump(mode="json")
        for row in hypotheses
    ]
    body["hypothesis_count"] = len(hypotheses)
    body["ready_hypothesis_count"] = ready_h
    body["not_ready_hypothesis_count"] = len(hypotheses) - ready_h
    body["claim_count"] = sum(row.claim_count for row in hypotheses)
    body["binding_ready_claim_count"] = binding_ready
    body["novelty_bearing_binding_ready_claim_count"] = novelty_ready
    body["hypothesis_status_counts"] = dict(
        sorted(Counter(row.binding_status for row in hypotheses).items())
    )
    body["claim_status_counts"] = dict(
        sorted(
            Counter(
                claim.binding_status
                for row in hypotheses
                for claim in row.claims
            ).items()
        )
    )

    digest = _sha256_json(body)
    return RelationalAtomicBindingPlan(
        **body,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )


def execute_decomposition_primary(
    *,
    binding_plan: RelationalAtomicBindingPlan,
    dispatch: ProspectiveRoutedDispatchReport,
) -> tuple[
    RelationalAtomicBindingPlan,
    PreVerifierContractGateV2Report,
    ProspectiveRoutedDecompositionPrimaryReport,
]:
    if dispatch.source_binding_plan_id != binding_plan.plan_id:
        raise ValueError("dispatch/binding-plan ID mismatch")
    if dispatch.source_binding_plan_sha256 != binding_plan.plan_sha256:
        raise ValueError("dispatch/binding-plan SHA mismatch")

    if any(
        row.route_action
        != "ATOMIC_DECOMPOSITION_THEN_REGENERATE_IF_UNAVAILABLE"
        for row in dispatch.hypotheses
    ):
        raise ValueError(
            "decomposition-primary executor requires every hypothesis "
            "to be on the frozen decomposition route"
        )

    plan_by_final = {
        row.final_hypothesis_id: row
        for row in binding_plan.hypotheses
    }
    if len(plan_by_final) != len(binding_plan.hypotheses):
        raise ValueError("duplicate final hypothesis IDs in binding plan")

    removed_by_final: dict[str, set[str]] = {}
    preliminary = {}

    for decision in dispatch.hypotheses:
        hypothesis = plan_by_final.get(decision.final_hypothesis_id)
        if hypothesis is None:
            raise ValueError(
                "dispatch hypothesis absent from binding plan: "
                + decision.final_hypothesis_id
            )

        source_claims_by_id: dict[str, NoveltyClaim] = {}
        for claim_plan in hypothesis.claims:
            _card, source_claim = _candidate_card_and_claim(
                hypothesis_plan=hypothesis,
                claim_plan=claim_plan,
            )
            source_claims_by_id[source_claim.claim_id] = source_claim

        assessments: list[DeterministicDecompositionAssessment] = []
        decomposition_work_items = [
            row
            for row in decision.claim_work_items
            if row.router_hint == "DECOMPOSE_OR_REGENERATE_REVIEW"
        ]
        if not decomposition_work_items:
            raise ValueError(
                decision.final_hypothesis_id
                + ": decomposition route has no decomposition work item"
            )

        for work_item in decomposition_work_items:
            source_claim = source_claims_by_id.get(work_item.claim_id)
            if source_claim is None:
                raise ValueError(
                    "decomposition work item source claim is missing: "
                    + work_item.claim_id
                )
            assessments.append(
                assess_deterministic_decomposition(
                    composite_claim=source_claim,
                    source_claims_by_id=source_claims_by_id,
                )
            )

        if all(
            row.status == "DETERMINISTIC_DECOMPOSITION_AVAILABLE"
            for row in assessments
        ):
            removed_by_final[decision.final_hypothesis_id] = {
                row.composite_claim_id
                for row in assessments
            }
            primary_status: HypothesisPrimaryStatus = "PRIMARY_MATERIALIZED"
        else:
            removed_by_final[decision.final_hypothesis_id] = set()
            primary_status = (
                "PRIMARY_UNAVAILABLE_REGENERATION_REQUIRED"
            )

        preliminary[decision.final_hypothesis_id] = (
            primary_status,
            assessments,
        )

    post_plan = rebuild_binding_plan_after_decomposition(
        source_plan=binding_plan,
        removed_claim_ids_by_final_hypothesis=removed_by_final,
    )
    post_gate = build_preverifier_contract_gate_v2(plan=post_plan)

    gate_by_final = {}
    for row in post_gate.rows:
        gate_by_final.setdefault(row.final_hypothesis_id, []).append(row)

    hypothesis_results: list[RoutedDecompositionHypothesisResult] = []
    source_by_final = {
        row.final_hypothesis_id: row
        for row in binding_plan.hypotheses
    }
    output_by_final = {
        row.final_hypothesis_id: row
        for row in post_plan.hypotheses
    }

    for decision in dispatch.hypotheses:
        primary_status, assessments = preliminary[
            decision.final_hypothesis_id
        ]
        gate_rows = gate_by_final.get(decision.final_hypothesis_id, [])
        ready_ids = [
            row.claim_id
            for row in gate_rows
            if row.gate_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
        ]
        novelty_ready_ids = [
            row.claim_id
            for row in gate_rows
            if row.gate_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            and row.novelty_selection_role == "NOVELTY_BEARING"
        ]

        fallback_required = (
            primary_status
            == "PRIMARY_UNAVAILABLE_REGENERATION_REQUIRED"
            or not novelty_ready_ids
        )

        hypothesis_results.append(
            RoutedDecompositionHypothesisResult(
                candidate_hypothesis_id=decision.candidate_hypothesis_id,
                final_hypothesis_id=decision.final_hypothesis_id,
                source_claim_ids=[
                    row.claim_id
                    for row in source_by_final[
                        decision.final_hypothesis_id
                    ].claims
                ],
                output_claim_ids=[
                    row.claim_id
                    for row in output_by_final[
                        decision.final_hypothesis_id
                    ].claims
                ],
                primary_status=primary_status,
                decompositions=assessments,
                post_gate_ready_claim_ids=ready_ids,
                post_gate_novelty_bearing_ready_claim_ids=novelty_ready_ids,
                regeneration_fallback_required=fallback_required,
            )
        )

    materialized_count = sum(
        row.primary_status == "PRIMARY_MATERIALIZED"
        for row in hypothesis_results
    )
    unavailable_count = len(hypothesis_results) - materialized_count
    fallback_count = sum(
        row.regeneration_fallback_required
        for row in hypothesis_results
    )
    available_decompositions = sum(
        item.status == "DETERMINISTIC_DECOMPOSITION_AVAILABLE"
        for row in hypothesis_results
        for item in row.decompositions
    )
    unavailable_decompositions = sum(
        item.status == "DETERMINISTIC_DECOMPOSITION_UNAVAILABLE"
        for row in hypothesis_results
        for item in row.decompositions
    )

    body = {
        "schema_version":
            "prospective-routed-decomposition-primary-report-v1",
        "case_id": dispatch.case_id,
        "source_dispatch_report_id": dispatch.report_id,
        "source_dispatch_report_sha256": dispatch.report_sha256,
        "source_binding_plan_id": binding_plan.plan_id,
        "source_binding_plan_sha256": binding_plan.plan_sha256,
        "post_route_binding_plan_id": post_plan.plan_id,
        "post_route_binding_plan_sha256": post_plan.plan_sha256,
        "post_route_gate_report_id": post_gate.report_id,
        "post_route_gate_report_sha256": post_gate.report_sha256,
        "hypotheses": [
            row.model_dump(mode="json")
            for row in hypothesis_results
        ],
        "hypothesis_count": len(hypothesis_results),
        "primary_materialized_count": materialized_count,
        "primary_unavailable_count": unavailable_count,
        "regeneration_fallback_required_count": fallback_count,
        "deterministic_decomposition_count": available_decompositions,
        "unavailable_decomposition_count": unavailable_decompositions,
        "all_input_hypotheses_on_decomposition_route": True,
        "deterministic_topology_only": True,
        "decomposition_llm_calls_performed": 0,
        "audit_llm_calls_performed": 0,
        "source_alignment_performed": False,
        "specification_repair_performed": False,
        "regeneration_performed": False,
        "endpoint_binding_performed": False,
        "novelty_assessment_performed": False,
        "verifier_result_observed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    report = ProspectiveRoutedDecompositionPrimaryReport(
        **body,
        report_id=(
            "prospective_routed_decomposition_primary_report:"
            + digest[:20]
        ),
        report_sha256=digest,
    )
    return post_plan, post_gate, report


__all__ = [
    "DeterministicDecompositionAssessment",
    "ProspectiveRoutedDecompositionPrimaryReport",
    "RoutedDecompositionHypothesisResult",
    "assess_deterministic_decomposition",
    "execute_decomposition_primary",
    "rebuild_binding_plan_after_decomposition",
]
