from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    PreVerifierContractGateV2Report,
    PreVerifierContractGateV2Row,
)
from pipeline_core.discovery.prospective_routed_campaign_freeze import (
    ProspectiveRepairRegenerationRouterPolicy,
)
from pipeline_core.discovery.prospective_routed_execution_plan_v2 import (
    ProspectiveRoutedCaseExecutionPlanV2,
)
from pipeline_core.discovery.prospective_routed_route_compiler import (
    _alignment_candidates,
    _dominant_hint,
    _editable_fields_from_binding_reasons,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
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


RouteActionV2 = Literal[
    "PROCEED_TO_LITERAL_ENDPOINT_BINDING",
    "ZERO_DELTA_SPECIFICATION_REPAIR",
    "ZERO_DELTA_SOURCE_ALIGNMENT_THEN_REGENERATE_IF_UNAVAILABLE",
    "ATOMIC_DECOMPOSITION_THEN_REGENERATE_IF_UNAVAILABLE",
]


class SourceAlignmentCandidatePairV2(StrictModel):
    prediction_observation_id: str
    prediction_observable: str
    falsification_criterion_id: str
    falsifier_observable: str
    falsifying_outcome: str

    shared_observable_identity: Literal[True] = True
    source_surfaces_only: Literal[True] = True


class ClaimRouteWorkItemV2(StrictModel):
    claim_id: str
    router_hint: str
    route_action: RouteActionV2

    binding_reason_codes: list[str]
    source_reason_codes: list[str]

    specification_repair_editable_fields: list[str]
    source_alignment_candidates: list[SourceAlignmentCandidatePairV2]

    decomposition_source_claim_text: str | None = None
    decomposition_source_required_bridge: str | None = None
    decomposition_source_prediction: str | None = None
    decomposition_source_falsifier: str | None = None

    claim_text_mutation_allowed: Literal[False] = False
    source_alignment_new_text_allowed: Literal[False] = False
    novelty_or_verifier_outcome_used: Literal[False] = False


class RegenerationFallbackLineageV2(StrictModel):
    operation: Literal[
        "PROSPECTIVE_REGENERATION_UNIT_V2"
    ] = "PROSPECTIVE_REGENERATION_UNIT_V2"
    downstream_operation: Literal[
        "PROSPECTIVE_REGENERATION_DOWNSTREAM_V2"
    ] = "PROSPECTIVE_REGENERATION_DOWNSTREAM_V2"

    input_hypothesis_context_path: str
    lineage_dir: str
    unit_result_path: str
    regenerated_portfolio_path: str
    downstream_dir: str
    downstream_report_path: str

    structured_generation_calls_max: Literal[1] = 1
    repair_calls_max: Literal[0] = 0

    full_e2e_argv_present: Literal[False] = False
    full_e2e_rerun_allowed: Literal[False] = False
    previous_hypothesis_text_allowed: Literal[False] = False
    novelty_or_verifier_outcome_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_paths(self) -> "RegenerationFallbackLineageV2":
        values = [
            self.lineage_dir,
            self.unit_result_path,
            self.regenerated_portfolio_path,
            self.downstream_dir,
            self.downstream_report_path,
        ]
        if any("{final_hypothesis_id_slug}" in value for value in values):
            raise ValueError(
                "compiled regeneration lineage must resolve hypothesis slug"
            )
        return self


class HypothesisRouteDecisionV2(StrictModel):
    candidate_hypothesis_id: str
    final_hypothesis_id: str
    claim_ids: list[str]

    dominant_router_hint: str
    route_action: RouteActionV2
    router_hint_counts: dict[str, int]

    claim_work_items: list[ClaimRouteWorkItemV2]

    regeneration_fallback_allowed: bool
    regeneration_fallback: RegenerationFallbackLineageV2 | None = None

    later_case_outcome_used: Literal[False] = False
    novelty_outcome_used: Literal[False] = False
    verifier_outcome_used: Literal[False] = False
    endpoint_outcome_used: Literal[False] = False

    @model_validator(mode="after")
    def validate_decision(self) -> "HypothesisRouteDecisionV2":
        if self.claim_ids != [row.claim_id for row in self.claim_work_items]:
            raise ValueError("claim work-item order mismatch")

        if self.regeneration_fallback_allowed:
            if self.regeneration_fallback is None:
                raise ValueError(
                    "regeneration route requires frozen v2 lineage contract"
                )
        elif self.regeneration_fallback is not None:
            raise ValueError(
                "non-regeneration route cannot carry regeneration lineage"
            )
        return self


class ProspectiveRoutedDispatchReportV2(StrictModel):
    schema_version: Literal[
        "prospective-routed-dispatch-report-v2"
    ] = "prospective-routed-dispatch-report-v2"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    case_id: Literal["P16", "P17", "P18", "P19", "P20"]

    source_binding_plan_id: str
    source_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_gate_report_id: str
    source_gate_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_execution_plan_id: str
    source_execution_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_campaign_freeze_id: str
    source_campaign_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_regeneration_unit_freeze_id: str
    source_regeneration_unit_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_regeneration_downstream_freeze_id: str
    source_regeneration_downstream_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    hypotheses: list[HypothesisRouteDecisionV2]
    hypothesis_count: int = Field(ge=0)
    claim_count: int = Field(ge=0)
    route_action_counts: dict[str, int]

    route_compiled_before_endpoint_binding: Literal[True] = True
    route_compiled_before_repair_execution: Literal[True] = True
    route_compiled_before_regeneration_execution: Literal[True] = True

    regeneration_full_e2e_argv_generated: Literal[False] = False
    regeneration_unit_v2_authoritative: Literal[True] = True
    regeneration_downstream_v2_authoritative: Literal[True] = True

    repair_performed: Literal[False] = False
    source_alignment_performed: Literal[False] = False
    decomposition_performed: Literal[False] = False
    regeneration_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "ProspectiveRoutedDispatchReportV2":
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError("hypothesis_count mismatch")
        if self.claim_count != sum(
            len(row.claim_ids) for row in self.hypotheses
        ):
            raise ValueError("claim_count mismatch")

        counts = Counter(row.route_action for row in self.hypotheses)
        if dict(sorted(counts.items())) != dict(
            sorted(self.route_action_counts.items())
        ):
            raise ValueError("route_action_counts mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective routed dispatch-v2 SHA mismatch")
        if observed_id != (
            "prospective_routed_dispatch_report_v2:" + expected_sha[:20]
        ):
            raise ValueError("prospective routed dispatch-v2 ID mismatch")
        return self


def _slug(final_hypothesis_id: str) -> str:
    slug = final_hypothesis_id.replace(":", "_").replace("/", "_")
    if not slug or slug in {".", ".."}:
        raise ValueError("invalid final hypothesis ID for lineage slug")
    return slug


def _resolve_template(template: str, final_hypothesis_id: str) -> str:
    token = "{final_hypothesis_id_slug}"
    if token not in template:
        raise ValueError(
            "regeneration path template lacks final hypothesis slug token"
        )
    return template.replace(token, _slug(final_hypothesis_id))


def _fallback_lineage(
    *,
    case: ProspectiveRoutedCaseExecutionPlanV2,
    final_hypothesis_id: str,
) -> RegenerationFallbackLineageV2:
    return RegenerationFallbackLineageV2(
        input_hypothesis_context_path=case.hypothesis_context_path,
        lineage_dir=_resolve_template(
            case.regeneration_lineage_dir_template,
            final_hypothesis_id,
        ),
        unit_result_path=_resolve_template(
            case.regeneration_unit_result_path_template,
            final_hypothesis_id,
        ),
        regenerated_portfolio_path=_resolve_template(
            case.regeneration_portfolio_path_template,
            final_hypothesis_id,
        ),
        downstream_dir=_resolve_template(
            case.regeneration_downstream_dir_template,
            final_hypothesis_id,
        ),
        downstream_report_path=_resolve_template(
            case.regeneration_downstream_report_path_template,
            final_hypothesis_id,
        ),
    )


def compile_routed_dispatch_v2(
    *,
    case: ProspectiveRoutedCaseExecutionPlanV2,
    binding_plan: RelationalAtomicBindingPlan,
    gate_report: PreVerifierContractGateV2Report,
    router_policy: ProspectiveRepairRegenerationRouterPolicy,
    execution_plan_id: str,
    execution_plan_sha256: str,
    campaign_freeze_id: str,
    campaign_freeze_sha256: str,
    regeneration_unit_freeze_id: str,
    regeneration_unit_freeze_sha256: str,
    regeneration_downstream_freeze_id: str,
    regeneration_downstream_freeze_sha256: str,
) -> ProspectiveRoutedDispatchReportV2:
    if gate_report.source_binding_plan_id != binding_plan.plan_id:
        raise ValueError("gate/binding plan ID mismatch")
    if gate_report.source_binding_plan_sha256 != binding_plan.plan_sha256:
        raise ValueError("gate/binding plan SHA mismatch")

    rows_by_final: dict[str, list[PreVerifierContractGateV2Row]] = {}
    for row in gate_report.rows:
        rows_by_final.setdefault(row.final_hypothesis_id, []).append(row)

    plan_claim_ids = {
        claim.claim_id
        for hypothesis in binding_plan.hypotheses
        for claim in hypothesis.claims
    }
    gate_claim_ids = {row.claim_id for row in gate_report.rows}
    if plan_claim_ids != gate_claim_ids:
        raise ValueError(
            "gate-v2 claim population does not match binding plan"
        )

    hypotheses: list[HypothesisRouteDecisionV2] = []

    for hypothesis in binding_plan.hypotheses:
        gate_rows = rows_by_final.get(hypothesis.final_hypothesis_id, [])
        if len(gate_rows) != len(hypothesis.claims):
            raise ValueError(
                hypothesis.final_hypothesis_id
                + ": gate-v2 rows do not cover all claims"
            )

        by_claim = {row.claim_id: row for row in gate_rows}
        if len(by_claim) != len(gate_rows):
            raise ValueError("duplicate gate-v2 claim rows")

        dominant = _dominant_hint(gate_rows)
        route_action = router_policy.route_for_hint(dominant)

        work_items: list[ClaimRouteWorkItemV2] = []
        for claim in hypothesis.claims:
            gate_row = by_claim[claim.claim_id]
            claim_action = router_policy.route_for_hint(
                gate_row.router_hint
            )

            editable = (
                _editable_fields_from_binding_reasons(
                    gate_row.binding_contract_reason_codes
                )
                if gate_row.router_hint == "SPECIFICATION_REPAIR_REVIEW"
                else []
            )

            old_candidates = (
                _alignment_candidates(
                    hypothesis_plan=hypothesis,
                    claim_plan=claim,
                )
                if gate_row.router_hint
                == "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"
                else []
            )
            candidates = [
                SourceAlignmentCandidatePairV2(
                    **row.model_dump(mode="json")
                )
                for row in old_candidates
            ]

            work_items.append(
                ClaimRouteWorkItemV2(
                    claim_id=claim.claim_id,
                    router_hint=gate_row.router_hint,
                    route_action=claim_action,
                    binding_reason_codes=list(
                        gate_row.binding_contract_reason_codes
                    ),
                    source_reason_codes=list(
                        gate_row.source_contract_reason_codes
                    ),
                    specification_repair_editable_fields=editable,
                    source_alignment_candidates=candidates,
                    decomposition_source_claim_text=(
                        claim.claim_text
                        if gate_row.router_hint
                        == "DECOMPOSE_OR_REGENERATE_REVIEW"
                        else None
                    ),
                    decomposition_source_required_bridge=(
                        claim.required_bridge
                        if gate_row.router_hint
                        == "DECOMPOSE_OR_REGENERATE_REVIEW"
                        else None
                    ),
                    decomposition_source_prediction=(
                        claim.predicted_observation
                        if gate_row.router_hint
                        == "DECOMPOSE_OR_REGENERATE_REVIEW"
                        else None
                    ),
                    decomposition_source_falsifier=(
                        claim.falsification_condition
                        if gate_row.router_hint
                        == "DECOMPOSE_OR_REGENERATE_REVIEW"
                        else None
                    ),
                )
            )

        regen_allowed = route_action in {
            "ZERO_DELTA_SOURCE_ALIGNMENT_THEN_REGENERATE_IF_UNAVAILABLE",
            "ATOMIC_DECOMPOSITION_THEN_REGENERATE_IF_UNAVAILABLE",
        }

        hypotheses.append(
            HypothesisRouteDecisionV2(
                candidate_hypothesis_id=hypothesis.candidate_hypothesis_id,
                final_hypothesis_id=hypothesis.final_hypothesis_id,
                claim_ids=[row.claim_id for row in hypothesis.claims],
                dominant_router_hint=dominant,
                route_action=route_action,
                router_hint_counts=dict(
                    sorted(
                        Counter(
                            row.router_hint for row in gate_rows
                        ).items()
                    )
                ),
                claim_work_items=work_items,
                regeneration_fallback_allowed=regen_allowed,
                regeneration_fallback=(
                    _fallback_lineage(
                        case=case,
                        final_hypothesis_id=hypothesis.final_hypothesis_id,
                    )
                    if regen_allowed
                    else None
                ),
            )
        )

    counts = Counter(row.route_action for row in hypotheses)
    body = {
        "schema_version": "prospective-routed-dispatch-report-v2",
        "case_id": case.case_id,
        "source_binding_plan_id": binding_plan.plan_id,
        "source_binding_plan_sha256": binding_plan.plan_sha256,
        "source_gate_report_id": gate_report.report_id,
        "source_gate_report_sha256": gate_report.report_sha256,
        "source_execution_plan_id": execution_plan_id,
        "source_execution_plan_sha256": execution_plan_sha256,
        "source_campaign_freeze_id": campaign_freeze_id,
        "source_campaign_freeze_sha256": campaign_freeze_sha256,
        "source_regeneration_unit_freeze_id": regeneration_unit_freeze_id,
        "source_regeneration_unit_freeze_sha256":
            regeneration_unit_freeze_sha256,
        "source_regeneration_downstream_freeze_id":
            regeneration_downstream_freeze_id,
        "source_regeneration_downstream_freeze_sha256":
            regeneration_downstream_freeze_sha256,
        "hypotheses": [row.model_dump(mode="json") for row in hypotheses],
        "hypothesis_count": len(hypotheses),
        "claim_count": sum(len(row.claim_ids) for row in hypotheses),
        "route_action_counts": dict(sorted(counts.items())),
        "route_compiled_before_endpoint_binding": True,
        "route_compiled_before_repair_execution": True,
        "route_compiled_before_regeneration_execution": True,
        "regeneration_full_e2e_argv_generated": False,
        "regeneration_unit_v2_authoritative": True,
        "regeneration_downstream_v2_authoritative": True,
        "repair_performed": False,
        "source_alignment_performed": False,
        "decomposition_performed": False,
        "regeneration_performed": False,
        "endpoint_binding_performed": False,
        "novelty_assessment_performed": False,
        "verifier_result_observed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveRoutedDispatchReportV2(
        **body,
        report_id=(
            "prospective_routed_dispatch_report_v2:" + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "ClaimRouteWorkItemV2",
    "HypothesisRouteDecisionV2",
    "ProspectiveRoutedDispatchReportV2",
    "RegenerationFallbackLineageV2",
    "RouteActionV2",
    "SourceAlignmentCandidatePairV2",
    "compile_routed_dispatch_v2",
]
