from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    PreVerifierContractGateV2Report,
    PreVerifierContractGateV2Row,
)
from pipeline_core.discovery.prospective_routed_campaign_freeze import (
    ProspectiveRepairRegenerationRouterPolicy,
)
from pipeline_core.discovery.prospective_routed_execution_plan import (
    ProspectiveRoutedCaseExecutionPlan,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_projection import (
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


RouteAction = Literal[
    "PROCEED_TO_LITERAL_ENDPOINT_BINDING",
    "ZERO_DELTA_SPECIFICATION_REPAIR",
    "ZERO_DELTA_SOURCE_ALIGNMENT_THEN_REGENERATE_IF_UNAVAILABLE",
    "ATOMIC_DECOMPOSITION_THEN_REGENERATE_IF_UNAVAILABLE",
]


_HINT_PRECEDENCE = {
    "PROCEED_TO_LITERAL_ENDPOINT_BINDING": 0,
    "SPECIFICATION_REPAIR_REVIEW": 1,
    "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW": 2,
    "DECOMPOSE_OR_REGENERATE_REVIEW": 3,
}


class SourceAlignmentCandidatePair(StrictModel):
    prediction_observation_id: str
    prediction_observable: str
    falsification_criterion_id: str
    falsifier_observable: str
    falsifying_outcome: str

    shared_observable_identity: Literal[True] = True
    source_surfaces_only: Literal[True] = True


class ClaimRouteWorkItem(StrictModel):
    claim_id: str
    router_hint: str
    route_action: RouteAction

    binding_reason_codes: list[str]
    source_reason_codes: list[str]

    specification_repair_editable_fields: list[str]
    source_alignment_candidates: list[SourceAlignmentCandidatePair]

    decomposition_source_claim_text: str | None = None
    decomposition_source_required_bridge: str | None = None
    decomposition_source_prediction: str | None = None
    decomposition_source_falsifier: str | None = None

    claim_text_mutation_allowed: Literal[False] = False
    source_alignment_new_text_allowed: Literal[False] = False
    novelty_or_verifier_outcome_used: Literal[False] = False


class HypothesisRouteDecision(StrictModel):
    candidate_hypothesis_id: str
    final_hypothesis_id: str
    claim_ids: list[str]

    dominant_router_hint: str
    route_action: RouteAction
    router_hint_counts: dict[str, int]

    claim_work_items: list[ClaimRouteWorkItem]

    regeneration_fallback_allowed: bool
    regeneration_argv: list[str] | None = None
    regeneration_run_dir: str | None = None

    later_case_outcome_used: Literal[False] = False
    novelty_outcome_used: Literal[False] = False
    verifier_outcome_used: Literal[False] = False
    endpoint_outcome_used: Literal[False] = False

    @model_validator(mode="after")
    def validate_decision(self) -> "HypothesisRouteDecision":
        if self.claim_ids != [row.claim_id for row in self.claim_work_items]:
            raise ValueError("claim work-item order mismatch")
        if self.regeneration_fallback_allowed:
            if not self.regeneration_argv or not self.regeneration_run_dir:
                raise ValueError(
                    "regeneration fallback requires frozen argv and run dir"
                )
        else:
            if self.regeneration_argv is not None:
                raise ValueError(
                    "non-regeneration route cannot carry regeneration argv"
                )
            if self.regeneration_run_dir is not None:
                raise ValueError(
                    "non-regeneration route cannot carry regeneration run dir"
                )
        return self


class ProspectiveRoutedDispatchReport(StrictModel):
    schema_version: Literal[
        "prospective-routed-dispatch-report-v1"
    ] = "prospective-routed-dispatch-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    case_id: Literal["P11", "P12", "P13", "P14", "P15"]

    source_binding_plan_id: str
    source_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_gate_report_id: str
    source_gate_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    hypotheses: list[HypothesisRouteDecision]
    hypothesis_count: int = Field(ge=0)
    claim_count: int = Field(ge=0)
    route_action_counts: dict[str, int]

    route_compiled_before_endpoint_binding: Literal[True] = True
    route_compiled_before_repair_execution: Literal[True] = True
    route_compiled_before_regeneration_execution: Literal[True] = True

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
    def validate_report(self) -> "ProspectiveRoutedDispatchReport":
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
            raise ValueError("prospective routed dispatch SHA mismatch")
        if observed_id != (
            "prospective_routed_dispatch_report:" + expected_sha[:20]
        ):
            raise ValueError("prospective routed dispatch ID mismatch")
        return self


def _dominant_hint(rows: list[PreVerifierContractGateV2Row]) -> str:
    if not rows:
        raise ValueError("cannot route hypothesis without gate rows")
    return max(
        (row.router_hint for row in rows),
        key=lambda value: _HINT_PRECEDENCE[value],
    )


def _editable_fields_from_binding_reasons(
    reasons: list[str],
) -> list[str]:
    fields: list[str] = []

    mapping = {
        "missing_required_bridge": "required_bridge",
        "missing_predicted_observation": "predicted_observation",
        "missing_falsification_condition": "falsification_condition",
    }
    for reason in reasons:
        field = mapping.get(reason)
        if field and field not in fields:
            fields.append(field)

        if reason.startswith("identity_not_literal_in_required_bridge:"):
            if "required_bridge" not in fields:
                fields.append("required_bridge")
        elif reason.startswith("identity_not_literal_in_predicted_observation:"):
            if "predicted_observation" not in fields:
                fields.append("predicted_observation")
        elif reason.startswith(
            "identity_not_literal_in_falsification_condition:"
        ):
            if "falsification_condition" not in fields:
                fields.append("falsification_condition")

    return fields


def _alignment_candidates(
    *,
    hypothesis_plan: object,
    claim_plan: object,
) -> list[SourceAlignmentCandidatePair]:
    candidate_card, _source_claim = _candidate_card_and_claim(
        hypothesis_plan=hypothesis_plan,
        claim_plan=claim_plan,
    )

    pairs: list[SourceAlignmentCandidatePair] = []
    seen: set[tuple[str, str]] = set()

    for prediction in candidate_card.predicted_observations:
        left = " ".join(
            str(prediction.observable or "").split()
        ).casefold()
        if not left:
            continue

        for falsifier in candidate_card.falsification_criteria:
            right = " ".join(
                str(falsifier.observable or "").split()
            ).casefold()
            if not right or left != right:
                continue

            key = (
                str(prediction.observation_id),
                str(falsifier.criterion_id),
            )
            if key in seen:
                continue
            seen.add(key)

            pairs.append(
                SourceAlignmentCandidatePair(
                    prediction_observation_id=prediction.observation_id,
                    prediction_observable=prediction.observable,
                    falsification_criterion_id=falsifier.criterion_id,
                    falsifier_observable=falsifier.observable,
                    falsifying_outcome=falsifier.falsifying_outcome,
                )
            )

    return pairs


def _replace_run_dir(
    argv: list[str],
    *,
    new_run_dir: str,
) -> list[str]:
    output = list(argv)
    if "--overwrite-run" in output:
        raise ValueError(
            "frozen routed regeneration must not use --overwrite-run"
        )
    try:
        index = output.index("--run-dir")
    except ValueError as exc:
        raise ValueError("main E2E argv lacks --run-dir") from exc
    if index + 1 >= len(output):
        raise ValueError("main E2E argv lacks run-dir value")
    output[index + 1] = new_run_dir
    return output


def compile_routed_dispatch(
    *,
    case: ProspectiveRoutedCaseExecutionPlan,
    binding_plan: RelationalAtomicBindingPlan,
    gate_report: PreVerifierContractGateV2Report,
    router_policy: ProspectiveRepairRegenerationRouterPolicy,
) -> ProspectiveRoutedDispatchReport:
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

    hypotheses: list[HypothesisRouteDecision] = []

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

        work_items: list[ClaimRouteWorkItem] = []
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
            candidates = (
                _alignment_candidates(
                    hypothesis_plan=hypothesis,
                    claim_plan=claim,
                )
                if gate_row.router_hint
                == "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"
                else []
            )

            work_items.append(
                ClaimRouteWorkItem(
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

        regen_dir = None
        regen_argv = None
        if regen_allowed:
            regen_dir = str(
                Path(case.routed_lineage_dir)
                / hypothesis.final_hypothesis_id.replace(":", "_")
                / "fresh_regeneration"
            )
            regen_argv = _replace_run_dir(
                case.main_e2e_argv,
                new_run_dir=regen_dir,
            )

        hypotheses.append(
            HypothesisRouteDecision(
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
                regeneration_argv=regen_argv,
                regeneration_run_dir=regen_dir,
            )
        )

    counts = Counter(row.route_action for row in hypotheses)
    body = {
        "schema_version": "prospective-routed-dispatch-report-v1",
        "case_id": case.case_id,
        "source_binding_plan_id": binding_plan.plan_id,
        "source_binding_plan_sha256": binding_plan.plan_sha256,
        "source_gate_report_id": gate_report.report_id,
        "source_gate_report_sha256": gate_report.report_sha256,
        "hypotheses": [row.model_dump(mode="json") for row in hypotheses],
        "hypothesis_count": len(hypotheses),
        "claim_count": sum(len(row.claim_ids) for row in hypotheses),
        "route_action_counts": dict(sorted(counts.items())),
        "route_compiled_before_endpoint_binding": True,
        "route_compiled_before_repair_execution": True,
        "route_compiled_before_regeneration_execution": True,
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
    return ProspectiveRoutedDispatchReport(
        **body,
        report_id="prospective_routed_dispatch_report:" + digest[:20],
        report_sha256=digest,
    )


__all__ = [
    "ClaimRouteWorkItem",
    "HypothesisRouteDecision",
    "ProspectiveRoutedDispatchReport",
    "RouteAction",
    "SourceAlignmentCandidatePair",
    "compile_routed_dispatch",
]
