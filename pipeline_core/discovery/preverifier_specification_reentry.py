from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.preverifier_specification_repair_executor import (
    SpecificationRepairCaseResult,
)
from pipeline_core.discovery.preverifier_specification_repair_policy import (
    SpecificationCaseRepairPlan,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingClaimPlan,
    RelationalAtomicBindingHypothesisPlan,
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


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _surface_contains(haystack: str, needle: str) -> bool:
    left = _normalize(haystack)
    right = _normalize(needle)
    return bool(right) and right in left


class RepairedClaimLineage(StrictModel):
    claim_id: str
    repair_action: str
    original_source_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repaired_source_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_claim_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    changed_fields: list[str]
    original_binding_status: str
    repaired_binding_status: str
    repaired_reason_codes: list[str]

    repair_status_required: Literal["MATERIALIZED_R1"] = "MATERIALIZED_R1"
    zero_scientific_delta_required: Literal[True] = True
    second_repair_attempt_performed: Literal[False] = False


class RepairedBindingPlanMaterialization(StrictModel):
    schema_version: Literal[
        "preverifier-repaired-binding-plan-materialization-v1"
    ] = "preverifier-repaired-binding-plan-materialization-v1"

    materialization_id: str
    materialization_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    case_id: Literal["P06", "P07", "P08", "P09", "P10"]
    source_binding_plan_id: str
    source_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repaired_binding_plan_id: str
    repaired_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_repair_execution_report_id: str

    claim_lineages: list[RepairedClaimLineage]
    applied_repair_count: int = Field(ge=0)

    original_plan_preserved: Literal[True] = True
    rejected_repairs_applied: Literal[False] = False
    second_repair_attempt_performed: Literal[False] = False
    literature_retrieval_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_materialization(
        self,
    ) -> "RepairedBindingPlanMaterialization":
        if self.applied_repair_count != len(self.claim_lineages):
            raise ValueError("applied_repair_count mismatch")
        body = self.model_dump(mode="json")
        observed_id = body.pop("materialization_id")
        observed_sha = body.pop("materialization_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("repaired binding materialization SHA mismatch")
        if observed_id != (
            "preverifier_repaired_binding_materialization:"
            + expected_sha[:20]
        ):
            raise ValueError("repaired binding materialization ID mismatch")
        return self


def _readiness_reasons(payload: dict) -> list[str]:
    reasons: list[str] = []

    if not str(payload.get("claim_text") or "").strip():
        reasons.append("missing_claim_text")
    if not str(payload.get("required_bridge") or "").strip():
        reasons.append("missing_required_bridge")
    if not str(payload.get("predicted_observation") or "").strip():
        reasons.append("missing_predicted_observation")
    if not str(payload.get("falsification_condition") or "").strip():
        reasons.append("missing_falsification_condition")

    identities = [
        str(value)
        for value in payload.get("prior_art_identity_terms", [])
        if str(value).strip()
    ]
    if not identities:
        reasons.append("missing_prior_art_identity_terms")
    if payload.get("novelty_selection_role") is None:
        reasons.append("missing_novelty_selection_role")

    for index, identity in enumerate(identities):
        for label in (
            "claim_text",
            "required_bridge",
            "predicted_observation",
            "falsification_condition",
        ):
            text = str(payload.get(label) or "")
            if text.strip() and not _surface_contains(text, identity):
                reasons.append(
                    f"identity_not_literal_in_{label}:{index}"
                )

    return list(dict.fromkeys(reasons))


def _repaired_claim_sha(payload: dict) -> str:
    source = dict(payload)
    source.pop("binding_status", None)
    source.pop("reason_codes", None)
    source.pop("source_claim_sha256", None)
    return _sha256_json(source)


def _rebuild_hypothesis(
    hypothesis: RelationalAtomicBindingHypothesisPlan,
    claims: list[RelationalAtomicBindingClaimPlan],
) -> RelationalAtomicBindingHypothesisPlan:
    ready = sum(
        row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
        for row in claims
    )
    novelty_ready = sum(
        row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
        and row.novelty_selection_role == "NOVELTY_BEARING"
        for row in claims
    )
    if ready == 0:
        status = "NO_BINDABLE_CLAIMS"
    elif novelty_ready == 0:
        status = "NO_NOVELTY_BEARING_BINDABLE_CLAIM"
    else:
        status = "READY_FOR_LITERAL_ENDPOINT_BINDING"

    payload = hypothesis.model_dump(mode="json")
    payload.update(
        {
            "claims": [row.model_dump(mode="json") for row in claims],
            "claim_count": len(claims),
            "binding_ready_claim_count": ready,
            "novelty_bearing_binding_ready_claim_count": novelty_ready,
            "binding_status": status,
        }
    )
    return RelationalAtomicBindingHypothesisPlan.model_validate(payload)


def _rebuild_plan(
    *,
    source: RelationalAtomicBindingPlan,
    hypotheses: list[RelationalAtomicBindingHypothesisPlan],
) -> RelationalAtomicBindingPlan:
    h_counts = Counter(row.binding_status for row in hypotheses)
    c_counts = Counter(
        claim.binding_status
        for row in hypotheses
        for claim in row.claims
    )
    body = {
        "schema_version": "relational-atomic-binding-plan-v1",
        "run_dir": source.run_dir,
        "source_alpha6_candidate_portfolio":
            source.source_alpha6_candidate_portfolio,
        "source_alpha6_candidate_portfolio_sha256":
            source.source_alpha6_candidate_portfolio_sha256,
        "source_certification_report":
            source.source_certification_report,
        "source_certification_report_sha256":
            source.source_certification_report_sha256,
        "hypotheses": [
            row.model_dump(mode="json")
            for row in hypotheses
        ],
        "hypothesis_count": len(hypotheses),
        "ready_hypothesis_count": sum(
            row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            for row in hypotheses
        ),
        "not_ready_hypothesis_count": sum(
            row.binding_status != "READY_FOR_LITERAL_ENDPOINT_BINDING"
            for row in hypotheses
        ),
        "claim_count": sum(row.claim_count for row in hypotheses),
        "binding_ready_claim_count": sum(
            row.binding_ready_claim_count
            for row in hypotheses
        ),
        "novelty_bearing_binding_ready_claim_count": sum(
            row.novelty_bearing_binding_ready_claim_count
            for row in hypotheses
        ),
        "hypothesis_status_counts": dict(sorted(h_counts.items())),
        "claim_status_counts": dict(sorted(c_counts.items())),
        "certification_only_source_required": True,
        "candidate_final_authority_equivalence_required": True,
        "endpoint_binding_performed": False,
        "verifier_result_observed": False,
        "production_selection_authority": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return RelationalAtomicBindingPlan(
        **body,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )


def materialize_repaired_binding_plan(
    *,
    source_plan: RelationalAtomicBindingPlan,
    case_plan: SpecificationCaseRepairPlan,
    case_result: SpecificationRepairCaseResult,
    repair_execution_report_id: str,
) -> tuple[
    RelationalAtomicBindingPlan,
    RepairedBindingPlanMaterialization,
]:
    if case_plan.case_id != case_result.case_id:
        raise ValueError("repair case plan/result case ID mismatch")
    if case_plan.source_case_result_id != case_result.source_case_result_id:
        raise ValueError("repair case source-result ID mismatch")
    if (
        case_plan.source_case_result_sha256
        != case_result.source_case_result_sha256
    ):
        raise ValueError("repair case source-result SHA mismatch")
    if case_plan.status != "PLANNED_AUTOMATIC_REPAIR":
        raise ValueError("cannot materialize excluded repair case")

    planned_by_id = {
        row.claim_id: row
        for row in case_plan.claim_repairs
    }
    result_by_id = {
        row.claim_id: row
        for row in case_result.claim_results
    }
    if set(planned_by_id) != set(result_by_id):
        raise ValueError("repair plan/result claim population mismatch")

    accepted = {
        claim_id: row
        for claim_id, row in result_by_id.items()
        if row.status == "MATERIALIZED_R1"
    }
    if not accepted:
        raise ValueError("case has no MATERIALIZED_R1 claims")

    source_claim_ids = {
        claim.claim_id
        for hypothesis in source_plan.hypotheses
        for claim in hypothesis.claims
    }
    missing = sorted(set(accepted) - source_claim_ids)
    if missing:
        raise ValueError(
            "materialized repair claims absent from source plan: "
            + repr(missing)
        )

    lineages: list[RepairedClaimLineage] = []
    hypotheses: list[RelationalAtomicBindingHypothesisPlan] = []

    for hypothesis in source_plan.hypotheses:
        rebuilt_claims: list[RelationalAtomicBindingClaimPlan] = []
        for claim in hypothesis.claims:
            result = accepted.get(claim.claim_id)
            if result is None:
                rebuilt_claims.append(claim)
                continue

            plan = planned_by_id[claim.claim_id]
            if (
                result.source_claim_snapshot_sha256
                != plan.source_claim_snapshot_sha256
            ):
                raise ValueError(
                    claim.claim_id + ": repair snapshot SHA mismatch"
                )
            if result.semantic_audit_passed is not True:
                raise ValueError(
                    claim.claim_id
                    + ": materialized result lacks passing semantic audit"
                )

            payload = claim.model_dump(mode="json")
            source_fields = {
                "required_bridge": claim.required_bridge,
                "predicted_observation": claim.predicted_observation,
                "falsification_condition": claim.falsification_condition,
            }
            repaired_fields = {
                "required_bridge":
                    str(result.accepted_required_bridge or ""),
                "predicted_observation":
                    str(result.accepted_predicted_observation or ""),
                "falsification_condition":
                    str(result.accepted_falsification_condition or ""),
            }

            for field in (
                "required_bridge",
                "predicted_observation",
                "falsification_condition",
            ):
                if field in plan.editable_fields:
                    payload[field] = repaired_fields[field]
                elif repaired_fields[field] != source_fields[field]:
                    raise ValueError(
                        claim.claim_id
                        + ": execution result mutated non-editable field "
                        + field
                    )

            reasons = _readiness_reasons(payload)
            payload["binding_status"] = (
                "READY_FOR_LITERAL_ENDPOINT_BINDING"
                if not reasons
                else "INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION"
            )
            payload["reason_codes"] = reasons
            repaired_sha = _repaired_claim_sha(payload)
            payload["source_claim_sha256"] = repaired_sha

            repaired_claim = RelationalAtomicBindingClaimPlan.model_validate(
                payload
            )
            changed_fields = [
                field
                for field in plan.editable_fields
                if source_fields[field] != repaired_fields[field]
            ]
            lineages.append(
                RepairedClaimLineage(
                    claim_id=claim.claim_id,
                    repair_action=result.repair_action,
                    original_source_claim_sha256=claim.source_claim_sha256,
                    repaired_source_claim_sha256=repaired_sha,
                    source_claim_snapshot_sha256=(
                        result.source_claim_snapshot_sha256
                    ),
                    changed_fields=changed_fields,
                    original_binding_status=claim.binding_status,
                    repaired_binding_status=repaired_claim.binding_status,
                    repaired_reason_codes=list(repaired_claim.reason_codes),
                )
            )
            rebuilt_claims.append(repaired_claim)

        hypotheses.append(
            _rebuild_hypothesis(
                hypothesis,
                rebuilt_claims,
            )
        )

    repaired_plan = _rebuild_plan(
        source=source_plan,
        hypotheses=hypotheses,
    )
    body = {
        "schema_version":
            "preverifier-repaired-binding-plan-materialization-v1",
        "case_id": case_plan.case_id,
        "source_binding_plan_id": source_plan.plan_id,
        "source_binding_plan_sha256": source_plan.plan_sha256,
        "repaired_binding_plan_id": repaired_plan.plan_id,
        "repaired_binding_plan_sha256": repaired_plan.plan_sha256,
        "source_repair_execution_report_id":
            repair_execution_report_id,
        "claim_lineages": [
            row.model_dump(mode="json")
            for row in lineages
        ],
        "applied_repair_count": len(lineages),
        "original_plan_preserved": True,
        "rejected_repairs_applied": False,
        "second_repair_attempt_performed": False,
        "literature_retrieval_performed": False,
        "verifier_result_observed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    materialization = RepairedBindingPlanMaterialization(
        **body,
        materialization_id=(
            "preverifier_repaired_binding_materialization:"
            + digest[:20]
        ),
        materialization_sha256=digest,
    )
    return repaired_plan, materialization


__all__ = [
    "RepairedBindingPlanMaterialization",
    "RepairedClaimLineage",
    "materialize_repaired_binding_plan",
]
