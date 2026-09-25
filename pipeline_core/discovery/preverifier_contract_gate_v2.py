from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingClaimPlan,
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_projection import (
    _ATOMIC_KINDS,
    _candidate_card_and_claim,
    _exact_observation_binding,
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


GateStatus = Literal[
    "READY_FOR_LITERAL_ENDPOINT_BINDING",
    "NOT_READY_FOR_LITERAL_ENDPOINT_BINDING",
]

RouterHint = Literal[
    "PROCEED_TO_LITERAL_ENDPOINT_BINDING",
    "SPECIFICATION_REPAIR_REVIEW",
    "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW",
    "DECOMPOSE_OR_REGENERATE_REVIEW",
]


_BINDING_SPECIFICATION_REASON_CODES = frozenset(
    {
        "missing_claim_text",
        "missing_required_bridge",
        "missing_predicted_observation",
        "missing_falsification_condition",
        "missing_prior_art_identity_terms",
        "missing_novelty_selection_role",
    }
)
_BINDING_SPECIFICATION_REASON_PREFIXES = (
    "identity_not_literal_in_claim_text:",
    "identity_not_literal_in_required_bridge:",
    "identity_not_literal_in_predicted_observation:",
    "identity_not_literal_in_falsification_condition:",
)

_SOURCE_DECOMPOSITION_REASON_PREFIXES = (
    "unsupported_atomic_claim_kind:",
)
_SOURCE_ALIGNMENT_REASON_PREFIXES = (
    "prediction_exact_source_binding_cardinality:",
    "falsifier_exact_source_binding_cardinality:",
)
_SOURCE_ALIGNMENT_REASON_CODES = frozenset(
    {
        "prediction_falsifier_observable_identity_mismatch",
        "observable_empty",
    }
)
_SOURCE_SPECIFICATION_REASON_CODES = frozenset(
    {
        "missing_novelty_selection_role",
    }
)


def _reason_known(
    reason: str,
    *,
    exact: frozenset[str],
    prefixes: tuple[str, ...],
) -> bool:
    return reason in exact or reason.startswith(prefixes)


class PreVerifierContractGateV2Row(StrictModel):
    candidate_hypothesis_id: str
    final_hypothesis_id: str
    claim_id: str
    novelty_selection_role: str | None
    claim_kind: str

    binding_plan_status: str
    binding_contract_reason_codes: list[str]
    source_contract_reason_codes: list[str]

    gate_status: GateStatus
    router_hint: RouterHint

    atomic_kind_supported: bool
    prediction_exact_source_binding_count: int = Field(ge=0)
    falsifier_exact_source_binding_count: int = Field(ge=0)
    shared_observable_identity_satisfied: bool | None

    endpoint_binding_performed: Literal[False] = False
    repair_performed: Literal[False] = False
    regeneration_performed: Literal[False] = False
    literature_retrieval_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_contract_route(self) -> "PreVerifierContractGateV2Row":
        has_contract_failure = bool(
            self.binding_contract_reason_codes
            or self.source_contract_reason_codes
        )
        ready = self.gate_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
        proceed = self.router_hint == "PROCEED_TO_LITERAL_ENDPOINT_BINDING"

        if ready == has_contract_failure:
            raise ValueError(
                "gate status must be ready iff contract reason codes are empty"
            )
        if proceed != ready:
            raise ValueError(
                "router may proceed iff claim is contract-ready"
            )
        return self


class PreVerifierContractGateV2Report(StrictModel):
    schema_version: Literal[
        "preverifier-contract-gate-v2"
    ] = "preverifier-contract-gate-v2"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_binding_plan_id: str
    source_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    rows: list[PreVerifierContractGateV2Row]
    claim_count: int = Field(ge=0)
    ready_claim_count: int = Field(ge=0)
    not_ready_claim_count: int = Field(ge=0)
    novelty_bearing_ready_claim_count: int = Field(ge=0)

    gate_status_counts: dict[str, int]
    router_hint_counts: dict[str, int]

    complete_source_contract_checked_before_endpoint_binding: Literal[
        True
    ] = True
    binding_contract_checked: Literal[True] = True
    atomic_kind_checked: Literal[True] = True
    exact_prediction_source_binding_checked: Literal[True] = True
    exact_falsifier_source_binding_checked: Literal[True] = True
    shared_observable_identity_checked: Literal[True] = True

    endpoint_binding_performed: Literal[False] = False
    repair_performed: Literal[False] = False
    regeneration_performed: Literal[False] = False
    literature_retrieval_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreVerifierContractGateV2Report":
        if self.claim_count != len(self.rows):
            raise ValueError("claim_count mismatch")
        if self.ready_claim_count != sum(
            row.gate_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            for row in self.rows
        ):
            raise ValueError("ready_claim_count mismatch")
        if self.not_ready_claim_count != sum(
            row.gate_status == "NOT_READY_FOR_LITERAL_ENDPOINT_BINDING"
            for row in self.rows
        ):
            raise ValueError("not_ready_claim_count mismatch")
        if self.novelty_bearing_ready_claim_count != sum(
            row.gate_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            and row.novelty_selection_role == "NOVELTY_BEARING"
            for row in self.rows
        ):
            raise ValueError("novelty_bearing_ready_claim_count mismatch")

        gate_counts = Counter(row.gate_status for row in self.rows)
        if dict(sorted(gate_counts.items())) != dict(
            sorted(self.gate_status_counts.items())
        ):
            raise ValueError("gate_status_counts mismatch")

        router_counts = Counter(row.router_hint for row in self.rows)
        if dict(sorted(router_counts.items())) != dict(
            sorted(self.router_hint_counts.items())
        ):
            raise ValueError("router_hint_counts mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("pre-verifier contract gate v2 SHA mismatch")
        if observed_id != (
            "preverifier_contract_gate_v2:" + expected_sha[:20]
        ):
            raise ValueError("pre-verifier contract gate v2 ID mismatch")
        return self


def _count_exact_observation_bindings(
    *,
    candidate_card: object,
    claim: object,
) -> tuple[int, int, bool | None]:
    prediction_key = " ".join(
        str(claim.predicted_observation or "").split()
    ).casefold()
    falsifier_key = " ".join(
        str(claim.falsification_condition or "").split()
    ).casefold()

    predictions = [
        row
        for row in candidate_card.predicted_observations
        if " ".join(str(row.observable or "").split()).casefold()
        == prediction_key
    ]
    falsifiers = [
        row
        for row in candidate_card.falsification_criteria
        if " ".join(str(row.falsifying_outcome or "").split()).casefold()
        == falsifier_key
    ]

    if len(predictions) != 1 or len(falsifiers) != 1:
        return len(predictions), len(falsifiers), None

    left = " ".join(
        str(predictions[0].observable or "").split()
    ).casefold()
    right = " ".join(
        str(falsifiers[0].observable or "").split()
    ).casefold()
    return len(predictions), len(falsifiers), bool(left and left == right)


def classify_router_hint(
    *,
    binding_reason_codes: list[str],
    source_reason_codes: list[str],
) -> RouterHint:
    unknown_binding = sorted(
        {
            reason
            for reason in binding_reason_codes
            if not _reason_known(
                reason,
                exact=_BINDING_SPECIFICATION_REASON_CODES,
                prefixes=_BINDING_SPECIFICATION_REASON_PREFIXES,
            )
        }
    )
    if unknown_binding:
        raise ValueError(
            "unclassified binding-contract reason codes: "
            + repr(unknown_binding)
        )

    source_decomposition = [
        reason
        for reason in source_reason_codes
        if reason.startswith(_SOURCE_DECOMPOSITION_REASON_PREFIXES)
    ]
    source_alignment = [
        reason
        for reason in source_reason_codes
        if (
            reason in _SOURCE_ALIGNMENT_REASON_CODES
            or reason.startswith(_SOURCE_ALIGNMENT_REASON_PREFIXES)
        )
    ]
    source_specification = [
        reason
        for reason in source_reason_codes
        if reason in _SOURCE_SPECIFICATION_REASON_CODES
    ]
    known_source = set(
        source_decomposition
        + source_alignment
        + source_specification
    )
    unknown_source = sorted(
        set(source_reason_codes) - known_source
    )
    if unknown_source:
        raise ValueError(
            "unclassified source-contract reason codes: "
            + repr(unknown_source)
        )

    if source_decomposition:
        return "DECOMPOSE_OR_REGENERATE_REVIEW"

    if source_alignment:
        return "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"

    if binding_reason_codes or source_specification:
        return "SPECIFICATION_REPAIR_REVIEW"

    return "PROCEED_TO_LITERAL_ENDPOINT_BINDING"


def assess_preverifier_claim_contract(
    *,
    hypothesis_plan: object,
    claim_plan: RelationalAtomicBindingClaimPlan,
) -> PreVerifierContractGateV2Row:
    candidate_card, source_claim = _candidate_card_and_claim(
        hypothesis_plan=hypothesis_plan,
        claim_plan=claim_plan,
    )

    source_reasons: list[str] = []
    if source_claim.kind not in _ATOMIC_KINDS:
        source_reasons.append(
            "unsupported_atomic_claim_kind:" + source_claim.kind
        )
    if source_claim.novelty_selection_role is None:
        source_reasons.append("missing_novelty_selection_role")

    _, _, observation_reasons = _exact_observation_binding(
        candidate_card=candidate_card,
        claim=source_claim,
    )
    source_reasons.extend(observation_reasons)
    source_reasons = list(dict.fromkeys(source_reasons))

    prediction_count, falsifier_count, shared_identity = (
        _count_exact_observation_bindings(
            candidate_card=candidate_card,
            claim=source_claim,
        )
    )

    binding_reasons = list(claim_plan.reason_codes)
    gate_ready = not binding_reasons and not source_reasons
    router_hint = classify_router_hint(
        binding_reason_codes=binding_reasons,
        source_reason_codes=source_reasons,
    )

    return PreVerifierContractGateV2Row(
        candidate_hypothesis_id=claim_plan.candidate_hypothesis_id,
        final_hypothesis_id=claim_plan.final_hypothesis_id,
        claim_id=claim_plan.claim_id,
        novelty_selection_role=claim_plan.novelty_selection_role,
        claim_kind=claim_plan.kind,
        binding_plan_status=claim_plan.binding_status,
        binding_contract_reason_codes=binding_reasons,
        source_contract_reason_codes=source_reasons,
        gate_status=(
            "READY_FOR_LITERAL_ENDPOINT_BINDING"
            if gate_ready
            else "NOT_READY_FOR_LITERAL_ENDPOINT_BINDING"
        ),
        router_hint=router_hint,
        atomic_kind_supported=(source_claim.kind in _ATOMIC_KINDS),
        prediction_exact_source_binding_count=prediction_count,
        falsifier_exact_source_binding_count=falsifier_count,
        shared_observable_identity_satisfied=shared_identity,
    )


def build_preverifier_contract_gate_v2(
    *,
    plan: RelationalAtomicBindingPlan,
) -> PreVerifierContractGateV2Report:
    hypothesis_by_final = {
        row.final_hypothesis_id: row
        for row in plan.hypotheses
    }
    if len(hypothesis_by_final) != len(plan.hypotheses):
        raise ValueError("duplicate final hypothesis IDs in binding plan")

    rows: list[PreVerifierContractGateV2Row] = []
    for hypothesis in plan.hypotheses:
        for claim in hypothesis.claims:
            rows.append(
                assess_preverifier_claim_contract(
                    hypothesis_plan=hypothesis,
                    claim_plan=claim,
                )
            )

    gate_counts = Counter(row.gate_status for row in rows)
    router_counts = Counter(row.router_hint for row in rows)

    body = {
        "schema_version": "preverifier-contract-gate-v2",
        "source_binding_plan_id": plan.plan_id,
        "source_binding_plan_sha256": plan.plan_sha256,
        "rows": [row.model_dump(mode="json") for row in rows],
        "claim_count": len(rows),
        "ready_claim_count": sum(
            row.gate_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            for row in rows
        ),
        "not_ready_claim_count": sum(
            row.gate_status == "NOT_READY_FOR_LITERAL_ENDPOINT_BINDING"
            for row in rows
        ),
        "novelty_bearing_ready_claim_count": sum(
            row.gate_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            and row.novelty_selection_role == "NOVELTY_BEARING"
            for row in rows
        ),
        "gate_status_counts": dict(sorted(gate_counts.items())),
        "router_hint_counts": dict(sorted(router_counts.items())),
        "complete_source_contract_checked_before_endpoint_binding": True,
        "binding_contract_checked": True,
        "atomic_kind_checked": True,
        "exact_prediction_source_binding_checked": True,
        "exact_falsifier_source_binding_checked": True,
        "shared_observable_identity_checked": True,
        "endpoint_binding_performed": False,
        "repair_performed": False,
        "regeneration_performed": False,
        "literature_retrieval_performed": False,
        "novelty_assessment_performed": False,
        "verifier_result_observed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return PreVerifierContractGateV2Report(
        **body,
        report_id="preverifier_contract_gate_v2:" + digest[:20],
        report_sha256=digest,
    )


__all__ = [
    "GateStatus",
    "PreVerifierContractGateV2Report",
    "PreVerifierContractGateV2Row",
    "RouterHint",
    "assess_preverifier_claim_contract",
    "build_preverifier_contract_gate_v2",
    "classify_router_hint",
]
