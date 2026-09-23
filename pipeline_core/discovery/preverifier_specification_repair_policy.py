from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.preverifier_specification_diagnostics import (
    PreVerifierClaimDiagnostic,
    PreVerifierSpecificationDiagnosticReport,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


EditableField = Literal[
    "required_bridge",
    "predicted_observation",
    "falsification_condition",
]

RepairPlanStatus = Literal[
    "PLANNED_AUTOMATIC_REPAIR",
    "EXCLUDED_MANUAL_DIAGNOSIS",
]


class SpecificationRepairPolicyV1(StrictModel):
    schema_version: Literal[
        "preverifier-specification-repair-policy-v1"
    ] = "preverifier-specification-repair-policy-v1"

    policy_id: Literal[
        "preverifier-specification-repair-policy:v1"
    ] = "preverifier-specification-repair-policy:v1"

    lexical_alignment_editable_fields: list[EditableField] = [
        "required_bridge"
    ]
    contract_completion_candidate_fields: list[EditableField] = [
        "required_bridge",
        "predicted_observation",
        "falsification_condition",
    ]

    claim_text_mutation_allowed: Literal[False] = False
    rationale_mutation_allowed: Literal[False] = False
    prior_art_identity_mutation_allowed: Literal[False] = False
    relation_nucleus_mutation_allowed: Literal[False] = False
    novelty_role_mutation_allowed: Literal[False] = False
    hypothesis_text_mutation_allowed: Literal[False] = False
    candidate_selection_mutation_allowed: Literal[False] = False
    task_mutation_allowed: Literal[False] = False

    new_scientific_concepts_allowed: Literal[False] = False
    new_mechanism_allowed: Literal[False] = False
    new_moderator_allowed: Literal[False] = False
    new_scope_allowed: Literal[False] = False
    relation_direction_change_allowed: Literal[False] = False

    lexical_repair_must_use_existing_surface_vocabulary: Literal[True] = True
    completion_repair_must_use_claim_supported_concepts_only: Literal[
        True
    ] = True
    completion_may_edit_only_originally_empty_fields: Literal[True] = True

    literature_retrieval_during_repair_allowed: Literal[False] = False
    external_novelty_outcome_as_repair_input_allowed: Literal[False] = False
    verifier_outcome_as_repair_input_allowed: Literal[False] = False
    old_n10_status_as_repair_input_allowed: Literal[False] = False

    max_repair_attempts_per_claim: Literal[1] = 1
    max_generation_calls_per_claim: Literal[1] = 1
    max_delta_audit_calls_per_claim: Literal[1] = 1
    semantic_delta_audit_required: Literal[True] = True
    strict_binding_reentry_required: Literal[True] = True
    original_case_result_preserved: Literal[True] = True
    repaired_artifacts_must_use_new_lineage: Literal[True] = True


class SpecificationClaimRepairPlan(StrictModel):
    claim_id: str
    repair_action: Literal[
        "LEXICAL_ALIGNMENT_REPAIR",
        "CONTRACT_COMPLETION_REPAIR",
    ]
    source_claim_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    editable_fields: list[EditableField]
    fill_only_fields: list[EditableField]
    immutable_fields: list[str]

    source_claim_text: str
    source_required_bridge: str
    source_predicted_observation: str
    source_falsification_condition: str
    source_prior_art_identity_terms: list[str]
    source_relation_nucleus_terms: list[str]
    source_reason_codes: list[str]
    source_abstention_reason: str | None = None

    max_repair_attempts: Literal[1] = 1
    semantic_delta_policy: Literal[
        "ZERO_SCIENTIFIC_DELTA_REQUIRED"
    ] = "ZERO_SCIENTIFIC_DELTA_REQUIRED"
    semantic_delta_audit_required: Literal[True] = True
    strict_binding_reentry_required: Literal[True] = True

    claim_text_mutation_allowed: Literal[False] = False
    identity_terms_mutation_allowed: Literal[False] = False
    novelty_role_mutation_allowed: Literal[False] = False
    new_scientific_concepts_allowed: Literal[False] = False
    relation_direction_change_allowed: Literal[False] = False
    scope_change_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_claim_plan(self) -> "SpecificationClaimRepairPlan":
        if not self.editable_fields:
            raise ValueError("automatic repair claim requires editable fields")
        if len(self.editable_fields) != len(set(self.editable_fields)):
            raise ValueError("editable_fields must be unique")
        if len(self.fill_only_fields) != len(set(self.fill_only_fields)):
            raise ValueError("fill_only_fields must be unique")
        if not set(self.fill_only_fields).issubset(self.editable_fields):
            raise ValueError("fill_only_fields must be editable")

        if self.repair_action == "LEXICAL_ALIGNMENT_REPAIR":
            if self.editable_fields != ["required_bridge"]:
                raise ValueError(
                    "v1 lexical repair may edit required_bridge only"
                )
            if self.fill_only_fields:
                raise ValueError(
                    "lexical alignment is not a missing-field completion"
                )

        if self.repair_action == "CONTRACT_COMPLETION_REPAIR":
            if self.fill_only_fields != self.editable_fields:
                raise ValueError(
                    "contract completion may edit only originally empty fields"
                )
        return self


class SpecificationCaseRepairPlan(StrictModel):
    case_id: Literal["P06", "P07", "P08", "P09", "P10"]
    source_case_result_id: str
    source_case_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_primary_diagnostic_class: str
    source_original_disposition: str

    status: RepairPlanStatus
    claim_repairs: list[SpecificationClaimRepairPlan]

    repair_output_dir: str
    excluded_reason: str | None = None

    case_replacement_allowed: Literal[False] = False
    source_case_mutation_allowed: Literal[False] = False
    source_case_result_preserved: Literal[True] = True
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_case_plan(self) -> "SpecificationCaseRepairPlan":
        if self.status == "PLANNED_AUTOMATIC_REPAIR":
            if not self.claim_repairs:
                raise ValueError(
                    "planned automatic repair requires claim repairs"
                )
            if self.excluded_reason is not None:
                raise ValueError(
                    "planned repair cannot carry excluded_reason"
                )
        else:
            if self.claim_repairs:
                raise ValueError(
                    "excluded case cannot carry automatic claim repairs"
                )
            if not (self.excluded_reason or "").strip():
                raise ValueError(
                    "excluded case requires excluded_reason"
                )
        return self


class SpecificationRepairCampaignPlan(StrictModel):
    schema_version: Literal[
        "preverifier-specification-repair-campaign-plan-v1"
    ] = "preverifier-specification-repair-campaign-plan-v1"

    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_diagnostic_report_id: str
    source_diagnostic_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diagnostic_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    repair_policy_repository_head_sha: str
    repository_tracked_worktree_dirty: Literal[False] = False

    repair_model: str
    audit_model: str
    base_url: str
    api_key_env: str
    temperature: Literal[0.0] = 0.0
    parse_retries: int = Field(default=3, ge=1)
    timeout_seconds: float = Field(default=180.0, gt=0)

    policy: SpecificationRepairPolicyV1
    cases: list[SpecificationCaseRepairPlan]
    case_ids: list[str]
    case_count: Literal[5] = 5

    automatic_case_ids: list[str]
    excluded_case_ids: list[str]
    automatic_claim_repair_count: int = Field(ge=0)
    repair_action_counts: dict[str, int]

    plan_frozen_before_repair_execution: Literal[True] = True
    repair_results_observed_before_plan_freeze: Literal[False] = False
    verifier_results_observed_for_repair_planning: Literal[False] = False
    external_novelty_outcomes_observed_for_repair_planning: Literal[
        False
    ] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(self) -> "SpecificationRepairCampaignPlan":
        expected = ["P06", "P07", "P08", "P09", "P10"]
        if self.case_ids != expected:
            raise ValueError("repair plan case IDs must be exactly P06-P10")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("repair cases must be ordered P06-P10")

        automatic = [
            row.case_id
            for row in self.cases
            if row.status == "PLANNED_AUTOMATIC_REPAIR"
        ]
        excluded = [
            row.case_id
            for row in self.cases
            if row.status == "EXCLUDED_MANUAL_DIAGNOSIS"
        ]
        if automatic != self.automatic_case_ids:
            raise ValueError("automatic_case_ids mismatch")
        if excluded != self.excluded_case_ids:
            raise ValueError("excluded_case_ids mismatch")

        claims = [
            claim
            for case in self.cases
            for claim in case.claim_repairs
        ]
        if len(claims) != self.automatic_claim_repair_count:
            raise ValueError("automatic_claim_repair_count mismatch")

        counts = Counter(claim.repair_action for claim in claims)
        if dict(sorted(counts.items())) != dict(
            sorted(self.repair_action_counts.items())
        ):
            raise ValueError("repair_action_counts mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("plan_id")
        observed_sha = body.pop("plan_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("specification repair campaign plan SHA mismatch")
        if observed_id != (
            "preverifier_specification_repair_campaign_plan:"
            + expected_sha[:20]
        ):
            raise ValueError("specification repair campaign plan ID mismatch")
        return self


def _snapshot_sha(claim: PreVerifierClaimDiagnostic) -> str:
    return _sha256_json(claim.source_claim.model_dump(mode="json"))


def _immutable_fields() -> list[str]:
    return [
        "claim_id",
        "claim_text",
        "rationale",
        "prior_art_identity_terms",
        "relation_nucleus_terms",
        "novelty_selection_role",
        "candidate_hypothesis_id",
        "final_hypothesis_id",
    ]


def _lexical_claim_plan(
    claim: PreVerifierClaimDiagnostic,
) -> SpecificationClaimRepairPlan:
    return SpecificationClaimRepairPlan(
        claim_id=claim.claim_id,
        repair_action="LEXICAL_ALIGNMENT_REPAIR",
        source_claim_snapshot_sha256=_snapshot_sha(claim),
        editable_fields=["required_bridge"],
        fill_only_fields=[],
        immutable_fields=_immutable_fields()
        + ["predicted_observation", "falsification_condition"],
        source_claim_text=claim.source_claim.claim_text,
        source_required_bridge=claim.source_claim.required_bridge,
        source_predicted_observation=(
            claim.source_claim.predicted_observation
        ),
        source_falsification_condition=(
            claim.source_claim.falsification_condition
        ),
        source_prior_art_identity_terms=list(
            claim.source_claim.prior_art_identity_terms
        ),
        source_relation_nucleus_terms=list(
            claim.source_claim.relation_nucleus_terms
        ),
        source_reason_codes=list(claim.source_claim.reason_codes),
        source_abstention_reason=claim.abstention_reason,
    )


def _completion_fields(
    claim: PreVerifierClaimDiagnostic,
) -> list[EditableField]:
    fields: list[EditableField] = []
    reason_to_field: dict[str, EditableField] = {
        "missing_required_bridge": "required_bridge",
        "missing_predicted_observation": "predicted_observation",
        "missing_falsification_condition": "falsification_condition",
    }
    for reason in claim.source_claim.reason_codes:
        field = reason_to_field.get(reason)
        if field is not None and field not in fields:
            fields.append(field)
    return fields


def _completion_claim_plan(
    claim: PreVerifierClaimDiagnostic,
) -> SpecificationClaimRepairPlan:
    if "missing_claim_text" in claim.source_claim.reason_codes:
        raise ValueError(
            claim.claim_id
            + ": v1 cannot automatically repair a missing claim_text"
        )
    fields = _completion_fields(claim)
    if not fields:
        raise ValueError(
            claim.claim_id
            + ": contract completion has no supported missing field"
        )
    return SpecificationClaimRepairPlan(
        claim_id=claim.claim_id,
        repair_action="CONTRACT_COMPLETION_REPAIR",
        source_claim_snapshot_sha256=_snapshot_sha(claim),
        editable_fields=fields,
        fill_only_fields=fields,
        immutable_fields=_immutable_fields()
        + [
            field
            for field in (
                "required_bridge",
                "predicted_observation",
                "falsification_condition",
            )
            if field not in fields
        ],
        source_claim_text=claim.source_claim.claim_text,
        source_required_bridge=claim.source_claim.required_bridge,
        source_predicted_observation=(
            claim.source_claim.predicted_observation
        ),
        source_falsification_condition=(
            claim.source_claim.falsification_condition
        ),
        source_prior_art_identity_terms=list(
            claim.source_claim.prior_art_identity_terms
        ),
        source_relation_nucleus_terms=list(
            claim.source_claim.relation_nucleus_terms
        ),
        source_reason_codes=list(claim.source_claim.reason_codes),
        source_abstention_reason=claim.abstention_reason,
    )


def build_specification_repair_campaign_plan(
    *,
    diagnostic: PreVerifierSpecificationDiagnosticReport,
    diagnostic_file_sha256: str,
    repair_policy_repository_head_sha: str,
    repository_tracked_worktree_dirty: bool,
    campaign_root: Path,
    repair_model: str,
    audit_model: str,
    base_url: str,
    api_key_env: str,
) -> SpecificationRepairCampaignPlan:
    if repository_tracked_worktree_dirty:
        raise ValueError(
            "specification repair plan requires a clean tracked worktree"
        )

    root = campaign_root.expanduser().resolve()
    policy = SpecificationRepairPolicyV1()
    cases: list[SpecificationCaseRepairPlan] = []

    for case in diagnostic.cases:
        repair_output_dir = str(
            root
            / "specification_repair_r1"
            / case.case_id
        )

        if not case.automatic_repair_eligible:
            cases.append(
                SpecificationCaseRepairPlan(
                    case_id=case.case_id,
                    source_case_result_id=case.source_case_result_id,
                    source_case_result_sha256=case.source_case_result_sha256,
                    source_primary_diagnostic_class=(
                        case.primary_diagnostic_class
                    ),
                    source_original_disposition=case.original_disposition,
                    status="EXCLUDED_MANUAL_DIAGNOSIS",
                    claim_repairs=[],
                    repair_output_dir=repair_output_dir,
                    excluded_reason=(
                        "diagnostic report did not authorize automatic "
                        "zero-scientific-delta repair"
                    ),
                )
            )
            continue

        claim_repairs: list[SpecificationClaimRepairPlan] = []
        for claim in case.claim_diagnostics:
            actions = set(claim.proposed_repair_actions)
            if actions == {"LEXICAL_ALIGNMENT_REPAIR"}:
                claim_repairs.append(_lexical_claim_plan(claim))
            elif actions == {"CONTRACT_COMPLETION_REPAIR"}:
                claim_repairs.append(_completion_claim_plan(claim))
            else:
                raise ValueError(
                    case.case_id
                    + "/"
                    + claim.claim_id
                    + ": v1 observed unsupported automatic repair action(s): "
                    + repr(sorted(actions))
                )

        cases.append(
            SpecificationCaseRepairPlan(
                case_id=case.case_id,
                source_case_result_id=case.source_case_result_id,
                source_case_result_sha256=case.source_case_result_sha256,
                source_primary_diagnostic_class=case.primary_diagnostic_class,
                source_original_disposition=case.original_disposition,
                status="PLANNED_AUTOMATIC_REPAIR",
                claim_repairs=claim_repairs,
                repair_output_dir=repair_output_dir,
            )
        )

    automatic_case_ids = [
        row.case_id
        for row in cases
        if row.status == "PLANNED_AUTOMATIC_REPAIR"
    ]
    excluded_case_ids = [
        row.case_id
        for row in cases
        if row.status == "EXCLUDED_MANUAL_DIAGNOSIS"
    ]
    claims = [
        claim
        for case in cases
        for claim in case.claim_repairs
    ]
    counts = Counter(claim.repair_action for claim in claims)

    body = {
        "schema_version":
            "preverifier-specification-repair-campaign-plan-v1",
        "source_diagnostic_report_id": diagnostic.report_id,
        "source_diagnostic_report_sha256": diagnostic.report_sha256,
        "source_diagnostic_file_sha256": diagnostic_file_sha256,
        "repair_policy_repository_head_sha":
            repair_policy_repository_head_sha,
        "repository_tracked_worktree_dirty": False,
        "repair_model": repair_model,
        "audit_model": audit_model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "temperature": 0.0,
        "parse_retries": 3,
        "timeout_seconds": 180.0,
        "policy": policy.model_dump(mode="json"),
        "cases": [row.model_dump(mode="json") for row in cases],
        "case_ids": ["P06", "P07", "P08", "P09", "P10"],
        "case_count": 5,
        "automatic_case_ids": automatic_case_ids,
        "excluded_case_ids": excluded_case_ids,
        "automatic_claim_repair_count": len(claims),
        "repair_action_counts": dict(sorted(counts.items())),
        "plan_frozen_before_repair_execution": True,
        "repair_results_observed_before_plan_freeze": False,
        "verifier_results_observed_for_repair_planning": False,
        "external_novelty_outcomes_observed_for_repair_planning": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return SpecificationRepairCampaignPlan(
        **body,
        plan_id=(
            "preverifier_specification_repair_campaign_plan:"
            + digest[:20]
        ),
        plan_sha256=digest,
    )


__all__ = [
    "EditableField",
    "SpecificationCaseRepairPlan",
    "SpecificationClaimRepairPlan",
    "SpecificationRepairCampaignPlan",
    "SpecificationRepairPolicyV1",
    "build_specification_repair_campaign_plan",
]
