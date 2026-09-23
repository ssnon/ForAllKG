from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.preverifier_specification_repair_policy import (
    SpecificationCaseRepairPlan,
    SpecificationClaimRepairPlan,
    SpecificationRepairCampaignPlan,
)
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


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
    return " ".join(
        re.findall(r"[a-z0-9]+", str(value).casefold())
    )


def _surface_contains(haystack: str, needle: str) -> bool:
    left = _normalize(haystack)
    right = _normalize(needle)
    return bool(right) and right in left


_GENERIC_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "if",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "than",
        "that",
        "the",
        "then",
        "to",
        "under",
        "with",
        "without",
        "within",
        "would",
        "not",
        "no",
        "does",
        "do",
        "did",
        "when",
        "where",
        "which",
        "while",
        "relative",
        "compared",
        "change",
        "changes",
        "changed",
        "changing",
        "increase",
        "increases",
        "increased",
        "increasing",
        "decrease",
        "decreases",
        "decreased",
        "decreasing",
        "higher",
        "lower",
        "greater",
        "smaller",
        "larger",
        "more",
        "less",
    }
)


def _content_tokens(value: str) -> set[str]:
    return {
        token
        for token in _normalize(value).split()
        if len(token) >= 2 and token not in _GENERIC_TOKENS
    }


class SpecificationRepairDraft(StrictModel):
    claim_id: str
    required_bridge: str
    predicted_observation: str
    falsification_condition: str


class SpecificationRepairAuditDraft(StrictModel):
    claim_id: str
    zero_scientific_delta: bool
    added_scientific_concepts: list[str] = Field(default_factory=list)
    removed_scientific_commitments: list[str] = Field(default_factory=list)
    new_mechanism_introduced: bool = False
    new_moderator_introduced: bool = False
    scope_changed: bool = False
    relation_direction_changed: bool = False
    scientific_entity_set_changed: bool = False
    rationale: str = Field(min_length=1)


RepairExecutionStatus = Literal[
    "MATERIALIZED_R1",
    "REJECTED_DETERMINISTIC_POLICY",
    "REJECTED_SEMANTIC_DELTA",
    "GENERATION_CALL_FAILED",
    "AUDIT_CALL_FAILED",
]


class SpecificationRepairClaimResult(StrictModel):
    claim_id: str
    source_claim_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repair_action: str

    status: RepairExecutionStatus
    proposed_required_bridge: str | None = None
    proposed_predicted_observation: str | None = None
    proposed_falsification_condition: str | None = None

    deterministic_policy_passed: bool
    deterministic_reason_codes: list[str]

    semantic_audit_performed: bool
    semantic_audit_passed: bool | None = None
    semantic_audit: SpecificationRepairAuditDraft | None = None

    generation_llm_calls: int = Field(ge=0, le=1)
    audit_llm_calls: int = Field(ge=0, le=1)
    generation_input_tokens: int | None = Field(default=None, ge=0)
    generation_output_tokens: int | None = Field(default=None, ge=0)
    audit_input_tokens: int | None = Field(default=None, ge=0)
    audit_output_tokens: int | None = Field(default=None, ge=0)

    accepted_required_bridge: str | None = None
    accepted_predicted_observation: str | None = None
    accepted_falsification_condition: str | None = None

    source_claim_text_preserved: Literal[True] = True
    prior_art_identity_terms_preserved: Literal[True] = True
    relation_nucleus_terms_preserved: Literal[True] = True
    novelty_role_preserved: Literal[True] = True
    literature_retrieval_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False
    external_novelty_outcome_used: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> "SpecificationRepairClaimResult":
        if self.status == "MATERIALIZED_R1":
            if not self.deterministic_policy_passed:
                raise ValueError(
                    "materialized repair requires deterministic policy pass"
                )
            if self.semantic_audit_passed is not True:
                raise ValueError(
                    "materialized repair requires semantic-delta audit pass"
                )
            if any(
                value is None
                for value in (
                    self.accepted_required_bridge,
                    self.accepted_predicted_observation,
                    self.accepted_falsification_condition,
                )
            ):
                raise ValueError(
                    "materialized repair requires complete accepted fields"
                )
        else:
            if any(
                value is not None
                for value in (
                    self.accepted_required_bridge,
                    self.accepted_predicted_observation,
                    self.accepted_falsification_condition,
                )
            ):
                raise ValueError(
                    "non-materialized repair cannot carry accepted fields"
                )
        return self


class SpecificationRepairCaseResult(StrictModel):
    case_id: Literal["P06", "P07", "P08", "P09", "P10"]
    source_case_result_id: str
    source_case_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_case_plan_status: str
    excluded_reason: str | None = None

    claim_results: list[SpecificationRepairClaimResult]
    planned_claim_count: int = Field(ge=0)
    materialized_claim_count: int = Field(ge=0)
    status_counts: dict[str, int]

    source_case_preserved: Literal[True] = True
    repair_attempts_per_claim_at_most_one: Literal[True] = True
    literature_retrieval_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_case(self) -> "SpecificationRepairCaseResult":
        if self.planned_claim_count != len(self.claim_results):
            raise ValueError("planned_claim_count mismatch")
        materialized = sum(
            row.status == "MATERIALIZED_R1"
            for row in self.claim_results
        )
        if self.materialized_claim_count != materialized:
            raise ValueError("materialized_claim_count mismatch")
        counts = Counter(row.status for row in self.claim_results)
        if dict(sorted(counts.items())) != dict(
            sorted(self.status_counts.items())
        ):
            raise ValueError("status_counts mismatch")
        return self


class SpecificationRepairExecutionReport(StrictModel):
    schema_version: Literal[
        "preverifier-specification-repair-execution-report-v1"
    ] = "preverifier-specification-repair-execution-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_repair_plan_id: str
    source_repair_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_repair_plan_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repair_policy_repository_head_sha: str

    cases: list[SpecificationRepairCaseResult]
    case_ids: list[str]
    case_count: Literal[5] = 5

    planned_claim_count: int = Field(ge=0)
    materialized_claim_count: int = Field(ge=0)
    claim_status_counts: dict[str, int]
    generation_llm_call_count: int = Field(ge=0)
    audit_llm_call_count: int = Field(ge=0)

    original_case_results_preserved: Literal[True] = True
    repair_results_do_not_replace_prospective_results: Literal[True] = True
    literature_retrieval_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False
    external_novelty_outcome_used: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "SpecificationRepairExecutionReport":
        expected = ["P06", "P07", "P08", "P09", "P10"]
        if self.case_ids != expected:
            raise ValueError("repair execution case IDs must be P06-P10")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("repair execution cases must be ordered P06-P10")
        if self.planned_claim_count != sum(
            row.planned_claim_count for row in self.cases
        ):
            raise ValueError("execution planned_claim_count mismatch")
        if self.materialized_claim_count != sum(
            row.materialized_claim_count for row in self.cases
        ):
            raise ValueError("execution materialized_claim_count mismatch")

        claim_rows = [
            claim
            for case in self.cases
            for claim in case.claim_results
        ]
        counts = Counter(row.status for row in claim_rows)
        if dict(sorted(counts.items())) != dict(
            sorted(self.claim_status_counts.items())
        ):
            raise ValueError("claim_status_counts mismatch")
        if self.generation_llm_call_count != sum(
            row.generation_llm_calls for row in claim_rows
        ):
            raise ValueError("generation_llm_call_count mismatch")
        if self.audit_llm_call_count != sum(
            row.audit_llm_calls for row in claim_rows
        ):
            raise ValueError("audit_llm_call_count mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("specification repair execution SHA mismatch")
        if observed_id != (
            "preverifier_specification_repair_execution_report:"
            + expected_sha[:20]
        ):
            raise ValueError("specification repair execution ID mismatch")
        return self


def _source_field_values(
    plan: SpecificationClaimRepairPlan,
) -> dict[str, str]:
    return {
        "required_bridge": plan.source_required_bridge,
        "predicted_observation": plan.source_predicted_observation,
        "falsification_condition": plan.source_falsification_condition,
    }


def _draft_field_values(
    draft: SpecificationRepairDraft,
) -> dict[str, str]:
    return {
        "required_bridge": draft.required_bridge,
        "predicted_observation": draft.predicted_observation,
        "falsification_condition": draft.falsification_condition,
    }


def validate_repair_draft(
    *,
    plan: SpecificationClaimRepairPlan,
    draft: SpecificationRepairDraft,
) -> list[str]:
    reasons: list[str] = []

    if draft.claim_id != plan.claim_id:
        reasons.append("claim_id_mismatch")

    source = _source_field_values(plan)
    proposed = _draft_field_values(draft)
    editable = set(plan.editable_fields)
    fill_only = set(plan.fill_only_fields)

    for field, source_value in source.items():
        proposed_value = proposed[field]
        if field not in editable and proposed_value != source_value:
            reasons.append(f"immutable_field_changed:{field}")
        if field in editable and not proposed_value.strip():
            reasons.append(f"editable_field_blank:{field}")
        if field in fill_only and source_value.strip():
            reasons.append(f"fill_only_source_was_nonempty:{field}")
        if field in fill_only and proposed_value == source_value:
            reasons.append(f"fill_only_field_not_filled:{field}")

    if plan.repair_action == "LEXICAL_ALIGNMENT_REPAIR":
        if draft.required_bridge == plan.source_required_bridge:
            reasons.append("lexical_repair_did_not_change_required_bridge")

    source_surface = " ".join(
        [
            plan.source_claim_text,
            plan.source_required_bridge,
            plan.source_predicted_observation,
            plan.source_falsification_condition,
            *plan.source_prior_art_identity_terms,
            *plan.source_relation_nucleus_terms,
        ]
    )
    allowed_tokens = _content_tokens(source_surface)

    for field in plan.editable_fields:
        proposed_tokens = _content_tokens(proposed[field])
        added_tokens = sorted(proposed_tokens - allowed_tokens)
        if added_tokens:
            reasons.append(
                "new_surface_content_tokens:"
                + field
                + ":"
                + ",".join(added_tokens)
            )

    for identity in plan.source_prior_art_identity_terms:
        for field in plan.editable_fields:
            if (
                proposed[field].strip()
                and not _surface_contains(proposed[field], identity)
            ):
                reasons.append(
                    "identity_not_literal_in_repaired_field:"
                    + field
                    + ":"
                    + identity
                )

    return list(dict.fromkeys(reasons))


def semantic_audit_passes(
    audit: SpecificationRepairAuditDraft,
    *,
    claim_id: str,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if audit.claim_id != claim_id:
        reasons.append("audit_claim_id_mismatch")
    if not audit.zero_scientific_delta:
        reasons.append("audit_nonzero_scientific_delta")
    if audit.added_scientific_concepts:
        reasons.append("audit_added_scientific_concepts")
    if audit.removed_scientific_commitments:
        reasons.append("audit_removed_scientific_commitments")
    if audit.new_mechanism_introduced:
        reasons.append("audit_new_mechanism")
    if audit.new_moderator_introduced:
        reasons.append("audit_new_moderator")
    if audit.scope_changed:
        reasons.append("audit_scope_changed")
    if audit.relation_direction_changed:
        reasons.append("audit_relation_direction_changed")
    if audit.scientific_entity_set_changed:
        reasons.append("audit_scientific_entity_set_changed")
    return not reasons, reasons


def build_repair_prompt(
    plan: SpecificationClaimRepairPlan,
) -> tuple[str, str]:
    system = """You perform one bounded scientific specification repair.

You are NOT creating a new hypothesis and NOT improving novelty.

Hard rules:
- Preserve claim_text exactly as the scientific proposition authority.
- Do not add a new scientific entity, mechanism, moderator, scope, regime,
  direction, comparison, threshold, causal claim, or measurement construct.
- Do not use literature, prior-art outcomes, verifier outcomes, or outside facts.
- You may edit ONLY the fields listed in editable_fields.
- Every non-editable field must be returned exactly unchanged.
- For fill_only_fields, the source is empty and you may fill it only with a
  formulation strictly supported by the supplied claim_text and existing
  specification fields.
- Preserve every prior_art_identity_term literally in every repaired editable
  field. This is a downstream provenance contract, not permission to add meaning.
- For LEXICAL_ALIGNMENT_REPAIR, align required_bridge wording to literal
  scientific vocabulary already present in the supplied frozen source surfaces.
- Return exactly one row for the supplied claim_id.
"""
    payload = {
        "claim_id": plan.claim_id,
        "repair_action": plan.repair_action,
        "editable_fields": plan.editable_fields,
        "fill_only_fields": plan.fill_only_fields,
        "claim_text": plan.source_claim_text,
        "required_bridge": plan.source_required_bridge,
        "predicted_observation": plan.source_predicted_observation,
        "falsification_condition": plan.source_falsification_condition,
        "prior_art_identity_terms": plan.source_prior_art_identity_terms,
        "relation_nucleus_terms": plan.source_relation_nucleus_terms,
        "source_reason_codes": plan.source_reason_codes,
        "source_abstention_reason": plan.source_abstention_reason,
    }
    user = (
        "Produce one zero-scientific-delta repaired specification row.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return system, user


def build_audit_prompt(
    *,
    plan: SpecificationClaimRepairPlan,
    draft: SpecificationRepairDraft,
) -> tuple[str, str]:
    system = """You audit whether a bounded specification repair has ZERO scientific delta.

Be conservative. PASS only when the repaired fields merely restate or complete
the SAME proposition already explicit in claim_text and the frozen source
specification.

A repair has nonzero scientific delta if it adds or removes any scientific
entity, mechanism, moderator, condition, scope, regime, direction, threshold,
comparison, causal commitment, measurement construct, or substantive predicted
effect not already explicit in the source.

For a missing prediction or falsifier, articulating a test statement is allowed
only when its scientific content and direction are already explicit in the
source claim/specification. Do not reward usefulness or plausibility.
Do not assess novelty, truth, prior art, or literature support.
"""
    payload = {
        "claim_id": plan.claim_id,
        "repair_action": plan.repair_action,
        "source": {
            "claim_text": plan.source_claim_text,
            "required_bridge": plan.source_required_bridge,
            "predicted_observation": plan.source_predicted_observation,
            "falsification_condition": plan.source_falsification_condition,
            "prior_art_identity_terms": plan.source_prior_art_identity_terms,
            "relation_nucleus_terms": plan.source_relation_nucleus_terms,
        },
        "proposed": draft.model_dump(mode="json"),
    }
    user = (
        "Audit this proposed repair for zero scientific delta.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return system, user


class InstructorSpecificationRepairBackend:
    def __init__(
        self,
        *,
        repair_model: str,
        audit_model: str,
        api_key_env: str,
        base_url: str,
        temperature: float,
        parse_retries: int,
        timeout_seconds: float,
        telemetry_path: str | Path | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.repair_model = repair_model
        self.audit_model = audit_model
        self.api_key_env = api_key_env
        self.api_key = os.getenv(api_key_env)
        self.base_url = base_url
        self.temperature = float(temperature)
        self.parse_retries = int(parse_retries)
        self.timeout_seconds = float(timeout_seconds)
        self.telemetry_path = telemetry_path
        self.telemetry_context = dict(telemetry_context or {})
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                f"No API key available. Set {self.api_key_env}."
            )
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Specification repair requires openai and instructor."
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout_seconds,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = instructor.from_openai(
            OpenAI(**kwargs),
            mode=instructor.Mode.JSON,
        )
        return self._client

    def generate(
        self,
        plan: SpecificationClaimRepairPlan,
    ) -> tuple[SpecificationRepairDraft, Any]:
        system, user = build_repair_prompt(plan)
        draft, event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.repair_model,
            response_model=SpecificationRepairDraft,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "preverifier_specification_repair",
                "stage": "repair_generation",
                "call_kind": "zero_scientific_delta_repair",
                "claim_id": plan.claim_id,
            },
            semantic_components={
                "repair_policy": plan.model_dump(mode="json"),
                "source_claim": plan.source_claim_text,
            },
        )
        if not isinstance(draft, SpecificationRepairDraft):
            draft = SpecificationRepairDraft.model_validate(draft)
        return draft, event

    def audit(
        self,
        *,
        plan: SpecificationClaimRepairPlan,
        draft: SpecificationRepairDraft,
    ) -> tuple[SpecificationRepairAuditDraft, Any]:
        system, user = build_audit_prompt(plan=plan, draft=draft)
        audit, event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.audit_model,
            response_model=SpecificationRepairAuditDraft,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "preverifier_specification_repair",
                "stage": "semantic_delta_audit",
                "call_kind": "zero_scientific_delta_audit",
                "claim_id": plan.claim_id,
            },
            semantic_components={
                "source_claim": plan.source_claim_text,
                "proposed_repair": draft.model_dump(mode="json"),
            },
        )
        if not isinstance(audit, SpecificationRepairAuditDraft):
            audit = SpecificationRepairAuditDraft.model_validate(audit)
        return audit, event


def _token_value(event: Any, name: str) -> int | None:
    value = getattr(event, name, None)
    return int(value) if value is not None else None


def execute_claim_repair(
    *,
    plan: SpecificationClaimRepairPlan,
    backend: InstructorSpecificationRepairBackend,
) -> SpecificationRepairClaimResult:
    try:
        draft, generation_event = backend.generate(plan)
    except Exception as exc:
        return SpecificationRepairClaimResult(
            claim_id=plan.claim_id,
            source_claim_snapshot_sha256=plan.source_claim_snapshot_sha256,
            repair_action=plan.repair_action,
            status="GENERATION_CALL_FAILED",
            deterministic_policy_passed=False,
            deterministic_reason_codes=[
                "generation_call_failed:" + type(exc).__name__
            ],
            semantic_audit_performed=False,
            generation_llm_calls=1,
            audit_llm_calls=0,
        )

    deterministic_reasons = validate_repair_draft(
        plan=plan,
        draft=draft,
    )
    if deterministic_reasons:
        return SpecificationRepairClaimResult(
            claim_id=plan.claim_id,
            source_claim_snapshot_sha256=plan.source_claim_snapshot_sha256,
            repair_action=plan.repair_action,
            status="REJECTED_DETERMINISTIC_POLICY",
            proposed_required_bridge=draft.required_bridge,
            proposed_predicted_observation=draft.predicted_observation,
            proposed_falsification_condition=draft.falsification_condition,
            deterministic_policy_passed=False,
            deterministic_reason_codes=deterministic_reasons,
            semantic_audit_performed=False,
            generation_llm_calls=1,
            audit_llm_calls=0,
            generation_input_tokens=_token_value(
                generation_event, "provider_input_tokens"
            ),
            generation_output_tokens=_token_value(
                generation_event, "provider_output_tokens"
            ),
        )

    try:
        audit, audit_event = backend.audit(plan=plan, draft=draft)
    except Exception as exc:
        return SpecificationRepairClaimResult(
            claim_id=plan.claim_id,
            source_claim_snapshot_sha256=plan.source_claim_snapshot_sha256,
            repair_action=plan.repair_action,
            status="AUDIT_CALL_FAILED",
            proposed_required_bridge=draft.required_bridge,
            proposed_predicted_observation=draft.predicted_observation,
            proposed_falsification_condition=draft.falsification_condition,
            deterministic_policy_passed=True,
            deterministic_reason_codes=[],
            semantic_audit_performed=True,
            semantic_audit_passed=None,
            generation_llm_calls=1,
            audit_llm_calls=1,
            generation_input_tokens=_token_value(
                generation_event, "provider_input_tokens"
            ),
            generation_output_tokens=_token_value(
                generation_event, "provider_output_tokens"
            ),
        )

    audit_passed, audit_reasons = semantic_audit_passes(
        audit,
        claim_id=plan.claim_id,
    )
    if not audit_passed:
        return SpecificationRepairClaimResult(
            claim_id=plan.claim_id,
            source_claim_snapshot_sha256=plan.source_claim_snapshot_sha256,
            repair_action=plan.repair_action,
            status="REJECTED_SEMANTIC_DELTA",
            proposed_required_bridge=draft.required_bridge,
            proposed_predicted_observation=draft.predicted_observation,
            proposed_falsification_condition=draft.falsification_condition,
            deterministic_policy_passed=True,
            deterministic_reason_codes=audit_reasons,
            semantic_audit_performed=True,
            semantic_audit_passed=False,
            semantic_audit=audit,
            generation_llm_calls=1,
            audit_llm_calls=1,
            generation_input_tokens=_token_value(
                generation_event, "provider_input_tokens"
            ),
            generation_output_tokens=_token_value(
                generation_event, "provider_output_tokens"
            ),
            audit_input_tokens=_token_value(
                audit_event, "provider_input_tokens"
            ),
            audit_output_tokens=_token_value(
                audit_event, "provider_output_tokens"
            ),
        )

    return SpecificationRepairClaimResult(
        claim_id=plan.claim_id,
        source_claim_snapshot_sha256=plan.source_claim_snapshot_sha256,
        repair_action=plan.repair_action,
        status="MATERIALIZED_R1",
        proposed_required_bridge=draft.required_bridge,
        proposed_predicted_observation=draft.predicted_observation,
        proposed_falsification_condition=draft.falsification_condition,
        deterministic_policy_passed=True,
        deterministic_reason_codes=[],
        semantic_audit_performed=True,
        semantic_audit_passed=True,
        semantic_audit=audit,
        generation_llm_calls=1,
        audit_llm_calls=1,
        generation_input_tokens=_token_value(
            generation_event, "provider_input_tokens"
        ),
        generation_output_tokens=_token_value(
            generation_event, "provider_output_tokens"
        ),
        audit_input_tokens=_token_value(
            audit_event, "provider_input_tokens"
        ),
        audit_output_tokens=_token_value(
            audit_event, "provider_output_tokens"
        ),
        accepted_required_bridge=draft.required_bridge,
        accepted_predicted_observation=draft.predicted_observation,
        accepted_falsification_condition=draft.falsification_condition,
    )


def execute_case_repair(
    *,
    case_plan: SpecificationCaseRepairPlan,
    backend: InstructorSpecificationRepairBackend,
) -> SpecificationRepairCaseResult:
    if case_plan.status != "PLANNED_AUTOMATIC_REPAIR":
        return SpecificationRepairCaseResult(
            case_id=case_plan.case_id,
            source_case_result_id=case_plan.source_case_result_id,
            source_case_result_sha256=case_plan.source_case_result_sha256,
            source_case_plan_status=case_plan.status,
            excluded_reason=case_plan.excluded_reason,
            claim_results=[],
            planned_claim_count=0,
            materialized_claim_count=0,
            status_counts={},
        )

    results = [
        execute_claim_repair(plan=claim, backend=backend)
        for claim in case_plan.claim_repairs
    ]
    counts = Counter(row.status for row in results)
    return SpecificationRepairCaseResult(
        case_id=case_plan.case_id,
        source_case_result_id=case_plan.source_case_result_id,
        source_case_result_sha256=case_plan.source_case_result_sha256,
        source_case_plan_status=case_plan.status,
        claim_results=results,
        planned_claim_count=len(results),
        materialized_claim_count=sum(
            row.status == "MATERIALIZED_R1" for row in results
        ),
        status_counts=dict(sorted(counts.items())),
    )


def build_execution_report(
    *,
    plan: SpecificationRepairCampaignPlan,
    plan_file_sha256: str,
    cases: list[SpecificationRepairCaseResult],
) -> SpecificationRepairExecutionReport:
    claim_rows = [
        claim
        for case in cases
        for claim in case.claim_results
    ]
    counts = Counter(row.status for row in claim_rows)
    body = {
        "schema_version":
            "preverifier-specification-repair-execution-report-v1",
        "source_repair_plan_id": plan.plan_id,
        "source_repair_plan_sha256": plan.plan_sha256,
        "source_repair_plan_file_sha256": plan_file_sha256,
        "repair_policy_repository_head_sha":
            plan.repair_policy_repository_head_sha,
        "cases": [row.model_dump(mode="json") for row in cases],
        "case_ids": ["P06", "P07", "P08", "P09", "P10"],
        "case_count": 5,
        "planned_claim_count": len(claim_rows),
        "materialized_claim_count": sum(
            row.status == "MATERIALIZED_R1"
            for row in claim_rows
        ),
        "claim_status_counts": dict(sorted(counts.items())),
        "generation_llm_call_count": sum(
            row.generation_llm_calls for row in claim_rows
        ),
        "audit_llm_call_count": sum(
            row.audit_llm_calls for row in claim_rows
        ),
        "original_case_results_preserved": True,
        "repair_results_do_not_replace_prospective_results": True,
        "literature_retrieval_performed": False,
        "verifier_result_observed": False,
        "external_novelty_outcome_used": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return SpecificationRepairExecutionReport(
        **body,
        report_id=(
            "preverifier_specification_repair_execution_report:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "InstructorSpecificationRepairBackend",
    "RepairExecutionStatus",
    "SpecificationRepairAuditDraft",
    "SpecificationRepairCaseResult",
    "SpecificationRepairClaimResult",
    "SpecificationRepairDraft",
    "SpecificationRepairExecutionReport",
    "build_audit_prompt",
    "build_execution_report",
    "build_repair_prompt",
    "execute_case_repair",
    "execute_claim_repair",
    "semantic_audit_passes",
    "validate_repair_draft",
]
