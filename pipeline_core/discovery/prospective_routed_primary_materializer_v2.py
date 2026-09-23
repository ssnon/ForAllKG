from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import LiteratureQueryPlan, NoveltyClaim
from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    PreVerifierContractGateV2Report,
    build_preverifier_contract_gate_v2,
)
from pipeline_core.discovery.prospective_routed_decomposition_primary import (
    assess_deterministic_decomposition,
)
from pipeline_core.discovery.prospective_routed_route_compiler_v2 import (
    HypothesisRouteDecisionV2,
    ProspectiveRoutedDispatchReportV2,
    SourceAlignmentCandidatePairV2,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingClaimPlan,
    RelationalAtomicBindingHypothesisPlan,
    RelationalAtomicBindingPlan,
    assess_claim_binding_readiness,
)
from pipeline_core.discovery.relational_atomic_projection import _candidate_card_and_claim
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _pretty_json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_exact_or_validate(path: Path, payload: object) -> str:
    expected = _pretty_json_bytes(payload)
    if path.exists():
        observed = path.read_bytes()
        if observed != expected:
            raise ValueError("existing write-once primary artifact differs: " + str(path))
        return hashlib.sha256(observed).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(expected)
    return hashlib.sha256(expected).hexdigest()


class SourceAlignmentAuditDraft(StrictModel):
    claim_id: str
    zero_scientific_delta: bool
    same_relation_commitment: bool
    same_scope: bool
    same_direction: bool
    no_new_mechanism: bool
    no_new_moderator: bool
    no_new_scientific_entity: bool
    rationale: str = Field(min_length=1)


@runtime_checkable
class SourceAlignmentAuditBackend(Protocol):
    backend_name: str
    model_name: str

    def audit(
        self,
        *,
        claim: NoveltyClaim,
        candidate: SourceAlignmentCandidatePairV2,
        candidate_card: object,
    ) -> tuple[SourceAlignmentAuditDraft, Any]: ...


def build_source_alignment_audit_prompt(
    *,
    claim: NoveltyClaim,
    candidate: SourceAlignmentCandidatePairV2,
    candidate_card: object,
) -> tuple[str, str]:
    system = (
        "You are a strict scientific provenance auditor. Evaluate only whether replacing "
        "the claim's prediction and falsification strings with the supplied existing "
        "HypothesisCard surfaces preserves exactly the same scientific commitment. "
        "Do not rewrite anything. Do not improve the hypothesis. "
        "Return zero_scientific_delta=true only when relation, scope, direction, "
        "mechanism/moderator commitments, and scientific entity set are unchanged. "
        "Any uncertainty must fail the audit."
    )
    payload = {
        "claim_id": claim.claim_id,
        "claim_kind": claim.kind,
        "novelty_selection_role": claim.novelty_selection_role,
        "claim_text": claim.text,
        "required_bridge": claim.required_bridge,
        "original_predicted_observation": claim.predicted_observation,
        "original_falsification_condition": claim.falsification_condition,
        "prior_art_identity_terms": list(claim.prior_art_identity_terms),
        "relation_nucleus_terms": list(claim.relation_nucleus_terms),
        "candidate_prediction_observation_id": candidate.prediction_observation_id,
        "candidate_prediction_observable": candidate.prediction_observable,
        "candidate_falsification_criterion_id": candidate.falsification_criterion_id,
        "candidate_falsifier_observable": candidate.falsifier_observable,
        "candidate_falsifying_outcome": candidate.falsifying_outcome,
        "candidate_card_title": str(getattr(candidate_card, "title", "") or ""),
        "candidate_card_hypothesis_statement": str(
            getattr(candidate_card, "hypothesis_statement", "")
            or getattr(candidate_card, "statement", "")
            or ""
        ),
    }
    return system, (
        "Audit this proposed source alignment. The proposed strings already exist on "
        "the frozen HypothesisCard; decide only whether substituting them into the "
        "canonical NoveltyClaim is scientifically zero-delta.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )


class InstructorSourceAlignmentAuditBackend:
    backend_name = "instructor_source_alignment_audit"

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        base_url: str,
        temperature: float,
        parse_retries: int,
        timeout_seconds: float,
        telemetry_path: str | Path | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = str(model)
        self.api_key_env = str(api_key_env)
        self.api_key = os.getenv(self.api_key_env)
        self.base_url = str(base_url)
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
            raise RuntimeError("No API key available. Set " + self.api_key_env + ".")
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Source-alignment audit requires openai and instructor.") from exc
        self._client = instructor.from_openai(
            OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_seconds),
            mode=instructor.Mode.JSON,
        )
        return self._client

    def audit(
        self,
        *,
        claim: NoveltyClaim,
        candidate: SourceAlignmentCandidatePairV2,
        candidate_card: object,
    ) -> tuple[SourceAlignmentAuditDraft, Any]:
        system, user = build_source_alignment_audit_prompt(
            claim=claim, candidate=candidate, candidate_card=candidate_card
        )
        draft, event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=SourceAlignmentAuditDraft,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "prospective_routed_primary_v2",
                "stage": "source_alignment_semantic_delta_audit",
                "claim_id": claim.claim_id,
            },
            semantic_components={
                "source_claim": claim.model_dump(mode="json"),
                "selected_existing_source_pair": candidate.model_dump(mode="json"),
            },
        )
        if not isinstance(draft, SourceAlignmentAuditDraft):
            draft = SourceAlignmentAuditDraft.model_validate(draft)
        return draft, event


def _audit_passes(audit: SourceAlignmentAuditDraft) -> bool:
    return all([
        audit.zero_scientific_delta,
        audit.same_relation_commitment,
        audit.same_scope,
        audit.same_direction,
        audit.no_new_mechanism,
        audit.no_new_moderator,
        audit.no_new_scientific_entity,
    ])


def _token_value(event: Any, name: str) -> int | None:
    value = getattr(event, name, None)
    return int(value) if value is not None else None


AlignmentStatus = Literal[
    "MATERIALIZED_ZERO_DELTA_SOURCE_ALIGNMENT",
    "UNAVAILABLE_NO_CANDIDATE",
    "UNAVAILABLE_AMBIGUOUS_CANDIDATES",
    "REJECTED_SEMANTIC_DELTA",
    "AUDIT_CALL_FAILED",
]


class SourceAlignmentClaimResultV2(StrictModel):
    claim_id: str
    status: AlignmentStatus
    candidate_count: int = Field(ge=0)
    selected_candidate: SourceAlignmentCandidatePairV2 | None = None
    semantic_audit_performed: bool = False
    semantic_audit_passed: bool | None = None
    semantic_audit: SourceAlignmentAuditDraft | None = None
    audit_llm_calls: int = Field(ge=0, le=1)
    audit_input_tokens: int | None = None
    audit_output_tokens: int | None = None
    source_claim_sha256_before: str
    source_claim_sha256_after: str | None = None
    aligned_query_plan_path: str | None = None
    aligned_query_plan_file_sha256: str | None = None
    new_text_generated: Literal[False] = False
    source_surfaces_only: Literal[True] = True


class DecompositionClaimResultV2(StrictModel):
    claim_id: str
    status: Literal[
        "MATERIALIZED_DETERMINISTIC_DECOMPOSITION",
        "UNAVAILABLE_DETERMINISTIC_DECOMPOSITION",
    ]
    component_claim_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    llm_calls: Literal[0] = 0
    scientific_content_added: Literal[False] = False


PrimaryStatusV2 = Literal[
    "PRIMARY_MATERIALIZED",
    "PRIMARY_UNAVAILABLE_REGENERATION_REQUIRED",
    "PRIMARY_PASSTHROUGH",
]


class RoutedPrimaryHypothesisResultV2(StrictModel):
    candidate_hypothesis_id: str
    final_hypothesis_id: str
    route_action: str
    primary_status: PrimaryStatusV2
    source_alignment_results: list[SourceAlignmentClaimResultV2]
    decomposition_results: list[DecompositionClaimResultV2]
    source_claim_ids: list[str]
    primary_output_claim_ids: list[str]
    primary_gate_ready_claim_ids: list[str]
    primary_gate_novelty_bearing_ready_claim_ids: list[str]
    regeneration_fallback_required: bool
    regeneration_fallback_available: bool
    scientific_content_added: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False


class ProspectiveRoutedPrimaryMaterializationReportV2(StrictModel):
    schema_version: Literal[
        "prospective-routed-primary-materialization-v2"
    ] = "prospective-routed-primary-materialization-v2"
    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: Literal["P16", "P17", "P18", "P19", "P20"]
    source_dispatch_report_id: str
    source_dispatch_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_binding_plan_id: str
    source_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_binding_plan_id: str
    primary_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_gate_report_id: str
    primary_gate_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hypotheses: list[RoutedPrimaryHypothesisResultV2]
    hypothesis_count: int = Field(ge=0)
    materialized_source_alignment_count: int = Field(ge=0)
    source_alignment_ambiguous_count: int = Field(ge=0)
    source_alignment_semantic_reject_count: int = Field(ge=0)
    source_alignment_audit_failure_count: int = Field(ge=0)
    materialized_decomposition_count: int = Field(ge=0)
    unavailable_decomposition_count: int = Field(ge=0)
    regeneration_fallback_required_count: int = Field(ge=0)
    recovered_without_regeneration_count: int = Field(ge=0)
    source_alignment_generation_llm_calls: Literal[0] = 0
    source_alignment_audit_llm_calls: int = Field(ge=0)
    decomposition_llm_calls: Literal[0] = 0
    regeneration_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self):
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError("hypothesis_count mismatch")
        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("primary materialization-v2 SHA mismatch")
        if observed_id != "prospective_routed_primary_materialization_v2:" + expected_sha[:20]:
            raise ValueError("primary materialization-v2 ID mismatch")
        return self


def _rebuild_hypothesis(
    source: RelationalAtomicBindingHypothesisPlan,
    claims: list[RelationalAtomicBindingClaimPlan],
    *,
    source_query_plan: str | None = None,
    source_query_plan_sha256: str | None = None,
) -> RelationalAtomicBindingHypothesisPlan:
    ready = sum(row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING" for row in claims)
    novelty_ready = sum(
        row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
        and row.novelty_selection_role == "NOVELTY_BEARING"
        for row in claims
    )
    status = (
        "NO_BINDABLE_CLAIMS"
        if ready == 0
        else (
            "NO_NOVELTY_BEARING_BINDABLE_CLAIM"
            if novelty_ready == 0
            else "READY_FOR_LITERAL_ENDPOINT_BINDING"
        )
    )
    payload = source.model_dump(mode="json")
    payload.update({
        "claims": [row.model_dump(mode="json") for row in claims],
        "claim_count": len(claims),
        "binding_ready_claim_count": ready,
        "novelty_bearing_binding_ready_claim_count": novelty_ready,
        "binding_status": status,
    })
    if source_query_plan is not None:
        payload["source_query_plan"] = source_query_plan
    if source_query_plan_sha256 is not None:
        payload["source_query_plan_sha256"] = source_query_plan_sha256
    return RelationalAtomicBindingHypothesisPlan.model_validate(payload)


def _rebuild_plan(
    source: RelationalAtomicBindingPlan,
    hypotheses: list[RelationalAtomicBindingHypothesisPlan],
) -> RelationalAtomicBindingPlan:
    body = source.model_dump(mode="json")
    body.pop("plan_id")
    body.pop("plan_sha256")
    body["hypotheses"] = [row.model_dump(mode="json") for row in hypotheses]
    body["hypothesis_count"] = len(hypotheses)
    body["ready_hypothesis_count"] = sum(
        row.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING" for row in hypotheses
    )
    body["not_ready_hypothesis_count"] = len(hypotheses) - body["ready_hypothesis_count"]
    body["claim_count"] = sum(row.claim_count for row in hypotheses)
    body["binding_ready_claim_count"] = sum(row.binding_ready_claim_count for row in hypotheses)
    body["novelty_bearing_binding_ready_claim_count"] = sum(
        row.novelty_bearing_binding_ready_claim_count for row in hypotheses
    )
    body["hypothesis_status_counts"] = dict(sorted(Counter(row.binding_status for row in hypotheses).items()))
    body["claim_status_counts"] = dict(sorted(Counter(
        claim.binding_status for row in hypotheses for claim in row.claims
    ).items()))
    digest = _sha256_json(body)
    return RelationalAtomicBindingPlan(
        **body,
        plan_id="relational_atomic_binding_plan:" + digest[:20],
        plan_sha256=digest,
    )


def _aligned_claim(
    *,
    claim: NoveltyClaim,
    candidate: SourceAlignmentCandidatePairV2,
) -> NoveltyClaim:
    """Rebind only canonical test-specification surfaces.

    NoveltyClaimDraft carries semantic_fidelity_binding source IDs, but the
    canonical NoveltyClaim schema intentionally does not.  The selected
    observation/falsifier IDs remain provenance in
    SourceAlignmentClaimResultV2.selected_candidate; injecting draft-only
    fields into the canonical claim would violate its strict schema.
    """
    payload = claim.model_dump(mode="json")
    payload["predicted_observation"] = candidate.prediction_observable
    payload["falsification_condition"] = candidate.falsifying_outcome
    return NoveltyClaim.model_validate(payload)


def _aligned_query_plan(
    *,
    source: LiteratureQueryPlan,
    candidate_hypothesis_id: str,
    aligned_claims: dict[str, NoveltyClaim],
) -> LiteratureQueryPlan:
    payload = source.model_dump(mode="json")
    changed = 0
    for group in payload["claims"]:
        if group["hypothesis_id"] != candidate_hypothesis_id:
            continue
        rewritten = []
        for claim in group["claims"]:
            cid = claim["claim_id"]
            if cid in aligned_claims:
                rewritten.append(aligned_claims[cid].model_dump(mode="json"))
                changed += 1
            else:
                rewritten.append(claim)
        group["claims"] = rewritten
    if changed != len(aligned_claims):
        raise ValueError("aligned source claims do not map exactly into query plan")
    payload.pop("plan_id")
    payload.pop("plan_sha256")
    digest = _sha256_json(payload)
    return LiteratureQueryPlan(
        **payload,
        plan_id="literature_query_plan:" + digest[:20],
        plan_sha256=digest,
    )


def _exact_candidate_exists(candidate_card: object, candidate: SourceAlignmentCandidatePairV2) -> bool:
    predictions = [
        row for row in candidate_card.predicted_observations
        if row.observation_id == candidate.prediction_observation_id
        and row.observable == candidate.prediction_observable
    ]
    falsifiers = [
        row for row in candidate_card.falsification_criteria
        if row.criterion_id == candidate.falsification_criterion_id
        and row.observable == candidate.falsifier_observable
        and row.falsifying_outcome == candidate.falsifying_outcome
    ]
    return len(predictions) == 1 and len(falsifiers) == 1


def execute_primary_materialization_v2(
    *,
    binding_plan: RelationalAtomicBindingPlan,
    dispatch: ProspectiveRoutedDispatchReportV2,
    audit_backend: SourceAlignmentAuditBackend,
    primary_root: Path,
) -> tuple[RelationalAtomicBindingPlan, PreVerifierContractGateV2Report, ProspectiveRoutedPrimaryMaterializationReportV2]:
    if dispatch.source_binding_plan_id != binding_plan.plan_id:
        raise ValueError("dispatch/binding-plan ID mismatch")
    if dispatch.source_binding_plan_sha256 != binding_plan.plan_sha256:
        raise ValueError("dispatch/binding-plan SHA mismatch")

    decisions = {row.final_hypothesis_id: row for row in dispatch.hypotheses}
    if len(decisions) != len(dispatch.hypotheses):
        raise ValueError("duplicate dispatch final hypothesis IDs")
    if set(decisions) != {row.final_hypothesis_id for row in binding_plan.hypotheses}:
        raise ValueError("dispatch/binding-plan hypothesis population mismatch")

    rebuilt = []
    preliminary = {}

    for source_hypothesis in binding_plan.hypotheses:
        decision = decisions[source_hypothesis.final_hypothesis_id]
        claims_by_id = {row.claim_id: row for row in source_hypothesis.claims}
        align_results = []
        decomp_results = []
        output_claims = list(source_hypothesis.claims)
        primary_status: PrimaryStatusV2 = "PRIMARY_PASSTHROUGH"

        if decision.route_action == "ZERO_DELTA_SOURCE_ALIGNMENT_THEN_REGENERATE_IF_UNAVAILABLE":
            primary_status = "PRIMARY_MATERIALIZED"
            aligned_source_claims = {}
            query_plan_source = LiteratureQueryPlan.model_validate_json(
                Path(source_hypothesis.source_query_plan).read_text(encoding="utf-8")
            )
            unavailable = False

            items = [
                row for row in decision.claim_work_items
                if row.router_hint == "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"
            ]
            if not items:
                raise ValueError("source-alignment route has no alignment work item")

            for item in items:
                claim_plan = claims_by_id[item.claim_id]
                candidate_card, source_claim = _candidate_card_and_claim(
                    hypothesis_plan=source_hypothesis,
                    claim_plan=claim_plan,
                )
                count = len(item.source_alignment_candidates)
                if count == 0:
                    align_results.append(SourceAlignmentClaimResultV2(
                        claim_id=item.claim_id,
                        status="UNAVAILABLE_NO_CANDIDATE",
                        candidate_count=0,
                        audit_llm_calls=0,
                        source_claim_sha256_before=claim_plan.source_claim_sha256,
                    ))
                    unavailable = True
                    continue
                if count != 1:
                    align_results.append(SourceAlignmentClaimResultV2(
                        claim_id=item.claim_id,
                        status="UNAVAILABLE_AMBIGUOUS_CANDIDATES",
                        candidate_count=count,
                        audit_llm_calls=0,
                        source_claim_sha256_before=claim_plan.source_claim_sha256,
                    ))
                    unavailable = True
                    continue

                candidate = item.source_alignment_candidates[0]
                if not _exact_candidate_exists(candidate_card, candidate):
                    raise ValueError(item.claim_id + ": frozen source-alignment candidate no longer resolves")

                try:
                    audit, event = audit_backend.audit(
                        claim=source_claim,
                        candidate=candidate,
                        candidate_card=candidate_card,
                    )
                except Exception:
                    align_results.append(SourceAlignmentClaimResultV2(
                        claim_id=item.claim_id,
                        status="AUDIT_CALL_FAILED",
                        candidate_count=1,
                        selected_candidate=candidate,
                        semantic_audit_performed=True,
                        semantic_audit_passed=None,
                        audit_llm_calls=1,
                        source_claim_sha256_before=claim_plan.source_claim_sha256,
                    ))
                    unavailable = True
                    continue

                if audit.claim_id != item.claim_id:
                    raise ValueError(item.claim_id + ": audit claim ID mismatch")

                if not _audit_passes(audit):
                    align_results.append(SourceAlignmentClaimResultV2(
                        claim_id=item.claim_id,
                        status="REJECTED_SEMANTIC_DELTA",
                        candidate_count=1,
                        selected_candidate=candidate,
                        semantic_audit_performed=True,
                        semantic_audit_passed=False,
                        semantic_audit=audit,
                        audit_llm_calls=1,
                        audit_input_tokens=_token_value(event, "provider_input_tokens"),
                        audit_output_tokens=_token_value(event, "provider_output_tokens"),
                        source_claim_sha256_before=claim_plan.source_claim_sha256,
                    ))
                    unavailable = True
                    continue

                aligned = _aligned_claim(claim=source_claim, candidate=candidate)
                aligned_source_claims[item.claim_id] = aligned
                align_results.append(SourceAlignmentClaimResultV2(
                    claim_id=item.claim_id,
                    status="MATERIALIZED_ZERO_DELTA_SOURCE_ALIGNMENT",
                    candidate_count=1,
                    selected_candidate=candidate,
                    semantic_audit_performed=True,
                    semantic_audit_passed=True,
                    semantic_audit=audit,
                    audit_llm_calls=1,
                    audit_input_tokens=_token_value(event, "provider_input_tokens"),
                    audit_output_tokens=_token_value(event, "provider_output_tokens"),
                    source_claim_sha256_before=claim_plan.source_claim_sha256,
                    source_claim_sha256_after=_sha256_json(aligned.model_dump(mode="json")),
                ))

            if unavailable:
                primary_status = "PRIMARY_UNAVAILABLE_REGENERATION_REQUIRED"
                rebuilt.append(source_hypothesis)
                output_claims = list(source_hypothesis.claims)
                align_results = [
                    row.model_copy(update={
                        "source_claim_sha256_after": None,
                        "aligned_query_plan_path": None,
                        "aligned_query_plan_file_sha256": None,
                    })
                    if row.status == "MATERIALIZED_ZERO_DELTA_SOURCE_ALIGNMENT"
                    else row
                    for row in align_results
                ]
            else:
                aligned_plan = _aligned_query_plan(
                    source=query_plan_source,
                    candidate_hypothesis_id=source_hypothesis.candidate_hypothesis_id,
                    aligned_claims=aligned_source_claims,
                )
                slug = source_hypothesis.final_hypothesis_id.replace(":", "_").replace("/", "_")
                sidecar = primary_root / "lineage" / slug / "source_alignment.query_plan.json"
                sidecar_sha = _write_exact_or_validate(sidecar, aligned_plan)

                groups = [
                    row for row in aligned_plan.claims
                    if row.hypothesis_id == source_hypothesis.candidate_hypothesis_id
                ]
                if len(groups) != 1:
                    raise ValueError("aligned query plan no longer resolves one claim group")
                source_claims = {row.claim_id: row for row in groups[0].claims}
                output_claims = [
                    assess_claim_binding_readiness(
                        claim=source_claims[claim_plan.claim_id],
                        candidate_hypothesis_id=source_hypothesis.candidate_hypothesis_id,
                        final_hypothesis_id=source_hypothesis.final_hypothesis_id,
                    )
                    for claim_plan in source_hypothesis.claims
                ]
                align_results = [
                    row.model_copy(update={
                        "aligned_query_plan_path": str(sidecar),
                        "aligned_query_plan_file_sha256": sidecar_sha,
                    })
                    if row.status == "MATERIALIZED_ZERO_DELTA_SOURCE_ALIGNMENT"
                    else row
                    for row in align_results
                ]
                rebuilt.append(_rebuild_hypothesis(
                    source_hypothesis,
                    output_claims,
                    source_query_plan=str(sidecar),
                    source_query_plan_sha256=sidecar_sha,
                ))

        elif decision.route_action == "ATOMIC_DECOMPOSITION_THEN_REGENERATE_IF_UNAVAILABLE":
            source_claims = {}
            for claim_plan in source_hypothesis.claims:
                _, source_claim = _candidate_card_and_claim(
                    hypothesis_plan=source_hypothesis,
                    claim_plan=claim_plan,
                )
                source_claims[source_claim.claim_id] = source_claim

            items = [
                row for row in decision.claim_work_items
                if row.router_hint == "DECOMPOSE_OR_REGENERATE_REVIEW"
            ]
            if not items:
                raise ValueError("decomposition route has no decomposition work item")

            removed = set()
            unavailable = False
            for item in items:
                assessment = assess_deterministic_decomposition(
                    composite_claim=source_claims[item.claim_id],
                    source_claims_by_id=source_claims,
                )
                if assessment.status == "DETERMINISTIC_DECOMPOSITION_AVAILABLE":
                    removed.add(item.claim_id)
                    decomp_results.append(DecompositionClaimResultV2(
                        claim_id=item.claim_id,
                        status="MATERIALIZED_DETERMINISTIC_DECOMPOSITION",
                        component_claim_ids=list(assessment.component_claim_ids),
                    ))
                else:
                    unavailable = True
                    decomp_results.append(DecompositionClaimResultV2(
                        claim_id=item.claim_id,
                        status="UNAVAILABLE_DETERMINISTIC_DECOMPOSITION",
                        component_claim_ids=list(assessment.component_claim_ids),
                        reason_codes=list(assessment.reason_codes),
                    ))

            if unavailable:
                primary_status = "PRIMARY_UNAVAILABLE_REGENERATION_REQUIRED"
                rebuilt.append(source_hypothesis)
            else:
                primary_status = "PRIMARY_MATERIALIZED"
                output_claims = [
                    row for row in source_hypothesis.claims
                    if row.claim_id not in removed
                ]
                rebuilt.append(_rebuild_hypothesis(source_hypothesis, output_claims))

        elif decision.route_action == "PROCEED_TO_LITERAL_ENDPOINT_BINDING":
            rebuilt.append(source_hypothesis)

        elif decision.route_action == "ZERO_DELTA_SPECIFICATION_REPAIR":
            raise ValueError(
                "specification-repair route requires its frozen LLM repair/audit executor"
            )
        else:
            raise ValueError("unsupported routed primary action")

        preliminary[source_hypothesis.final_hypothesis_id] = {
            "decision": decision,
            "primary_status": primary_status,
            "align_results": align_results,
            "decomp_results": decomp_results,
            "source_claim_ids": [row.claim_id for row in source_hypothesis.claims],
            "output_claim_ids": [row.claim_id for row in output_claims],
        }

    primary_plan = _rebuild_plan(binding_plan, rebuilt)
    primary_gate = build_preverifier_contract_gate_v2(plan=primary_plan)

    gate_by_final = {}
    for row in primary_gate.rows:
        gate_by_final.setdefault(row.final_hypothesis_id, []).append(row)

    results = []
    for source_hypothesis in binding_plan.hypotheses:
        final_id = source_hypothesis.final_hypothesis_id
        rec = preliminary[final_id]
        decision = rec["decision"]
        rows = gate_by_final.get(final_id, [])
        ready = [row.claim_id for row in rows if row.gate_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"]
        novelty_ready = [
            row.claim_id for row in rows
            if row.gate_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            and row.novelty_selection_role == "NOVELTY_BEARING"
        ]
        fallback = bool(
            decision.regeneration_fallback_allowed
            and (
                rec["primary_status"] == "PRIMARY_UNAVAILABLE_REGENERATION_REQUIRED"
                or not novelty_ready
            )
        )
        results.append(RoutedPrimaryHypothesisResultV2(
            candidate_hypothesis_id=decision.candidate_hypothesis_id,
            final_hypothesis_id=final_id,
            route_action=decision.route_action,
            primary_status=rec["primary_status"],
            source_alignment_results=rec["align_results"],
            decomposition_results=rec["decomp_results"],
            source_claim_ids=rec["source_claim_ids"],
            primary_output_claim_ids=rec["output_claim_ids"],
            primary_gate_ready_claim_ids=ready,
            primary_gate_novelty_bearing_ready_claim_ids=novelty_ready,
            regeneration_fallback_required=fallback,
            regeneration_fallback_available=decision.regeneration_fallback_allowed,
        ))

    aligns = [x for row in results for x in row.source_alignment_results]
    decomps = [x for row in results for x in row.decomposition_results]
    body = {
        "schema_version": "prospective-routed-primary-materialization-v2",
        "case_id": dispatch.case_id,
        "source_dispatch_report_id": dispatch.report_id,
        "source_dispatch_report_sha256": dispatch.report_sha256,
        "source_binding_plan_id": binding_plan.plan_id,
        "source_binding_plan_sha256": binding_plan.plan_sha256,
        "primary_binding_plan_id": primary_plan.plan_id,
        "primary_binding_plan_sha256": primary_plan.plan_sha256,
        "primary_gate_report_id": primary_gate.report_id,
        "primary_gate_report_sha256": primary_gate.report_sha256,
        "hypotheses": [row.model_dump(mode="json") for row in results],
        "hypothesis_count": len(results),
        "materialized_source_alignment_count": sum(
            x.status == "MATERIALIZED_ZERO_DELTA_SOURCE_ALIGNMENT"
            and x.source_claim_sha256_after is not None
            for x in aligns
        ),
        "source_alignment_ambiguous_count": sum(
            x.status == "UNAVAILABLE_AMBIGUOUS_CANDIDATES" for x in aligns
        ),
        "source_alignment_semantic_reject_count": sum(
            x.status == "REJECTED_SEMANTIC_DELTA" for x in aligns
        ),
        "source_alignment_audit_failure_count": sum(
            x.status == "AUDIT_CALL_FAILED" for x in aligns
        ),
        "materialized_decomposition_count": sum(
            x.status == "MATERIALIZED_DETERMINISTIC_DECOMPOSITION" for x in decomps
        ),
        "unavailable_decomposition_count": sum(
            x.status == "UNAVAILABLE_DETERMINISTIC_DECOMPOSITION" for x in decomps
        ),
        "regeneration_fallback_required_count": sum(
            row.regeneration_fallback_required for row in results
        ),
        "recovered_without_regeneration_count": sum(
            bool(row.primary_gate_novelty_bearing_ready_claim_ids)
            and not row.regeneration_fallback_required
            for row in results
        ),
        "source_alignment_generation_llm_calls": 0,
        "source_alignment_audit_llm_calls": sum(x.audit_llm_calls for x in aligns),
        "decomposition_llm_calls": 0,
        "regeneration_performed": False,
        "endpoint_binding_performed": False,
        "novelty_assessment_performed": False,
        "verifier_result_observed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    report = ProspectiveRoutedPrimaryMaterializationReportV2(
        **body,
        report_id="prospective_routed_primary_materialization_v2:" + digest[:20],
        report_sha256=digest,
    )
    return primary_plan, primary_gate, report
