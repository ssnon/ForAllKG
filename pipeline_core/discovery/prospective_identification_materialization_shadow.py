from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.relation_validity_aware_generation import IdentificationContract
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProspectiveIdentificationShadowResponse(StrictModel):
    contract: IdentificationContract


class ProspectiveIdentificationShadowArtifact(StrictModel):
    schema_version: Literal["prospective-identification-materialization-shadow-v1"] = (
        "prospective-identification-materialization-shadow-v1"
    )
    status: Literal["COMPLETE", "SHADOW_EVALUATION_FAILED"]
    source_stage: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_hypothesis_id: str = Field(min_length=1)
    contract: IdentificationContract | None = None
    contract_integrity_passed: bool = False
    contract_integrity_reasons: list[str] = Field(default_factory=list)
    current_evidence_status: str | None = None
    prospective_identifiability: str | None = None
    directionality_mode: str | None = None
    measurement_compatibility_mode: str | None = None
    would_abstain_if_authoritative: bool = False
    evaluation_error: str | None = None
    artifact_path: str | None = None
    positive_premise_statement_ids: list[str] = Field(default_factory=list)
    gap_statement_ids: list[str] = Field(default_factory=list)
    external_prior_art_as_positive_premise: Literal[False] = False
    prospective_design_as_positive_premise: Literal[False] = False
    shadow_has_generation_authority: Literal[False] = False
    shadow_has_selection_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False


SYSTEM_PROMPT = """
You are a scientific prospective-identification auditor operating as a
NON-AUTHORITATIVE shadow precondition.

Evaluate the already-generated candidate hypothesis. Do NOT replace it with a
different scientific hypothesis. Distinguish current grounded support from a
grounded prospective prediction and from an ungrounded proposal.

Absolute invariants:
- Only supplied GROUNDED PREMISE records are positive scientific evidence.
- GAP records are task/gap context, not positive evidence unless they are also
  candidate positive premises.
- External prior art is not supplied and must not be invented.
- Future experiments/controls/protocols establish TESTABILITY only; they never
  become current positive evidence.
- Do not invent variables, reporter properties, affinities, mechanisms,
  structural distinctions, observables, or comparison concepts absent from the
  supplied grounded premises.

Use the existing IdentificationContract semantics.

Current evidence status:
- DIRECT_RELATION_SUPPORTED: current grounded evidence supports the relation.
- PARTIAL_GROUNDING: variables/components/bridge are grounded but the full
  relation is not currently established.
- CONTEXT_ONLY: supplied evidence cannot ground the variables/bridge required.

Prospective identifiability:
- CURRENTLY_IDENTIFIED: current grounded evidence directly supports relation.
- PROSPECTIVELY_IDENTIFIABLE: relation is not currently established, but its
  variables/bridge are grounded and an explicit falsifiable future experiment
  can identify it without ungrounded concepts.
- NOT_OPERATIONALIZABLE: testing requires ungrounded concepts or no coherent
  falsifiable design can be specified.

Directionality:
- DIRECTLY_SUPPORTED: same direction is directly grounded.
- GROUNDED_PREDICTION: direction is a falsifiable grounded prediction, not an
  observed result.
- NON_DIRECTIONAL: no direction should be asserted.
- NOT_IDENTIFIABLE: relation is not operationalizable.

Measurement:
- DIRECTLY_COMPARABLE: current measurements are commensurate.
- CONTEXTUAL_ONLY: one current context supports a scoped relation; other records
  are contextual only.
- PROSPECTIVE_MATCH_REQUIRED: current records are not directly comparable but a
  grounded matched future measurement can test the relation.
- NOT_COMPARABLE: no grounded prospective matching can operationalize relation.

For PROSPECTIVELY_IDENTIFIABLE:
- assessment MUST be PARTIALLY_IDENTIFIED;
- grounded_bridge_statement_ids must identify positive-premise IDs grounding the
  proposed variables/bridge;
- prospective_test_design must explicitly state intervention/comparison,
  observables, matching/controls, and measurement level;
- prospective_falsifier must state an outcome that rejects the relation;
- ungrounded_required_concepts must be empty.

For NOT_OPERATIONALIZABLE:
- assessment MUST be NOT_IDENTIFIABLE;
- directionality_mode MUST be NOT_IDENTIFIABLE;
- prospective_identifiability MUST be NOT_OPERATIONALIZABLE;
- list missing scientific concepts in ungrounded_required_concepts.

For CURRENTLY_IDENTIFIED:
- assessment should be SUPPORTED;
- current_relation_support_statement_ids must list direct grounded support.

CRITICAL IDENTITY RULE:
IdentificationContract.premise_statement_ids must contain exactly the candidate
positive premise IDs in the same order. Do not add/remove/substitute IDs.

This audit has NO generation or selection authority.
"""


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _statement_id(row: Any) -> str:
    value = getattr(row, "statement_id", None)
    if value is not None:
        return str(value)
    payload = _dump(row)
    return str(payload.get("statement_id") or "") if isinstance(payload, dict) else ""


def _candidate_id(candidate: Any) -> str:
    value = getattr(candidate, "hypothesis_id", None)
    if value is not None:
        return str(value)
    payload = _dump(candidate)
    return str(payload.get("hypothesis_id") or "") if isinstance(payload, dict) else ""


def _candidate_ids(candidate: Any, field: str) -> list[str]:
    value = getattr(candidate, field, None)
    if value is None:
        payload = _dump(candidate)
        if isinstance(payload, dict):
            value = payload.get(field, [])
    return list(dict.fromkeys(map(str, value or [])))


def _evidence_payload(*, context: HypothesisContext, candidate: Any) -> dict[str, Any]:
    premise_ids = _candidate_ids(candidate, "premise_statement_ids")
    gap_ids = _candidate_ids(candidate, "gap_statement_ids")
    premise_set = set(premise_ids)
    wanted = premise_set | set(gap_ids)
    rows = []
    for row in context.evidence_statements:
        sid = _statement_id(row)
        if sid not in wanted:
            continue
        rows.append({
            "statement_id": sid,
            "candidate_role": "POSITIVE_PREMISE" if sid in premise_set else "GAP_CONTEXT",
            "record": _dump(row),
        })
    return {
        "candidate": _dump(candidate),
        "candidate_positive_premise_ids": premise_ids,
        "candidate_gap_statement_ids": gap_ids,
        "grounded_records": rows,
    }


def _integrity(*, candidate: Any, contract: IdentificationContract) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    expected = _candidate_ids(candidate, "premise_statement_ids")
    actual = list(map(str, contract.premise_statement_ids))
    if actual != expected:
        reasons.append(
            "premise_statement_ids_order_changed"
            if set(actual) == set(expected)
            else "premise_statement_ids_identity_changed"
        )
    if contract.prospective_identifiability == "PROSPECTIVELY_IDENTIFIABLE" and contract.ungrounded_required_concepts:
        reasons.append("prospective_identifiable_has_ungrounded_required_concepts")
    if contract.prospective_identifiability == "NOT_OPERATIONALIZABLE" and contract.assessment != "NOT_IDENTIFIABLE":
        reasons.append("not_operationalizable_assessment_mismatch")
    return not reasons, reasons


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_from_contract(*, source_stage: str, context: HypothesisContext, candidate: Any, contract: IdentificationContract, artifact_path: str | None = None) -> ProspectiveIdentificationShadowArtifact:
    passed, reasons = _integrity(candidate=candidate, contract=contract)
    return ProspectiveIdentificationShadowArtifact(
        status="COMPLETE",
        source_stage=str(source_stage),
        source_context_id=str(context.context_id),
        source_hypothesis_id=_candidate_id(candidate),
        contract=contract,
        contract_integrity_passed=passed,
        contract_integrity_reasons=reasons,
        current_evidence_status=str(contract.current_evidence_status),
        prospective_identifiability=str(contract.prospective_identifiability),
        directionality_mode=str(contract.directionality_mode),
        measurement_compatibility_mode=str(contract.measurement_compatibility_mode),
        would_abstain_if_authoritative=(contract.prospective_identifiability == "NOT_OPERATIONALIZABLE"),
        artifact_path=artifact_path,
        positive_premise_statement_ids=_candidate_ids(candidate, "premise_statement_ids"),
        gap_statement_ids=_candidate_ids(candidate, "gap_statement_ids"),
    )


def run_prospective_identification_shadow(*, context: HypothesisContext, candidate: Any, source_stage: str, model: str, api_key_env: str = "OPENROUTER_API_KEY", base_url: str | None = None, parse_retries: int = 3, output_prefix: str | Path) -> ProspectiveIdentificationShadowArtifact:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prompt_path = Path(str(prefix) + ".prompt.json")
    artifact_path = Path(str(prefix) + ".result.json")
    telemetry_path = Path(str(prefix) + ".telemetry.jsonl")

    payload = _evidence_payload(context=context, candidate=candidate)
    user_payload = {
        "research_question": str(context.question),
        "source_stage": str(source_stage),
        **payload,
    }
    _write_json(prompt_path, {
        "system_prompt": SYSTEM_PROMPT,
        "user_payload": user_payload,
        "external_prior_art_included": False,
        "prospective_design_is_positive_evidence": False,
    })

    try:
        key = os.getenv(api_key_env)
        if not key:
            raise RuntimeError(f"No API key available in {api_key_env}")
        import instructor
        from openai import OpenAI
        kwargs: dict[str, Any] = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        client = instructor.from_openai(OpenAI(**kwargs), mode=instructor.Mode.JSON)
        response, _event = run_instructor_structured_call(
            client.chat.completions,
            model=str(model),
            response_model=ProspectiveIdentificationShadowResponse,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
            ],
            temperature=0.0,
            max_retries=int(parse_retries),
            telemetry_path=telemetry_path,
            telemetry_context={
                "pipeline": "prospective_identification_materialization_shadow_v1",
                "source_stage": str(source_stage),
                "source_hypothesis_id": _candidate_id(candidate),
            },
        )
        if not isinstance(response, ProspectiveIdentificationShadowResponse):
            response = ProspectiveIdentificationShadowResponse.model_validate(response)
        artifact = artifact_from_contract(
            source_stage=source_stage,
            context=context,
            candidate=candidate,
            contract=response.contract,
            artifact_path=str(artifact_path),
        )
    except Exception as exc:
        artifact = ProspectiveIdentificationShadowArtifact(
            status="SHADOW_EVALUATION_FAILED",
            source_stage=str(source_stage),
            source_context_id=str(context.context_id),
            source_hypothesis_id=_candidate_id(candidate),
            contract=None,
            contract_integrity_passed=False,
            contract_integrity_reasons=["shadow_evaluation_failed"],
            evaluation_error=f"{type(exc).__name__}:{exc}",
            artifact_path=str(artifact_path),
            positive_premise_statement_ids=_candidate_ids(candidate, "premise_statement_ids"),
            gap_statement_ids=_candidate_ids(candidate, "gap_statement_ids"),
        )
    _write_json(artifact_path, artifact)
    return artifact


def compact_shadow_record(artifact: ProspectiveIdentificationShadowArtifact) -> dict[str, Any]:
    return {
        "status": artifact.status,
        "artifact_path": artifact.artifact_path,
        "contract_integrity_passed": artifact.contract_integrity_passed,
        "contract_integrity_reasons": list(artifact.contract_integrity_reasons),
        "current_evidence_status": artifact.current_evidence_status,
        "prospective_identifiability": artifact.prospective_identifiability,
        "directionality_mode": artifact.directionality_mode,
        "measurement_compatibility_mode": artifact.measurement_compatibility_mode,
        "would_abstain_if_authoritative": artifact.would_abstain_if_authoritative,
        "shadow_has_generation_authority": False,
        "shadow_has_selection_authority": False,
        "production_selection_changed": False,
    }
