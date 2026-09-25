from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    PreN10ScientificContractReportV1,
    build_pre_n10_scientific_contract_v1,
)
from pipeline_core.discovery.preverifier_specification_repair_executor import (
    SpecificationRepairAuditDraft,
    SpecificationRepairClaimResult,
    SpecificationRepairDraft,
    execute_claim_repair,
)
from pipeline_core.discovery.preverifier_specification_repair_policy import (
    SpecificationClaimRepairPlan,
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


def _pretty_json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_exact_or_validate(path: Path, payload: object) -> str:
    expected = _pretty_json_bytes(payload)
    path = path.expanduser().resolve()
    if path.exists():
        observed = path.read_bytes()
        if observed != expected:
            raise ValueError(
                "existing write-once pre-N10 specification-repair artifact differs: "
                + str(path)
            )
        return hashlib.sha256(observed).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(expected)
    return hashlib.sha256(expected).hexdigest()


class SpecificationRepairBackend(Protocol):
    def generate(
        self,
        plan: SpecificationClaimRepairPlan,
    ) -> tuple[SpecificationRepairDraft, Any]: ...

    def audit(
        self,
        *,
        plan: SpecificationClaimRepairPlan,
        draft: SpecificationRepairDraft,
    ) -> tuple[SpecificationRepairAuditDraft, Any]: ...


_SUPPORTED_COMPLETION_REASON_TO_FIELD = {
    "missing_required_bridge": "required_bridge",
    "missing_predicted_observation": "predicted_observation",
    "missing_falsification_condition": "falsification_condition",
}


SpecificationRepairPrimaryStatusV1 = Literal[
    "MATERIALIZED_ZERO_DELTA_SPECIFICATION_REPAIR",
    "UNSUPPORTED_AUTOMATIC_REPAIR",
    "REJECTED_DETERMINISTIC_POLICY",
    "REJECTED_SEMANTIC_DELTA",
    "GENERATION_CALL_FAILED",
    "AUDIT_CALL_FAILED",
]


class PreN10SpecificationRepairClaimResultV1(StrictModel):
    hypothesis_id: str
    claim_id: str
    status: SpecificationRepairPrimaryStatusV1
    source_reason_codes: list[str] = Field(default_factory=list)
    unsupported_reason_codes: list[str] = Field(default_factory=list)
    repair_plan: SpecificationClaimRepairPlan | None = None
    repair_execution: SpecificationRepairClaimResult | None = None
    source_claim_sha256_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_claim_sha256_after: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    scientific_content_added: Literal[False] = False
    claim_text_mutated: Literal[False] = False
    identity_terms_mutated: Literal[False] = False
    novelty_role_mutated: Literal[False] = False
    retrieval_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> "PreN10SpecificationRepairClaimResultV1":
        materialized = (
            self.status == "MATERIALIZED_ZERO_DELTA_SPECIFICATION_REPAIR"
        )
        if materialized:
            if self.repair_plan is None or self.repair_execution is None:
                raise ValueError(
                    "materialized pre-N10 specification repair requires plan and execution"
                )
            if self.repair_execution.status != "MATERIALIZED_R1":
                raise ValueError(
                    "materialized pre-N10 specification repair requires MATERIALIZED_R1"
                )
            if self.source_claim_sha256_after is None:
                raise ValueError(
                    "materialized pre-N10 specification repair requires output SHA"
                )
        if self.status == "UNSUPPORTED_AUTOMATIC_REPAIR":
            if not self.unsupported_reason_codes:
                raise ValueError(
                    "unsupported pre-N10 specification repair requires blockers"
                )
            if self.repair_plan is not None or self.repair_execution is not None:
                raise ValueError(
                    "unsupported pre-N10 specification repair cannot execute repair"
                )
        return self


class PreN10SpecificationRepairHypothesisResultV1(StrictModel):
    hypothesis_id: str
    source_contract_status: str
    dominant_router_hint: Literal[
        "SPECIFICATION_REPAIR_REVIEW"
    ] = "SPECIFICATION_REPAIR_REVIEW"
    repair_results: list[PreN10SpecificationRepairClaimResultV1] = Field(
        default_factory=list
    )
    post_contract_status: str
    recovered_for_n10: bool
    regeneration_fallback_required: bool

    @model_validator(mode="after")
    def validate_state(self) -> "PreN10SpecificationRepairHypothesisResultV1":
        if self.recovered_for_n10 != (
            self.post_contract_status == "READY_FOR_N10"
        ):
            raise ValueError("recovered_for_n10 mismatch")
        if self.regeneration_fallback_required == self.recovered_for_n10:
            raise ValueError(
                "regeneration fallback must be exact complement of recovery"
            )
        return self


class PreN10SpecificationRepairPrimaryReportV1(StrictModel):
    schema_version: Literal[
        "pre-n10-specification-repair-primary-report-v1"
    ] = "pre-n10-specification-repair-primary-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_contract_report_id: str
    source_contract_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_query_plan_id: str
    source_query_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repaired_query_plan_id: str
    repaired_query_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    post_contract_report_id: str
    post_contract_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    hypotheses: list[PreN10SpecificationRepairHypothesisResultV1]
    hypothesis_count: int = Field(ge=0)

    materialized_repair_count: int = Field(ge=0)
    unsupported_automatic_repair_count: int = Field(ge=0)
    deterministic_reject_count: int = Field(ge=0)
    semantic_reject_count: int = Field(ge=0)
    generation_failure_count: int = Field(ge=0)
    audit_failure_count: int = Field(ge=0)
    repair_generation_llm_call_count: int = Field(ge=0)
    repair_audit_llm_call_count: int = Field(ge=0)

    recovered_for_n10_count: int = Field(ge=0)
    regeneration_fallback_required_count: int = Field(ge=0)

    claim_generation_llm_calls: Literal[0] = 0
    retrieval_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    regeneration_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreN10SpecificationRepairPrimaryReportV1":
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError("hypothesis_count mismatch")
        claim_results = [
            claim
            for hypothesis in self.hypotheses
            for claim in hypothesis.repair_results
        ]
        counts = Counter(row.status for row in claim_results)
        expected = {
            "MATERIALIZED_ZERO_DELTA_SPECIFICATION_REPAIR":
                self.materialized_repair_count,
            "UNSUPPORTED_AUTOMATIC_REPAIR":
                self.unsupported_automatic_repair_count,
            "REJECTED_DETERMINISTIC_POLICY": self.deterministic_reject_count,
            "REJECTED_SEMANTIC_DELTA": self.semantic_reject_count,
            "GENERATION_CALL_FAILED": self.generation_failure_count,
            "AUDIT_CALL_FAILED": self.audit_failure_count,
        }
        for status, count in expected.items():
            if counts.get(status, 0) != count:
                raise ValueError("specification repair status count mismatch: " + status)
        if self.repair_generation_llm_call_count != sum(
            (
                row.repair_execution.generation_llm_calls
                if row.repair_execution is not None
                else 0
            )
            for row in claim_results
        ):
            raise ValueError("repair generation LLM count mismatch")
        if self.repair_audit_llm_call_count != sum(
            (
                row.repair_execution.audit_llm_calls
                if row.repair_execution is not None
                else 0
            )
            for row in claim_results
        ):
            raise ValueError("repair audit LLM count mismatch")
        if self.recovered_for_n10_count != sum(
            row.recovered_for_n10 for row in self.hypotheses
        ):
            raise ValueError("recovered_for_n10_count mismatch")
        if self.regeneration_fallback_required_count != sum(
            row.regeneration_fallback_required for row in self.hypotheses
        ):
            raise ValueError("regeneration fallback count mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("pre-N10 specification-repair primary SHA mismatch")
        if observed_id != (
            "pre_n10_specification_repair_primary_v1:"
            + expected_sha[:20]
        ):
            raise ValueError("pre-N10 specification-repair primary ID mismatch")
        return self


def _repair_plan_for_claim(
    *,
    claim: NoveltyClaim,
    binding_reason_codes: list[str],
    source_reason_codes: list[str],
) -> tuple[SpecificationClaimRepairPlan | None, list[str]]:
    reasons = list(
        dict.fromkeys([*binding_reason_codes, *source_reason_codes])
    )
    unsupported = [
        reason
        for reason in reasons
        if reason not in _SUPPORTED_COMPLETION_REASON_TO_FIELD
    ]
    if unsupported:
        return None, unsupported

    editable_fields: list[str] = []
    for reason in binding_reason_codes:
        field = _SUPPORTED_COMPLETION_REASON_TO_FIELD.get(reason)
        if field is not None and field not in editable_fields:
            editable_fields.append(field)
    if not editable_fields:
        return None, ["no_supported_zero_delta_completion_field"]

    payload = claim.model_dump(mode="json")
    for field in editable_fields:
        if str(payload[field] or "").strip():
            return None, ["completion_source_field_not_empty:" + field]

    immutable_fields = [
        "claim_id",
        "claim_text",
        "rationale",
        "prior_art_identity_terms",
        "relation_nucleus_terms",
        "novelty_selection_role",
        "candidate_hypothesis_id",
        "final_hypothesis_id",
    ] + [
        field
        for field in (
            "required_bridge",
            "predicted_observation",
            "falsification_condition",
        )
        if field not in editable_fields
    ]

    return (
        SpecificationClaimRepairPlan(
            claim_id=claim.claim_id,
            repair_action="CONTRACT_COMPLETION_REPAIR",
            source_claim_snapshot_sha256=_sha256_json(
                claim.model_dump(mode="json")
            ),
            editable_fields=editable_fields,  # type: ignore[arg-type]
            fill_only_fields=editable_fields,  # type: ignore[arg-type]
            immutable_fields=immutable_fields,
            source_claim_text=claim.text,
            source_required_bridge=claim.required_bridge,
            source_predicted_observation=claim.predicted_observation,
            source_falsification_condition=claim.falsification_condition,
            source_prior_art_identity_terms=list(
                claim.prior_art_identity_terms
            ),
            source_relation_nucleus_terms=list(
                claim.relation_nucleus_terms
            ),
            source_reason_codes=reasons,
        ),
        [],
    )


def _materialize_repaired_claim(
    *,
    source: NoveltyClaim,
    result: SpecificationRepairClaimResult,
) -> NoveltyClaim:
    if result.status != "MATERIALIZED_R1":
        raise ValueError("cannot materialize non-accepted specification repair")
    payload = source.model_dump(mode="json")
    payload["required_bridge"] = result.accepted_required_bridge
    payload["predicted_observation"] = result.accepted_predicted_observation
    payload["falsification_condition"] = (
        result.accepted_falsification_condition
    )
    repaired = NoveltyClaim.model_validate(payload)
    if repaired.text != source.text:
        raise ValueError("specification repair mutated claim text")
    if repaired.prior_art_identity_terms != source.prior_art_identity_terms:
        raise ValueError("specification repair mutated identity terms")
    if repaired.novelty_selection_role != source.novelty_selection_role:
        raise ValueError("specification repair mutated novelty role")
    return repaired


def _rewrite_query_plan(
    *,
    source: LiteratureQueryPlan,
    replacements: dict[str, NoveltyClaim],
) -> LiteratureQueryPlan:
    payload = source.model_dump(mode="json")
    changed = 0
    for group in payload["claims"]:
        rewritten = []
        for claim in group["claims"]:
            claim_id = claim["claim_id"]
            replacement = replacements.get(claim_id)
            if replacement is None:
                rewritten.append(claim)
            else:
                rewritten.append(replacement.model_dump(mode="json"))
                changed += 1
        group["claims"] = rewritten
    if changed != len(replacements):
        raise ValueError("repaired claims do not map exactly into query plan")
    payload.pop("plan_id")
    payload.pop("plan_sha256")
    digest = _sha256_json(payload)
    return LiteratureQueryPlan(
        **payload,
        plan_id="literature_query_plan:" + digest[:20],
        plan_sha256=digest,
    )


def execute_pre_n10_specification_repair_primary_v1(
    *,
    portfolio_path: Path,
    query_plan_path: Path,
    contract_report: PreN10ScientificContractReportV1,
    repair_backend: SpecificationRepairBackend,
    output_root: Path,
) -> tuple[
    LiteratureQueryPlan,
    PreN10ScientificContractReportV1,
    PreN10SpecificationRepairPrimaryReportV1,
]:
    portfolio_file = portfolio_path.expanduser().resolve()
    query_file = query_plan_path.expanduser().resolve()
    root = output_root.expanduser().resolve()

    if _sha256_file(portfolio_file) != contract_report.source_portfolio_sha256:
        raise ValueError("source portfolio changed after V_pre freeze")
    if _sha256_file(query_file) != contract_report.source_query_plan_sha256:
        raise ValueError("source query plan changed after V_pre freeze")

    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_file.read_text(encoding="utf-8")
    )
    query_plan = LiteratureQueryPlan.model_validate_json(
        query_file.read_text(encoding="utf-8")
    )
    if portfolio.portfolio_id != contract_report.source_portfolio_id:
        raise ValueError("portfolio/contract ID mismatch")
    if query_plan.plan_id != contract_report.source_query_plan_id:
        raise ValueError("query-plan/contract ID mismatch")

    cards = {row.hypothesis_id: row for row in portfolio.hypotheses}
    groups = {row.hypothesis_id: row for row in query_plan.claims}
    contract_by_hypothesis = {
        row.hypothesis_id: row for row in contract_report.hypotheses
    }
    expected_ids = set(cards)
    if set(groups) != expected_ids or set(contract_by_hypothesis) != expected_ids:
        raise ValueError(
            "portfolio/query-plan/contract hypothesis populations differ"
        )

    replacements: dict[str, NoveltyClaim] = {}
    preliminary: dict[
        str,
        list[PreN10SpecificationRepairClaimResultV1],
    ] = {}

    for hypothesis_id in [row.hypothesis_id for row in portfolio.hypotheses]:
        contract_h = contract_by_hypothesis[hypothesis_id]
        hints = [
            claim.router_hint
            for claim in contract_h.claims
            if claim.contract_status == "NOT_READY_FOR_N10_CONTRACT"
        ]
        if not hints:
            raise ValueError(
                "specification-repair primary received already-ready hypothesis: "
                + hypothesis_id
            )
        if any(hint != "SPECIFICATION_REPAIR_REVIEW" for hint in hints):
            raise ValueError(
                "specification-repair primary requires specification-repair "
                "population only; observed="
                + repr(sorted(set(hints)))
            )

        claims_by_id = {
            row.claim_id: row for row in groups[hypothesis_id].claims
        }
        results: list[PreN10SpecificationRepairClaimResultV1] = []

        for contract_claim in contract_h.claims:
            if contract_claim.contract_status == "READY_FOR_N10_CONTRACT":
                continue
            claim = claims_by_id[contract_claim.claim_id]
            before_sha = _sha256_json(claim.model_dump(mode="json"))
            plan, unsupported = _repair_plan_for_claim(
                claim=claim,
                binding_reason_codes=list(
                    contract_claim.binding_contract_reason_codes
                ),
                source_reason_codes=list(
                    contract_claim.source_contract_reason_codes
                ),
            )
            if plan is None:
                results.append(
                    PreN10SpecificationRepairClaimResultV1(
                        hypothesis_id=hypothesis_id,
                        claim_id=claim.claim_id,
                        status="UNSUPPORTED_AUTOMATIC_REPAIR",
                        source_reason_codes=list(
                            dict.fromkeys(
                                [
                                    *contract_claim.binding_contract_reason_codes,
                                    *contract_claim.source_contract_reason_codes,
                                ]
                            )
                        ),
                        unsupported_reason_codes=unsupported,
                        source_claim_sha256_before=before_sha,
                    )
                )
                continue

            execution = execute_claim_repair(
                plan=plan,
                backend=repair_backend,  # type: ignore[arg-type]
            )
            status_map = {
                "MATERIALIZED_R1":
                    "MATERIALIZED_ZERO_DELTA_SPECIFICATION_REPAIR",
                "REJECTED_DETERMINISTIC_POLICY":
                    "REJECTED_DETERMINISTIC_POLICY",
                "REJECTED_SEMANTIC_DELTA": "REJECTED_SEMANTIC_DELTA",
                "GENERATION_CALL_FAILED": "GENERATION_CALL_FAILED",
                "AUDIT_CALL_FAILED": "AUDIT_CALL_FAILED",
            }
            status = status_map[execution.status]
            after_sha = None
            if execution.status == "MATERIALIZED_R1":
                repaired = _materialize_repaired_claim(
                    source=claim,
                    result=execution,
                )
                replacements[claim.claim_id] = repaired
                after_sha = _sha256_json(repaired.model_dump(mode="json"))

            results.append(
                PreN10SpecificationRepairClaimResultV1(
                    hypothesis_id=hypothesis_id,
                    claim_id=claim.claim_id,
                    status=status,  # type: ignore[arg-type]
                    source_reason_codes=list(
                        dict.fromkeys(
                            [
                                *contract_claim.binding_contract_reason_codes,
                                *contract_claim.source_contract_reason_codes,
                            ]
                        )
                    ),
                    repair_plan=plan,
                    repair_execution=execution,
                    source_claim_sha256_before=before_sha,
                    source_claim_sha256_after=after_sha,
                )
            )
        preliminary[hypothesis_id] = results

    repaired_plan = _rewrite_query_plan(
        source=query_plan,
        replacements=replacements,
    )
    repaired_path = root / "repaired.claims_queries.json"
    _write_exact_or_validate(repaired_path, repaired_plan)

    post_contract = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_file,
        query_plan_path=repaired_path,
        claim_decomposition_request_count=0,
    )
    post_contract_path = root / "contract.after_primary.json"
    _write_exact_or_validate(post_contract_path, post_contract)

    post_by_hypothesis = {
        row.hypothesis_id: row for row in post_contract.hypotheses
    }
    hypothesis_results: list[
        PreN10SpecificationRepairHypothesisResultV1
    ] = []
    for source_h in contract_report.hypotheses:
        post_h = post_by_hypothesis[source_h.hypothesis_id]
        recovered = post_h.contract_status == "READY_FOR_N10"
        hypothesis_results.append(
            PreN10SpecificationRepairHypothesisResultV1(
                hypothesis_id=source_h.hypothesis_id,
                source_contract_status=source_h.contract_status,
                repair_results=preliminary[source_h.hypothesis_id],
                post_contract_status=post_h.contract_status,
                recovered_for_n10=recovered,
                regeneration_fallback_required=not recovered,
            )
        )

    claim_results = [
        claim
        for hypothesis in hypothesis_results
        for claim in hypothesis.repair_results
    ]
    counts = Counter(row.status for row in claim_results)
    body = {
        "schema_version": "pre-n10-specification-repair-primary-report-v1",
        "source_contract_report_id": contract_report.report_id,
        "source_contract_report_sha256": contract_report.report_sha256,
        "source_query_plan_id": query_plan.plan_id,
        "source_query_plan_sha256": query_plan.plan_sha256,
        "repaired_query_plan_id": repaired_plan.plan_id,
        "repaired_query_plan_sha256": repaired_plan.plan_sha256,
        "post_contract_report_id": post_contract.report_id,
        "post_contract_report_sha256": post_contract.report_sha256,
        "hypotheses": [row.model_dump(mode="json") for row in hypothesis_results],
        "hypothesis_count": len(hypothesis_results),
        "materialized_repair_count": counts.get(
            "MATERIALIZED_ZERO_DELTA_SPECIFICATION_REPAIR", 0
        ),
        "unsupported_automatic_repair_count": counts.get(
            "UNSUPPORTED_AUTOMATIC_REPAIR", 0
        ),
        "deterministic_reject_count": counts.get(
            "REJECTED_DETERMINISTIC_POLICY", 0
        ),
        "semantic_reject_count": counts.get("REJECTED_SEMANTIC_DELTA", 0),
        "generation_failure_count": counts.get("GENERATION_CALL_FAILED", 0),
        "audit_failure_count": counts.get("AUDIT_CALL_FAILED", 0),
        "repair_generation_llm_call_count": sum(
            (
                row.repair_execution.generation_llm_calls
                if row.repair_execution is not None
                else 0
            )
            for row in claim_results
        ),
        "repair_audit_llm_call_count": sum(
            (
                row.repair_execution.audit_llm_calls
                if row.repair_execution is not None
                else 0
            )
            for row in claim_results
        ),
        "recovered_for_n10_count": sum(
            row.recovered_for_n10 for row in hypothesis_results
        ),
        "regeneration_fallback_required_count": sum(
            row.regeneration_fallback_required for row in hypothesis_results
        ),
        "claim_generation_llm_calls": 0,
        "retrieval_performed": False,
        "novelty_assessment_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "regeneration_performed": False,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    report = PreN10SpecificationRepairPrimaryReportV1(
        **body,
        report_id=(
            "pre_n10_specification_repair_primary_v1:" + digest[:20]
        ),
        report_sha256=digest,
    )
    _write_exact_or_validate(root / "primary.report.json", report)
    return repaired_plan, post_contract, report


__all__ = [
    "PreN10SpecificationRepairClaimResultV1",
    "PreN10SpecificationRepairHypothesisResultV1",
    "PreN10SpecificationRepairPrimaryReportV1",
    "SpecificationRepairBackend",
    "SpecificationRepairPrimaryStatusV1",
    "execute_pre_n10_specification_repair_primary_v1",
]
