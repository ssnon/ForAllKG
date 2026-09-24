from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

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
from pipeline_core.discovery.prospective_routed_primary_materializer_v2 import (
    SourceAlignmentAuditBackend,
    SourceAlignmentAuditDraft,
    _aligned_query_plan,
    _audit_passes,
)
from pipeline_core.discovery.prospective_routed_route_compiler_v2 import (
    SourceAlignmentCandidatePairV2,
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
                "existing write-once pre-N10 primary artifact differs: "
                + str(path)
            )
        return hashlib.sha256(observed).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(expected)
    return hashlib.sha256(expected).hexdigest()


def _token_value(event: object, name: str) -> int | None:
    value = getattr(event, name, None)
    return int(value) if value is not None else None


def _normalize_observable(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _alignment_candidates_for_card(
    card: object,
) -> list[SourceAlignmentCandidatePairV2]:
    pairs: list[SourceAlignmentCandidatePairV2] = []
    seen: set[tuple[str, str]] = set()

    for prediction in card.predicted_observations:
        left = _normalize_observable(prediction.observable)
        if not left:
            continue
        for falsifier in card.falsification_criteria:
            right = _normalize_observable(falsifier.observable)
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
                SourceAlignmentCandidatePairV2(
                    prediction_observation_id=prediction.observation_id,
                    prediction_observable=prediction.observable,
                    falsification_criterion_id=falsifier.criterion_id,
                    falsifier_observable=falsifier.observable,
                    falsifying_outcome=falsifier.falsifying_outcome,
                )
            )
    return pairs


AlignmentStatusV1 = Literal[
    "MATERIALIZED_ZERO_DELTA_SOURCE_ALIGNMENT",
    "UNAVAILABLE_NO_CANDIDATE",
    "UNAVAILABLE_AMBIGUOUS_CANDIDATES",
    "REJECTED_SEMANTIC_DELTA",
    "AUDIT_CALL_FAILED",
]


class PreN10SourceAlignmentClaimResultV1(StrictModel):
    hypothesis_id: str
    claim_id: str
    status: AlignmentStatusV1
    candidate_count: int = Field(ge=0)
    selected_candidate: SourceAlignmentCandidatePairV2 | None = None

    semantic_audit_performed: bool
    semantic_audit_passed: bool | None = None
    semantic_audit: SourceAlignmentAuditDraft | None = None
    audit_llm_calls: int = Field(ge=0, le=1)
    audit_input_tokens: int | None = Field(default=None, ge=0)
    audit_output_tokens: int | None = Field(default=None, ge=0)

    source_claim_sha256_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_claim_sha256_after: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    new_text_generated: Literal[False] = False
    source_surfaces_only: Literal[True] = True
    external_novelty_outcome_used: Literal[False] = False
    n10_outcome_used: Literal[False] = False


class PreN10SourceAlignmentHypothesisResultV1(StrictModel):
    hypothesis_id: str
    source_contract_status: str
    dominant_router_hint: Literal[
        "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"
    ] = "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"

    source_alignment_results: list[
        PreN10SourceAlignmentClaimResultV1
    ] = Field(default_factory=list)

    post_contract_status: str
    recovered_for_n10: bool
    regeneration_fallback_required: bool

    @model_validator(mode="after")
    def validate_state(self) -> "PreN10SourceAlignmentHypothesisResultV1":
        if self.recovered_for_n10 != (
            self.post_contract_status == "READY_FOR_N10"
        ):
            raise ValueError("recovered_for_n10 mismatch")
        if self.regeneration_fallback_required == self.recovered_for_n10:
            raise ValueError(
                "regeneration fallback must be exact complement of recovery"
            )
        return self


class PreN10SourceAlignmentPrimaryReportV1(StrictModel):
    schema_version: Literal[
        "pre-n10-source-alignment-primary-report-v1"
    ] = "pre-n10-source-alignment-primary-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_contract_report_id: str
    source_contract_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_query_plan_id: str
    source_query_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    aligned_query_plan_id: str
    aligned_query_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    post_contract_report_id: str
    post_contract_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    hypotheses: list[PreN10SourceAlignmentHypothesisResultV1]
    hypothesis_count: int = Field(ge=0)

    materialized_alignment_count: int = Field(ge=0)
    unavailable_alignment_count: int = Field(ge=0)
    ambiguous_alignment_count: int = Field(ge=0)
    semantic_reject_count: int = Field(ge=0)
    audit_failure_count: int = Field(ge=0)

    recovered_for_n10_count: int = Field(ge=0)
    regeneration_fallback_required_count: int = Field(ge=0)
    audit_llm_call_count: int = Field(ge=0)

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
    def validate_report(self) -> "PreN10SourceAlignmentPrimaryReportV1":
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError("hypothesis_count mismatch")

        claim_results = [
            claim
            for hypothesis in self.hypotheses
            for claim in hypothesis.source_alignment_results
        ]
        counts = Counter(row.status for row in claim_results)

        expected = {
            "MATERIALIZED_ZERO_DELTA_SOURCE_ALIGNMENT":
                self.materialized_alignment_count,
            "UNAVAILABLE_NO_CANDIDATE":
                self.unavailable_alignment_count,
            "UNAVAILABLE_AMBIGUOUS_CANDIDATES":
                self.ambiguous_alignment_count,
            "REJECTED_SEMANTIC_DELTA":
                self.semantic_reject_count,
            "AUDIT_CALL_FAILED":
                self.audit_failure_count,
        }
        for status, count in expected.items():
            if counts.get(status, 0) != count:
                raise ValueError("alignment status count mismatch: " + status)

        if self.recovered_for_n10_count != sum(
            row.recovered_for_n10 for row in self.hypotheses
        ):
            raise ValueError("recovered_for_n10_count mismatch")
        if self.regeneration_fallback_required_count != sum(
            row.regeneration_fallback_required for row in self.hypotheses
        ):
            raise ValueError("regeneration fallback count mismatch")
        if self.audit_llm_call_count != sum(
            row.audit_llm_calls for row in claim_results
        ):
            raise ValueError("audit LLM call count mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("pre-N10 source-alignment primary SHA mismatch")
        if observed_id != (
            "pre_n10_source_alignment_primary_v1:"
            + expected_sha[:20]
        ):
            raise ValueError("pre-N10 source-alignment primary ID mismatch")
        return self


def execute_pre_n10_source_alignment_primary_v1(
    *,
    portfolio_path: Path,
    query_plan_path: Path,
    contract_report: PreN10ScientificContractReportV1,
    audit_backend: SourceAlignmentAuditBackend,
    output_root: Path,
) -> tuple[
    LiteratureQueryPlan,
    PreN10ScientificContractReportV1,
    PreN10SourceAlignmentPrimaryReportV1,
]:
    portfolio_file = portfolio_path.expanduser().resolve()
    query_file = query_plan_path.expanduser().resolve()
    root = output_root.expanduser().resolve()

    if _sha256_file(portfolio_file) != (
        contract_report.source_portfolio_sha256
    ):
        raise ValueError("source portfolio changed after V_pre freeze")
    if _sha256_file(query_file) != (
        contract_report.source_query_plan_sha256
    ):
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

    cards = {
        row.hypothesis_id: row
        for row in portfolio.hypotheses
    }
    groups = {
        row.hypothesis_id: row
        for row in query_plan.claims
    }
    contract_by_hypothesis = {
        row.hypothesis_id: row
        for row in contract_report.hypotheses
    }

    expected_ids = set(cards)
    if set(groups) != expected_ids or set(contract_by_hypothesis) != expected_ids:
        raise ValueError(
            "portfolio/query-plan/contract hypothesis populations differ"
        )

    current_plan = query_plan
    preliminary: dict[
        str,
        list[PreN10SourceAlignmentClaimResultV1],
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
                "source-alignment primary received already-ready hypothesis: "
                + hypothesis_id
            )
        if any(
            hint != "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"
            for hint in hints
        ):
            raise ValueError(
                "source-alignment primary requires source-alignment dominant "
                "population only; observed="
                + repr(sorted(set(hints)))
            )

        card = cards[hypothesis_id]
        group = next(
            row
            for row in current_plan.claims
            if row.hypothesis_id == hypothesis_id
        )
        claims_by_id = {
            row.claim_id: row
            for row in group.claims
        }

        aligned: dict[str, NoveltyClaim] = {}
        results: list[PreN10SourceAlignmentClaimResultV1] = []

        for contract_claim in contract_h.claims:
            if contract_claim.contract_status == "READY_FOR_N10_CONTRACT":
                continue
            claim = claims_by_id[contract_claim.claim_id]
            before_sha = _sha256_json(claim.model_dump(mode="json"))
            candidates = _alignment_candidates_for_card(card)

            if not candidates:
                results.append(
                    PreN10SourceAlignmentClaimResultV1(
                        hypothesis_id=hypothesis_id,
                        claim_id=claim.claim_id,
                        status="UNAVAILABLE_NO_CANDIDATE",
                        candidate_count=0,
                        semantic_audit_performed=False,
                        audit_llm_calls=0,
                        source_claim_sha256_before=before_sha,
                    )
                )
                continue

            if len(candidates) != 1:
                results.append(
                    PreN10SourceAlignmentClaimResultV1(
                        hypothesis_id=hypothesis_id,
                        claim_id=claim.claim_id,
                        status="UNAVAILABLE_AMBIGUOUS_CANDIDATES",
                        candidate_count=len(candidates),
                        semantic_audit_performed=False,
                        audit_llm_calls=0,
                        source_claim_sha256_before=before_sha,
                    )
                )
                continue

            candidate = candidates[0]
            try:
                audit, event = audit_backend.audit(
                    claim=claim,
                    candidate=candidate,
                    candidate_card=card,
                )
            except Exception:
                results.append(
                    PreN10SourceAlignmentClaimResultV1(
                        hypothesis_id=hypothesis_id,
                        claim_id=claim.claim_id,
                        status="AUDIT_CALL_FAILED",
                        candidate_count=1,
                        selected_candidate=candidate,
                        semantic_audit_performed=True,
                        semantic_audit_passed=None,
                        audit_llm_calls=1,
                        source_claim_sha256_before=before_sha,
                    )
                )
                continue

            if not _audit_passes(audit):
                results.append(
                    PreN10SourceAlignmentClaimResultV1(
                        hypothesis_id=hypothesis_id,
                        claim_id=claim.claim_id,
                        status="REJECTED_SEMANTIC_DELTA",
                        candidate_count=1,
                        selected_candidate=candidate,
                        semantic_audit_performed=True,
                        semantic_audit_passed=False,
                        semantic_audit=audit,
                        audit_llm_calls=1,
                        audit_input_tokens=_token_value(
                            event,
                            "provider_input_tokens",
                        ),
                        audit_output_tokens=_token_value(
                            event,
                            "provider_output_tokens",
                        ),
                        source_claim_sha256_before=before_sha,
                    )
                )
                continue

            payload = claim.model_dump(mode="json")
            payload["predicted_observation"] = (
                candidate.prediction_observable
            )
            payload["falsification_condition"] = (
                candidate.falsifying_outcome
            )
            after = NoveltyClaim.model_validate(payload)
            after_sha = _sha256_json(after.model_dump(mode="json"))
            aligned[claim.claim_id] = after

            results.append(
                PreN10SourceAlignmentClaimResultV1(
                    hypothesis_id=hypothesis_id,
                    claim_id=claim.claim_id,
                    status="MATERIALIZED_ZERO_DELTA_SOURCE_ALIGNMENT",
                    candidate_count=1,
                    selected_candidate=candidate,
                    semantic_audit_performed=True,
                    semantic_audit_passed=True,
                    semantic_audit=audit,
                    audit_llm_calls=1,
                    audit_input_tokens=_token_value(
                        event,
                        "provider_input_tokens",
                    ),
                    audit_output_tokens=_token_value(
                        event,
                        "provider_output_tokens",
                    ),
                    source_claim_sha256_before=before_sha,
                    source_claim_sha256_after=after_sha,
                )
            )

        if aligned:
            current_plan = _aligned_query_plan(
                source=current_plan,
                candidate_hypothesis_id=hypothesis_id,
                aligned_claims=aligned,
            )
        preliminary[hypothesis_id] = results

    aligned_path = root / "aligned.claims_queries.json"
    aligned_sha = _write_exact_or_validate(
        aligned_path,
        current_plan,
    )

    post_contract = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_file,
        query_plan_path=aligned_path,
        claim_decomposition_request_count=0,
    )
    post_contract_path = root / "contract.after_primary.json"
    _write_exact_or_validate(
        post_contract_path,
        post_contract,
    )

    post_by_hypothesis = {
        row.hypothesis_id: row
        for row in post_contract.hypotheses
    }
    hypothesis_results: list[
        PreN10SourceAlignmentHypothesisResultV1
    ] = []
    for source_h in contract_report.hypotheses:
        post_h = post_by_hypothesis[source_h.hypothesis_id]
        recovered = post_h.contract_status == "READY_FOR_N10"
        hypothesis_results.append(
            PreN10SourceAlignmentHypothesisResultV1(
                hypothesis_id=source_h.hypothesis_id,
                source_contract_status=source_h.contract_status,
                source_alignment_results=preliminary[
                    source_h.hypothesis_id
                ],
                post_contract_status=post_h.contract_status,
                recovered_for_n10=recovered,
                regeneration_fallback_required=not recovered,
            )
        )

    claim_results = [
        claim
        for hypothesis in hypothesis_results
        for claim in hypothesis.source_alignment_results
    ]
    counts = Counter(row.status for row in claim_results)

    body = {
        "schema_version": "pre-n10-source-alignment-primary-report-v1",
        "source_contract_report_id": contract_report.report_id,
        "source_contract_report_sha256": contract_report.report_sha256,
        "source_query_plan_id": query_plan.plan_id,
        "source_query_plan_sha256": query_plan.plan_sha256,
        "aligned_query_plan_id": current_plan.plan_id,
        "aligned_query_plan_sha256": current_plan.plan_sha256,
        "post_contract_report_id": post_contract.report_id,
        "post_contract_report_sha256": post_contract.report_sha256,
        "hypotheses": [
            row.model_dump(mode="json")
            for row in hypothesis_results
        ],
        "hypothesis_count": len(hypothesis_results),
        "materialized_alignment_count": counts.get(
            "MATERIALIZED_ZERO_DELTA_SOURCE_ALIGNMENT",
            0,
        ),
        "unavailable_alignment_count": counts.get(
            "UNAVAILABLE_NO_CANDIDATE",
            0,
        ),
        "ambiguous_alignment_count": counts.get(
            "UNAVAILABLE_AMBIGUOUS_CANDIDATES",
            0,
        ),
        "semantic_reject_count": counts.get(
            "REJECTED_SEMANTIC_DELTA",
            0,
        ),
        "audit_failure_count": counts.get(
            "AUDIT_CALL_FAILED",
            0,
        ),
        "recovered_for_n10_count": sum(
            row.recovered_for_n10 for row in hypothesis_results
        ),
        "regeneration_fallback_required_count": sum(
            row.regeneration_fallback_required
            for row in hypothesis_results
        ),
        "audit_llm_call_count": sum(
            row.audit_llm_calls for row in claim_results
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
    report = PreN10SourceAlignmentPrimaryReportV1(
        **body,
        report_id=(
            "pre_n10_source_alignment_primary_v1:"
            + digest[:20]
        ),
        report_sha256=digest,
    )
    report_path = root / "primary.report.json"
    _write_exact_or_validate(report_path, report)

    if _sha256_file(aligned_path) != aligned_sha:
        raise ValueError("aligned query plan changed during primary execution")

    return current_plan, post_contract, report


__all__ = [
    "PreN10SourceAlignmentClaimResultV1",
    "PreN10SourceAlignmentHypothesisResultV1",
    "PreN10SourceAlignmentPrimaryReportV1",
    "execute_pre_n10_source_alignment_primary_v1",
]
