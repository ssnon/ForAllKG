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
from pipeline_core.discovery.prospective_routed_decomposition_primary import (
    DeterministicDecompositionAssessment,
    assess_deterministic_decomposition,
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
                "existing write-once pre-N10 decomposition artifact differs: "
                + str(path)
            )
        return hashlib.sha256(observed).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(expected)
    return hashlib.sha256(expected).hexdigest()


DecompositionPrimaryStatusV1 = Literal[
    "MATERIALIZED_DETERMINISTIC_DECOMPOSITION",
    "UNAVAILABLE_DETERMINISTIC_DECOMPOSITION",
]


class PreN10DecompositionClaimResultV1(StrictModel):
    hypothesis_id: str
    claim_id: str
    status: DecompositionPrimaryStatusV1
    assessment: DeterministicDecompositionAssessment
    scientific_content_added: Literal[False] = False
    new_claim_generated: Literal[False] = False
    llm_calls: Literal[0] = 0
    retrieval_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> "PreN10DecompositionClaimResultV1":
        expected = (
            "MATERIALIZED_DETERMINISTIC_DECOMPOSITION"
            if self.assessment.status
            == "DETERMINISTIC_DECOMPOSITION_AVAILABLE"
            else "UNAVAILABLE_DETERMINISTIC_DECOMPOSITION"
        )
        if self.status != expected:
            raise ValueError("decomposition result/assessment status mismatch")
        return self


class PreN10DecompositionHypothesisResultV1(StrictModel):
    hypothesis_id: str
    source_contract_status: str
    dominant_router_hint: Literal[
        "DECOMPOSE_OR_REGENERATE_REVIEW"
    ] = "DECOMPOSE_OR_REGENERATE_REVIEW"
    source_claim_ids: list[str]
    output_claim_ids: list[str]
    decomposition_results: list[PreN10DecompositionClaimResultV1] = Field(
        default_factory=list
    )
    post_contract_status: str
    recovered_for_n10: bool
    regeneration_fallback_required: bool

    @model_validator(mode="after")
    def validate_state(self) -> "PreN10DecompositionHypothesisResultV1":
        if self.recovered_for_n10 != (
            self.post_contract_status == "READY_FOR_N10"
        ):
            raise ValueError("recovered_for_n10 mismatch")
        if self.regeneration_fallback_required == self.recovered_for_n10:
            raise ValueError(
                "regeneration fallback must be exact complement of recovery"
            )
        return self


class PreN10DecompositionPrimaryReportV1(StrictModel):
    schema_version: Literal[
        "pre-n10-decomposition-primary-report-v1"
    ] = "pre-n10-decomposition-primary-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_contract_report_id: str
    source_contract_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_query_plan_id: str
    source_query_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decomposed_query_plan_id: str
    decomposed_query_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    post_contract_report_id: str
    post_contract_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    hypotheses: list[PreN10DecompositionHypothesisResultV1]
    hypothesis_count: int = Field(ge=0)
    materialized_decomposition_count: int = Field(ge=0)
    unavailable_decomposition_count: int = Field(ge=0)
    removed_composite_claim_count: int = Field(ge=0)
    recovered_for_n10_count: int = Field(ge=0)
    regeneration_fallback_required_count: int = Field(ge=0)

    deterministic_topology_only: Literal[True] = True
    decomposition_llm_calls: Literal[0] = 0
    audit_llm_calls: Literal[0] = 0
    scientific_content_added: Literal[False] = False
    new_claim_generated: Literal[False] = False
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
    def validate_report(self) -> "PreN10DecompositionPrimaryReportV1":
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError("hypothesis_count mismatch")
        results = [
            claim
            for hypothesis in self.hypotheses
            for claim in hypothesis.decomposition_results
        ]
        materialized = sum(
            row.status == "MATERIALIZED_DETERMINISTIC_DECOMPOSITION"
            for row in results
        )
        unavailable = sum(
            row.status == "UNAVAILABLE_DETERMINISTIC_DECOMPOSITION"
            for row in results
        )
        if self.materialized_decomposition_count != materialized:
            raise ValueError("materialized_decomposition_count mismatch")
        if self.unavailable_decomposition_count != unavailable:
            raise ValueError("unavailable_decomposition_count mismatch")
        removed = sum(
            len(set(row.source_claim_ids) - set(row.output_claim_ids))
            for row in self.hypotheses
        )
        if self.removed_composite_claim_count != removed:
            raise ValueError("removed_composite_claim_count mismatch")
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
            raise ValueError("pre-N10 decomposition primary SHA mismatch")
        if observed_id != (
            "pre_n10_decomposition_primary_v1:" + expected_sha[:20]
        ):
            raise ValueError("pre-N10 decomposition primary ID mismatch")
        return self


def _rewrite_query_plan_without_claims(
    *,
    source: LiteratureQueryPlan,
    removed_by_hypothesis: dict[str, set[str]],
) -> LiteratureQueryPlan:
    payload = source.model_dump(mode="json")
    observed_removed = 0
    expected_removed = sum(len(values) for values in removed_by_hypothesis.values())
    all_removed: set[str] = set()
    for group in payload["claims"]:
        removed = removed_by_hypothesis.get(group["hypothesis_id"], set())
        if not removed:
            continue
        before = list(group["claims"])
        group["claims"] = [
            claim for claim in before if claim["claim_id"] not in removed
        ]
        observed_removed += len(before) - len(group["claims"])
        all_removed.update(removed)
    if observed_removed != expected_removed:
        raise ValueError("decomposition removals do not map exactly into query plan")
    payload["queries"] = [
        query
        for query in payload.get("queries", [])
        if query.get("claim_id") not in all_removed
    ]
    payload.pop("plan_id")
    payload.pop("plan_sha256")
    digest = _sha256_json(payload)
    return LiteratureQueryPlan(
        **payload,
        plan_id="literature_query_plan:" + digest[:20],
        plan_sha256=digest,
    )


def execute_pre_n10_decomposition_primary_v1(
    *,
    portfolio_path: Path,
    query_plan_path: Path,
    contract_report: PreN10ScientificContractReportV1,
    output_root: Path,
) -> tuple[
    LiteratureQueryPlan,
    PreN10ScientificContractReportV1,
    PreN10DecompositionPrimaryReportV1,
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

    removed_by_hypothesis: dict[str, set[str]] = {}
    preliminary: dict[
        str,
        list[PreN10DecompositionClaimResultV1],
    ] = {}
    source_claim_ids: dict[str, list[str]] = {}

    for hypothesis_id in [row.hypothesis_id for row in portfolio.hypotheses]:
        contract_h = contract_by_hypothesis[hypothesis_id]
        hints = [
            claim.router_hint
            for claim in contract_h.claims
            if claim.contract_status == "NOT_READY_FOR_N10_CONTRACT"
        ]
        if not hints:
            raise ValueError(
                "decomposition primary received already-ready hypothesis: "
                + hypothesis_id
            )
        if any(hint != "DECOMPOSE_OR_REGENERATE_REVIEW" for hint in hints):
            raise ValueError(
                "decomposition primary requires decomposition population only; observed="
                + repr(sorted(set(hints)))
            )

        claims_by_id: dict[str, NoveltyClaim] = {
            row.claim_id: row for row in groups[hypothesis_id].claims
        }
        source_claim_ids[hypothesis_id] = list(claims_by_id)
        results: list[PreN10DecompositionClaimResultV1] = []

        for contract_claim in contract_h.claims:
            if contract_claim.contract_status == "READY_FOR_N10_CONTRACT":
                continue
            claim = claims_by_id[contract_claim.claim_id]
            assessment = assess_deterministic_decomposition(
                composite_claim=claim,
                source_claims_by_id=claims_by_id,
            )
            results.append(
                PreN10DecompositionClaimResultV1(
                    hypothesis_id=hypothesis_id,
                    claim_id=claim.claim_id,
                    status=(
                        "MATERIALIZED_DETERMINISTIC_DECOMPOSITION"
                        if assessment.status
                        == "DETERMINISTIC_DECOMPOSITION_AVAILABLE"
                        else "UNAVAILABLE_DETERMINISTIC_DECOMPOSITION"
                    ),
                    assessment=assessment,
                )
            )

        if results and all(
            row.status == "MATERIALIZED_DETERMINISTIC_DECOMPOSITION"
            for row in results
        ):
            removed_by_hypothesis[hypothesis_id] = {
                row.claim_id for row in results
            }
        else:
            removed_by_hypothesis[hypothesis_id] = set()
        preliminary[hypothesis_id] = results

    decomposed_plan = _rewrite_query_plan_without_claims(
        source=query_plan,
        removed_by_hypothesis=removed_by_hypothesis,
    )
    decomposed_path = root / "decomposed.claims_queries.json"
    _write_exact_or_validate(decomposed_path, decomposed_plan)

    post_contract = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_file,
        query_plan_path=decomposed_path,
        claim_decomposition_request_count=0,
    )
    post_contract_path = root / "contract.after_primary.json"
    _write_exact_or_validate(post_contract_path, post_contract)

    post_by_hypothesis = {
        row.hypothesis_id: row for row in post_contract.hypotheses
    }
    output_groups = {
        row.hypothesis_id: row for row in decomposed_plan.claims
    }
    hypothesis_results: list[PreN10DecompositionHypothesisResultV1] = []
    for source_h in contract_report.hypotheses:
        post_h = post_by_hypothesis[source_h.hypothesis_id]
        recovered = post_h.contract_status == "READY_FOR_N10"
        hypothesis_results.append(
            PreN10DecompositionHypothesisResultV1(
                hypothesis_id=source_h.hypothesis_id,
                source_contract_status=source_h.contract_status,
                source_claim_ids=source_claim_ids[source_h.hypothesis_id],
                output_claim_ids=[
                    row.claim_id
                    for row in output_groups[source_h.hypothesis_id].claims
                ],
                decomposition_results=preliminary[source_h.hypothesis_id],
                post_contract_status=post_h.contract_status,
                recovered_for_n10=recovered,
                regeneration_fallback_required=not recovered,
            )
        )

    results = [
        claim
        for hypothesis in hypothesis_results
        for claim in hypothesis.decomposition_results
    ]
    body = {
        "schema_version": "pre-n10-decomposition-primary-report-v1",
        "source_contract_report_id": contract_report.report_id,
        "source_contract_report_sha256": contract_report.report_sha256,
        "source_query_plan_id": query_plan.plan_id,
        "source_query_plan_sha256": query_plan.plan_sha256,
        "decomposed_query_plan_id": decomposed_plan.plan_id,
        "decomposed_query_plan_sha256": decomposed_plan.plan_sha256,
        "post_contract_report_id": post_contract.report_id,
        "post_contract_report_sha256": post_contract.report_sha256,
        "hypotheses": [row.model_dump(mode="json") for row in hypothesis_results],
        "hypothesis_count": len(hypothesis_results),
        "materialized_decomposition_count": sum(
            row.status == "MATERIALIZED_DETERMINISTIC_DECOMPOSITION"
            for row in results
        ),
        "unavailable_decomposition_count": sum(
            row.status == "UNAVAILABLE_DETERMINISTIC_DECOMPOSITION"
            for row in results
        ),
        "removed_composite_claim_count": sum(
            len(values) for values in removed_by_hypothesis.values()
        ),
        "recovered_for_n10_count": sum(
            row.recovered_for_n10 for row in hypothesis_results
        ),
        "regeneration_fallback_required_count": sum(
            row.regeneration_fallback_required for row in hypothesis_results
        ),
        "deterministic_topology_only": True,
        "decomposition_llm_calls": 0,
        "audit_llm_calls": 0,
        "scientific_content_added": False,
        "new_claim_generated": False,
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
    report = PreN10DecompositionPrimaryReportV1(
        **body,
        report_id="pre_n10_decomposition_primary_v1:" + digest[:20],
        report_sha256=digest,
    )
    _write_exact_or_validate(root / "primary.report.json", report)
    return decomposed_plan, post_contract, report


__all__ = [
    "DecompositionPrimaryStatusV1",
    "PreN10DecompositionClaimResultV1",
    "PreN10DecompositionHypothesisResultV1",
    "PreN10DecompositionPrimaryReportV1",
    "execute_pre_n10_decomposition_primary_v1",
]
