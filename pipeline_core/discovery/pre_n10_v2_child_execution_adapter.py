from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import LiteratureQueryPlan
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.pre_n10_scientific_contract_v2 import (
    PreN10ScientificContractReportV2,
)
from pipeline_core.discovery.preverifier_contract_gate_v2 import RouterHint


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
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PreN10V2ChildExecutionClaimAdapterV1(StrictModel):
    hypothesis_id: str
    claim_id: str
    claim_rank: int = Field(ge=1)
    kind: str
    importance: str
    novelty_selection_role: str | None = None
    binding_contract_reason_codes: list[str] = Field(default_factory=list)
    source_contract_reason_codes: list[str] = Field(default_factory=list)
    contract_status: Literal[
        "READY_FOR_N10_CONTRACT",
        "NOT_READY_FOR_N10_CONTRACT",
    ]
    router_hint: RouterHint
    atomic_kind_supported: bool
    stable_source_ids_are_authority: Literal[True] = True
    exact_text_reconstruction_used_for_authority: Literal[False] = False
    legacy_child_execution_only: Literal[True] = True

    @model_validator(mode="after")
    def validate_route(self) -> "PreN10V2ChildExecutionClaimAdapterV1":
        ready = self.contract_status == "READY_FOR_N10_CONTRACT"
        if (self.router_hint == "PROCEED_TO_LITERAL_ENDPOINT_BINDING") != ready:
            raise ValueError("V2 child execution adapter route/status mismatch")
        return self


class PreN10V2ChildExecutionHypothesisAdapterV1(StrictModel):
    hypothesis_id: str
    claim_count: int = Field(ge=0)
    ready_claim_count: int = Field(ge=0)
    novelty_bearing_ready_claim_count: int = Field(ge=0)
    contract_status: Literal[
        "READY_FOR_N10",
        "REQUIRES_PRE_N10_INTERVENTION",
    ]
    claims: list[PreN10V2ChildExecutionClaimAdapterV1] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_counts(self) -> "PreN10V2ChildExecutionHypothesisAdapterV1":
        if self.claim_count != len(self.claims):
            raise ValueError("V2 child adapter claim_count mismatch")
        ready = sum(
            row.contract_status == "READY_FOR_N10_CONTRACT"
            for row in self.claims
        )
        if self.ready_claim_count != ready:
            raise ValueError("V2 child adapter ready count mismatch")
        novelty_ready = sum(
            row.contract_status == "READY_FOR_N10_CONTRACT"
            and row.novelty_selection_role == "NOVELTY_BEARING"
            for row in self.claims
        )
        if self.novelty_bearing_ready_claim_count != novelty_ready:
            raise ValueError("V2 child adapter novelty-ready count mismatch")
        expected_ready = bool(
            self.claim_count > 0
            and ready == self.claim_count
            and novelty_ready > 0
        )
        if (self.contract_status == "READY_FOR_N10") != expected_ready:
            raise ValueError("V2 child adapter hypothesis status mismatch")
        return self


class PreN10V2ChildExecutionContractAdapterV1(StrictModel):
    schema_version: Literal[
        "pre-n10-v2-child-execution-adapter-v1"
    ] = "pre-n10-v2-child-execution-adapter-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_authority_report_id: str
    source_authority_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_authority_query_plan_id: str
    source_authority_query_plan_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    source_portfolio_path: str
    source_portfolio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_portfolio_id: str
    source_query_plan_path: str
    source_query_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_query_plan_id: str

    hypotheses: list[PreN10V2ChildExecutionHypothesisAdapterV1]
    hypothesis_count: Literal[1] = 1

    stable_source_ids_are_authority: Literal[True] = True
    exact_text_reconstruction_used_for_authority: Literal[False] = False
    exact_text_diagnostics_used_for_routing: Literal[False] = False
    legacy_child_execution_only: Literal[True] = True
    scientific_authority: Literal[False] = False
    production_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreN10V2ChildExecutionContractAdapterV1":
        if len(self.hypotheses) != 1:
            raise ValueError(
                "V2 child execution adapter requires one hypothesis"
            )
        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("V2 child execution adapter SHA mismatch")
        if observed_id != (
            "pre_n10_v2_child_execution_adapter:" + expected_sha[:20]
        ):
            raise ValueError("V2 child execution adapter ID mismatch")
        return self


def _source_execution_reason_codes(row: object) -> list[str]:
    reasons: list[str] = []
    if not row.atomic_kind_supported:
        reasons.append("unsupported_atomic_claim_kind:" + str(row.kind))
    if row.source_reference_status != "READY":
        reasons.append(
            "stable_source_reference_status:"
            + str(row.source_reference_status)
        )
        reasons.extend(
            "stable_source_reference_reason:" + str(reason)
            for reason in row.source_reference_reason_codes
        )
    return list(dict.fromkeys(reasons))


def build_pre_n10_v2_child_execution_adapter_v1(
    *,
    authority_report: PreN10ScientificContractReportV2,
    hypothesis_id: str,
    subset_portfolio_path: Path,
    subset_query_plan_path: Path,
) -> PreN10V2ChildExecutionContractAdapterV1:
    portfolio_file = subset_portfolio_path.expanduser().resolve()
    query_file = subset_query_plan_path.expanduser().resolve()

    if not portfolio_file.is_file():
        raise ValueError(
            "missing V2 child adapter subset portfolio: "
            + str(portfolio_file)
        )
    if not query_file.is_file():
        raise ValueError(
            "missing V2 child adapter subset query plan: "
            + str(query_file)
        )

    subset_portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_file.read_text(encoding="utf-8")
    )
    subset_plan = LiteratureQueryPlan.model_validate_json(
        query_file.read_text(encoding="utf-8")
    )

    if len(subset_portfolio.hypotheses) != 1:
        raise ValueError(
            "V2 child adapter subset portfolio must contain one hypothesis"
        )
    if subset_portfolio.hypotheses[0].hypothesis_id != hypothesis_id:
        raise ValueError(
            "V2 child adapter subset portfolio hypothesis mismatch"
        )
    if subset_plan.source_portfolio_id != subset_portfolio.portfolio_id:
        raise ValueError(
            "V2 child adapter subset query-plan/portfolio mismatch"
        )

    authority_hypotheses = [
        row
        for row in authority_report.hypotheses
        if row.hypothesis_id == hypothesis_id
    ]
    if len(authority_hypotheses) != 1:
        raise ValueError(
            "V2 child adapter authority hypothesis must resolve once"
        )
    authority_h = authority_hypotheses[0]

    authority_plan_file = Path(
        authority_report.source_query_plan_path
    ).expanduser().resolve()
    if not authority_plan_file.is_file():
        raise ValueError(
            "missing V2 authority source query plan: "
            + str(authority_plan_file)
        )
    if _sha256_file(authority_plan_file) != (
        authority_report.source_query_plan_file_sha256
    ):
        raise ValueError(
            "V2 authority source query plan changed before child adaptation"
        )
    authority_plan = LiteratureQueryPlan.model_validate_json(
        authority_plan_file.read_text(encoding="utf-8")
    )
    if authority_plan.plan_id != authority_report.source_query_plan_id:
        raise ValueError(
            "V2 authority query-plan ID mismatch in child adapter"
        )
    if authority_plan.plan_sha256 != authority_report.source_query_plan_sha256:
        raise ValueError(
            "V2 authority query-plan SHA mismatch in child adapter"
        )

    authority_groups = [
        row
        for row in authority_plan.claims
        if row.hypothesis_id == hypothesis_id
    ]
    subset_groups = [
        row
        for row in subset_plan.claims
        if row.hypothesis_id == hypothesis_id
    ]
    if len(authority_groups) != 1 or len(subset_groups) != 1:
        raise ValueError(
            "V2 child adapter claim group must resolve once"
        )
    if (
        authority_groups[0].model_dump(mode="json")
        != subset_groups[0].model_dump(mode="json")
    ):
        raise ValueError(
            "V2 child adapter subset claims differ from authority claims"
        )

    authority_queries = [
        row.model_dump(mode="json")
        for row in authority_plan.queries
        if row.hypothesis_id == hypothesis_id
    ]
    subset_queries = [
        row.model_dump(mode="json")
        for row in subset_plan.queries
        if row.hypothesis_id == hypothesis_id
    ]
    if authority_queries != subset_queries:
        raise ValueError(
            "V2 child adapter subset queries differ from authority queries"
        )

    subset_claim_ids = [
        row.claim_id for row in subset_groups[0].claims
    ]
    authority_contract_ids = [
        row.claim_id for row in authority_h.claims
    ]
    if subset_claim_ids != authority_contract_ids:
        raise ValueError(
            "V2 child adapter claim order differs from authority"
        )

    claim_rows = [
        PreN10V2ChildExecutionClaimAdapterV1(
            hypothesis_id=row.hypothesis_id,
            claim_id=row.claim_id,
            claim_rank=row.claim_rank,
            kind=row.kind,
            importance=row.importance,
            novelty_selection_role=row.novelty_selection_role,
            binding_contract_reason_codes=list(
                row.binding_contract_reason_codes
            ),
            source_contract_reason_codes=_source_execution_reason_codes(row),
            contract_status=row.contract_status,
            router_hint=row.router_hint,
            atomic_kind_supported=row.atomic_kind_supported,
        )
        for row in authority_h.claims
    ]

    hypothesis = PreN10V2ChildExecutionHypothesisAdapterV1(
        hypothesis_id=authority_h.hypothesis_id,
        claim_count=authority_h.claim_count,
        ready_claim_count=authority_h.ready_claim_count,
        novelty_bearing_ready_claim_count=(
            authority_h.novelty_bearing_ready_claim_count
        ),
        contract_status=authority_h.contract_status,
        claims=claim_rows,
    )

    body = {
        "schema_version": "pre-n10-v2-child-execution-adapter-v1",
        "source_authority_report_id": authority_report.report_id,
        "source_authority_report_sha256": authority_report.report_sha256,
        "source_authority_query_plan_id": authority_report.source_query_plan_id,
        "source_authority_query_plan_sha256": (
            authority_report.source_query_plan_sha256
        ),
        "source_portfolio_path": str(portfolio_file),
        "source_portfolio_sha256": _sha256_file(portfolio_file),
        "source_portfolio_id": subset_portfolio.portfolio_id,
        "source_query_plan_path": str(query_file),
        "source_query_plan_sha256": _sha256_file(query_file),
        "source_query_plan_id": subset_plan.plan_id,
        "hypotheses": [hypothesis.model_dump(mode="json")],
        "hypothesis_count": 1,
        "stable_source_ids_are_authority": True,
        "exact_text_reconstruction_used_for_authority": False,
        "exact_text_diagnostics_used_for_routing": False,
        "legacy_child_execution_only": True,
        "scientific_authority": False,
        "production_authority": False,
    }
    digest = _sha256_json(body)
    return PreN10V2ChildExecutionContractAdapterV1(
        **body,
        report_id="pre_n10_v2_child_execution_adapter:" + digest[:20],
        report_sha256=digest,
    )


__all__ = [
    "PreN10V2ChildExecutionClaimAdapterV1",
    "PreN10V2ChildExecutionContractAdapterV1",
    "PreN10V2ChildExecutionHypothesisAdapterV1",
    "build_pre_n10_v2_child_execution_adapter_v1",
]
