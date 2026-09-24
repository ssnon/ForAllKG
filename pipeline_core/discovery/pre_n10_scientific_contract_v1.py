from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import LiteratureQueryPlan, NoveltyClaim
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    _count_exact_observation_bindings,
    classify_router_hint,
)
from pipeline_core.discovery.relational_atomic_binding_plan import assess_claim_binding_readiness
from pipeline_core.discovery.relational_atomic_projection import _ATOMIC_KINDS, _exact_observation_binding


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ClaimContractStatus = Literal["READY_FOR_N10_CONTRACT", "NOT_READY_FOR_N10_CONTRACT"]
HypothesisContractStatus = Literal["READY_FOR_N10", "REQUIRES_PRE_N10_INTERVENTION"]
PreN10Disposition = Literal["READY_FOR_N10", "INTERVENTION_REQUIRED", "NO_HYPOTHESES"]


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PreN10ClaimContractRowV1(StrictModel):
    hypothesis_id: str
    claim_id: str
    claim_rank: int = Field(ge=1)
    kind: str
    importance: str
    novelty_selection_role: str | None = None
    binding_contract_reason_codes: list[str] = Field(default_factory=list)
    source_contract_reason_codes: list[str] = Field(default_factory=list)
    contract_status: ClaimContractStatus
    router_hint: str
    atomic_kind_supported: bool
    prediction_exact_source_binding_count: int = Field(ge=0)
    falsifier_exact_source_binding_count: int = Field(ge=0)
    shared_observable_identity_satisfied: bool | None
    retrieval_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    scientific_content_mutated: Literal[False] = False


class PreN10HypothesisContractV1(StrictModel):
    hypothesis_id: str
    claim_count: int = Field(ge=0)
    ready_claim_count: int = Field(ge=0)
    novelty_bearing_ready_claim_count: int = Field(ge=0)
    contract_status: HypothesisContractStatus
    claims: list[PreN10ClaimContractRowV1] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> "PreN10HypothesisContractV1":
        if self.claim_count != len(self.claims):
            raise ValueError("claim_count mismatch")
        ready = sum(row.contract_status == "READY_FOR_N10_CONTRACT" for row in self.claims)
        if self.ready_claim_count != ready:
            raise ValueError("ready_claim_count mismatch")
        novelty_ready = sum(
            row.contract_status == "READY_FOR_N10_CONTRACT"
            and row.novelty_selection_role == "NOVELTY_BEARING"
            for row in self.claims
        )
        if self.novelty_bearing_ready_claim_count != novelty_ready:
            raise ValueError("novelty-bearing ready count mismatch")
        expected_ready = (
            self.claim_count > 0
            and self.ready_claim_count == self.claim_count
            and self.novelty_bearing_ready_claim_count > 0
        )
        if (self.contract_status == "READY_FOR_N10") != expected_ready:
            raise ValueError("hypothesis pre-N10 contract status mismatch")
        return self


class PreN10ScientificContractReportV1(StrictModel):
    schema_version: Literal["pre-n10-scientific-contract-report-v1"] = "pre-n10-scientific-contract-report-v1"
    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_portfolio_path: str
    source_portfolio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_portfolio_id: str
    source_query_plan_path: str
    source_query_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_query_plan_id: str
    hypothesis_count: int = Field(ge=0)
    ready_hypothesis_count: int = Field(ge=0)
    intervention_required_hypothesis_count: int = Field(ge=0)
    claim_count: int = Field(ge=0)
    ready_claim_count: int = Field(ge=0)
    not_ready_claim_count: int = Field(ge=0)
    novelty_bearing_ready_claim_count: int = Field(ge=0)
    hypothesis_status_counts: dict[str, int]
    router_hint_counts: dict[str, int]
    disposition: PreN10Disposition
    hypotheses: list[PreN10HypothesisContractV1] = Field(default_factory=list)
    claim_decomposition_performed: Literal[True] = True
    claim_decomposition_request_count: int = Field(ge=0)
    all_claims_ready_required_for_n10: Literal[True] = True
    novelty_bearing_ready_claim_required_for_n10: Literal[True] = True
    exact_prediction_source_binding_checked: Literal[True] = True
    exact_falsifier_source_binding_checked: Literal[True] = True
    shared_observable_identity_checked: Literal[True] = True
    atomic_kind_checked: Literal[True] = True
    retrieval_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    repair_performed: Literal[False] = False
    regeneration_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreN10ScientificContractReportV1":
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError("hypothesis_count mismatch")
        ready_h = sum(row.contract_status == "READY_FOR_N10" for row in self.hypotheses)
        if self.ready_hypothesis_count != ready_h:
            raise ValueError("ready_hypothesis_count mismatch")
        if self.intervention_required_hypothesis_count != self.hypothesis_count - ready_h:
            raise ValueError("intervention-required hypothesis count mismatch")
        rows = [claim for hypothesis in self.hypotheses for claim in hypothesis.claims]
        if self.claim_count != len(rows):
            raise ValueError("claim_count mismatch")
        ready_c = sum(row.contract_status == "READY_FOR_N10_CONTRACT" for row in rows)
        if self.ready_claim_count != ready_c:
            raise ValueError("ready_claim_count mismatch")
        if self.not_ready_claim_count != self.claim_count - ready_c:
            raise ValueError("not_ready_claim_count mismatch")
        novelty_ready = sum(
            row.contract_status == "READY_FOR_N10_CONTRACT"
            and row.novelty_selection_role == "NOVELTY_BEARING"
            for row in rows
        )
        if self.novelty_bearing_ready_claim_count != novelty_ready:
            raise ValueError("novelty-bearing ready claim count mismatch")
        if dict(sorted(Counter(row.contract_status for row in self.hypotheses).items())) != dict(
            sorted(self.hypothesis_status_counts.items())
        ):
            raise ValueError("hypothesis_status_counts mismatch")
        if dict(sorted(Counter(row.router_hint for row in rows).items())) != dict(
            sorted(self.router_hint_counts.items())
        ):
            raise ValueError("router_hint_counts mismatch")
        if self.hypothesis_count == 0:
            expected_disposition = "NO_HYPOTHESES"
        elif self.ready_hypothesis_count == self.hypothesis_count:
            expected_disposition = "READY_FOR_N10"
        else:
            expected_disposition = "INTERVENTION_REQUIRED"
        if self.disposition != expected_disposition:
            raise ValueError("pre-N10 disposition mismatch")
        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("pre-N10 scientific contract SHA mismatch")
        if observed_id != "pre_n10_scientific_contract_v1:" + expected_sha[:20]:
            raise ValueError("pre-N10 scientific contract ID mismatch")
        return self


def _assess_claim(*, hypothesis: object, claim: NoveltyClaim) -> PreN10ClaimContractRowV1:
    binding = assess_claim_binding_readiness(
        claim=claim,
        candidate_hypothesis_id=hypothesis.hypothesis_id,
        final_hypothesis_id=hypothesis.hypothesis_id,
    )
    source_reasons: list[str] = []
    if claim.kind not in _ATOMIC_KINDS:
        source_reasons.append("unsupported_atomic_claim_kind:" + claim.kind)
    if claim.novelty_selection_role is None:
        source_reasons.append("missing_novelty_selection_role")
    _, _, observation_reasons = _exact_observation_binding(candidate_card=hypothesis, claim=claim)
    source_reasons.extend(observation_reasons)
    source_reasons = list(dict.fromkeys(source_reasons))
    prediction_count, falsifier_count, shared_identity = _count_exact_observation_bindings(
        candidate_card=hypothesis,
        claim=claim,
    )
    binding_reasons = list(binding.reason_codes)
    ready = not binding_reasons and not source_reasons
    router_hint = classify_router_hint(
        binding_reason_codes=binding_reasons,
        source_reason_codes=source_reasons,
    )
    return PreN10ClaimContractRowV1(
        hypothesis_id=hypothesis.hypothesis_id,
        claim_id=claim.claim_id,
        claim_rank=claim.claim_rank,
        kind=claim.kind,
        importance=claim.importance,
        novelty_selection_role=claim.novelty_selection_role,
        binding_contract_reason_codes=binding_reasons,
        source_contract_reason_codes=source_reasons,
        contract_status="READY_FOR_N10_CONTRACT" if ready else "NOT_READY_FOR_N10_CONTRACT",
        router_hint=router_hint,
        atomic_kind_supported=(claim.kind in _ATOMIC_KINDS),
        prediction_exact_source_binding_count=prediction_count,
        falsifier_exact_source_binding_count=falsifier_count,
        shared_observable_identity_satisfied=shared_identity,
    )


def build_pre_n10_scientific_contract_v1(
    *,
    portfolio_path: Path,
    query_plan_path: Path,
    claim_decomposition_request_count: int = 0,
) -> PreN10ScientificContractReportV1:
    portfolio_file = portfolio_path.expanduser().resolve()
    plan_file = query_plan_path.expanduser().resolve()
    if not portfolio_file.is_file():
        raise ValueError("missing pre-N10 source portfolio: " + str(portfolio_file))
    if not plan_file.is_file():
        raise ValueError("missing pre-N10 query plan: " + str(plan_file))
    portfolio = HypothesisPortfolio.model_validate_json(portfolio_file.read_text(encoding="utf-8"))
    plan = LiteratureQueryPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
    if plan.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError("query-plan/source-portfolio provenance mismatch")

    cards = {row.hypothesis_id: row for row in portfolio.hypotheses}
    if len(cards) != len(portfolio.hypotheses):
        raise ValueError("duplicate hypothesis IDs in pre-N10 portfolio")
    groups = {row.hypothesis_id: row for row in plan.claims}
    if len(groups) != len(plan.claims):
        raise ValueError("duplicate hypothesis claim groups in query plan")
    if set(groups) != set(cards):
        raise ValueError("query-plan hypothesis population differs from source portfolio")

    hypotheses: list[PreN10HypothesisContractV1] = []
    for card in portfolio.hypotheses:
        group = groups[card.hypothesis_id]
        claim_rows = [_assess_claim(hypothesis=card, claim=claim) for claim in group.claims]
        ready_count = sum(row.contract_status == "READY_FOR_N10_CONTRACT" for row in claim_rows)
        novelty_ready_count = sum(
            row.contract_status == "READY_FOR_N10_CONTRACT"
            and row.novelty_selection_role == "NOVELTY_BEARING"
            for row in claim_rows
        )
        fully_ready = bool(claim_rows) and ready_count == len(claim_rows) and novelty_ready_count > 0
        hypotheses.append(
            PreN10HypothesisContractV1(
                hypothesis_id=card.hypothesis_id,
                claim_count=len(claim_rows),
                ready_claim_count=ready_count,
                novelty_bearing_ready_claim_count=novelty_ready_count,
                contract_status="READY_FOR_N10" if fully_ready else "REQUIRES_PRE_N10_INTERVENTION",
                claims=claim_rows,
            )
        )

    all_rows = [claim for hypothesis in hypotheses for claim in hypothesis.claims]
    ready_h = sum(row.contract_status == "READY_FOR_N10" for row in hypotheses)
    ready_c = sum(row.contract_status == "READY_FOR_N10_CONTRACT" for row in all_rows)
    novelty_ready = sum(
        row.contract_status == "READY_FOR_N10_CONTRACT"
        and row.novelty_selection_role == "NOVELTY_BEARING"
        for row in all_rows
    )
    if not hypotheses:
        disposition: PreN10Disposition = "NO_HYPOTHESES"
    elif ready_h == len(hypotheses):
        disposition = "READY_FOR_N10"
    else:
        disposition = "INTERVENTION_REQUIRED"

    body = {
        "schema_version": "pre-n10-scientific-contract-report-v1",
        "source_portfolio_path": str(portfolio_file),
        "source_portfolio_sha256": _sha256_file(portfolio_file),
        "source_portfolio_id": portfolio.portfolio_id,
        "source_query_plan_path": str(plan_file),
        "source_query_plan_sha256": _sha256_file(plan_file),
        "source_query_plan_id": plan.plan_id,
        "hypothesis_count": len(hypotheses),
        "ready_hypothesis_count": ready_h,
        "intervention_required_hypothesis_count": len(hypotheses) - ready_h,
        "claim_count": len(all_rows),
        "ready_claim_count": ready_c,
        "not_ready_claim_count": len(all_rows) - ready_c,
        "novelty_bearing_ready_claim_count": novelty_ready,
        "hypothesis_status_counts": dict(Counter(row.contract_status for row in hypotheses)),
        "router_hint_counts": dict(Counter(row.router_hint for row in all_rows)),
        "disposition": disposition,
        "hypotheses": [row.model_dump(mode="json") for row in hypotheses],
        "claim_decomposition_performed": True,
        "claim_decomposition_request_count": int(claim_decomposition_request_count),
        "all_claims_ready_required_for_n10": True,
        "novelty_bearing_ready_claim_required_for_n10": True,
        "exact_prediction_source_binding_checked": True,
        "exact_falsifier_source_binding_checked": True,
        "shared_observable_identity_checked": True,
        "atomic_kind_checked": True,
        "retrieval_performed": False,
        "novelty_assessment_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "repair_performed": False,
        "regeneration_performed": False,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return PreN10ScientificContractReportV1(
        **body,
        report_id="pre_n10_scientific_contract_v1:" + digest[:20],
        report_sha256=digest,
    )


__all__ = [
    "ClaimContractStatus",
    "HypothesisContractStatus",
    "PreN10Disposition",
    "PreN10ClaimContractRowV1",
    "PreN10HypothesisContractV1",
    "PreN10ScientificContractReportV1",
    "build_pre_n10_scientific_contract_v1",
]
