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
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.pre_n10_canonical_source_reference_v1 import (
    PreN10CanonicalSourceReferenceReportV1,
    PreN10CanonicalSourceReferenceRowV1,
    SourceIdentityComparison,
)
from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    RouterHint,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    assess_claim_binding_readiness,
)
from pipeline_core.discovery.relational_atomic_projection import (
    _ATOMIC_KINDS,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ClaimContractStatus = Literal[
    "READY_FOR_N10_CONTRACT",
    "NOT_READY_FOR_N10_CONTRACT",
]
HypothesisContractStatus = Literal[
    "READY_FOR_N10",
    "REQUIRES_PRE_N10_INTERVENTION",
]
PreN10Disposition = Literal[
    "READY_FOR_N10",
    "INTERVENTION_REQUIRED",
    "NO_HYPOTHESES",
]


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


def classify_pre_n10_router_v2(
    *,
    atomic_kind_supported: bool,
    source_reference_status: str,
    binding_reason_codes: list[str],
) -> RouterHint:
    """Route from structured contract dimensions, not legacy reason strings."""

    if not atomic_kind_supported:
        return "DECOMPOSE_OR_REGENERATE_REVIEW"

    if source_reference_status != "READY":
        return "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"

    if binding_reason_codes:
        return "SPECIFICATION_REPAIR_REVIEW"

    return "PROCEED_TO_LITERAL_ENDPOINT_BINDING"


class PreN10ClaimContractRowV2(StrictModel):
    hypothesis_id: str
    claim_id: str
    claim_rank: int = Field(ge=1)
    kind: str
    importance: str
    novelty_selection_role: str | None = None

    binding_contract_reason_codes: list[str] = Field(
        default_factory=list
    )

    source_reference_status: Literal[
        "READY",
        "INCOMPLETE",
        "INVALID",
    ]
    source_reference_reason_codes: list[str] = Field(
        default_factory=list
    )
    prediction_observation_id: str | None = None
    falsification_criterion_id: str | None = None

    source_identity_comparison: SourceIdentityComparison
    exact_text_source_reference_ready: bool
    exact_text_source_reference_reason_codes: list[str] = Field(
        default_factory=list
    )

    contract_status: ClaimContractStatus
    router_hint: RouterHint
    atomic_kind_supported: bool

    stable_source_ids_used_for_readiness: Literal[True] = True
    exact_text_reconstruction_used_for_readiness: Literal[False] = False
    exact_text_comparison_is_diagnostic_only: Literal[True] = True

    retrieval_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    scientific_content_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_contract_route(self) -> "PreN10ClaimContractRowV2":
        source_ready = self.source_reference_status == "READY"

        if source_ready and self.source_reference_reason_codes:
            raise ValueError(
                "READY stable source reference cannot carry failures"
            )
        if (
            not source_ready
            and not self.source_reference_reason_codes
        ):
            raise ValueError(
                "non-READY stable source reference requires failures"
            )

        expected_ready = bool(
            self.atomic_kind_supported
            and source_ready
            and not self.binding_contract_reason_codes
        )
        ready = self.contract_status == "READY_FOR_N10_CONTRACT"
        if ready != expected_ready:
            raise ValueError(
                "pre-N10 v2 contract status mismatch"
            )

        expected_route = classify_pre_n10_router_v2(
            atomic_kind_supported=self.atomic_kind_supported,
            source_reference_status=self.source_reference_status,
            binding_reason_codes=list(
                self.binding_contract_reason_codes
            ),
        )
        if self.router_hint != expected_route:
            raise ValueError(
                "pre-N10 v2 router hint mismatch"
            )
        return self


class PreN10HypothesisContractV2(StrictModel):
    hypothesis_id: str
    claim_count: int = Field(ge=0)
    ready_claim_count: int = Field(ge=0)
    novelty_bearing_ready_claim_count: int = Field(ge=0)
    contract_status: HypothesisContractStatus
    claims: list[PreN10ClaimContractRowV2] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_counts(self) -> "PreN10HypothesisContractV2":
        if self.claim_count != len(self.claims):
            raise ValueError("claim_count mismatch")

        ready = sum(
            row.contract_status == "READY_FOR_N10_CONTRACT"
            for row in self.claims
        )
        if self.ready_claim_count != ready:
            raise ValueError("ready_claim_count mismatch")

        novelty_ready = sum(
            row.contract_status == "READY_FOR_N10_CONTRACT"
            and row.novelty_selection_role == "NOVELTY_BEARING"
            for row in self.claims
        )
        if self.novelty_bearing_ready_claim_count != novelty_ready:
            raise ValueError(
                "novelty-bearing ready count mismatch"
            )

        expected_ready = bool(
            self.claim_count > 0
            and ready == self.claim_count
            and novelty_ready > 0
        )
        if (
            self.contract_status == "READY_FOR_N10"
        ) != expected_ready:
            raise ValueError(
                "hypothesis pre-N10 v2 contract status mismatch"
            )
        return self


class PreN10ScientificContractReportV2(StrictModel):
    schema_version: Literal[
        "pre-n10-scientific-contract-report-v2"
    ] = "pre-n10-scientific-contract-report-v2"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_portfolio_path: str
    source_portfolio_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_portfolio_id: str

    source_query_plan_path: str
    source_query_plan_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_query_plan_id: str
    source_query_plan_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    canonical_source_reference_path: str
    canonical_source_reference_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    canonical_source_reference_report_id: str
    canonical_source_reference_report_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_binding_bundle_id: str
    source_binding_bundle_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    hypothesis_count: int = Field(ge=0)
    ready_hypothesis_count: int = Field(ge=0)
    intervention_required_hypothesis_count: int = Field(ge=0)

    claim_count: int = Field(ge=0)
    ready_claim_count: int = Field(ge=0)
    not_ready_claim_count: int = Field(ge=0)
    novelty_bearing_ready_claim_count: int = Field(ge=0)

    hypothesis_status_counts: dict[str, int]
    router_hint_counts: dict[str, int]
    source_reference_status_counts: dict[str, int]
    source_identity_comparison_counts: dict[str, int]

    disposition: PreN10Disposition
    hypotheses: list[PreN10HypothesisContractV2] = Field(
        default_factory=list
    )

    claim_decomposition_performed: Literal[True] = True
    claim_decomposition_request_count: int = Field(ge=0)

    all_claims_ready_required_for_n10: Literal[True] = True
    novelty_bearing_ready_claim_required_for_n10: Literal[True] = True

    canonical_source_reference_report_required: Literal[True] = True
    stable_source_ids_used_for_readiness: Literal[True] = True
    exact_text_reconstruction_used_for_readiness: Literal[False] = False
    exact_text_comparison_is_diagnostic_only: Literal[True] = True
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
    def validate_report(
        self,
    ) -> "PreN10ScientificContractReportV2":
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError("hypothesis_count mismatch")

        rows = [
            claim
            for hypothesis in self.hypotheses
            for claim in hypothesis.claims
        ]
        if self.claim_count != len(rows):
            raise ValueError("claim_count mismatch")

        ready_h = sum(
            row.contract_status == "READY_FOR_N10"
            for row in self.hypotheses
        )
        if self.ready_hypothesis_count != ready_h:
            raise ValueError("ready_hypothesis_count mismatch")
        if (
            self.intervention_required_hypothesis_count
            != self.hypothesis_count - ready_h
        ):
            raise ValueError(
                "intervention-required hypothesis count mismatch"
            )

        ready_c = sum(
            row.contract_status == "READY_FOR_N10_CONTRACT"
            for row in rows
        )
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
            raise ValueError(
                "novelty-bearing ready claim count mismatch"
            )

        expected_h = Counter(
            row.contract_status for row in self.hypotheses
        )
        if dict(sorted(expected_h.items())) != dict(
            sorted(self.hypothesis_status_counts.items())
        ):
            raise ValueError(
                "hypothesis_status_counts mismatch"
            )

        expected_routes = Counter(
            row.router_hint for row in rows
        )
        if dict(sorted(expected_routes.items())) != dict(
            sorted(self.router_hint_counts.items())
        ):
            raise ValueError(
                "router_hint_counts mismatch"
            )

        expected_source = Counter(
            row.source_reference_status for row in rows
        )
        if dict(sorted(expected_source.items())) != dict(
            sorted(self.source_reference_status_counts.items())
        ):
            raise ValueError(
                "source_reference_status_counts mismatch"
            )

        expected_comparisons = Counter(
            row.source_identity_comparison for row in rows
        )
        if dict(sorted(expected_comparisons.items())) != dict(
            sorted(self.source_identity_comparison_counts.items())
        ):
            raise ValueError(
                "source_identity_comparison_counts mismatch"
            )

        if self.hypothesis_count == 0:
            expected_disposition: PreN10Disposition = (
                "NO_HYPOTHESES"
            )
        elif ready_h == self.hypothesis_count:
            expected_disposition = "READY_FOR_N10"
        else:
            expected_disposition = "INTERVENTION_REQUIRED"

        if self.disposition != expected_disposition:
            raise ValueError(
                "pre-N10 v2 disposition mismatch"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError(
                "pre-N10 scientific contract v2 SHA mismatch"
            )
        if observed_id != (
            "pre_n10_scientific_contract_v2:"
            + expected_sha[:20]
        ):
            raise ValueError(
                "pre-N10 scientific contract v2 ID mismatch"
            )
        return self


def _assess_claim_v2(
    *,
    hypothesis: object,
    claim: NoveltyClaim,
    source_row: PreN10CanonicalSourceReferenceRowV1,
) -> PreN10ClaimContractRowV2:
    if source_row.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError(
            "pre-N10 v2 source row/hypothesis mismatch"
        )
    if source_row.claim_id != claim.claim_id:
        raise ValueError(
            "pre-N10 v2 source row/claim mismatch"
        )
    if source_row.claim_rank != claim.claim_rank:
        raise ValueError(
            "pre-N10 v2 source row/claim-rank mismatch"
        )
    if source_row.source_claim_sha256 != _sha256_json(
        claim.model_dump(mode="json")
    ):
        raise ValueError(
            "pre-N10 v2 canonical source row claim SHA mismatch: "
            + claim.claim_id
        )

    binding = assess_claim_binding_readiness(
        claim=claim,
        candidate_hypothesis_id=hypothesis.hypothesis_id,
        final_hypothesis_id=hypothesis.hypothesis_id,
    )
    binding_reasons = list(binding.reason_codes)

    atomic_kind_supported = claim.kind in _ATOMIC_KINDS
    source_status = source_row.stable_source_reference_status
    ready = bool(
        atomic_kind_supported
        and source_status == "READY"
        and not binding_reasons
    )

    route = classify_pre_n10_router_v2(
        atomic_kind_supported=atomic_kind_supported,
        source_reference_status=source_status,
        binding_reason_codes=binding_reasons,
    )

    return PreN10ClaimContractRowV2(
        hypothesis_id=hypothesis.hypothesis_id,
        claim_id=claim.claim_id,
        claim_rank=claim.claim_rank,
        kind=claim.kind,
        importance=claim.importance,
        novelty_selection_role=claim.novelty_selection_role,
        binding_contract_reason_codes=binding_reasons,
        source_reference_status=source_status,
        source_reference_reason_codes=list(
            source_row.stable_source_reference_reason_codes
        ),
        prediction_observation_id=(
            source_row.prediction_observation_id
        ),
        falsification_criterion_id=(
            source_row.falsification_criterion_id
        ),
        source_identity_comparison=source_row.comparison,
        exact_text_source_reference_ready=(
            source_row.exact_text_source_reference_ready
        ),
        exact_text_source_reference_reason_codes=list(
            source_row.exact_text_source_reference_reason_codes
        ),
        contract_status=(
            "READY_FOR_N10_CONTRACT"
            if ready
            else "NOT_READY_FOR_N10_CONTRACT"
        ),
        router_hint=route,
        atomic_kind_supported=atomic_kind_supported,
    )


def build_pre_n10_scientific_contract_v2(
    *,
    portfolio_path: Path,
    query_plan_path: Path,
    canonical_source_reference_path: Path,
    claim_decomposition_request_count: int = 0,
) -> PreN10ScientificContractReportV2:
    portfolio_file = portfolio_path.expanduser().resolve()
    plan_file = query_plan_path.expanduser().resolve()
    canonical_file = (
        canonical_source_reference_path.expanduser().resolve()
    )

    for path, label in (
        (portfolio_file, "source portfolio"),
        (plan_file, "query plan"),
        (canonical_file, "canonical source-reference report"),
    ):
        if not path.is_file():
            raise ValueError(
                "missing pre-N10 v2 "
                + label
                + ": "
                + str(path)
            )

    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_file.read_text(encoding="utf-8")
    )
    plan = LiteratureQueryPlan.model_validate_json(
        plan_file.read_text(encoding="utf-8")
    )
    canonical = (
        PreN10CanonicalSourceReferenceReportV1.model_validate_json(
            canonical_file.read_text(encoding="utf-8")
        )
    )

    portfolio_file_sha = _sha256_file(portfolio_file)
    plan_file_sha = _sha256_file(plan_file)

    if plan.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError(
            "pre-N10 v2 query-plan/portfolio mismatch"
        )
    if canonical.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError(
            "pre-N10 v2 canonical source-reference/portfolio mismatch"
        )
    if canonical.source_portfolio_file_sha256 != portfolio_file_sha:
        raise ValueError(
            "pre-N10 v2 canonical source-reference portfolio file changed"
        )
    if canonical.source_query_plan_id != plan.plan_id:
        raise ValueError(
            "pre-N10 v2 canonical source-reference query-plan ID mismatch"
        )
    if canonical.source_query_plan_sha256 != plan.plan_sha256:
        raise ValueError(
            "pre-N10 v2 canonical source-reference query-plan SHA mismatch"
        )
    if canonical.source_query_plan_file_sha256 != plan_file_sha:
        raise ValueError(
            "pre-N10 v2 canonical source-reference query-plan file changed"
        )

    cards = {
        row.hypothesis_id: row
        for row in portfolio.hypotheses
    }
    if len(cards) != len(portfolio.hypotheses):
        raise ValueError(
            "duplicate hypothesis IDs in pre-N10 v2 portfolio"
        )

    groups = {
        row.hypothesis_id: row
        for row in plan.claims
    }
    if len(groups) != len(plan.claims):
        raise ValueError(
            "duplicate hypothesis claim groups in pre-N10 v2 query plan"
        )
    if set(groups) != set(cards):
        raise ValueError(
            "pre-N10 v2 query-plan hypothesis population differs "
            "from source portfolio"
        )

    source_rows = {
        (row.hypothesis_id, row.claim_id): row
        for row in canonical.rows
    }
    if len(source_rows) != len(canonical.rows):
        raise ValueError(
            "duplicate canonical source-reference claim identity"
        )

    plan_keys = {
        (claim.hypothesis_id, claim.claim_id)
        for group in plan.claims
        for claim in group.claims
    }
    if set(source_rows) != plan_keys:
        missing = sorted(plan_keys - set(source_rows))
        extra = sorted(set(source_rows) - plan_keys)
        raise ValueError(
            "pre-N10 v2 canonical source-reference claim population "
            "mismatch: missing="
            + repr(missing)
            + " extra="
            + repr(extra)
        )

    hypotheses: list[PreN10HypothesisContractV2] = []

    for card in portfolio.hypotheses:
        group = groups[card.hypothesis_id]
        claim_rows = [
            _assess_claim_v2(
                hypothesis=card,
                claim=claim,
                source_row=source_rows[
                    (claim.hypothesis_id, claim.claim_id)
                ],
            )
            for claim in group.claims
        ]

        ready_count = sum(
            row.contract_status == "READY_FOR_N10_CONTRACT"
            for row in claim_rows
        )
        novelty_ready_count = sum(
            row.contract_status == "READY_FOR_N10_CONTRACT"
            and row.novelty_selection_role == "NOVELTY_BEARING"
            for row in claim_rows
        )
        fully_ready = bool(
            claim_rows
            and ready_count == len(claim_rows)
            and novelty_ready_count > 0
        )

        hypotheses.append(
            PreN10HypothesisContractV2(
                hypothesis_id=card.hypothesis_id,
                claim_count=len(claim_rows),
                ready_claim_count=ready_count,
                novelty_bearing_ready_claim_count=(
                    novelty_ready_count
                ),
                contract_status=(
                    "READY_FOR_N10"
                    if fully_ready
                    else "REQUIRES_PRE_N10_INTERVENTION"
                ),
                claims=claim_rows,
            )
        )

    all_rows = [
        claim
        for hypothesis in hypotheses
        for claim in hypothesis.claims
    ]
    ready_h = sum(
        row.contract_status == "READY_FOR_N10"
        for row in hypotheses
    )
    ready_c = sum(
        row.contract_status == "READY_FOR_N10_CONTRACT"
        for row in all_rows
    )
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
        "schema_version": "pre-n10-scientific-contract-report-v2",
        "source_portfolio_path": str(portfolio_file),
        "source_portfolio_file_sha256": portfolio_file_sha,
        "source_portfolio_id": portfolio.portfolio_id,
        "source_query_plan_path": str(plan_file),
        "source_query_plan_file_sha256": plan_file_sha,
        "source_query_plan_id": plan.plan_id,
        "source_query_plan_sha256": plan.plan_sha256,
        "canonical_source_reference_path": str(canonical_file),
        "canonical_source_reference_file_sha256": _sha256_file(
            canonical_file
        ),
        "canonical_source_reference_report_id": canonical.report_id,
        "canonical_source_reference_report_sha256": (
            canonical.report_sha256
        ),
        "source_binding_bundle_id": canonical.source_binding_bundle_id,
        "source_binding_bundle_sha256": (
            canonical.source_binding_bundle_sha256
        ),
        "hypothesis_count": len(hypotheses),
        "ready_hypothesis_count": ready_h,
        "intervention_required_hypothesis_count": (
            len(hypotheses) - ready_h
        ),
        "claim_count": len(all_rows),
        "ready_claim_count": ready_c,
        "not_ready_claim_count": len(all_rows) - ready_c,
        "novelty_bearing_ready_claim_count": novelty_ready,
        "hypothesis_status_counts": dict(
            Counter(
                row.contract_status
                for row in hypotheses
            )
        ),
        "router_hint_counts": dict(
            Counter(row.router_hint for row in all_rows)
        ),
        "source_reference_status_counts": dict(
            Counter(
                row.source_reference_status
                for row in all_rows
            )
        ),
        "source_identity_comparison_counts": dict(
            Counter(
                row.source_identity_comparison
                for row in all_rows
            )
        ),
        "disposition": disposition,
        "hypotheses": [
            row.model_dump(mode="json")
            for row in hypotheses
        ],
        "claim_decomposition_performed": True,
        "claim_decomposition_request_count": int(
            claim_decomposition_request_count
        ),
        "all_claims_ready_required_for_n10": True,
        "novelty_bearing_ready_claim_required_for_n10": True,
        "canonical_source_reference_report_required": True,
        "stable_source_ids_used_for_readiness": True,
        "exact_text_reconstruction_used_for_readiness": False,
        "exact_text_comparison_is_diagnostic_only": True,
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

    return PreN10ScientificContractReportV2(
        **body,
        report_id=(
            "pre_n10_scientific_contract_v2:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "PreN10ClaimContractRowV2",
    "PreN10HypothesisContractV2",
    "PreN10ScientificContractReportV2",
    "classify_pre_n10_router_v2",
    "build_pre_n10_scientific_contract_v2",
]
