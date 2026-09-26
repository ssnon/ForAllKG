from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.atomic_scientific_source_binding import (
    SourceReferenceStatus,
    resolve_atomic_source_reference,
)
from pipeline_core.discovery.atomic_scientific_source_provenance import (
    AtomicScientificSourceBindingBundle,
)
from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    _count_exact_observation_bindings,
)
from pipeline_core.discovery.relational_atomic_projection import (
    _exact_observation_binding,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SourceIdentityComparison = Literal[
    "AGREE_READY",
    "AGREE_NOT_READY",
    "STABLE_READY_EXACT_NOT_READY",
    "STABLE_NOT_READY_EXACT_READY",
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


class PreN10CanonicalSourceReferenceRowV1(StrictModel):
    schema_version: Literal[
        "pre-n10-canonical-source-reference-row-v1"
    ] = "pre-n10-canonical-source-reference-row-v1"

    hypothesis_id: str
    claim_id: str
    claim_rank: int = Field(ge=1)
    claim_local_id: str
    source_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    prediction_observation_id: str | None = None
    falsification_criterion_id: str | None = None

    stable_source_reference_status: SourceReferenceStatus
    stable_source_reference_reason_codes: list[str] = Field(
        default_factory=list
    )

    exact_text_source_reference_ready: bool
    exact_text_source_reference_reason_codes: list[str] = Field(
        default_factory=list
    )
    prediction_exact_text_binding_count: int = Field(ge=0)
    falsifier_exact_text_binding_count: int = Field(ge=0)
    exact_text_shared_observable_identity: bool | None = None

    comparison: SourceIdentityComparison

    stable_ids_are_identity_authority_for_this_assessment: Literal[
        True
    ] = True
    exact_text_reconstruction_used_for_identity: Literal[False] = False
    exact_text_comparison_is_diagnostic_only: Literal[True] = True
    contract_readiness_authority: Literal[False] = False
    router_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_comparison(
        self,
    ) -> "PreN10CanonicalSourceReferenceRowV1":
        stable_ready = (
            self.stable_source_reference_status == "READY"
        )
        exact_ready = self.exact_text_source_reference_ready

        expected: SourceIdentityComparison
        if stable_ready and exact_ready:
            expected = "AGREE_READY"
        elif not stable_ready and not exact_ready:
            expected = "AGREE_NOT_READY"
        elif stable_ready:
            expected = "STABLE_READY_EXACT_NOT_READY"
        else:
            expected = "STABLE_NOT_READY_EXACT_READY"

        if self.comparison != expected:
            raise ValueError(
                "canonical source-reference comparison mismatch"
            )

        if stable_ready and self.stable_source_reference_reason_codes:
            raise ValueError(
                "stable READY row cannot carry source-reference failures"
            )
        if (
            not stable_ready
            and not self.stable_source_reference_reason_codes
        ):
            raise ValueError(
                "stable non-READY row requires source-reference failures"
            )
        if exact_ready == bool(
            self.exact_text_source_reference_reason_codes
        ):
            raise ValueError(
                "exact-text ready state/reason-code mismatch"
            )
        return self


class PreN10CanonicalSourceReferenceReportV1(StrictModel):
    schema_version: Literal[
        "pre-n10-canonical-source-reference-report-v1"
    ] = "pre-n10-canonical-source-reference-report-v1"

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

    source_binding_bundle_path: str
    source_binding_bundle_file_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    source_binding_bundle_id: str
    source_binding_bundle_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    claim_count: int = Field(ge=0)
    stable_ready_count: int = Field(ge=0)
    stable_incomplete_count: int = Field(ge=0)
    stable_invalid_count: int = Field(ge=0)
    exact_text_ready_count: int = Field(ge=0)
    agreement_count: int = Field(ge=0)
    disagreement_count: int = Field(ge=0)
    comparison_counts: dict[str, int]

    rows: list[PreN10CanonicalSourceReferenceRowV1] = Field(
        default_factory=list
    )

    stable_source_ids_are_identity_authority: Literal[True] = True
    exact_text_reconstruction_used_for_identity: Literal[False] = False
    exact_text_comparison_is_diagnostic_only: Literal[True] = True

    diagnostic_only: Literal[True] = True
    contract_readiness_authority: Literal[False] = False
    router_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_authority: Literal[False] = False
    retrieval_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "PreN10CanonicalSourceReferenceReportV1":
        if self.claim_count != len(self.rows):
            raise ValueError(
                "canonical source-reference claim_count mismatch"
            )

        observed_status = Counter(
            row.stable_source_reference_status
            for row in self.rows
        )
        if self.stable_ready_count != observed_status["READY"]:
            raise ValueError(
                "canonical source-reference READY count mismatch"
            )
        if self.stable_incomplete_count != observed_status["INCOMPLETE"]:
            raise ValueError(
                "canonical source-reference INCOMPLETE count mismatch"
            )
        if self.stable_invalid_count != observed_status["INVALID"]:
            raise ValueError(
                "canonical source-reference INVALID count mismatch"
            )

        exact_ready = sum(
            row.exact_text_source_reference_ready
            for row in self.rows
        )
        if self.exact_text_ready_count != exact_ready:
            raise ValueError(
                "canonical source-reference exact-text count mismatch"
            )

        comparisons = Counter(row.comparison for row in self.rows)
        if dict(sorted(comparisons.items())) != dict(
            sorted(self.comparison_counts.items())
        ):
            raise ValueError(
                "canonical source-reference comparison counts mismatch"
            )

        agreement = sum(
            row.comparison in {"AGREE_READY", "AGREE_NOT_READY"}
            for row in self.rows
        )
        if self.agreement_count != agreement:
            raise ValueError(
                "canonical source-reference agreement count mismatch"
            )
        if self.disagreement_count != self.claim_count - agreement:
            raise ValueError(
                "canonical source-reference disagreement count mismatch"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError(
                "canonical source-reference report SHA mismatch"
            )
        if observed_id != (
            "pre_n10_canonical_source_reference:"
            + expected_sha[:20]
        ):
            raise ValueError(
                "canonical source-reference report ID mismatch"
            )
        return self


def build_pre_n10_canonical_source_reference_report_v1(
    *,
    portfolio_path: Path,
    query_plan_path: Path,
    source_binding_bundle_path: Path,
) -> PreN10CanonicalSourceReferenceReportV1:
    portfolio_file = portfolio_path.expanduser().resolve()
    plan_file = query_plan_path.expanduser().resolve()
    bundle_file = source_binding_bundle_path.expanduser().resolve()

    for path, label in (
        (portfolio_file, "portfolio"),
        (plan_file, "query plan"),
        (bundle_file, "source-binding bundle"),
    ):
        if not path.is_file():
            raise ValueError(
                "missing pre-N10 canonical source-reference "
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
    bundle = AtomicScientificSourceBindingBundle.model_validate_json(
        bundle_file.read_text(encoding="utf-8")
    )

    if plan.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError(
            "canonical source-reference query-plan/portfolio mismatch"
        )
    if bundle.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError(
            "canonical source-reference bundle/portfolio mismatch"
        )
    if bundle.source_query_plan_id != plan.plan_id:
        raise ValueError(
            "canonical source-reference bundle/query-plan ID mismatch"
        )
    if bundle.source_query_plan_sha256 != plan.plan_sha256:
        raise ValueError(
            "canonical source-reference bundle/query-plan SHA mismatch"
        )

    cards = {
        row.hypothesis_id: row
        for row in portfolio.hypotheses
    }
    if len(cards) != len(portfolio.hypotheses):
        raise ValueError(
            "duplicate hypothesis IDs in canonical source-reference portfolio"
        )

    claims = [
        claim
        for group in plan.claims
        for claim in group.claims
    ]
    claims_by_key = {
        (claim.hypothesis_id, claim.claim_id): claim
        for claim in claims
    }
    if len(claims_by_key) != len(claims):
        raise ValueError(
            "duplicate query-plan claim identity in canonical source-reference"
        )

    records_by_key = {
        (row.hypothesis_id, row.claim_id): row
        for row in bundle.records
    }
    if len(records_by_key) != len(bundle.records):
        raise ValueError(
            "duplicate source-binding identity in canonical source-reference"
        )
    if set(records_by_key) != set(claims_by_key):
        missing = sorted(set(claims_by_key) - set(records_by_key))
        extra = sorted(set(records_by_key) - set(claims_by_key))
        raise ValueError(
            "canonical source-reference bundle/query-plan population mismatch: "
            + "missing="
            + repr(missing)
            + " extra="
            + repr(extra)
        )

    rows: list[PreN10CanonicalSourceReferenceRowV1] = []

    for group in plan.claims:
        card = cards.get(group.hypothesis_id)
        if card is None:
            raise ValueError(
                "canonical source-reference query plan contains "
                "unknown hypothesis: "
                + group.hypothesis_id
            )

        for claim in sorted(
            group.claims,
            key=lambda row: row.claim_rank,
        ):
            key = (claim.hypothesis_id, claim.claim_id)
            record = records_by_key[key]

            if record.claim_rank != claim.claim_rank:
                raise ValueError(
                    "canonical source-reference claim-rank mismatch: "
                    + claim.claim_id
                )
            observed_claim_sha = _sha256_json(
                claim.model_dump(mode="json")
            )
            if record.source_claim_sha256 != observed_claim_sha:
                raise ValueError(
                    "canonical source-reference claim SHA mismatch: "
                    + claim.claim_id
                )

            (
                stable_status,
                stable_reasons,
                _prediction,
                _falsifier,
            ) = resolve_atomic_source_reference(
                hypothesis=card,
                prediction_observation_id=(
                    record.prediction_observation_id
                ),
                falsification_criterion_id=(
                    record.falsification_criterion_id
                ),
            )

            (
                _exact_prediction,
                _exact_falsifier,
                exact_reasons,
            ) = _exact_observation_binding(
                candidate_card=card,
                claim=claim,
            )
            (
                prediction_count,
                falsifier_count,
                shared_identity,
            ) = _count_exact_observation_bindings(
                candidate_card=card,
                claim=claim,
            )

            stable_ready = stable_status == "READY"
            exact_ready = not exact_reasons
            if stable_ready and exact_ready:
                comparison: SourceIdentityComparison = "AGREE_READY"
            elif not stable_ready and not exact_ready:
                comparison = "AGREE_NOT_READY"
            elif stable_ready:
                comparison = "STABLE_READY_EXACT_NOT_READY"
            else:
                comparison = "STABLE_NOT_READY_EXACT_READY"

            rows.append(
                PreN10CanonicalSourceReferenceRowV1(
                    hypothesis_id=claim.hypothesis_id,
                    claim_id=claim.claim_id,
                    claim_rank=claim.claim_rank,
                    claim_local_id=record.claim_local_id,
                    source_claim_sha256=record.source_claim_sha256,
                    prediction_observation_id=(
                        record.prediction_observation_id
                    ),
                    falsification_criterion_id=(
                        record.falsification_criterion_id
                    ),
                    stable_source_reference_status=stable_status,
                    stable_source_reference_reason_codes=list(
                        stable_reasons
                    ),
                    exact_text_source_reference_ready=exact_ready,
                    exact_text_source_reference_reason_codes=list(
                        exact_reasons
                    ),
                    prediction_exact_text_binding_count=(
                        prediction_count
                    ),
                    falsifier_exact_text_binding_count=(
                        falsifier_count
                    ),
                    exact_text_shared_observable_identity=(
                        shared_identity
                    ),
                    comparison=comparison,
                )
            )

    status_counts = Counter(
        row.stable_source_reference_status
        for row in rows
    )
    comparison_counts = Counter(
        row.comparison
        for row in rows
    )
    agreement = sum(
        row.comparison in {"AGREE_READY", "AGREE_NOT_READY"}
        for row in rows
    )

    body = {
        "schema_version":
            "pre-n10-canonical-source-reference-report-v1",
        "source_portfolio_path": str(portfolio_file),
        "source_portfolio_file_sha256": _sha256_file(portfolio_file),
        "source_portfolio_id": portfolio.portfolio_id,
        "source_query_plan_path": str(plan_file),
        "source_query_plan_file_sha256": _sha256_file(plan_file),
        "source_query_plan_id": plan.plan_id,
        "source_query_plan_sha256": plan.plan_sha256,
        "source_binding_bundle_path": str(bundle_file),
        "source_binding_bundle_file_sha256": _sha256_file(bundle_file),
        "source_binding_bundle_id": bundle.bundle_id,
        "source_binding_bundle_sha256": bundle.bundle_sha256,
        "claim_count": len(rows),
        "stable_ready_count": status_counts["READY"],
        "stable_incomplete_count": status_counts["INCOMPLETE"],
        "stable_invalid_count": status_counts["INVALID"],
        "exact_text_ready_count": sum(
            row.exact_text_source_reference_ready
            for row in rows
        ),
        "agreement_count": agreement,
        "disagreement_count": len(rows) - agreement,
        "comparison_counts": dict(sorted(comparison_counts.items())),
        "rows": [row.model_dump(mode="json") for row in rows],
        "stable_source_ids_are_identity_authority": True,
        "exact_text_reconstruction_used_for_identity": False,
        "exact_text_comparison_is_diagnostic_only": True,
        "diagnostic_only": True,
        "contract_readiness_authority": False,
        "router_authority": False,
        "novelty_authority": False,
        "production_authority": False,
        "retrieval_performed": False,
        "n9_performed": False,
        "n10_performed": False,
    }
    digest = _sha256_json(body)
    return PreN10CanonicalSourceReferenceReportV1(
        **body,
        report_id=(
            "pre_n10_canonical_source_reference:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "PreN10CanonicalSourceReferenceReportV1",
    "PreN10CanonicalSourceReferenceRowV1",
    "SourceIdentityComparison",
    "build_pre_n10_canonical_source_reference_report_v1",
]
