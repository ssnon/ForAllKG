from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from typing import get_args

from pipeline_core.discovery.atomic_scientific_source_binding import (
    resolve_atomic_source_reference,
)
from pipeline_core.discovery.atomic_scientific_source_provenance import (
    AtomicScientificSourceBindingBundle,
)
from pipeline_core.discovery.atomic_scientific_specification import (
    AtomicClaimKind,
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
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


class AtomicScientificSpecificationBundleHypothesis(StrictModel):
    hypothesis_id: str
    specifications: list[CompiledAtomicSpecification] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_claim_ids(
        self,
    ) -> "AtomicScientificSpecificationBundleHypothesis":
        ids = [row.claim_id for row in self.specifications]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "duplicate claim IDs within atomic specification hypothesis"
            )
        return self


class AtomicScientificSpecificationBundle(StrictModel):
    schema_version: Literal[
        "atomic-scientific-specification-bundle-v1"
    ] = "atomic-scientific-specification-bundle-v1"

    bundle_id: str
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_report_id: str
    source_contract: str

    hypotheses: list[AtomicScientificSpecificationBundleHypothesis]
    hypothesis_count: int = Field(ge=0)
    atomic_specification_count: int = Field(ge=0)

    canonical_scientific_representation: Literal[True] = True
    stable_source_identity_preserved: Literal[True] = True
    exact_text_is_identity_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_bundle(
        self,
    ) -> "AtomicScientificSpecificationBundle":
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError(
                "atomic specification bundle hypothesis_count mismatch"
            )
        total = sum(
            len(row.specifications) for row in self.hypotheses
        )
        if self.atomic_specification_count != total:
            raise ValueError(
                "atomic specification bundle specification_count mismatch"
            )

        hypothesis_ids = [
            row.hypothesis_id for row in self.hypotheses
        ]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError(
                "duplicate hypothesis IDs in atomic specification bundle"
            )

        claim_ids = [
            spec.claim_id
            for row in self.hypotheses
            for spec in row.specifications
        ]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError(
                "duplicate global claim IDs in atomic specification bundle"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("bundle_id")
        observed_sha = body.pop("bundle_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError(
                "atomic specification bundle SHA mismatch"
            )
        if observed_id != (
            "atomic_scientific_specification_bundle:"
            + expected_sha[:20]
        ):
            raise ValueError(
                "atomic specification bundle ID mismatch"
            )
        return self


def build_atomic_scientific_specification_bundle(
    *,
    source_report_id: str,
    source_contract: str,
    hypotheses: list[
        tuple[str, list[CompiledAtomicSpecification]]
    ],
) -> AtomicScientificSpecificationBundle:
    rows = [
        AtomicScientificSpecificationBundleHypothesis(
            hypothesis_id=hypothesis_id,
            specifications=list(specifications),
        )
        for hypothesis_id, specifications in hypotheses
    ]
    body = {
        "schema_version":
            "atomic-scientific-specification-bundle-v1",
        "source_report_id": source_report_id,
        "source_contract": source_contract,
        "hypotheses": [
            row.model_dump(mode="json") for row in rows
        ],
        "hypothesis_count": len(rows),
        "atomic_specification_count": sum(
            len(row.specifications) for row in rows
        ),
        "canonical_scientific_representation": True,
        "stable_source_identity_preserved": True,
        "exact_text_is_identity_authority": False,
        "novelty_authority": False,
        "production_authority": False,
    }
    digest = _sha256_json(body)
    return AtomicScientificSpecificationBundle(
        **body,
        bundle_id=(
            "atomic_scientific_specification_bundle:"
            + digest[:20]
        ),
        bundle_sha256=digest,
    )


_ATOMIC_KINDS = frozenset(get_args(AtomicClaimKind))


def compile_atomic_specifications_from_source_binding(
    *,
    portfolio: HypothesisPortfolio,
    query_plan: LiteratureQueryPlan,
    source_binding_bundle: AtomicScientificSourceBindingBundle,
) -> list[tuple[str, list[CompiledAtomicSpecification]]]:
    """Compile canonical atomic specifications from stable source IDs.

    Identity comes only from the source-binding bundle's stable
    prediction/falsifier IDs. Exact source-text reconstruction is not used.
    """

    if query_plan.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError(
            "canonical specification query-plan/portfolio mismatch"
        )
    if source_binding_bundle.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError(
            "canonical specification source-bundle/portfolio mismatch"
        )
    if source_binding_bundle.source_query_plan_id != query_plan.plan_id:
        raise ValueError(
            "canonical specification source-bundle/query-plan ID mismatch"
        )
    if (
        source_binding_bundle.source_query_plan_sha256
        != query_plan.plan_sha256
    ):
        raise ValueError(
            "canonical specification source-bundle/query-plan SHA mismatch"
        )

    cards = {row.hypothesis_id: row for row in portfolio.hypotheses}
    if len(cards) != len(portfolio.hypotheses):
        raise ValueError(
            "duplicate hypothesis IDs in canonical specification portfolio"
        )

    records = {
        (row.hypothesis_id, row.claim_id): row
        for row in source_binding_bundle.records
    }
    if len(records) != len(source_binding_bundle.records):
        raise ValueError(
            "duplicate source-binding identity in canonical specification"
        )

    plan_claims = {
        (claim.hypothesis_id, claim.claim_id): claim
        for group in query_plan.claims
        for claim in group.claims
    }
    if sum(len(group.claims) for group in query_plan.claims) != len(
        plan_claims
    ):
        raise ValueError(
            "duplicate query-plan claim identity in canonical specification"
        )
    if set(records) != set(plan_claims):
        missing = sorted(set(plan_claims) - set(records))
        extra = sorted(set(records) - set(plan_claims))
        raise ValueError(
            "canonical specification source population mismatch: "
            + "missing="
            + repr(missing)
            + " extra="
            + repr(extra)
        )

    hypotheses: list[
        tuple[str, list[CompiledAtomicSpecification]]
    ] = []

    for group in query_plan.claims:
        if group.hypothesis_id not in cards:
            raise ValueError(
                "canonical specification query plan references unknown "
                "hypothesis: "
                + group.hypothesis_id
            )
        card = cards[group.hypothesis_id]
        specifications: list[CompiledAtomicSpecification] = []

        for claim in sorted(group.claims, key=lambda row: row.claim_rank):
            key = (claim.hypothesis_id, claim.claim_id)
            record = records[key]
            if record.claim_rank != claim.claim_rank:
                raise ValueError(
                    "canonical specification claim-rank mismatch: "
                    + claim.claim_id
                )
            observed_claim_sha = _sha256_json(
                claim.model_dump(mode="json")
            )
            if record.source_claim_sha256 != observed_claim_sha:
                raise ValueError(
                    "canonical specification source claim SHA mismatch: "
                    + claim.claim_id
                )
            if claim.kind not in _ATOMIC_KINDS:
                raise ValueError(
                    "canonical specification unsupported atomic kind: "
                    + claim.kind
                )
            if claim.novelty_selection_role is None:
                raise ValueError(
                    "canonical specification missing novelty selection role: "
                    + claim.claim_id
                )

            status, reasons, prediction, falsifier = (
                resolve_atomic_source_reference(
                    hypothesis=card,
                    prediction_observation_id=(
                        record.prediction_observation_id
                    ),
                    falsification_criterion_id=(
                        record.falsification_criterion_id
                    ),
                )
            )
            if status != "READY":
                raise ValueError(
                    "canonical specification stable source reference is "
                    "not READY: "
                    + claim.claim_id
                    + "; "
                    + ",".join(reasons)
                )
            assert prediction is not None
            assert falsifier is not None

            specifications.append(
                CompiledAtomicSpecification(
                    local_id=record.claim_local_id,
                    claim_id=claim.claim_id,
                    kind=claim.kind,
                    importance=claim.importance,
                    novelty_selection_role=(
                        claim.novelty_selection_role
                    ),
                    text=claim.text,
                    rationale=claim.rationale,
                    source_candidate_ids=[],
                    premise_statement_ids=list(
                        card.premise_statement_ids
                    ),
                    gap_statement_ids=list(
                        card.gap_statement_ids
                    ),
                    prior_art_identity_terms=list(
                        claim.prior_art_identity_terms
                    ),
                    relation_endpoint_anchors=list(
                        record.relation_endpoint_anchors
                    ),
                    scope_qualifier_spans=list(
                        record.scope_qualifier_spans
                    ),
                    directional_qualifier_spans=list(
                        record.directional_qualifier_spans
                    ),
                    relation_nucleus_terms=list(
                        claim.relation_nucleus_terms
                    ),
                    distinguishing_terms=list(
                        claim.distinguishing_terms
                    ),
                    required_bridge=claim.required_bridge,
                    observable=prediction.observable,
                    predicted_observation=(
                        claim.predicted_observation
                    ),
                    falsification_condition=(
                        claim.falsification_condition
                    ),
                    prediction_observation_id=(
                        prediction.observation_id
                    ),
                    falsification_criterion_id=(
                        falsifier.criterion_id
                    ),
                    search_concepts=list(claim.search_concepts),
                    search_queries=list(claim.search_queries),
                    scientific_structure=(
                        claim.scientific_structure
                    ),
                    scientific_structure_reason_codes=list(
                        claim.scientific_structure_reason_codes
                    ),
                )
            )

        hypotheses.append((group.hypothesis_id, specifications))

    if set(cards) != {hypothesis_id for hypothesis_id, _ in hypotheses}:
        raise ValueError(
            "canonical specification hypothesis population mismatch"
        )
    return hypotheses


def build_atomic_scientific_specification_bundle_from_source_bindings(
    *,
    source_report_id: str,
    source_contract: str,
    inputs: list[
        tuple[
            HypothesisPortfolio,
            LiteratureQueryPlan,
            AtomicScientificSourceBindingBundle,
        ]
    ],
) -> AtomicScientificSpecificationBundle:
    hypotheses: list[
        tuple[str, list[CompiledAtomicSpecification]]
    ] = []
    for portfolio, query_plan, source_binding_bundle in inputs:
        hypotheses.extend(
            compile_atomic_specifications_from_source_binding(
                portfolio=portfolio,
                query_plan=query_plan,
                source_binding_bundle=source_binding_bundle,
            )
        )
    return build_atomic_scientific_specification_bundle(
        source_report_id=source_report_id,
        source_contract=source_contract,
        hypotheses=hypotheses,
    )


__all__ = [
    "AtomicScientificSpecificationBundle",
    "AtomicScientificSpecificationBundleHypothesis",
    "build_atomic_scientific_specification_bundle",
    "build_atomic_scientific_specification_bundle_from_source_bindings",
    "compile_atomic_specifications_from_source_binding",
]
