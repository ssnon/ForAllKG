from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
    NoveltyClaim,
    NoveltyClaimSemanticFidelityBindingDraft,
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


class AtomicScientificSourceBindingRecord(StrictModel):
    """Stable source provenance captured before NoveltyClaim loses source IDs.

    This record carries identity/provenance only. It does not establish
    scientific truth, semantic admissibility, novelty, readiness, or
    production authority.
    """

    schema_version: Literal[
        "atomic-scientific-source-binding-v1"
    ] = "atomic-scientific-source-binding-v1"

    hypothesis_id: str
    claim_id: str
    claim_rank: int = Field(ge=1)
    claim_local_id: str
    source_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    proposition_basis: str = ""
    relation_endpoint_anchors: list[str] = Field(default_factory=list)
    scope_qualifier_spans: list[str] = Field(default_factory=list)
    directional_qualifier_spans: list[str] = Field(default_factory=list)
    prediction_observation_id: str | None = None
    falsification_criterion_id: str | None = None

    source_identity_provenance_only: Literal[True] = True
    exact_text_reconstruction_used_for_identity: Literal[False] = False
    scientific_truth_assessed: Literal[False] = False
    semantic_admissibility_assessed: Literal[False] = False
    novelty_authority: Literal[False] = False
    readiness_authority: Literal[False] = False
    production_authority: Literal[False] = False


class AtomicScientificSourceBindingBundle(StrictModel):
    schema_version: Literal[
        "atomic-scientific-source-binding-bundle-v1"
    ] = "atomic-scientific-source-binding-bundle-v1"

    bundle_id: str
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_portfolio_id: str
    source_query_plan_id: str
    source_query_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    claim_count: int = Field(ge=0)
    prediction_source_id_present_count: int = Field(ge=0)
    falsifier_source_id_present_count: int = Field(ge=0)
    records: list[AtomicScientificSourceBindingRecord] = Field(
        default_factory=list
    )

    source_identity_provenance_only: Literal[True] = True
    exact_text_reconstruction_used_for_identity: Literal[False] = False
    scientific_truth_assessed: Literal[False] = False
    semantic_admissibility_assessed: Literal[False] = False
    novelty_authority: Literal[False] = False
    readiness_authority: Literal[False] = False
    production_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_bundle(
        self,
    ) -> "AtomicScientificSourceBindingBundle":
        if self.claim_count != len(self.records):
            raise ValueError(
                "atomic source-binding bundle claim_count mismatch"
            )

        keys = [
            (row.hypothesis_id, row.claim_id)
            for row in self.records
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "duplicate hypothesis/claim identity in source-binding bundle"
            )

        if self.prediction_source_id_present_count != sum(
            bool(str(row.prediction_observation_id or "").strip())
            for row in self.records
        ):
            raise ValueError(
                "source-binding bundle prediction-ID count mismatch"
            )
        if self.falsifier_source_id_present_count != sum(
            bool(str(row.falsification_criterion_id or "").strip())
            for row in self.records
        ):
            raise ValueError(
                "source-binding bundle falsifier-ID count mismatch"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("bundle_id")
        observed_sha = body.pop("bundle_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError(
                "atomic source-binding bundle SHA mismatch"
            )
        if observed_id != (
            "atomic_scientific_source_binding_bundle:"
            + expected_sha[:20]
        ):
            raise ValueError(
                "atomic source-binding bundle ID mismatch"
            )
        return self


def build_atomic_scientific_source_binding_record(
    *,
    hypothesis_id: str,
    claim_local_id: str,
    claim: NoveltyClaim,
    binding: NoveltyClaimSemanticFidelityBindingDraft,
) -> AtomicScientificSourceBindingRecord:
    if claim.hypothesis_id != hypothesis_id:
        raise ValueError(
            "source-binding record claim/hypothesis ID mismatch"
        )
    return AtomicScientificSourceBindingRecord(
        hypothesis_id=hypothesis_id,
        claim_id=claim.claim_id,
        claim_rank=claim.claim_rank,
        claim_local_id=claim_local_id,
        source_claim_sha256=_sha256_json(
            claim.model_dump(mode="json")
        ),
        proposition_basis=binding.proposition_basis,
        relation_endpoint_anchors=list(
            binding.relation_endpoint_anchors
        ),
        scope_qualifier_spans=list(
            binding.scope_qualifier_spans
        ),
        directional_qualifier_spans=list(
            binding.directional_qualifier_spans
        ),
        prediction_observation_id=(
            str(binding.prediction_observation_id).strip()
            if binding.prediction_observation_id is not None
            and str(binding.prediction_observation_id).strip()
            else None
        ),
        falsification_criterion_id=(
            str(binding.falsification_criterion_id).strip()
            if binding.falsification_criterion_id is not None
            and str(binding.falsification_criterion_id).strip()
            else None
        ),
    )


def build_atomic_scientific_source_binding_bundle(
    *,
    source_portfolio_id: str,
    query_plan: LiteratureQueryPlan,
    records: list[AtomicScientificSourceBindingRecord],
) -> AtomicScientificSourceBindingBundle:
    if query_plan.source_portfolio_id != source_portfolio_id:
        raise ValueError(
            "source-binding bundle query-plan/portfolio mismatch"
        )

    claims = [
        claim
        for group in query_plan.claims
        for claim in group.claims
    ]
    claim_by_key = {
        (claim.hypothesis_id, claim.claim_id): claim
        for claim in claims
    }
    if len(claim_by_key) != len(claims):
        raise ValueError(
            "duplicate query-plan claim identity in source-binding bundle"
        )

    record_by_key = {
        (row.hypothesis_id, row.claim_id): row
        for row in records
    }
    if len(record_by_key) != len(records):
        raise ValueError(
            "duplicate source-binding record identity"
        )

    if set(record_by_key) != set(claim_by_key):
        missing = sorted(set(claim_by_key) - set(record_by_key))
        extra = sorted(set(record_by_key) - set(claim_by_key))
        raise ValueError(
            "source-binding/query-plan claim population mismatch: "
            + "missing="
            + repr(missing)
            + " extra="
            + repr(extra)
        )

    ordered_records: list[AtomicScientificSourceBindingRecord] = []
    for group in query_plan.claims:
        for claim in sorted(
            group.claims,
            key=lambda row: row.claim_rank,
        ):
            key = (claim.hypothesis_id, claim.claim_id)
            record = record_by_key[key]
            if record.claim_rank != claim.claim_rank:
                raise ValueError(
                    "source-binding/query-plan claim-rank mismatch: "
                    + claim.claim_id
                )
            observed_claim_sha = _sha256_json(
                claim.model_dump(mode="json")
            )
            if record.source_claim_sha256 != observed_claim_sha:
                raise ValueError(
                    "source-binding/query-plan claim SHA mismatch: "
                    + claim.claim_id
                )
            ordered_records.append(record)

    body = {
        "schema_version":
            "atomic-scientific-source-binding-bundle-v1",
        "source_portfolio_id": source_portfolio_id,
        "source_query_plan_id": query_plan.plan_id,
        "source_query_plan_sha256": query_plan.plan_sha256,
        "claim_count": len(ordered_records),
        "prediction_source_id_present_count": sum(
            bool(str(row.prediction_observation_id or "").strip())
            for row in ordered_records
        ),
        "falsifier_source_id_present_count": sum(
            bool(str(row.falsification_criterion_id or "").strip())
            for row in ordered_records
        ),
        "records": [
            row.model_dump(mode="json")
            for row in ordered_records
        ],
        "source_identity_provenance_only": True,
        "exact_text_reconstruction_used_for_identity": False,
        "scientific_truth_assessed": False,
        "semantic_admissibility_assessed": False,
        "novelty_authority": False,
        "readiness_authority": False,
        "production_authority": False,
    }
    digest = _sha256_json(body)
    return AtomicScientificSourceBindingBundle(
        **body,
        bundle_id=(
            "atomic_scientific_source_binding_bundle:"
            + digest[:20]
        ),
        bundle_sha256=digest,
    )


__all__ = [
    "AtomicScientificSourceBindingBundle",
    "AtomicScientificSourceBindingRecord",
    "build_atomic_scientific_source_binding_bundle",
    "build_atomic_scientific_source_binding_record",
]
