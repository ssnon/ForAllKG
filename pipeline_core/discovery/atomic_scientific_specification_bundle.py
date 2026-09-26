from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.atomic_scientific_specification import (
    CompiledAtomicSpecification,
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


__all__ = [
    "AtomicScientificSpecificationBundle",
    "AtomicScientificSpecificationBundleHypothesis",
    "build_atomic_scientific_specification_bundle",
]
