from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.hypothesis_semantic_contracts import (
    HypothesisSemanticReview,
    SEMANTIC_DIMENSIONS,
    SemanticDimension,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SemanticScientificDisposition = Literal[
    "PASS",
    "REQUIRES_SEMANTIC_INTERVENTION",
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


class HypothesisSemanticDispositionV1(StrictModel):
    schema_version: Literal[
        "hypothesis-semantic-disposition-v1"
    ] = "hypothesis-semantic-disposition-v1"

    disposition_id: str
    disposition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_review_id: str
    source_portfolio_id: str
    source_portfolio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    disposition: SemanticScientificDisposition
    semantic_admissible_for_pre_n10: bool

    failed_dimensions: list[SemanticDimension] = Field(default_factory=list)
    warning_dimensions: list[SemanticDimension] = Field(default_factory=list)
    failed_hypothesis_ids: list[str] = Field(default_factory=list)
    portfolio_level_failed_dimensions: list[SemanticDimension] = Field(
        default_factory=list
    )

    semantic_review_validated: Literal[True] = True
    fail_blocks_pre_n10: Literal[True] = True
    warning_blocks_pre_n10: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    final_scientific_rejection_authority_created: Literal[False] = False
    repair_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_disposition(self) -> "HypothesisSemanticDispositionV1":
        expected_admissible = not self.failed_dimensions
        if self.semantic_admissible_for_pre_n10 != expected_admissible:
            raise ValueError("semantic admissibility mismatch")
        expected_disposition: SemanticScientificDisposition = (
            "PASS"
            if expected_admissible
            else "REQUIRES_SEMANTIC_INTERVENTION"
        )
        if self.disposition != expected_disposition:
            raise ValueError("semantic disposition mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("disposition_id")
        observed_sha = body.pop("disposition_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("semantic disposition SHA mismatch")
        if observed_id != (
            "hypothesis_semantic_disposition_v1:" + expected_sha[:20]
        ):
            raise ValueError("semantic disposition ID mismatch")
        return self


def compile_hypothesis_semantic_disposition_v1(
    *,
    review: HypothesisSemanticReview,
    portfolio: HypothesisPortfolio,
) -> HypothesisSemanticDispositionV1:
    if review.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError(
            "semantic review/portfolio ID mismatch"
        )

    portfolio_sha = _sha256_json(portfolio)
    if review.source_portfolio_sha256 != portfolio_sha:
        raise ValueError(
            "semantic review/portfolio SHA mismatch"
        )

    dimensions = [row.dimension for row in review.dimensions]
    if len(dimensions) != len(set(dimensions)):
        raise ValueError("semantic disposition received duplicate dimensions")
    if set(dimensions) != set(SEMANTIC_DIMENSIONS):
        raise ValueError("semantic disposition dimension set mismatch")

    failed_rows = [
        row for row in review.dimensions if row.verdict == "fail"
    ]
    failed_dimensions = sorted(
        {row.dimension for row in failed_rows}
    )
    warning_dimensions = sorted(
        {
            row.dimension
            for row in review.dimensions
            if row.verdict == "warning"
        }
    )
    failed_hypothesis_ids = sorted(
        {
            hypothesis_id
            for row in failed_rows
            for hypothesis_id in row.hypothesis_ids
        }
    )
    portfolio_level_failed_dimensions = sorted(
        {
            row.dimension
            for row in failed_rows
            if not row.hypothesis_ids
        }
    )

    admissible = not failed_dimensions
    body = {
        "schema_version": "hypothesis-semantic-disposition-v1",
        "source_review_id": review.review_id,
        "source_portfolio_id": portfolio.portfolio_id,
        "source_portfolio_sha256": portfolio_sha,
        "disposition": (
            "PASS"
            if admissible
            else "REQUIRES_SEMANTIC_INTERVENTION"
        ),
        "semantic_admissible_for_pre_n10": admissible,
        "failed_dimensions": failed_dimensions,
        "warning_dimensions": warning_dimensions,
        "failed_hypothesis_ids": failed_hypothesis_ids,
        "portfolio_level_failed_dimensions": (
            portfolio_level_failed_dimensions
        ),
        "semantic_review_validated": True,
        "fail_blocks_pre_n10": True,
        "warning_blocks_pre_n10": False,
        "novelty_authority_created": False,
        "final_scientific_rejection_authority_created": False,
        "repair_performed": False,
    }
    digest = _sha256_json(body)
    return HypothesisSemanticDispositionV1(
        **body,
        disposition_id=(
            "hypothesis_semantic_disposition_v1:" + digest[:20]
        ),
        disposition_sha256=digest,
    )


__all__ = [
    "HypothesisSemanticDispositionV1",
    "SemanticScientificDisposition",
    "compile_hypothesis_semantic_disposition_v1",
]
