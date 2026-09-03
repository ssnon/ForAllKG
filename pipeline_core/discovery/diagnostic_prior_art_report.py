from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.diagnostic_prior_art_review import (
    DiagnosticClaimPriorArtReview,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )


class DiagnosticPriorArtReviewReport(
    _StrictModel
):
    schema_version: Literal[
        "diagnostic-prior-art-review-report-v1"
    ] = (
        "diagnostic-prior-art-review-report-v1"
    )

    report_id: str
    report_sha256: str

    source_portfolio_id: str
    source_diagnostic_query_plan_id: str
    source_diagnostic_query_plan_sha256: str
    source_diagnostic_prior_art_packet_id: str
    source_diagnostic_prior_art_packet_sha256: str

    reviews: list[
        DiagnosticClaimPriorArtReview
    ] = Field(
        default_factory=list
    )

    reviewed_claim_count: int
    signal_claim_count: int
    signal_work_count: int

    shadow_only: Literal[
        True
    ] = True

    scientific_selection_changed: Literal[
        False
    ] = False

    epistemic_usage: Literal[
        "diagnostic_prior_art_only_not_full_claim_status_authority"
    ] = (
        "diagnostic_prior_art_only_not_full_claim_status_authority"
    )


def _canonical_json(
    value: object,
) -> str:
    if hasattr(
        value,
        "model_dump",
    ):
        value = value.model_dump(
            mode="json"
        )

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(
    value: object,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            value
        ).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def build_diagnostic_review_report(
    *,
    source_portfolio_id: str,
    source_query_plan_id: str,
    source_query_plan_sha256: str,
    source_prior_art_packet_id: str,
    source_prior_art_packet_sha256: str,
    reviews: list[
        DiagnosticClaimPriorArtReview
    ],
) -> DiagnosticPriorArtReviewReport:
    ordered = sorted(
        reviews,
        key=lambda row: (
            row.hypothesis_id,
            row.claim_id,
        ),
    )

    signal_claim_count = sum(
        bool(
            row.signal_work_ids
        )
        for row in ordered
    )

    signal_work_ids = sorted(
        {
            work_id
            for row in ordered
            for work_id
            in row.signal_work_ids
        }
    )

    body = {
        "schema_version":
            "diagnostic-prior-art-review-report-v1",

        "source_portfolio_id":
            source_portfolio_id,

        "source_diagnostic_query_plan_id":
            source_query_plan_id,

        "source_diagnostic_query_plan_sha256":
            source_query_plan_sha256,

        "source_diagnostic_prior_art_packet_id":
            source_prior_art_packet_id,

        "source_diagnostic_prior_art_packet_sha256":
            source_prior_art_packet_sha256,

        "reviews": [
            row.model_dump(
                mode="json"
            )
            for row in ordered
        ],

        "reviewed_claim_count":
            len(ordered),

        "signal_claim_count":
            signal_claim_count,

        "signal_work_count":
            len(signal_work_ids),

        "shadow_only":
            True,

        "scientific_selection_changed":
            False,

        "epistemic_usage": (
            "diagnostic_prior_art_only_"
            "not_full_claim_status_authority"
        ),
    }

    report_id = _stable_id(
        "diagnostic_prior_art_review_report",
        source_query_plan_id,
        source_prior_art_packet_id,
        *[
            (
                row.claim_id
                + ":"
                + ",".join(
                    row.signal_work_ids
                )
            )
            for row in ordered
        ],
    )

    hashed_body = {
        **body,
        "report_id":
            report_id,
    }

    return (
        DiagnosticPriorArtReviewReport(
            **hashed_body,
            report_sha256=(
                _sha256_json(
                    hashed_body
                )
            ),
        )
    )
