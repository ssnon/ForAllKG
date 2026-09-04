"""Candidate-local discovery-lineage projection for N10 continuation.

This artifact is NOT a discovery-axis synthesis report and must never claim
that axis synthesis was rerun for the generated Alpha6 candidate.

It preserves one already-established discovery-axis lineage while rebinding
only the local hypothesis identity from the first-pass hypothesis to the
post-generation continuation input.

The source synthesis report remains immutable and authoritative for the
original axis provenance.  This projection has no production-selection
authority.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxisPlan,
    DiscoveryAxisSynthesisReport,
    DiscoveryHypothesisLineage,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )


class N10PostGenerationLineageProjection(
    StrictModel
):
    schema_version: Literal[
        "n10-post-generation-lineage-projection-v1"
    ] = (
        "n10-post-generation-lineage-projection-v1"
    )

    projection_id: str
    projection_sha256: str

    source_lineage_report_id: str
    source_lineage_report_sha256: str

    source_axis_plan_id: str
    source_axis_plan_sha256: str

    source_dual_context_id: str
    source_dual_context_sha256: str

    source_hypothesis_id: str
    projected_hypothesis_id: str

    projected_portfolio_id: str
    projected_portfolio_sha256: str

    axis_id: str

    lineages: list[
        DiscoveryHypothesisLineage
    ]

    identity_projection_only: Literal[
        True
    ] = True

    production_authority: Literal[
        False
    ] = False


def _canonical_json(
    value: Any,
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
    value: Any,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            value
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(value)
        for value in parts
    ).encode(
        "utf-8"
    )

    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def build_n10_post_generation_lineage_projection(
    *,
    source_lineage_report:
        DiscoveryAxisSynthesisReport,

    axis_plan:
        DiscoveryAxisPlan,

    source_hypothesis_id:
        str,

    projected_portfolio:
        HypothesisPortfolio,
) -> N10PostGenerationLineageProjection:
    """Project exactly one frozen source lineage onto one H1 portfolio.

    Only ``hypothesis_id`` may change inside the lineage row.

    Discovery axis identity, inspiration identity, candidate-unit identity,
    epistemic status, fidelity state, inference state, and novelty state are
    inherited exactly from the source lineage.

    This is provenance context only.  It grants no generation or production
    authority.
    """

    if (
        source_lineage_report.axis_plan_id
        != axis_plan.plan_id
    ):
        raise ValueError(
            "source lineage / axis-plan ID mismatch"
        )

    if (
        source_lineage_report.axis_plan_sha256
        != axis_plan.plan_sha256
    ):
        raise ValueError(
            "source lineage / axis-plan SHA mismatch"
        )

    matches = [
        row
        for row
        in source_lineage_report.lineages
        if (
            row.hypothesis_id
            == source_hypothesis_id
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            "source lineage must contain exactly "
            "one matching hypothesis row"
        )

    if (
        len(
            projected_portfolio.hypotheses
        )
        != 1
    ):
        raise ValueError(
            "projected continuation portfolio "
            "must contain exactly one hypothesis"
        )

    projected_hypothesis_id = (
        projected_portfolio
        .hypotheses[0]
        .hypothesis_id
    )

    if (
        not projected_hypothesis_id
        or projected_hypothesis_id
        == source_hypothesis_id
    ):
        raise ValueError(
            "projection requires a distinct "
            "post-generation hypothesis identity"
        )

    source_row = matches[0]

    axis_matches = [
        row
        for row
        in axis_plan.axes
        if (
            row.axis_id
            == source_row.axis_id
        )
    ]

    if len(axis_matches) != 1:
        raise ValueError(
            "source lineage axis must resolve "
            "exactly once in axis plan"
        )

    source_dump = (
        source_row.model_dump(
            mode="json"
        )
    )

    projected_dump = deepcopy(
        source_dump
    )

    projected_dump[
        "hypothesis_id"
    ] = projected_hypothesis_id

    changed_fields = [
        key
        for key
        in (
            set(source_dump)
            | set(projected_dump)
        )
        if (
            source_dump.get(key)
            != projected_dump.get(key)
        )
    ]

    if (
        changed_fields
        != ["hypothesis_id"]
    ):
        raise ValueError(
            "lineage projection may change "
            "hypothesis_id only"
        )

    projected_row = (
        DiscoveryHypothesisLineage
        .model_validate(
            projected_dump
        )
    )

    if (
        projected_row.axis_id
        != source_row.axis_id
    ):
        raise ValueError(
            "lineage projection changed axis identity"
        )

    projected_portfolio_sha256 = (
        _sha256_json(
            projected_portfolio
        )
    )

    projection_id = _stable_id(
        "n10_post_generation_lineage_projection",
        source_lineage_report.report_id,
        source_lineage_report.report_sha256,
        source_hypothesis_id,
        projected_hypothesis_id,
        projected_portfolio.portfolio_id,
        projected_portfolio_sha256,
        projected_row.axis_id,
    )

    body = {
        "schema_version":
            "n10-post-generation-lineage-projection-v1",

        "projection_id":
            projection_id,

        "source_lineage_report_id":
            source_lineage_report.report_id,

        "source_lineage_report_sha256":
            source_lineage_report.report_sha256,

        "source_axis_plan_id":
            axis_plan.plan_id,

        "source_axis_plan_sha256":
            axis_plan.plan_sha256,

        "source_dual_context_id":
            source_lineage_report.source_dual_context_id,

        "source_dual_context_sha256":
            source_lineage_report.source_dual_context_sha256,

        "source_hypothesis_id":
            source_hypothesis_id,

        "projected_hypothesis_id":
            projected_hypothesis_id,

        "projected_portfolio_id":
            projected_portfolio.portfolio_id,

        "projected_portfolio_sha256":
            projected_portfolio_sha256,

        "axis_id":
            projected_row.axis_id,

        "lineages": [
            projected_row.model_dump(
                mode="json"
            )
        ],

        "identity_projection_only":
            True,

        "production_authority":
            False,
    }

    return (
        N10PostGenerationLineageProjection(
            **body,
            projection_sha256=(
                _sha256_json(
                    body
                )
            ),
        )
    )
