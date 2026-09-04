"""Fail-closed Alpha6 lineage input loader.

Alpha6 normally consumes the frozen discovery-axis synthesis report.
A bounded post-generation continuation may instead consume an explicit
N10 identity-only lineage projection.

Both inputs expose ``.lineages``.  No projection metadata grants scientific,
generation, or production authority.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxisSynthesisReport,
)
from pipeline_core.discovery.n10_post_generation_lineage_projection import (
    N10PostGenerationLineageProjection,
)


Alpha6LineageInput: TypeAlias = (
    DiscoveryAxisSynthesisReport
    | N10PostGenerationLineageProjection
)


def load_alpha6_lineage_input(
    path: Path,
) -> Alpha6LineageInput:
    """Load one explicitly supported Alpha6 lineage artifact."""

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Alpha6 lineage input must be a JSON object"
        )

    schema = payload.get(
        "schema_version"
    )

    if (
        schema
        == "discovery-axis-synthesis-report-v1"
    ):
        return (
            DiscoveryAxisSynthesisReport
            .model_validate(
                payload
            )
        )

    if (
        schema
        == "n10-post-generation-lineage-projection-v1"
    ):
        projection = (
            N10PostGenerationLineageProjection
            .model_validate(
                payload
            )
        )

        if (
            projection.production_authority
            is not False
        ):
            raise ValueError(
                "N10 lineage projection must not carry "
                "production authority"
            )

        if (
            projection.identity_projection_only
            is not True
        ):
            raise ValueError(
                "N10 lineage projection must be "
                "identity-only"
            )

        if (
            len(
                projection.lineages
            )
            != 1
        ):
            raise ValueError(
                "N10 continuation lineage projection "
                "must contain exactly one lineage row"
            )

        if (
            projection.lineages[0].hypothesis_id
            != projection.projected_hypothesis_id
        ):
            raise ValueError(
                "projected lineage identity does not "
                "match projected hypothesis"
            )

        return projection

    raise ValueError(
        "unsupported Alpha6 lineage schema: "
        + repr(schema)
    )
