from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import networkx as nx

from domains.sers.trend_precision_alpha4c212 import (
    SERS_AU_AG_TREND_PRECISION_ADAPTER as _ALPHA4C212_PRECISION_ADAPTER,
)
from dac_her.trend_precision import (
    PaperLocalTrendResult,
    TrendEvidenceAnnotation,
    TrendPrecisionAdapter,
)


SERS_AU_AG_TREND_SEMANTICS_ID = "sers_au_ag_trend_v5_alpha4c2121"
SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID = (
    "sers_au_ag_trend_precision_v4_alpha4c2121"
)


def _annotation(
    row: Mapping[str, Any],
    graph: nx.Graph,
) -> TrendEvidenceAnnotation:
    old = _ALPHA4C212_PRECISION_ADAPTER.annotate(row, graph)
    return replace(
        old,
        precision_semantics_id=
            SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
    )


def _consolidate(
    rows: list[Mapping[str, Any]],
    annotations: list[TrendEvidenceAnnotation],
    graphs: dict[str, nx.Graph],
) -> list[PaperLocalTrendResult]:
    # Rebuild historical alpha4c212 annotations so the historical consolidator
    # sees exactly the semantics it was written for.
    historical_annotations = []
    for row in rows:
        paper_id = str(row.get("paper_id", "")).strip()
        graph = graphs.get(paper_id)
        if graph is None:
            raise ValueError(
                f"Missing canonical graph for paper {paper_id!r}."
            )
        historical_annotations.append(
            _ALPHA4C212_PRECISION_ADAPTER.annotate(row, graph)
        )

    historical_results = (
        _ALPHA4C212_PRECISION_ADAPTER.consolidate(
            rows,
            historical_annotations,
            graphs,
        )
    )

    return [
        replace(
            result,
            trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
            precision_semantics_id=
                SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
        )
        for result in historical_results
    ]


SERS_AU_AG_TREND_PRECISION_ADAPTER = TrendPrecisionAdapter(
    adapter_id=_ALPHA4C212_PRECISION_ADAPTER.adapter_id,
    domain_profile_id=
        _ALPHA4C212_PRECISION_ADAPTER.domain_profile_id,
    trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
    precision_semantics_id=
        SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
    annotate_fn=_annotation,
    consolidate_fn=_consolidate,
)
