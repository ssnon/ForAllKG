from __future__ import annotations

from dataclasses import replace

from campaigns.sers_alpha4_epoch.legacy.trend.sers_au_ag_trend_alpha4c5g2r2 import (
    SERS_AU_AG_TREND_ADAPTER as V6R2_TREND_ADAPTER,
)
from campaigns.sers_alpha4_epoch.legacy.trend.sers_au_ag_trend_precision_alpha4c21211 import (
    SERS_AU_AG_TREND_PRECISION_ADAPTER as V5_PRECISION_ADAPTER,
)
from dac_her.trend_precision import TrendPrecisionAdapter


V6R2_TREND_SEMANTICS_ID = (
    "sers_au_ag_trend_v6r2_alpha4c5g2r2"
)
V5_PRECISION_SEMANTICS_ID = (
    V5_PRECISION_ADAPTER.precision_semantics_id
)


def _annotate(row, graph):
    # Scientific annotation behavior is exactly the already-frozen v5
    # precision adapter.
    return V5_PRECISION_ADAPTER.annotate(row, graph)


def _consolidate(rows, annotations, graphs):
    # Scientific consolidation behavior is exactly the already-frozen v5
    # precision adapter. Only the parent Trend semantics metadata is rebound
    # to the frozen v6r2 source semantics.
    results = V5_PRECISION_ADAPTER.consolidate(
        rows,
        annotations,
        graphs,
    )
    return [
        replace(
            result,
            trend_semantics_id=V6R2_TREND_SEMANTICS_ID,
        )
        for result in results
    ]


V6R2_RUNTIME_PRECISION_ADAPTER = TrendPrecisionAdapter(
    adapter_id=V5_PRECISION_ADAPTER.adapter_id,
    domain_profile_id=V5_PRECISION_ADAPTER.domain_profile_id,
    trend_semantics_id=V6R2_TREND_SEMANTICS_ID,
    precision_semantics_id=V5_PRECISION_SEMANTICS_ID,
    annotate_fn=_annotate,
    consolidate_fn=_consolidate,
)
