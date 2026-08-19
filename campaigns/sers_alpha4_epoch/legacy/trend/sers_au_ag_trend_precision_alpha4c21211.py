from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import networkx as nx

from campaigns.sers_alpha4_epoch.legacy.trend.sers_au_ag_trend_precision_alpha4c2121 import (
    SERS_AU_AG_TREND_PRECISION_ADAPTER as _ALPHA4C2121_PRECISION_ADAPTER,
)
from dac_her.trend_precision import (
    PaperLocalTrendResult,
    TrendEvidenceAnnotation,
    TrendPrecisionAdapter,
)


SERS_AU_AG_TREND_SEMANTICS_ID = "sers_au_ag_trend_v5_alpha4c2121"
SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID = (
    "sers_au_ag_trend_precision_v5_alpha4c21211"
)


def _annotation(
    row: Mapping[str, Any],
    graph: nx.Graph,
) -> TrendEvidenceAnnotation:
    old = _ALPHA4C2121_PRECISION_ADAPTER.annotate(row, graph)
    return replace(
        old,
        precision_semantics_id=
            SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
    )


def _member_annotations(
    result: PaperLocalTrendResult,
    annotation_by_id: dict[str, TrendEvidenceAnnotation],
) -> list[TrendEvidenceAnnotation]:
    values: list[TrendEvidenceAnnotation] = []
    for trend_id in result.member_trend_ids:
        annotation = annotation_by_id.get(trend_id)
        if annotation is None:
            raise ValueError(
                "alpha4c21211 precision reconciliation is missing "
                f"annotation for member trend_id={trend_id!r} "
                f"in result_id={result.result_id!r}."
            )
        values.append(annotation)
    if not values:
        raise ValueError(
            "alpha4c21211 precision reconciliation requires every "
            f"PaperLocalTrendResult to contain at least one member: "
            f"{result.result_id!r}."
        )
    return values


def _unique_member_semantic(
    result: PaperLocalTrendResult,
    annotations: list[TrendEvidenceAnnotation],
    field: str,
) -> str:
    values = {
        str(getattr(annotation, field)).strip()
        for annotation in annotations
    }
    if len(values) != 1:
        raise ValueError(
            "alpha4c21211 refuses to collapse incompatible member "
            f"{field} values for result_id={result.result_id!r}: "
            f"{sorted(values)!r}."
        )
    return next(iter(values))


def _reconcile_result_semantics(
    result: PaperLocalTrendResult,
    annotation_by_id: dict[str, TrendEvidenceAnnotation],
) -> PaperLocalTrendResult:
    """
    Reconcile paper-local semantic labels from the ACTIVE annotations.

    alpha4c2121 trend re-grounding can introduce a new independent variable
    (e.g. spr_excitation_detuning). Its historical consolidation ancestry may
    still materialize a stale local-result control_family such as "other".

    The active annotations are the precision-layer source of truth for:
      - control_family
      - observable_semantics

    Reconciliation is fail-closed: all member annotations must agree.
    """
    annotations = _member_annotations(
        result,
        annotation_by_id,
    )
    control_family = _unique_member_semantic(
        result,
        annotations,
        "control_family",
    )
    observable_semantics = _unique_member_semantic(
        result,
        annotations,
        "observable_semantics",
    )

    return replace(
        result,
        trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
        precision_semantics_id=
            SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
        control_family=control_family,
        observable_semantics=observable_semantics,
    )


def _consolidate(
    rows: list[Mapping[str, Any]],
    annotations: list[TrendEvidenceAnnotation],
    graphs: dict[str, nx.Graph],
) -> list[PaperLocalTrendResult]:
    # alpha4c2121 expects annotations carrying its historical precision
    # semantics. Rebuild those only for the historical consolidator.
    historical_annotations: list[TrendEvidenceAnnotation] = []
    for row in rows:
        paper_id = str(row.get("paper_id", "")).strip()
        graph = graphs.get(paper_id)
        if graph is None:
            raise ValueError(
                f"Missing canonical graph for paper {paper_id!r}."
            )
        historical_annotations.append(
            _ALPHA4C2121_PRECISION_ADAPTER.annotate(
                row,
                graph,
            )
        )

    historical_results = (
        _ALPHA4C2121_PRECISION_ADAPTER.consolidate(
            rows,
            historical_annotations,
            graphs,
        )
    )

    annotation_by_id = {
        annotation.trend_id: annotation
        for annotation in annotations
    }

    reconciled = [
        _reconcile_result_semantics(
            result,
            annotation_by_id,
        )
        for result in historical_results
    ]

    return sorted(
        reconciled,
        key=lambda row: (
            row.paper_id,
            row.independent_variable_key,
            row.dependent_observable_key,
            row.result_lane,
            row.direction,
            row.shape,
            row.result_id,
        ),
    )


SERS_AU_AG_TREND_PRECISION_ADAPTER = TrendPrecisionAdapter(
    adapter_id=_ALPHA4C2121_PRECISION_ADAPTER.adapter_id,
    domain_profile_id=
        _ALPHA4C2121_PRECISION_ADAPTER.domain_profile_id,
    trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
    precision_semantics_id=
        SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
    annotate_fn=_annotation,
    consolidate_fn=_consolidate,
)
