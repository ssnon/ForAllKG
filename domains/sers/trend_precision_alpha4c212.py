from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Any, Mapping

import networkx as nx

from domains.sers.trend_precision_alpha4c211 import (
    SERS_AU_AG_TREND_PRECISION_ADAPTER as _ALPHA4C211_PRECISION_ADAPTER,
)
from dac_her.trend_precision import (
    PaperLocalTrendResult,
    TrendEvidenceAnnotation,
    TrendPrecisionAdapter,
    stable_local_trend_result_id,
)


SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID = (
    "sers_au_ag_trend_precision_v3_alpha4c212"
)
SERS_AU_AG_TREND_SEMANTICS_ID = "sers_au_ag_trend_v4_alpha4c212"


def _annotation(
    row: Mapping[str, Any],
    graph: nx.Graph,
) -> TrendEvidenceAnnotation:
    old = _ALPHA4C211_PRECISION_ADAPTER.annotate(row, graph)

    independent = str(
        row.get("independent_variable_key", "")
    ).strip()
    dependent = str(
        row.get("dependent_observable_key", "")
    ).strip()

    updates: dict[str, object] = {
        "precision_semantics_id":
            SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
    }

    if independent == "spr_excitation_detuning":
        updates.update({
            "control_family": "optical_alignment",
            "canonical_control_unit": "nm",
            # A raw laser wavelength such as 532 nm is not a detuning value.
            "source_control_value_text": "",
            "canonical_control_value_numeric": None,
            "normalization_transform": "",
        })

    if (
        independent == "spr_excitation_detuning"
        and dependent == "sers_enhancement_factor"
    ):
        updates["observable_semantics"] = (
            "formal_sers_enhancement_factor"
        )

    return replace(old, **updates)


def _subject_text(
    graph: nx.Graph,
    subject_ids: tuple[str, ...],
) -> str:
    parts: list[str] = []
    for node_id in subject_ids:
        parts.append(str(node_id))
        if node_id in graph:
            attrs = graph.nodes[node_id]
            for key in ("label", "name", "description"):
                value = str(attrs.get(key, "")).strip()
                if value:
                    parts.append(value)
    return " ".join(parts).casefold()


def _material_signature(
    graph: nx.Graph,
    subject_ids: tuple[str, ...],
) -> tuple[str, ...]:
    text = _subject_text(graph, subject_ids)
    metals: set[str] = set()
    if re.search(
        r"(?<![a-z])au(?:\d+(?:[._-]\d+)*)?(?![a-z])",
        text,
    ):
        metals.add("au")
    if re.search(
        r"(?<![a-z])ag(?:\d+(?:[._-]\d+)*)?(?![a-z])",
        text,
    ):
        metals.add("ag")
    return tuple(sorted(metals))


def _structural_form_signature(
    graph: nx.Graph,
    subject_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """
    Conservative morphology/architecture signature.

    This prevents same-landmark Au/Ag claims from being merged across
    unrelated nanobox/nanocube/nanorod/etc. systems.
    """
    text = _subject_text(graph, subject_ids)
    features: set[str] = set()

    patterns = {
        "nanocube": r"\bnano\s*cube(?:s)?\b|\bnanocube(?:s)?\b",
        "nanobox": r"\bnano\s*box(?:es)?\b|\bnanobox(?:es)?\b",
        "nanostar": r"\bnano\s*star(?:s)?\b|\bnanostar(?:s)?\b",
        "nanoplate": r"\bnano\s*plate(?:s)?\b|\bnanoplate(?:s)?\b",
        "nanorod": r"\bnano\s*rod(?:s)?\b|\bnanorod(?:s)?\b",
        "nanoparticle":
            r"\bnano\s*particle(?:s)?\b|\bnanoparticle(?:s)?\b",
        "core_shell":
            r"\bcore[-\s@/]*shell\b|\bcore@shell\b|\bau@ag\b|\bag@au\b",
        "double_shelled": r"\bdouble[-\s]*shell(?:ed)?\b",
        "alloy": r"\balloy\b",
    }
    for key, pattern in patterns.items():
        if re.search(pattern, text):
            features.add(key)

    return tuple(sorted(features))


def _landmark_signature(
    result: PaperLocalTrendResult,
    annotation_by_id: dict[str, TrendEvidenceAnnotation],
) -> tuple[float, str, str] | None:
    values: set[tuple[float, str, str]] = set()
    for trend_id in result.member_trend_ids:
        annotation = annotation_by_id.get(trend_id)
        if annotation is None:
            continue
        value = annotation.canonical_control_value_numeric
        if value is None:
            continue
        values.add((
            round(float(value), 12),
            str(annotation.canonical_control_unit),
            str(annotation.normalization_transform),
        ))
    if len(values) != 1:
        return None
    return next(iter(values))


def _union(
    results: list[PaperLocalTrendResult],
    field: str,
) -> tuple[str, ...]:
    return tuple(sorted({
        str(value)
        for result in results
        for value in getattr(result, field)
        if str(value).strip()
    }))


def _merge_group(
    results: list[PaperLocalTrendResult],
) -> PaperLocalTrendResult:
    first = results[0]
    members = _union(results, "member_trend_ids")
    result_id = stable_local_trend_result_id(
        paper_id=first.paper_id,
        result_lane=first.result_lane,
        independent_variable_key=first.independent_variable_key,
        dependent_observable_key=first.dependent_observable_key,
        direction=first.direction,
        shape=first.shape,
        member_trend_ids=members,
    )
    return replace(
        first,
        result_id=result_id,
        trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
        precision_semantics_id=
            SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
        member_trend_ids=members,
        evidence_kinds=_union(results, "evidence_kinds"),
        trend_subject_ids=_union(results, "trend_subject_ids"),
        reference_subject_ids=_union(
            results, "reference_subject_ids"
        ),
        source_claim_ids=_union(results, "source_claim_ids"),
        source_measurement_ids=_union(
            results, "source_measurement_ids"
        ),
        source_measurement_result_ids=_union(
            results, "source_measurement_result_ids"
        ),
        source_calculation_ids=_union(
            results, "source_calculation_ids"
        ),
        source_node_ids=_union(results, "source_node_ids"),
        support_mention_count=len(members),
    )


def _single_evidence_base_results(
    rows: list[Mapping[str, Any]],
    graphs: dict[str, nx.Graph],
) -> list[PaperLocalTrendResult]:
    """
    Critical alpha4c.2.1.2 invariant:

    Never allow the alpha4c211 consolidator to merge two raw trend mentions
    before alpha4c212 evaluates their explicit landmarks.

    Each raw TrendEvidence is first converted independently into exactly one
    historical paper-local result. Only then does alpha4c212 apply its own
    stricter grouping rule.
    """
    results: list[PaperLocalTrendResult] = []

    for row in rows:
        paper_id = str(row.get("paper_id", "")).strip()
        graph = graphs.get(paper_id)
        if graph is None:
            raise ValueError(
                f"Missing canonical graph for paper {paper_id!r}."
            )

        old_annotation = (
            _ALPHA4C211_PRECISION_ADAPTER.annotate(
                row,
                graph,
            )
        )
        per_row = (
            _ALPHA4C211_PRECISION_ADAPTER.consolidate(
                [row],
                [old_annotation],
                graphs,
            )
        )
        if len(per_row) != 1:
            raise ValueError(
                "alpha4c212 requires one historical paper-local result "
                f"per raw TrendEvidence before regrouping; got "
                f"{len(per_row)} for trend_id="
                f"{row.get('trend_id')!r}."
            )

        results.append(
            replace(
                per_row[0],
                trend_semantics_id=
                    SERS_AU_AG_TREND_SEMANTICS_ID,
                precision_semantics_id=
                    SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
            )
        )

    return results


def _claim_group_key(
    result: PaperLocalTrendResult,
    *,
    graph: nx.Graph,
    annotation_by_id: dict[str, TrendEvidenceAnnotation],
) -> tuple[object, ...]:
    relation = (
        result.paper_id,
        result.independent_variable_key,
        result.dependent_observable_key,
        result.direction,
        result.shape,
        result.control_family,
        result.observable_semantics,
    )

    landmark = _landmark_signature(
        result,
        annotation_by_id,
    )

    if (
        result.control_family == "structural"
        and landmark is not None
    ):
        material = _material_signature(
            graph,
            result.trend_subject_ids,
        )
        structural_form = _structural_form_signature(
            graph,
            result.trend_subject_ids,
        )

        # The new alias rule is fail-closed:
        # require both explicit metal family and structural-form support.
        if material and structural_form:
            return (
                "structural_alias",
                *relation,
                landmark,
                material,
                structural_form,
            )

    # Default paper-local identity remains strict on exact normalized subjects.
    return (
        "exact_subject",
        *relation,
        tuple(sorted(result.trend_subject_ids)),
    )


def _consolidate(
    rows: list[Mapping[str, Any]],
    annotations: list[TrendEvidenceAnnotation],
    graphs: dict[str, nx.Graph],
) -> list[PaperLocalTrendResult]:
    base_results = _single_evidence_base_results(
        rows,
        graphs,
    )
    annotation_by_id = {
        annotation.trend_id: annotation
        for annotation in annotations
    }

    numeric: list[PaperLocalTrendResult] = []
    claim_groups: dict[
        tuple[object, ...],
        list[PaperLocalTrendResult],
    ] = defaultdict(list)

    for result in base_results:
        if result.result_lane != "claim":
            numeric.append(result)
            continue

        graph = graphs[result.paper_id]
        key = _claim_group_key(
            result,
            graph=graph,
            annotation_by_id=annotation_by_id,
        )
        claim_groups[key].append(result)

    merged: list[PaperLocalTrendResult] = list(numeric)
    for _key, group in sorted(
        claim_groups.items(),
        key=lambda item: str(item[0]),
    ):
        if len(group) == 1:
            merged.append(group[0])
        else:
            merged.append(_merge_group(group))

    return sorted(
        merged,
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
    adapter_id=_ALPHA4C211_PRECISION_ADAPTER.adapter_id,
    domain_profile_id=
        _ALPHA4C211_PRECISION_ADAPTER.domain_profile_id,
    trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
    precision_semantics_id=
        SERS_AU_AG_TREND_PRECISION_SEMANTICS_ID,
    annotate_fn=_annotation,
    consolidate_fn=_consolidate,
)
