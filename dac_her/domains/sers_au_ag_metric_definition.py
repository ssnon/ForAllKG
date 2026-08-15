from __future__ import annotations

import json
import re
from typing import Any, Iterable

import networkx as nx

from dac_her.metric_definition_context import stable_metric_definition_id
from dac_her.metric_definition_domain import (
    MetricDefinitionContext,
    MetricDefinitionDomainAdapter,
)


SERS_METRIC_DEFINITION_SEMANTICS_ID = (
    "sers_au_ag_metric_definition_v3_alpha4c4c1"
)

SERS_SUPPORTED_OBSERVABLES = frozenset({
    "sers_enhancement_factor",
    "detection_limit",
})

SERS_DEFINITION_FAMILIES = frozenset({
    "molecule_normalized_intensity_ratio",
    "concentration_normalized_intensity_ratio",
    "reported_ef_unspecified",
    "calibration_curve_statistical",
    "lowest_detected_concentration",
    "reported_lod_unspecified",
})

SERS_AGGREGATION_SCOPES = frozenset({
    "single_particle",
    "population_mean",
    "population_distribution",
    "lower_bound",
    "maximum",
    "substrate_summary",
    "unspecified",
    "not_applicable",
})

SERS_NORMALIZATION_BASES = frozenset({
    "molecule_count",
    "concentration",
    "unspecified",
    "not_applicable",
})

SERS_REFERENCE_BASES = frozenset({
    "normal_raman",
    "normal_raman_on_glass",
    "unspecified",
    "not_applicable",
})

_RAMAN_PEAK_RE = re.compile(
    r"(?<![\d.])(?P<number>\d+(?:\.\d+)?)\s*cm\s*(?:\^?\s*[-−]\s*1|⁻¹)\b",
    re.I,
)


def _relation(attrs: dict[str, Any]) -> str:
    return str(attrs.get("relation", "")).strip()


def _incoming(graph: nx.Graph, node_id: str, relation: str) -> list[str]:
    if not graph.is_directed():
        return []
    if graph.is_multigraph():
        iterator = graph.in_edges(node_id, keys=True, data=True)
        return sorted({
            str(left)
            for left, _right, _key, attrs in iterator
            if _relation(dict(attrs)) == relation
        })
    return sorted({
        str(left)
        for left, _right, attrs in graph.in_edges(node_id, data=True)
        if _relation(dict(attrs)) == relation
    })


def _outgoing(graph: nx.Graph, node_id: str, relation: str) -> list[str]:
    if not graph.is_directed():
        return []
    if graph.is_multigraph():
        iterator = graph.out_edges(node_id, keys=True, data=True)
        return sorted({
            str(right)
            for _left, right, _key, attrs in iterator
            if _relation(dict(attrs)) == relation
        })
    return sorted({
        str(right)
        for _left, right, attrs in graph.out_edges(node_id, data=True)
        if _relation(dict(attrs)) == relation
    })


def _observable_key(graph: nx.Graph, measurement_id: str) -> str:
    attrs = graph.nodes[measurement_id]
    raw = str(
        attrs.get("metric_id")
        or attrs.get("metric")
        or attrs.get("label")
        or ""
    ).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")


def _node_text(graph: nx.Graph, node_id: str) -> str:
    attrs = graph.nodes[node_id]
    keys = (
        "label",
        "source_expression",
        "description",
        "qualifier",
        "value_text",
        "method_details",
        "basis",
        "calculation_type",
        "raw_method_name",
        "method_label",
    )
    return " ".join(
        str(attrs.get(key, "")).strip()
        for key in keys
        if str(attrs.get(key, "")).strip()
    )


def _source_text(graph: nx.Graph, node_ids: Iterable[str]) -> str:
    return " ".join(
        _node_text(graph, node_id)
        for node_id in sorted(set(map(str, node_ids)))
        if node_id in graph
    ).strip()


def _typed_sources(
    graph: nx.Graph,
    node_ids: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    measurements: list[str] = []
    groups: list[str] = []
    experiments: list[str] = []
    calculations: list[str] = []
    for node_id in sorted(set(map(str, node_ids))):
        if node_id not in graph:
            continue
        node_type = str(graph.nodes[node_id].get("type", ""))
        if node_type == "Measurement":
            measurements.append(node_id)
        elif node_type == "MeasurementGroup":
            groups.append(node_id)
        elif node_type == "Experiment":
            experiments.append(node_id)
        elif node_type == "Calculation":
            calculations.append(node_id)
    return (
        tuple(measurements),
        tuple(groups),
        tuple(experiments),
        tuple(calculations),
    )


def _local_sources(graph: nx.Graph, measurement_id: str) -> tuple[str, ...]:
    producers = _incoming(graph, measurement_id, "HAS_MEASUREMENT")
    groups = _outgoing(graph, measurement_id, "IN_MEASUREMENT_GROUP")
    allowed_producers = [
        node_id
        for node_id in producers
        if node_id in graph
        and str(graph.nodes[node_id].get("type", ""))
        in {"Experiment", "Calculation"}
    ]
    allowed_groups = [
        node_id
        for node_id in groups
        if node_id in graph
        and str(graph.nodes[node_id].get("type", "")) == "MeasurementGroup"
    ]
    return tuple(sorted({measurement_id, *allowed_producers, *allowed_groups}))


def _parse_conditions(graph: nx.Graph, node_id: str) -> list[dict[str, Any]]:
    if node_id not in graph:
        return []
    raw = str(graph.nodes[node_id].get("conditions_json", "")).strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed conditions_json on node {node_id!r}.") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"conditions_json on node {node_id!r} must be a list.")
    rows: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError(
                f"conditions_json item on node {node_id!r} must be an object."
            )
        rows.append(dict(item))
    return rows


def _condition_peak_values(
    graph: nx.Graph,
    node_ids: Iterable[str],
) -> set[str]:
    peaks: set[str] = set()
    accepted = {
        "raman band",
        "raman peak",
        "raman shift",
        "measurement peak",
    }
    for node_id in sorted(set(map(str, node_ids))):
        if node_id not in graph:
            continue
        for condition in _parse_conditions(graph, node_id):
            name = str(condition.get("name", "")).strip().lower()
            if name not in accepted:
                continue
            value = condition.get("value_numeric")
            unit = str(condition.get("unit", "")).strip()
            if value is None or str(value).strip() == "":
                continue
            if "cm" not in unit.lower():
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            peaks.add(f"{number:g} cm^-1")
    return peaks


def _raman_peak(
    graph: nx.Graph,
    source_ids: Iterable[str],
) -> str:
    peaks = _condition_peak_values(graph, source_ids)
    for node_id in sorted(set(map(str, source_ids))):
        if node_id not in graph:
            continue
        for match in _RAMAN_PEAK_RE.finditer(_node_text(graph, node_id)):
            peaks.add(f"{float(match.group('number')):g} cm^-1")
    return next(iter(peaks)) if len(peaks) == 1 else ""


def _formula_text(graph: nx.Graph, source_ids: Iterable[str]) -> str:
    calculation_details: list[str] = []
    measurement_details: list[str] = []
    for node_id in sorted(set(map(str, source_ids))):
        if node_id not in graph:
            continue
        attrs = graph.nodes[node_id]
        node_type = str(attrs.get("type", ""))
        for key in ("method_details", "description", "basis"):
            value = str(attrs.get(key, "")).strip()
            lower = value.lower()
            if not value:
                continue
            qualifies = (
                re.search(r"\bef\s*=\s*\(", lower) is not None
                or (
                    "calculated from" in lower
                    and "raman" in lower
                    and "intens" in lower
                )
                or (
                    "normalized" in lower
                    and "molecule" in lower
                )
                or (
                    "isers" in lower
                    and ("insers" in lower or "innr" in lower)
                )
            )
            if not qualifies:
                continue
            if node_type == "Calculation":
                calculation_details.append(value)
            else:
                measurement_details.append(value)
    for candidates in (calculation_details, measurement_details):
        unique = sorted(set(candidates))
        if len(unique) == 1:
            return unique[0]
    return ""


def _ef_aggregation_scope(text: str) -> str:
    lower = text.lower()
    if re.search(r"\b(?:lower[- ]bound|minimum|≥|at least|or larger)\b", lower):
        return "lower_bound"
    if re.search(r"\b(?:highest|maximum|max\b|up to|as high as)\b", lower):
        return "maximum"
    if (
        re.search(r"\b(?:range|ranged|distribution|population)\b", lower)
        and not re.search(r"\bmean\b|\baverage\b", lower)
    ):
        return "population_distribution"
    if re.search(r"\b(?:mean|average)\b", lower) and re.search(
        r"\b(?:single|individual)\b.{0,50}\b(?:particle|nanoparticle|dip)s?\b",
        lower,
    ):
        return "population_mean"
    if re.search(
        r"\b(?:single|individual)\b.{0,50}\b(?:particle|nanoparticle|dip)s?\b",
        lower,
    ):
        return "single_particle"
    if re.search(r"\b(?:estimated|reported|calculated)\s+ef\b", lower):
        return "substrate_summary"
    return "unspecified"


def _ef_definition(text: str, formula: str) -> tuple[str, str, str, str]:
    lower = f"{text} {formula}".lower()
    if (
        ("molecule" in lower and "normal raman" in lower and "intens" in lower)
        or "insers" in lower
        or "innr" in lower
    ):
        reference = "normal_raman"
        return (
            "known",
            "molecule_normalized_intensity_ratio",
            "molecule_count",
            reference,
        )
    if (
        ("c_nor" in lower and "c_sers" in lower)
        or (
            "normal raman concentration" in lower
            and "sers concentration" in lower
        )
    ):
        reference = (
            "normal_raman_on_glass"
            if "glass" in lower
            else "normal_raman"
        )
        return (
            "known",
            "concentration_normalized_intensity_ratio",
            "concentration",
            reference,
        )

    # alpha4b.3b.4b.1: "calculated/estimated EF" is not definition evidence.
    # Partial is reserved for an explicit but incomplete definition component.
    if (
        re.search(r"\bnormaliz(?:e|ed|ation|ing)\b.{0,60}\bmolecule", lower)
        or re.search(r"\bmolecule(?:s|\s+count|\s+number)", lower)
        and "normalized" in lower
    ):
        return (
            "partial",
            "reported_ef_unspecified",
            "molecule_count",
            "unspecified",
        )
    if (
        re.search(r"\bnormaliz(?:e|ed|ation|ing)\b.{0,60}\bconcentration", lower)
        or (
            ("c_sers" in lower or "c_nor" in lower)
            and not ("c_sers" in lower and "c_nor" in lower)
        )
    ):
        return (
            "partial",
            "reported_ef_unspecified",
            "concentration",
            "unspecified",
        )
    if "normal raman" in lower and "intens" in lower:
        return (
            "partial",
            "reported_ef_unspecified",
            "unspecified",
            "normal_raman_on_glass" if "glass" in lower else "normal_raman",
        )

    return (
        "unknown",
        "reported_ef_unspecified",
        "unspecified",
        "unspecified",
    )


def _lod_definition(text: str) -> tuple[str, str, str]:
    lower = text.lower()
    if (
        "calibration" in lower
        and "slope" in lower
        and ("standard deviation" in lower or "response deviation" in lower)
    ):
        return (
            "known",
            "calibration_curve_statistical",
            "response_standard_deviation_and_calibration_slope",
        )

    # alpha4b.3b.4b.1: the concentration must be explicitly tied to
    # detection. "lowest concentration used/adsorbed/tested" alone is not LOD.
    explicit_detection_patterns = (
        r"\blowest\s+(?:concentration|level)\b.{0,100}"
        r"\b(?:that\s+)?(?:can|could|was|were)?\s*"
        r"(?:be\s+)?(?:detected|identified|observed)\b",
        r"\b(?:detected|identified|observed)\b.{0,100}"
        r"\b(?:at|down\s+to|as\s+low\s+as)\b",
        r"\b(?:can|could)\s+be\s+(?:detected|identified|observed)"
        r"\s+(?:at|down\s+to)\b",
        r"\bdetection\s+level\b",
    )
    if any(re.search(pattern, lower) for pattern in explicit_detection_patterns):
        return (
            "known",
            "lowest_detected_concentration",
            "lowest_observed_detection",
        )

    # "theoretical/calculated LOD" names a result, not its criterion.
    return (
        "unknown",
        "reported_lod_unspecified",
        "",
    )


def _finalize_definition_interpretation(
    *,
    status: str,
    criterion: str,
    formula_text: str,
    normalization_basis: str,
    reference_basis: str,
) -> tuple[str, str, str, str]:
    """Conservatively enforce the generic unknown-definition contract.

    Raw source_expression and provenance are intentionally preserved elsewhere.
    Only interpreted definition fields are cleared when status is unknown.
    """
    if status != "unknown":
        return (
            criterion,
            formula_text,
            normalization_basis,
            reference_basis,
        )

    safe_normalization = (
        normalization_basis
        if normalization_basis in {"unspecified", "not_applicable"}
        else "unspecified"
    )
    safe_reference = (
        reference_basis
        if reference_basis in {"unspecified", "not_applicable"}
        else "unspecified"
    )
    return "", "", safe_normalization, safe_reference


def _build_context(
    *,
    graph: nx.Graph,
    paper_id: str,
    measurement_id: str,
) -> MetricDefinitionContext:
    observable = _observable_key(graph, measurement_id)
    source_ids = _local_sources(graph, measurement_id)
    source_text = _source_text(graph, source_ids)
    formula = _formula_text(graph, source_ids)
    measurements, groups, experiments, calculations = _typed_sources(
        graph,
        source_ids,
    )
    measurement_source = str(
        graph.nodes[measurement_id].get("source_expression", "")
    ).strip()

    if observable == "sers_enhancement_factor":
        status, family, normalization, reference = _ef_definition(
            source_text,
            formula,
        )
        aggregation = _ef_aggregation_scope(source_text)
        criterion = ""
    elif observable == "detection_limit":
        status, family, criterion = _lod_definition(source_text)
        aggregation = "not_applicable"
        normalization = "not_applicable"
        reference = "not_applicable"
        formula = ""
    else:
        raise ValueError(f"Unsupported SERS metric definition observable: {observable!r}")

    criterion, formula, normalization, reference = (
        _finalize_definition_interpretation(
            status=status,
            criterion=criterion,
            formula_text=formula,
            normalization_basis=normalization,
            reference_basis=reference,
        )
    )

    return MetricDefinitionContext(
        context_id=stable_metric_definition_id(
            paper_id=paper_id,
            measurement_id=measurement_id,
            semantics_id=SERS_METRIC_DEFINITION_SEMANTICS_ID,
        ),
        domain_profile_id="sers_au_ag",
        metric_definition_semantics_id=SERS_METRIC_DEFINITION_SEMANTICS_ID,
        paper_id=paper_id,
        measurement_id=measurement_id,
        observable_key=observable,
        definition_status=status,
        definition_family=family,
        aggregation_scope=aggregation,
        normalization_basis=normalization,
        reference_basis=reference,
        criterion=criterion,
        formula_text=formula,
        raman_peak=_raman_peak(graph, source_ids),
        source_expression=measurement_source,
        source_measurement_ids=measurements,
        source_measurement_group_ids=groups,
        source_experiment_ids=experiments,
        source_calculation_ids=calculations,
        source_node_ids=source_ids,
    )


def extract_sers_metric_definition_contexts(
    graph: nx.Graph,
    paper_id: str,
) -> list[MetricDefinitionContext]:
    contexts: list[MetricDefinitionContext] = []
    for node_id, attrs in sorted(graph.nodes(data=True), key=lambda item: str(item[0])):
        if str(attrs.get("type", "")) != "Measurement":
            continue
        observable = _observable_key(graph, str(node_id))
        if observable not in SERS_SUPPORTED_OBSERVABLES:
            continue
        contexts.append(
            _build_context(
                graph=graph,
                paper_id=paper_id,
                measurement_id=str(node_id),
            )
        )
    return contexts


SERS_AU_AG_METRIC_DEFINITION_ADAPTER = MetricDefinitionDomainAdapter(
    adapter_id="sers_au_ag",
    domain_profile_id="sers_au_ag",
    semantics_id=SERS_METRIC_DEFINITION_SEMANTICS_ID,
    supported_observable_keys=SERS_SUPPORTED_OBSERVABLES,
    definition_families=SERS_DEFINITION_FAMILIES,
    aggregation_scopes=SERS_AGGREGATION_SCOPES,
    normalization_bases=SERS_NORMALIZATION_BASES,
    reference_bases=SERS_REFERENCE_BASES,
    extract_contexts_fn=extract_sers_metric_definition_contexts,
)
