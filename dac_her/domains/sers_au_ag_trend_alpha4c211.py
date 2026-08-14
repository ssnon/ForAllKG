from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import networkx as nx

from dac_her.domains import sers_au_ag_trend as v1
from dac_her.trend_domain import (
    TrendDomainAdapter,
    TrendEvidence,
    TrendEvidenceSource,
    TrendSeriesPoint,
)
from dac_her.trend_evidence import stable_trend_id


SERS_AU_AG_TREND_SEMANTICS_ID = "sers_au_ag_trend_v3_alpha4c211"

_SUPPORTED_EVIDENCE_BASES = v1._SUPPORTED_EVIDENCE_BASES
_REQUIRED_INPUTS = v1._REQUIRED_INPUTS
_METHOD_GUARD_DIMENSIONS = v1._METHOD_GUARD_DIMENSIONS
_SUBJECT_TYPES = v1._SUBJECT_TYPES
_CLAIM_TYPES = v1._CLAIM_TYPES

_NUMERIC_RESPONSE_KEYS = frozenset({
    "raman_intensity",
    "sers_enhancement_factor",
    "relative_sers_intensity_ratio",
})

_CONTROL_TO_METHOD_DIMENSION = {
    "analyte_concentration": "analyte_concentration",
    "excitation_wavelength": "excitation_wavelength",
    "laser_power": "laser_power",
    "integration_time": "integration_time",
}

_CALCULATION_RELATIONS = frozenset({
    "SIMULATED_BY",
    "CALCULATED_BY",
    "COMPUTED_BY",
    "MODELED_BY",
    "MODELLED_BY",
    "HAS_CALCULATION",
    "DERIVED_FROM_CALCULATION",
})


@dataclass(frozen=True)
class SersTrendControlSpec:
    key: str
    label: str
    family: str
    patterns: tuple[str, ...]
    unit_kind: str
    canonical_unit: str = ""


SERS_TREND_CONTROL_SPECS: dict[str, SersTrendControlSpec] = {
    "shell_thickness": SersTrendControlSpec(
        "shell_thickness", "Ag shell thickness", "structural",
        (r"\b(?:ag|silver)\s+shell\s+thickness\b", r"\bshell\s+thickness\b"),
        "length", "nm",
    ),
    "nanogap_size": SersTrendControlSpec(
        "nanogap_size", "nanogap size", "structural",
        (
            r"\b(?:interior\s+)?(?:nano\s*)?gap\s+(?:size|width|distance)\b",
            r"\b(?:nanogap|nano\s+gap)\s+(?:size|width|distance)\b",
        ),
        "length", "nm",
    ),
    "nanogap_presence": SersTrendControlSpec(
        "nanogap_presence", "interior nanogap presence", "structural",
        (
            r"\b(?:interior\s+)?nanogap\b",
            r"\b(?:interior\s+)?nano\s+gap\b",
            r"\bnanogap[-\s]*less\b",
            r"\bgap[-\s]*less\b",
        ),
        "categorical", "",
    ),
    "particle_size": SersTrendControlSpec(
        "particle_size", "particle size", "structural",
        (r"\b(?:nano)?particle\s+(?:size|diameter)\b", r"\bau@ag\s+diameter\b", r"\bparticle\s+diameter\b"),
        "length", "nm",
    ),
    "ag_to_au_ratio": SersTrendControlSpec(
        "ag_to_au_ratio", "Ag/Au ratio", "composition",
        (
            r"\bau\s*[:/\-]\s*ag\s+ratios?\b", r"\bag\s*[:/]\s*au\s+ratios?\b",
            r"\bau[-\s]*ag\s+ratios?\b", r"\bag[-\s]*au\s+ratios?\b",
            r"\bratios?\s+of\s+au\s+to\s+ag\b", r"\bratios?\s+of\s+ag\s+to\s+au\b",
            r"\bgold[-\s]+silver\s+ratios?\b",
        ),
        "ratio", "Ag/Au ratio",
    ),
    "au_content": SersTrendControlSpec(
        "au_content", "Au content", "composition",
        (r"\b(?:au|gold)\s+content\b", r"\bau\s+atomic\s*(?:%|percent|fraction)\b"),
        "fraction", "fraction",
    ),
    "gold_precursor_amount": SersTrendControlSpec(
        "gold_precursor_amount", "gold precursor amount", "synthesis",
        (
            r"\bhaucl4\s+(?:concentration|amount|dosage)\b",
            r"\b(?:concentration|amount|dosage)\s+of\s+haucl4\b",
            r"\b(?:gold|au)\s+precursor\s+(?:concentration|amount|dosage)\b",
            r"\bamount\s+of\s+(?:added\s+)?gold(?:\s+salt)?\b",
            r"\badded\s+gold\s+amount\b",
        ),
        "concentration_or_amount", "",
    ),
    "silver_precursor_concentration": SersTrendControlSpec(
        "silver_precursor_concentration", "AgNO3 concentration", "synthesis",
        (r"\bagno3\s+(?:concentration|amount|dosage)\b", r"\b(?:concentration|amount|dosage)s?\s+of\s+agno3\b", r"\bsilver\s+nitrate\s+(?:concentration|amount|dosage)\b"),
        "concentration_or_amount", "",
    ),
    "particle_concentration": SersTrendControlSpec(
        "particle_concentration", "particle concentration", "concentration",
        (r"\b(?:nano)?particle\s+concentration\b", r"\bnp\s+concentration\b", r"\bconcentration\s+of\s+(?:nano)?particles?\b"),
        "particle_concentration", "",
    ),
    "analyte_concentration": SersTrendControlSpec(
        "analyte_concentration", "analyte concentration", "concentration",
        (
            r"\banalyte\s+concentration\b", r"\btarget\s+(?:dna\s+)?concentration\b",
            r"\bprobe\s+(?:molecule\s+)?concentration\b",
            r"\b(?:atp|4-atp|r6g|rhodamine\s*6g|methylene\s+blue|mb|dna)\s+concentration\b",
            r"\bconcentration\s+of\s+(?:atp|4-atp|r6g|rhodamine\s*6g|methylene\s+blue|mb|dna)\b",
        ),
        "concentration", "M",
    ),
    "laser_power": SersTrendControlSpec(
        "laser_power", "laser power", "measurement",
        (r"\blaser\s+power\b", r"\bexcitation\s+power\b"), "power", "W",
    ),
    "excitation_wavelength": SersTrendControlSpec(
        "excitation_wavelength", "excitation wavelength", "measurement",
        (r"\bexcitation\s+wavelength\b", r"\blaser\s+wavelength\b"), "length", "nm",
    ),
    "integration_time": SersTrendControlSpec(
        "integration_time", "integration time", "measurement",
        (r"\bintegration\s+time\b", r"\bacquisition\s+time\b", r"\bexposure\s+time\b"), "time", "s",
    ),
}

_CONTROL_PRIORITY = tuple(SERS_TREND_CONTROL_SPECS)


@dataclass(frozen=True)
class ControlObservation:
    key: str
    label: str
    family: str
    value_numeric: float
    unit: str
    normalization_transform: str
    source_node_id: str
    source_scope: str
    source_value_text: str = ""


def _norm(value: Any) -> str:
    return v1._norm(value)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _control_key_from_name(name: Any) -> str:
    text = _norm(name)
    for key in _CONTROL_PRIORITY:
        spec = SERS_TREND_CONTROL_SPECS[key]
        if any(re.search(pattern, text) for pattern in spec.patterns):
            return key
    return ""


def _length_nm(value: Any, unit: Any) -> float | None:
    number = _finite(value)
    if number is None:
        return None
    normalized = _norm(unit).replace(" ", "")
    factors = {"nm": 1.0, "å": 0.1, "angstrom": 0.1, "angstroms": 0.1, "µm": 1000.0, "um": 1000.0}
    return None if normalized not in factors else number * factors[normalized]


def _concentration_m(value: Any, unit: Any) -> float | None:
    number = _finite(value)
    if number is None:
        return None
    raw = str(unit or "").strip().replace("μ", "µ").replace(" ", "")
    factors = {"M": 1.0, "mM": 1e-3, "µM": 1e-6, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12, "fM": 1e-15}
    return None if raw not in factors else number * factors[raw]


def _power_w(value: Any, unit: Any) -> float | None:
    number = _finite(value)
    if number is None:
        return None
    raw = str(unit or "").strip().replace("μ", "µ").replace(" ", "")
    factors = {"W": 1.0, "mW": 1e-3, "µW": 1e-6, "uW": 1e-6}
    return None if raw not in factors else number * factors[raw]


def _time_s(value: Any, unit: Any) -> float | None:
    number = _finite(value)
    if number is None:
        return None
    raw = _norm(unit).replace(" ", "")
    factors = {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0, "ms": 1e-3}
    return None if raw not in factors else number * factors[raw]


def _parse_number_unit(text: str) -> tuple[float, str] | None:
    match = re.search(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>fM|pM|nM|[µu]M|mM|M|nm|[µu]m|mW|[µu]W|W|ms|s|%|at%)\b",
        str(text or ""), re.I,
    )
    if not match:
        return None
    value = _finite(match.group("value"))
    return None if value is None else (value, match.group("unit"))


def _parse_ag_to_au_ratio_text(text: str) -> tuple[float, str] | None:
    raw = str(text or "").replace("−", "-").replace("–", "-")
    patterns = (
        (r"(?:au\s*[:/\-]\s*ag(?:\s+ratios?)?|au[-\s]*ag\s+ratios?).{0,30}?(?P<a>\d+(?:\.\d+)?)\s*:\s*(?P<b>\d+(?:\.\d+)?)", "au_ag_to_ag_over_au"),
        (r"ratios?\s+of\s+au\s+to\s+ag.{0,30}?(?P<a>\d+(?:\.\d+)?)\s*:\s*(?P<b>\d+(?:\.\d+)?)", "au_ag_to_ag_over_au"),
        (r"(?:ag\s*[:/]\s*au(?:\s+ratios?)?|ag[-\s]*au\s+ratios?).{0,30}?(?P<a>\d+(?:\.\d+)?)\s*:\s*(?P<b>\d+(?:\.\d+)?)", "ag_au_identity"),
    )
    for pattern, transform in patterns:
        match = re.search(pattern, raw, re.I)
        if not match:
            continue
        first, second = float(match.group("a")), float(match.group("b"))
        if transform == "au_ag_to_ag_over_au":
            return None if first <= 0 else (second / first, transform)
        return None if second <= 0 else (first / second, transform)
    return None


def _condition_raw_value(condition: Mapping[str, Any]) -> str:
    return v1._condition_raw_value(condition)


def _control_from_condition(condition: Mapping[str, Any], *, source_node_id: str, source_scope: str) -> ControlObservation | None:
    key = _control_key_from_name(condition.get("name", ""))
    if not key:
        return None
    spec = SERS_TREND_CONTROL_SPECS[key]
    numeric = condition.get("value_numeric")
    unit = str(condition.get("unit") or "").strip()
    raw = _condition_raw_value(condition)

    if spec.unit_kind == "ratio":
        parsed = _parse_ag_to_au_ratio_text(f"{condition.get('name', '')} {raw}")
        if parsed:
            value, transform = parsed
            return ControlObservation(key, spec.label, spec.family, value, spec.canonical_unit, transform, source_node_id, source_scope, raw)
        number = _finite(numeric)
        name = _norm(condition.get("name", ""))
        if number is not None and ("ag/au" in name or "ag to au" in name):
            return ControlObservation(key, spec.label, spec.family, number, spec.canonical_unit, "ag_au_identity", source_node_id, source_scope, raw)
        return None

    if spec.unit_kind == "length":
        value = _length_nm(numeric, unit)
        if value is None and (parsed := _parse_number_unit(raw)):
            value = _length_nm(*parsed)
        return None if value is None else ControlObservation(key, spec.label, spec.family, value, "nm", "length_to_nm", source_node_id, source_scope, raw)

    if spec.unit_kind == "concentration":
        value = _concentration_m(numeric, unit)
        if value is None and (parsed := _parse_number_unit(raw)):
            value = _concentration_m(*parsed)
        return None if value is None else ControlObservation(key, spec.label, spec.family, value, "M", "molar_to_M", source_node_id, source_scope, raw)

    if spec.unit_kind == "particle_concentration":
        value = _concentration_m(numeric, unit)
        if value is not None:
            return ControlObservation(key, spec.label, spec.family, value, "M", "molar_to_M", source_node_id, source_scope, raw)
        number = _finite(numeric)
        normalized_unit = _norm(unit)
        return None if number is None or not normalized_unit else ControlObservation(key, spec.label, spec.family, number, normalized_unit, "identity_unit", source_node_id, source_scope, raw)

    if spec.unit_kind == "power":
        value = _power_w(numeric, unit)
        if value is None and (parsed := _parse_number_unit(raw)):
            value = _power_w(*parsed)
        return None if value is None else ControlObservation(key, spec.label, spec.family, value, "W", "power_to_W", source_node_id, source_scope, raw)

    if spec.unit_kind == "time":
        value = _time_s(numeric, unit)
        if value is None and (parsed := _parse_number_unit(raw)):
            value = _time_s(*parsed)
        return None if value is None else ControlObservation(key, spec.label, spec.family, value, "s", "time_to_s", source_node_id, source_scope, raw)

    if spec.unit_kind == "fraction":
        number = _finite(numeric)
        if number is None and (parsed := _parse_number_unit(raw)):
            number, unit = parsed
        if number is None:
            return None
        transform = "identity_fraction"
        if "%" in str(unit) or "at%" in str(unit).casefold():
            number /= 100.0
            transform = "percent_to_fraction"
        return ControlObservation(key, spec.label, spec.family, number, "fraction", transform, source_node_id, source_scope, raw)

    if spec.unit_kind == "concentration_or_amount":
        value = _concentration_m(numeric, unit)
        if value is not None:
            return ControlObservation(key, spec.label, spec.family, value, "M", "molar_to_M", source_node_id, source_scope, raw)
        number = _finite(numeric)
        normalized_unit = _norm(unit)
        return None if number is None or not normalized_unit else ControlObservation(key, spec.label, spec.family, number, normalized_unit, "identity_unit", source_node_id, source_scope, raw)
    return None


def _node_text(graph: nx.Graph, node_id: str) -> str:
    if node_id not in graph:
        return ""
    attrs = graph.nodes[node_id]
    values = [str(attrs.get(key, "")).strip() for key in ("label", "name", "description", "source_expression", "statement", "method", "node_text")]
    return " | ".join(value for value in values if value)


def _control_from_text(text: str, *, source_node_id: str, source_scope: str) -> tuple[ControlObservation, ...]:
    normalized = _norm(text)
    found: list[ControlObservation] = []
    for key in _CONTROL_PRIORITY:
        spec = SERS_TREND_CONTROL_SPECS[key]
        if not any(re.search(pattern, normalized) for pattern in spec.patterns):
            continue
        if spec.unit_kind == "ratio":
            if parsed := _parse_ag_to_au_ratio_text(text):
                value, transform = parsed
                found.append(ControlObservation(key, spec.label, spec.family, value, spec.canonical_unit, transform, source_node_id, source_scope, text))
            continue
        parsed_num = _parse_number_unit(text)
        if not parsed_num:
            continue
        parsed = _control_from_condition(
            {"name": spec.label, "value_numeric": parsed_num[0], "unit": parsed_num[1], "value_text": f"{parsed_num[0]} {parsed_num[1]}"},
            source_node_id=source_node_id, source_scope=source_scope,
        )
        if parsed:
            found.append(parsed)
    return tuple(found)


def extract_control_landmark_from_text(control_key: str, text: str) -> dict[str, object]:
    spec = SERS_TREND_CONTROL_SPECS.get(control_key)
    empty = {"source_value_text": "", "canonical_value_numeric": None, "canonical_unit": spec.canonical_unit if spec else "", "normalization_transform": ""}
    if spec is None:
        return empty
    if control_key == "ag_to_au_ratio":
        parsed = _parse_ag_to_au_ratio_text(text)
        if not parsed:
            return empty
        raw = re.search(r"\b\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?\b", text)
        return {"source_value_text": raw.group(0) if raw else "", "canonical_value_numeric": parsed[0], "canonical_unit": spec.canonical_unit, "normalization_transform": parsed[1]}
    if not any(re.search(pattern, _norm(text)) for pattern in spec.patterns):
        return empty
    parsed_num = _parse_number_unit(text)
    if not parsed_num:
        return empty
    obs = _control_from_condition(
        {"name": spec.label, "value_numeric": parsed_num[0], "unit": parsed_num[1], "value_text": f"{parsed_num[0]} {parsed_num[1]}"},
        source_node_id="claim", source_scope="claim_text",
    )
    if obs is None:
        return empty
    return {"source_value_text": f"{parsed_num[0]} {parsed_num[1]}", "canonical_value_numeric": obs.value_numeric, "canonical_unit": obs.unit, "normalization_transform": obs.normalization_transform}


def _method_dimension_map(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return v1._method_dimension_map(row)


def _methods_compatible(rows: Iterable[Mapping[str, Any]], *, varied_control_key: str) -> bool:
    rows = list(rows)
    ignored = _CONTROL_TO_METHOD_DIMENSION.get(varied_control_key, "")
    for name in _METHOD_GUARD_DIMENSIONS:
        if name == ignored:
            continue
        known: set[str] = set()
        for row in rows:
            item = _method_dimension_map(row).get(name)
            if item is None:
                continue
            status = str(item.get("status", "unknown"))
            if status == "ambiguous":
                return False
            if status == "known":
                value = str(item.get("normalized_value", "")).strip()
                if value:
                    known.add(value)
        if len(known) > 1:
            return False
    return True


def _measurement_control_seed_nodes(graph: nx.Graph, measurement_id: str, identity_row: Mapping[str, Any], method_row: Mapping[str, Any], context_row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    seeds: dict[str, str] = {}
    for mention_id in v1._measurement_mentions(measurement_id, identity_row):
        if mention_id in graph:
            seeds[mention_id] = "measurement"
        for group_id in v1._outgoing(graph, mention_id, "IN_MEASUREMENT_GROUP"):
            if group_id in graph:
                seeds[group_id] = "measurement_group"
        for producer in v1._incoming(graph, mention_id, "HAS_MEASUREMENT"):
            if producer in graph:
                seeds[producer] = "producer"
        for subject in v1._outgoing(graph, mention_id, "MEASURED_FOR"):
            if subject in graph:
                seeds[subject] = "measurement_subject"
    for node_id in list(method_row.get("source_node_ids", []) or []) + list(context_row.get("source_node_ids", []) or []) + list(method_row.get("producer_ids", []) or []):
        node_id = str(node_id)
        if node_id in graph:
            seeds.setdefault(node_id, "sidecar_provenance")
    return tuple(sorted(seeds.items()))


def _measurement_controls(graph: nx.Graph, measurement_id: str, identity_row: Mapping[str, Any], method_row: Mapping[str, Any], context_row: Mapping[str, Any]) -> dict[str, ControlObservation]:
    # Point-local provenance outranks broader producer/sidecar prose. A shared
    # MeasurementGroup defines lineage, not a per-point x value, so its text is
    # never projected onto every member measurement. This avoids turning the
    # first number mentioned in a sweep description into all series points.
    observed: dict[str, list[ControlObservation]] = defaultdict(list)
    priority = {
        "measurement_conditions_json": 0,
        "measurement_text": 1,
        "measurement_subject_conditions_json": 2,
        "measurement_subject_text": 3,
        "producer_conditions_json": 4,
        "producer_text": 5,
        "sidecar_provenance_conditions_json": 6,
        "sidecar_provenance_text": 7,
    }
    for node_id, scope in _measurement_control_seed_nodes(graph, measurement_id, identity_row, method_row, context_row):
        node_type = str(graph.nodes[node_id].get("type", "")) if node_id in graph else ""
        if scope == "measurement_group" or node_type == "MeasurementGroup":
            continue
        for condition in v1._structured_conditions(graph, node_id):
            parsed = _control_from_condition(condition, source_node_id=node_id, source_scope=f"{scope}_conditions_json")
            if parsed:
                observed[parsed.key].append(parsed)
        for parsed in _control_from_text(_node_text(graph, node_id), source_node_id=node_id, source_scope=f"{scope}_text"):
            observed[parsed.key].append(parsed)
    resolved: dict[str, ControlObservation] = {}
    for key, rows in observed.items():
        best_rank = min(priority.get(row.source_scope, 99) for row in rows)
        best = [row for row in rows if priority.get(row.source_scope, 99) == best_rank]
        signatures = {(round(row.value_numeric, 15), row.unit) for row in best}
        if len(signatures) != 1:
            continue
        resolved[key] = sorted(best, key=lambda row: (row.source_scope, row.source_node_id))[0]
    return resolved


def _incident_edges(graph: nx.Graph, node_id: str):
    if node_id not in graph:
        return ()
    rows = []
    if graph.is_multigraph():
        if graph.is_directed():
            rows.extend((str(a), str(b), dict(attrs)) for a, b, _k, attrs in graph.in_edges(node_id, keys=True, data=True))
            rows.extend((str(a), str(b), dict(attrs)) for a, b, _k, attrs in graph.out_edges(node_id, keys=True, data=True))
        else:
            rows.extend((str(a), str(b), dict(attrs)) for a, b, _k, attrs in graph.edges(node_id, keys=True, data=True))
    else:
        if graph.is_directed():
            rows.extend((str(a), str(b), dict(attrs)) for a, b, attrs in graph.in_edges(node_id, data=True))
            rows.extend((str(a), str(b), dict(attrs)) for a, b, attrs in graph.out_edges(node_id, data=True))
        else:
            rows.extend((str(a), str(b), dict(attrs)) for a, b, attrs in graph.edges(node_id, data=True))
    return tuple(rows)


def _calculation_ids(graph: nx.Graph, *, measurement_ids: Iterable[str], lineage_ids: Iterable[str], subject_ids: Iterable[str]) -> tuple[str, ...]:
    seeds = {str(value) for value in (*tuple(measurement_ids), *tuple(lineage_ids), *tuple(subject_ids)) if str(value) in graph}
    for measurement_id in tuple(measurement_ids):
        seeds.update(v1._incoming(graph, str(measurement_id), "HAS_MEASUREMENT"))
    calculations = {node_id for node_id in seeds if node_id in graph and str(graph.nodes[node_id].get("type", "")) == "Calculation"}
    for seed in tuple(seeds):
        for left, right, attrs in _incident_edges(graph, seed):
            if str(attrs.get("relation", "")).upper() not in _CALCULATION_RELATIONS:
                continue
            other = right if left == seed else left
            if other in graph and str(graph.nodes[other].get("type", "")) == "Calculation":
                calculations.add(other)
    return tuple(sorted(calculations))


def _numeric_trends(source: TrendEvidenceSource) -> list[TrendEvidence]:
    graph = source.graph
    identity_by_rep = v1._identity_by_representative(source.measurement_result_rows)
    method_by_id = v1._method_by_id(source.method_context_rows)
    candidates: dict[tuple[str, str, str, tuple[str, ...], str, str], list[dict[str, Any]]] = defaultdict(list)

    for context in source.comparison_context_rows:
        measurement_id = str(context.get("measurement_id", "")).strip()
        observable_key = str(context.get("observable_key", "")).strip()
        dependent_value = _finite(context.get("value_numeric"))
        if not measurement_id or observable_key not in _NUMERIC_RESPONSE_KEYS or dependent_value is None:
            continue
        method_row = method_by_id.get(str(context.get("method_context_id", "")).strip())
        identity_row = identity_by_rep.get(measurement_id)
        if method_row is None or identity_row is None:
            continue
        lineage = v1._lineage(graph, measurement_id, identity_row, method_row, context)
        if lineage is None:
            continue
        lineage_kind, lineage_ids = lineage
        dependent_unit = str(context.get("unit", "")).strip()
        for control_key, control in _measurement_controls(graph, measurement_id, identity_row, method_row, context).items():
            key = (control_key, observable_key, lineage_kind, lineage_ids, dependent_unit, control.unit)
            candidates[key].append({"measurement_id": measurement_id, "identity_id": str(identity_row.get("identity_id", "")), "context": context, "method": method_row, "control": control, "dependent_value": dependent_value})

    evidence: list[TrendEvidence] = []
    for (control_key, observable_key, lineage_kind, lineage_ids, dependent_unit, control_unit), rows in sorted(candidates.items(), key=lambda item: str(item[0])):
        if len(rows) < 2 or not _methods_compatible((row["method"] for row in rows), varied_control_key=control_key):
            continue
        x_values = [float(row["control"].value_numeric) for row in rows]
        if len(x_values) != len(set(x_values)):
            continue
        ordered = sorted(rows, key=lambda row: float(row["control"].value_numeric))
        direction, shape = v1._numeric_direction_shape([(float(row["control"].value_numeric), float(row["dependent_value"])) for row in ordered])
        basis = "controlled_numeric_pair" if len(ordered) == 2 else "controlled_numeric_series"
        measurement_ids = tuple(sorted({str(row["measurement_id"]) for row in ordered}))
        result_ids = tuple(sorted({str(row["identity_id"]) for row in ordered if str(row["identity_id"]).strip()}))
        method_ids = tuple(sorted({str(row["method"].get("method_context_id", "")) for row in ordered if str(row["method"].get("method_context_id", "")).strip()}))
        context_ids = tuple(sorted({str(row["context"].get("context_id", "")) for row in ordered if str(row["context"].get("context_id", "")).strip()}))
        subject_ids = tuple(sorted({str(subject_id) for row in ordered for subject_id in row["context"].get("subject_ids", []) or [] if str(subject_id).strip()}))
        source_expressions = tuple(sorted({str(row["context"].get("source_expression", "")).strip() for row in ordered if str(row["context"].get("source_expression", "")).strip()}))
        calculation_ids = _calculation_ids(graph, measurement_ids=measurement_ids, lineage_ids=lineage_ids, subject_ids=subject_ids)
        source_node_ids = tuple(sorted({*measurement_ids, *lineage_ids, *calculation_ids}))
        points = tuple(TrendSeriesPoint(
            point_id=f"{source.paper_id}:{row['measurement_id']}",
            independent_value_numeric=float(row["control"].value_numeric), independent_unit=control_unit,
            dependent_value_numeric=float(row["dependent_value"]), dependent_unit=dependent_unit,
            source_measurement_result_ids=(str(row["identity_id"]),), source_measurement_ids=(str(row["measurement_id"]),), source_node_ids=(str(row["measurement_id"]),),
        ) for row in ordered)
        trend_id = stable_trend_id(paper_id=source.paper_id, independent_variable_key=control_key, dependent_observable_key=observable_key, evidence_basis=basis, source_node_ids=source_node_ids)
        evidence.append(TrendEvidence(
            trend_id=trend_id, domain_profile_id="sers_au_ag", trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID, paper_id=source.paper_id,
            independent_variable_key=control_key, independent_variable_label=ordered[0]["control"].label,
            dependent_observable_key=observable_key, dependent_observable_label=str(ordered[0]["context"].get("observable_label", observable_key)).strip() or observable_key,
            direction=direction, shape=shape, evidence_basis=basis, causal_status="not_asserted", varied_dimension=control_key,
            subject_ids=subject_ids, series_points=points, source_expression=source_expressions[0] if source_expressions else "", source_expressions=source_expressions,
            source_measurement_ids=measurement_ids,
            source_measurement_group_ids=lineage_ids if lineage_kind == "measurement_group" else (),
            source_experiment_ids=lineage_ids if lineage_kind == "experiment" else (),
            source_calculation_ids=calculation_ids, source_measurement_result_ids=result_ids, source_method_context_ids=method_ids, source_comparison_context_ids=context_ids, source_node_ids=source_node_ids,
        ))
    return evidence


def _claim_control(text: str) -> tuple[str, str] | None:
    normalized = _norm(text)
    matches = []
    for key in _CONTROL_PRIORITY:
        spec = SERS_TREND_CONTROL_SPECS[key]
        if any(re.search(pattern, normalized) for pattern in spec.patterns):
            matches.append((key, spec.label))

    # A bare "nanogap" is a structural-presence control, not a size control.
    # Size requires explicit size/width/distance language. This prevents
    # presence-vs-absence comparisons from masquerading as continuous trends.
    matched_keys = {key for key, _label in matches}
    if "nanogap_size" in matched_keys and "nanogap_presence" in matched_keys:
        matches = [
            item for item in matches
            if item[0] != "nanogap_presence"
        ]

    unique = {key: (key, label) for key, label in matches}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _relative_fold_expression(normalized: str) -> bool:
    return bool(
        re.search(r"\b\d+(?:\.\d+)?\s*[- ]?fold\b", normalized)
        or re.search(r"\b\d+(?:\.\d+)?\s*times?\b", normalized)
        or (
            re.search(
                r"\b(?:relative\s+to|compared\s+to|versus|vs\.?)\b",
                normalized,
            )
            and re.search(
                r"\benhanc\w*|\bintensit\w*|\bsignal\b",
                normalized,
            )
        )
    )


_RAMAN_SIGNAL_PATTERN = (
    r"(?:"
    r"(?:sers|serrs)(?:\s+(?:signal|intensit\w*))?"
    r"|raman(?:\s+peak)?(?:\s+(?:signal|intensit\w*))?"
    r"|(?:sers|serrs|raman)\s+signal\s+intensit\w*"
    r")"
)

_FORMAL_EF_PATTERN = (
    r"(?:"
    r"(?:sers\s+)?enhancement[-\s]+factor"
    r"|sers\s+ef"
    r"|ef\s+coefficient"
    r")"
)


def _formal_ef_directional_relation(
    normalized: str,
    *,
    control_pattern: str,
) -> bool:
    # Formal EF wins only when the EF term itself participates in the
    # directional clause. A later baseline-relative "factor of 5.8" detail
    # must not steal a Raman-intensity trend.
    change = r"(?:increas\w*|decreas\w*|higher|lower|rise\w*|fall\w*|grow\w*|declin\w*|var\w*)"
    return bool(
        re.search(
            rf"{_FORMAL_EF_PATTERN}.{{0,85}}{change}.{{0,85}}"
            rf"(?:with|as|when).{{0,55}}{control_pattern}",
            normalized,
        )
        or re.search(
            rf"{control_pattern}.{{0,85}}{change}.{{0,85}}"
            rf"{_FORMAL_EF_PATTERN}.{{0,65}}{change}",
            normalized,
        )
        or re.search(
            rf"(?:as|with).{{0,45}}{control_pattern}.{{0,65}}{change}"
            rf".{{0,85}}{_FORMAL_EF_PATTERN}.{{0,65}}{change}",
            normalized,
        )
    )


def _directional_intensity_relation(
    normalized: str,
    *,
    control_pattern: str,
) -> bool:
    change = r"(?:increas\w*|decreas\w*|higher|lower|stronger|weaker|rise\w*|fall\w*|grow\w*|declin\w*)"
    return bool(
        re.search(
            rf"{_RAMAN_SIGNAL_PATTERN}.{{0,80}}{change}.{{0,80}}"
            rf"(?:with|as|when).{{0,55}}{control_pattern}",
            normalized,
        )
        or re.search(
            rf"{control_pattern}.{{0,65}}{change}.{{0,90}}"
            rf"{_RAMAN_SIGNAL_PATTERN}.{{0,55}}{change}",
            normalized,
        )
        or re.search(
            rf"{_RAMAN_SIGNAL_PATTERN}.{{0,85}}(?:proportional|linear)"
            rf".{{0,70}}{control_pattern}",
            normalized,
        )
        or re.search(
            rf"{control_pattern}.{{0,70}}(?:proportional|linear)"
            rf".{{0,85}}{_RAMAN_SIGNAL_PATTERN}",
            normalized,
        )
    )


def _claim_response(
    text: str,
    *,
    control_key: str,
) -> tuple[str, str] | None:
    normalized = _norm(text)
    spec = SERS_TREND_CONTROL_SPECS[control_key]
    control = "(?:" + "|".join(spec.patterns) + ")"

    has_formal_ef = bool(re.search(_FORMAL_EF_PATTERN, normalized))
    has_raman_or_sers = bool(re.search(r"\b(?:sers|serrs|raman)\b", normalized))
    if not has_formal_ef and not has_raman_or_sers:
        return None

    # Stronger precedence: when an explicit EF/EF-coefficient clause itself
    # changes with the control, preserve formal EF semantics.
    if has_formal_ef and _formal_ef_directional_relation(
        normalized,
        control_pattern=control,
    ):
        return "sers_enhancement_factor", "SERS enhancement factor"

    # Raman/SERS signal intensity is checked next, including "Raman peak
    # intensity" syntax. This protects shell-thickness claims whose trailing
    # "5.8 relative to Au" is merely a baseline-relative fold detail.
    if _directional_intensity_relation(
        normalized,
        control_pattern=control,
    ):
        return "raman_intensity", "SERS/Raman intensity"

    if _relative_fold_expression(normalized):
        return (
            "relative_sers_intensity_ratio",
            "relative SERS intensity ratio",
        )

    if has_formal_ef:
        return "sers_enhancement_factor", "SERS enhancement factor"

    if (
        re.search(
            r"\b(?:sers|serrs)\b.{0,65}\b(?:intensit|signal)",
            normalized,
        )
        or re.search(
            r"\braman(?:\s+peak)?\s+intensit\w*\b",
            normalized,
        )
        or re.search(
            r"\b(?:intensit|signal)\w*\b.{0,65}\b(?:sers|serrs|raman)\b",
            normalized,
        )
    ):
        return "raman_intensity", "SERS/Raman intensity"

    if (
        re.search(
            r"\b(?:sers|serrs)\b.{0,65}\b"
            r"(?:performance|activity|enhancement|sensitivity)",
            normalized,
        )
        or re.search(
            r"\b(?:performance|activity|enhancement|sensitivity)\b"
            r".{0,65}\b(?:sers|serrs)\b",
            normalized,
        )
    ):
        return "sers_performance", "SERS performance"
    return None


def _saturation_marker(text: str) -> bool:
    normalized = _norm(text)
    return bool(
        re.search(
            r"\b(?:approach(?:es|ed|ing)?|reach(?:es|ed|ing)?)\b"
            r".{0,40}\b(?:maximum|maximal|optimal|optimum)\b",
            normalized,
        )
        or re.search(r"\bplateau(?:s|ed|ing)?\b", normalized)
        or "close to the maximum" in normalized
        or "essentially the same" in normalized
        or "critical thickness" in normalized
    )


def _rise_peak_fall_marker(
    normalized: str,
    *,
    control_pattern: str,
) -> bool:
    response = (
        rf"(?:{_RAMAN_SIGNAL_PATTERN}|{_FORMAL_EF_PATTERN}|"
        r"sers\s+performance|sers\s+activity|signal|intensit\w*)"
    )
    rise = r"(?:increas\w*|rise\w*|grow\w*|higher|stronger)"
    peak = r"(?:highest|maximum|maximal|strongest|best|optimal|optimum)"
    fall = r"(?:decreas\w*|declin\w*|fall\w*|lower|weaker)"
    return bool(
        re.search(
            rf"{response}.{{0,100}}{rise}.{{0,100}}{control_pattern}"
            rf".{{0,130}}{peak}.{{0,130}}{fall}",
            normalized,
        )
        or re.search(
            rf"{control_pattern}.{{0,100}}{rise}.{{0,130}}{peak}"
            rf".{{0,130}}{fall}",
            normalized,
        )
        or (
            re.search(rf"{response}.{{0,160}}{peak}", normalized)
            and re.search(rf"(?:whereas|but|then|subsequently).{{0,80}}{fall}", normalized)
            and re.search(control_pattern, normalized)
        )
    )


def _tested_optimum_marker(
    normalized: str,
    *,
    control_pattern: str,
) -> bool:
    return bool(
        re.search(control_pattern, normalized)
        and re.search(r"\b(?:among|tested|evaluated|investigated|series)\b", normalized)
        and re.search(
            r"\b(?:strongest|highest|best|maximum|maximal|optimal|optimum)\b",
            normalized,
        )
    )


def _presence_direction_shape(text: str) -> tuple[str, str] | None:
    normalized = _norm(text)
    has_gap = bool(re.search(r"\b(?:interior\s+)?(?:nano\s*)?gap\b", normalized))
    has_absent = bool(
        re.search(r"\b(?:nano)?gap[-\s]*less\b", normalized)
        or re.search(r"\bwithout\b.{0,30}\b(?:nano)?gap\b", normalized)
        or re.search(r"\babsence\s+of\b.{0,30}\b(?:nano)?gap\b", normalized)
    )
    stronger_present = bool(
        re.search(
            r"\b(?:stronger|higher|enhanced|greater|improved)\b",
            normalized,
        )
        or re.search(r"\bpromot(?:e|es|ed|ing)\b", normalized)
    )
    weaker_present = bool(
        re.search(
            r"\b(?:weaker|lower|reduced|decreased)\b",
            normalized,
        )
    )
    if has_gap and stronger_present and (
        has_absent
        or re.search(r"\b(?:presence|present|with|inside|within)\b", normalized)
    ):
        return "positive", "unspecified"
    if has_gap and weaker_present and (
        has_absent
        or re.search(r"\b(?:presence|present|with|inside|within)\b", normalized)
    ):
        return "negative", "unspecified"
    return None


def _direction_shape(text: str, control_key: str) -> tuple[str, str] | None:
    normalized = _norm(text)

    if control_key == "nanogap_presence":
        return _presence_direction_shape(text)

    spec = SERS_TREND_CONTROL_SPECS[control_key]
    control = "(?:" + "|".join(spec.patterns) + ")"
    response = (
        rf"(?:{_RAMAN_SIGNAL_PATTERN}|{_FORMAL_EF_PATTERN}|"
        r"sers\s+performance|sers\s+activity|signal|intensit\w*)"
    )

    # A genuine rise -> peak -> fall relation is stronger than a local
    # positive clause. Likewise, "among the tested ratios, X was strongest"
    # denotes a single optimum over the tested series. Saturation language
    # remains distinct: increase -> plateau is not a reversal.
    if _rise_peak_fall_marker(
        normalized,
        control_pattern=control,
    ):
        return "non_monotonic", "single_optimum"
    if (
        _tested_optimum_marker(
            normalized,
            control_pattern=control,
        )
        and not _saturation_marker(text)
    ):
        return "non_monotonic", "single_optimum"

    negative = bool(
        re.search(
            rf"{response}.{{0,90}}(?:increas\w*|higher|stronger|enhanc\w*)"
            rf".{{0,90}}(?:as|with).{{0,55}}{control}.{{0,40}}"
            rf"(?:decreas\w*|smaller|lower|narrower)",
            normalized,
        )
        or re.search(
            rf"(?:smaller|decreasing|decreased|lower|narrower).{{0,40}}"
            rf"{control}.{{0,100}}{response}.{{0,55}}"
            rf"(?:increas\w*|higher|stronger|enhanc\w*)",
            normalized,
        )
        or re.search(
            rf"{control}.{{0,55}}(?:increas\w*|larger|higher|wider)"
            rf".{{0,100}}{response}.{{0,55}}(?:decreas\w*|lower|weaker)",
            normalized,
        )
        or re.search(
            rf"{response}.{{0,90}}(?:decreas\w*|lower|weaker)"
            rf".{{0,90}}(?:with|as).{{0,55}}(?:increasing\s+)?{control}",
            normalized,
        )
    )
    positive = bool(
        re.search(
            rf"{response}.{{0,90}}(?:increas\w*|higher|stronger|enhanc\w*)"
            rf".{{0,90}}(?:with|as).{{0,60}}(?:increasing\s+)?{control}",
            normalized,
        )
        or re.search(
            rf"{control}.{{0,60}}(?:increas\w*|higher|larger)"
            rf".{{0,100}}{response}.{{0,55}}"
            rf"(?:increas\w*|higher|stronger|enhanc\w*)",
            normalized,
        )
        or re.search(
            rf"{response}.{{0,65}}\bproportional\b.{{0,65}}{control}",
            normalized,
        )
        or re.search(
            rf"{control}.{{0,65}}\bproportional\b.{{0,65}}{response}",
            normalized,
        )
    )

    if positive and not negative:
        return (
            "positive",
            "saturating" if _saturation_marker(text) else "monotonic",
        )
    if negative and not positive:
        return "negative", "monotonic"

    explicit_optimum = bool(
        re.search(r"\b(?:optimal|optimum)\b", normalized)
        and re.search(control, normalized)
    )
    if explicit_optimum and not _saturation_marker(text):
        return "non_monotonic", "single_optimum"

    if (
        "linear relationship" in normalized
        or "linear correlation" in normalized
        or re.search(r"\bcorrelat\w*\b", normalized)
    ):
        return "unspecified", "monotonic"
    return None

def _claim_trends(source: TrendEvidenceSource) -> list[TrendEvidence]:
    evidence: list[TrendEvidence] = []
    for claim_id, attrs in sorted(source.graph.nodes(data=True), key=lambda item: str(item[0])):
        if str(attrs.get("type", "")) not in _CLAIM_TYPES:
            continue
        text = v1._claim_text(attrs)
        if not text or (control := _claim_control(text)) is None:
            continue
        response = _claim_response(text, control_key=control[0])
        direction_shape = _direction_shape(text, control[0])
        if response is None or direction_shape is None:
            continue
        direction, shape = direction_shape
        normalized = _norm(text)
        basis = "reported_correlation" if re.search(r"\bcorrelat\w*\b", normalized) or "linear relationship" in normalized else "reported_directional_claim"
        causal_status = "source_asserted" if basis == "reported_directional_claim" and v1._explicit_causal_language(text) else "not_asserted"
        trend_id = stable_trend_id(paper_id=source.paper_id, independent_variable_key=control[0], dependent_observable_key=response[0], evidence_basis=basis, source_node_ids=(str(claim_id),))
        evidence.append(TrendEvidence(
            trend_id=trend_id, domain_profile_id="sers_au_ag", trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID, paper_id=source.paper_id,
            independent_variable_key=control[0], independent_variable_label=control[1], dependent_observable_key=response[0], dependent_observable_label=response[1],
            direction=direction, shape=shape, evidence_basis=basis, causal_status=causal_status, varied_dimension=control[0],
            subject_ids=v1._claim_subjects(source.graph, str(claim_id)), source_expression=text, source_expressions=(text,), source_claim_ids=(str(claim_id),), source_node_ids=(str(claim_id),), requires_verification=v1._requires_verification(attrs),
        ))
    return evidence


def extract_sers_au_ag_trend_evidence(source: TrendEvidenceSource) -> list[TrendEvidence]:
    evidence = [*_numeric_trends(source), *_claim_trends(source)]
    return sorted(evidence, key=lambda item: (item.paper_id, item.independent_variable_key, item.dependent_observable_key, item.evidence_basis, item.trend_id))


SERS_AU_AG_TREND_ADAPTER = TrendDomainAdapter(
    adapter_id="sers_au_ag",
    domain_profile_id="sers_au_ag",
    semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
    supported_evidence_bases=_SUPPORTED_EVIDENCE_BASES,
    required_inputs=_REQUIRED_INPUTS,
    extract_evidence_fn=extract_sers_au_ag_trend_evidence,
)
