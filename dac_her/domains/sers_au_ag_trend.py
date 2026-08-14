from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping

import networkx as nx

from dac_her.trend_domain import (
    TrendDomainAdapter,
    TrendEvidence,
    TrendEvidenceSource,
    TrendSeriesPoint,
)
from dac_her.trend_evidence import stable_trend_id


SERS_AU_AG_TREND_SEMANTICS_ID = "sers_au_ag_trend_v1_alpha4c2"

_SUPPORTED_EVIDENCE_BASES = frozenset({
    "controlled_numeric_series",
    "controlled_numeric_pair",
    "reported_directional_claim",
    "reported_correlation",
})

_REQUIRED_INPUTS = frozenset({
    "canonical_graph",
    "measurement_result_identity",
    "method_context",
    "comparison_context",
})

_NUMERIC_RESPONSE_KEYS = frozenset({
    "raman_intensity",
    "sers_enhancement_factor",
})

_METHOD_GUARD_DIMENSIONS = (
    "analyte",
    "reporter",
    "analyte_concentration",
    "excitation_wavelength",
    "laser_power",
    "integration_time",
    "sample_preparation",
    "preparation_medium",
    "measurement_environment",
    "sample_state",
    "substrate_condition",
)

_SUBJECT_TYPES = frozenset({
    "PlasmonicSubstrate",
    "Nanostructure",
    "Metal",
    "Material",
    "Support",
    "StructuralMotif",
    "Morphology",
    "SynthesisMethod",
})

_CLAIM_TYPES = frozenset({"ObservationClaim", "MechanismClaim"})


def _norm(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("μ", "µ")
        .replace("×", "x")
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _relation(attrs: Mapping[str, Any]) -> str:
    return str(attrs.get("relation", "")).strip()


def _incoming(
    graph: nx.Graph,
    node_id: str,
    relation: str,
) -> tuple[str, ...]:
    if node_id not in graph or not graph.is_directed():
        return ()
    if graph.is_multigraph():
        iterator = graph.in_edges(node_id, keys=True, data=True)
        values = {
            str(left)
            for left, _right, _key, attrs in iterator
            if _relation(attrs) == relation
        }
    else:
        iterator = graph.in_edges(node_id, data=True)
        values = {
            str(left)
            for left, _right, attrs in iterator
            if _relation(attrs) == relation
        }
    return tuple(sorted(values))


def _outgoing(
    graph: nx.Graph,
    node_id: str,
    relation: str,
) -> tuple[str, ...]:
    if node_id not in graph or not graph.is_directed():
        return ()
    if graph.is_multigraph():
        iterator = graph.out_edges(node_id, keys=True, data=True)
        values = {
            str(right)
            for _left, right, _key, attrs in iterator
            if _relation(attrs) == relation
        }
    else:
        iterator = graph.out_edges(node_id, data=True)
        values = {
            str(right)
            for _left, right, attrs in iterator
            if _relation(attrs) == relation
        }
    return tuple(sorted(values))


def _structured_conditions(
    graph: nx.Graph,
    node_id: str,
) -> tuple[dict[str, Any], ...]:
    if node_id not in graph:
        return ()
    raw = graph.nodes[node_id].get("conditions_json", "")
    if isinstance(raw, list):
        parsed = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return ()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(dict(row) for row in parsed if isinstance(row, dict))


def _condition_raw_value(condition: Mapping[str, Any]) -> str:
    text = str(condition.get("value_text") or "").strip()
    if text:
        return text
    numeric = condition.get("value_numeric")
    if numeric is None or not str(numeric).strip():
        return ""
    unit = str(condition.get("unit") or "").strip()
    return f"{numeric} {unit}".strip()


def _control_key_from_name(name: Any) -> str:
    text = _norm(name)
    if re.search(r"\b(?:ag|silver)?\s*shell\s+thickness\b", text):
        return "shell_thickness"
    if re.search(
        r"\b(?:interior\s+)?(?:nano\s*)?gap(?:\s+(?:size|width|distance))?\b",
        text,
    ):
        return "nanogap_size"
    if (
        re.search(r"\bau\s*[:/\-]\s*ag\b", text)
        or "au ag ratio" in text
        or "ratio of au to ag" in text
        or "gold silver ratio" in text
    ) and "ratio" in text:
        return "ag_to_au_ratio"
    if "ag/au ratio" in text or "ag to au ratio" in text:
        return "ag_to_au_ratio"
    return ""


def _length_nm(value: Any, unit: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    normalized = _norm(unit).replace(" ", "")
    factors = {
        "nm": 1.0,
        "å": 0.1,
        "angstrom": 0.1,
        "angstroms": 0.1,
        "µm": 1000.0,
        "um": 1000.0,
    }
    if normalized not in factors:
        return None
    return number * factors[normalized]


def _parse_length_text(text: str) -> float | None:
    match = re.fullmatch(
        r"\s*(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>nm|å|angstroms?|µm|um)\s*",
        _norm(text),
        re.I,
    )
    if not match:
        return None
    return _length_nm(match.group("value"), match.group("unit"))


def _parse_ag_to_au_ratio(condition: Mapping[str, Any]) -> float | None:
    raw = _condition_raw_value(condition)
    match = re.search(
        r"(?P<au>\d+(?:\.\d+)?)\s*:\s*(?P<ag>\d+(?:\.\d+)?)",
        raw,
    )
    if match:
        au = float(match.group("au"))
        ag = float(match.group("ag"))
        if au > 0 and math.isfinite(au) and math.isfinite(ag):
            return ag / au

    name = _norm(condition.get("name", ""))
    numeric = condition.get("value_numeric")
    if (
        numeric is not None
        and ("ag/au ratio" in name or "ag to au ratio" in name)
    ):
        try:
            value = float(numeric)
        except (TypeError, ValueError):
            return None
        if math.isfinite(value):
            return value
    return None


def _control_value(
    condition: Mapping[str, Any],
) -> tuple[str, str, float, str] | None:
    key = _control_key_from_name(condition.get("name", ""))
    if not key:
        return None
    if key in {"shell_thickness", "nanogap_size"}:
        numeric = condition.get("value_numeric")
        unit = condition.get("unit", "")
        value = _length_nm(numeric, unit)
        if value is None:
            value = _parse_length_text(_condition_raw_value(condition))
        if value is None:
            return None
        label = (
            "Ag shell thickness"
            if key == "shell_thickness"
            else "nanogap size"
        )
        return key, label, value, "nm"
    ratio = _parse_ag_to_au_ratio(condition)
    if ratio is None:
        return None
    return key, "Ag/Au ratio", ratio, "Ag/Au ratio"


def _identity_by_representative(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        representative = str(
            row.get("representative_measurement_id", "")
        ).strip()
        if representative:
            result[representative] = row
    return result


def _method_by_id(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("method_context_id", "")): row
        for row in rows
        if str(row.get("method_context_id", "")).strip()
    }


def _method_dimension_map(
    row: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for item in row.get("dimensions", []) or []:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).strip()
        if name:
            result[name] = item
    return result


def _methods_compatible(rows: Iterable[Mapping[str, Any]]) -> bool:
    rows = list(rows)
    for name in _METHOD_GUARD_DIMENSIONS:
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


def _measurement_mentions(
    measurement_id: str,
    identity_row: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if identity_row is None:
        return (measurement_id,)
    values = tuple(
        str(value)
        for value in identity_row.get("source_mention_ids", [])
        if str(value).strip()
    )
    return values or (measurement_id,)


def _measurement_control(
    graph: nx.Graph,
    measurement_id: str,
    identity_row: Mapping[str, Any] | None,
) -> tuple[str, str, float, str] | None:
    observed: set[tuple[str, str, float, str]] = set()
    for mention_id in _measurement_mentions(measurement_id, identity_row):
        for condition in _structured_conditions(graph, mention_id):
            parsed = _control_value(condition)
            if parsed is not None:
                observed.add(parsed)
    if len(observed) != 1:
        # Missing control or multiple explicit controls both fail closed.
        return None
    return next(iter(observed))


def _lineage(
    graph: nx.Graph,
    measurement_id: str,
    identity_row: Mapping[str, Any] | None,
    method_row: Mapping[str, Any],
    context_row: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]] | None:
    mentions = _measurement_mentions(measurement_id, identity_row)
    group_ids: set[str] = set()
    experiment_ids: set[str] = set()

    for mention_id in mentions:
        group_ids.update(_outgoing(graph, mention_id, "IN_MEASUREMENT_GROUP"))
        for producer in _incoming(graph, mention_id, "HAS_MEASUREMENT"):
            if (
                producer in graph
                and str(graph.nodes[producer].get("type", "")) == "Experiment"
            ):
                experiment_ids.add(producer)

    for node_id in (
        list(method_row.get("source_node_ids", []) or [])
        + list(context_row.get("source_node_ids", []) or [])
    ):
        node_id = str(node_id)
        if node_id not in graph:
            continue
        if str(graph.nodes[node_id].get("type", "")) == "MeasurementGroup":
            group_ids.add(node_id)

    for producer in method_row.get("producer_ids", []) or []:
        producer = str(producer)
        if (
            producer in graph
            and str(graph.nodes[producer].get("type", "")) == "Experiment"
        ):
            experiment_ids.add(producer)

    if group_ids:
        return "measurement_group", tuple(sorted(group_ids))
    if experiment_ids:
        return "experiment", tuple(sorted(experiment_ids))
    return None


def _numeric_direction_shape(
    points: list[tuple[float, float]],
) -> tuple[str, str]:
    ordered = sorted(points)
    deltas = [
        ordered[index + 1][1] - ordered[index][1]
        for index in range(len(ordered) - 1)
    ]
    signs = [0 if delta == 0 else (1 if delta > 0 else -1) for delta in deltas]
    if all(sign == 0 for sign in signs):
        return "unchanged", "unspecified"
    if all(sign >= 0 for sign in signs) and any(sign > 0 for sign in signs):
        return "positive", "monotonic"
    if all(sign <= 0 for sign in signs) and any(sign < 0 for sign in signs):
        return "negative", "monotonic"

    nonzero = [sign for sign in signs if sign]
    if 1 in nonzero and -1 in nonzero:
        first_negative = nonzero.index(-1)
        if (
            all(sign == 1 for sign in nonzero[:first_negative])
            and all(sign == -1 for sign in nonzero[first_negative:])
        ):
            return "non_monotonic", "single_optimum"
        first_positive = nonzero.index(1)
        if (
            all(sign == -1 for sign in nonzero[:first_positive])
            and all(sign == 1 for sign in nonzero[first_positive:])
        ):
            return "non_monotonic", "u_shaped"
    return "non_monotonic", "unspecified"


def _numeric_trends(source: TrendEvidenceSource) -> list[TrendEvidence]:
    graph = source.graph
    identity_by_rep = _identity_by_representative(
        source.measurement_result_rows
    )
    method_by_id = _method_by_id(source.method_context_rows)

    candidates: dict[
        tuple[str, str, str, tuple[str, ...], str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for context in source.comparison_context_rows:
        measurement_id = str(context.get("measurement_id", "")).strip()
        observable_key = str(context.get("observable_key", "")).strip()
        value = context.get("value_numeric")
        if (
            not measurement_id
            or observable_key not in _NUMERIC_RESPONSE_KEYS
            or value is None
        ):
            continue
        try:
            dependent_value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(dependent_value):
            continue

        method_id = str(context.get("method_context_id", "")).strip()
        method_row = method_by_id.get(method_id)
        if method_row is None:
            continue
        identity_row = identity_by_rep.get(measurement_id)
        if identity_row is None:
            continue
        control = _measurement_control(
            graph,
            measurement_id,
            identity_row,
        )
        if control is None:
            continue
        control_key, control_label, control_value, control_unit = control
        lineage = _lineage(
            graph,
            measurement_id,
            identity_row,
            method_row,
            context,
        )
        if lineage is None:
            continue
        lineage_kind, lineage_ids = lineage
        dependent_unit = str(context.get("unit", "")).strip()
        key = (
            control_key,
            observable_key,
            lineage_kind,
            lineage_ids,
            dependent_unit,
        )
        candidates[key].append(
            {
                "measurement_id": measurement_id,
                "identity_id": str(identity_row.get("identity_id", "")),
                "context": context,
                "method": method_row,
                "control_label": control_label,
                "control_value": control_value,
                "control_unit": control_unit,
                "dependent_value": dependent_value,
                "dependent_unit": dependent_unit,
            }
        )

    evidence: list[TrendEvidence] = []
    for (
        control_key,
        observable_key,
        lineage_kind,
        lineage_ids,
        dependent_unit,
    ), rows in sorted(candidates.items(), key=lambda item: str(item[0])):
        if len(rows) < 2:
            continue
        if not _methods_compatible(row["method"] for row in rows):
            continue

        x_values = [float(row["control_value"]) for row in rows]
        if len(x_values) != len(set(x_values)):
            # Never average or silently select repeated x values.
            continue

        ordered = sorted(rows, key=lambda row: float(row["control_value"]))
        direction, shape = _numeric_direction_shape([
            (float(row["control_value"]), float(row["dependent_value"]))
            for row in ordered
        ])
        basis = (
            "controlled_numeric_pair"
            if len(ordered) == 2
            else "controlled_numeric_series"
        )

        measurement_ids = tuple(sorted({
            str(row["measurement_id"]) for row in ordered
        }))
        result_ids = tuple(sorted({
            str(row["identity_id"])
            for row in ordered
            if str(row["identity_id"]).strip()
        }))
        method_ids = tuple(sorted({
            str(row["method"].get("method_context_id", ""))
            for row in ordered
            if str(row["method"].get("method_context_id", "")).strip()
        }))
        context_ids = tuple(sorted({
            str(row["context"].get("context_id", ""))
            for row in ordered
            if str(row["context"].get("context_id", "")).strip()
        }))
        subject_ids = tuple(sorted({
            str(subject_id)
            for row in ordered
            for subject_id in row["context"].get("subject_ids", []) or []
            if str(subject_id).strip()
        }))
        source_expressions = tuple(sorted({
            str(row["context"].get("source_expression", "")).strip()
            for row in ordered
            if str(row["context"].get("source_expression", "")).strip()
        }))

        source_node_ids = tuple(sorted({
            *measurement_ids,
            *lineage_ids,
        }))
        group_ids = (
            lineage_ids if lineage_kind == "measurement_group" else ()
        )
        experiment_ids = (
            lineage_ids if lineage_kind == "experiment" else ()
        )

        points = tuple(
            TrendSeriesPoint(
                point_id=f"{source.paper_id}:{row['measurement_id']}",
                independent_value_numeric=float(row["control_value"]),
                independent_unit=str(row["control_unit"]),
                dependent_value_numeric=float(row["dependent_value"]),
                dependent_unit=dependent_unit,
                source_measurement_result_ids=(str(row["identity_id"]),),
                source_measurement_ids=(str(row["measurement_id"]),),
                source_node_ids=(str(row["measurement_id"]),),
            )
            for row in ordered
        )

        observable_label = str(
            ordered[0]["context"].get("observable_label", observable_key)
        ).strip() or observable_key
        trend_id = stable_trend_id(
            paper_id=source.paper_id,
            independent_variable_key=control_key,
            dependent_observable_key=observable_key,
            evidence_basis=basis,
            source_node_ids=source_node_ids,
        )
        evidence.append(
            TrendEvidence(
                trend_id=trend_id,
                domain_profile_id="sers_au_ag",
                trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
                paper_id=source.paper_id,
                independent_variable_key=control_key,
                independent_variable_label=str(ordered[0]["control_label"]),
                dependent_observable_key=observable_key,
                dependent_observable_label=observable_label,
                direction=direction,
                shape=shape,
                evidence_basis=basis,
                causal_status="not_asserted",
                varied_dimension=control_key,
                subject_ids=subject_ids,
                series_points=points,
                source_expression=(source_expressions[0] if source_expressions else ""),
                source_expressions=source_expressions,
                source_measurement_ids=measurement_ids,
                source_measurement_group_ids=tuple(group_ids),
                source_experiment_ids=tuple(experiment_ids),
                source_measurement_result_ids=result_ids,
                source_method_context_ids=method_ids,
                source_comparison_context_ids=context_ids,
                source_node_ids=source_node_ids,
            )
        )
    return evidence


def _claim_text(attrs: Mapping[str, Any]) -> str:
    for key in (
        "statement",
        "source_expression",
        "description",
        "label",
        "node_text",
    ):
        value = str(attrs.get(key, "")).strip()
        if value:
            return value
    return ""


def _claim_control(text: str) -> tuple[str, str] | None:
    normalized = _norm(text)
    matches: list[tuple[str, str]] = []
    if re.search(r"\b(?:ag|silver)?\s*shell\s+thickness\b", normalized):
        matches.append(("shell_thickness", "Ag shell thickness"))
    if re.search(
        r"\b(?:interior\s+)?(?:nano\s*)?gap(?:\s+(?:size|width|distance))?\b",
        normalized,
    ):
        matches.append(("nanogap_size", "nanogap size"))
    if (
        re.search(r"\bau\s*[:/\-]\s*ag\b", normalized)
        or "au ag ratio" in normalized
        or "au-ag ratio" in normalized
        or "ratio of au to ag" in normalized
    ) and "ratio" in normalized:
        matches.append(("ag_to_au_ratio", "Ag/Au ratio"))
    unique = {item[0]: item for item in matches}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _claim_response(text: str) -> tuple[str, str] | None:
    normalized = _norm(text)
    if not re.search(r"\b(?:sers|serrs|raman)\b", normalized):
        return None
    if re.search(r"\benhancement\s+factor\b|\bsers\s+ef\b", normalized):
        return "sers_enhancement_factor", "SERS enhancement factor"
    if (
        re.search(
            r"\b(?:sers|serrs|raman)\b.{0,45}\b(?:intensit|signal)",
            normalized,
        )
        or re.search(
            r"\b(?:intensit|signal)\w*\b.{0,45}\b(?:sers|serrs|raman)\b",
            normalized,
        )
    ):
        return "raman_intensity", "SERS/Raman intensity"
    if (
        re.search(
            r"\b(?:sers|serrs)\b.{0,50}\b(?:performance|activity|enhancement|sensitivity)",
            normalized,
        )
        or re.search(
            r"\b(?:performance|activity|enhancement|sensitivity)\b.{0,50}\b(?:sers|serrs)\b",
            normalized,
        )
    ):
        return "sers_performance", "SERS performance"
    return None


def _saturation_marker(text: str) -> bool:
    normalized = _norm(text)
    return bool(
        re.search(r"\b(?:approach(?:es|ed|ing)?|reach(?:es|ed|ing)?)\b.{0,35}\b(?:maximum|maximal)\b", normalized)
        or re.search(r"\bplateau(?:s|ed|ing)?\b", normalized)
        or "close to the maximum" in normalized
        or "essentially the same" in normalized
        or "critical thickness" in normalized
    )


def _explicit_causal_language(text: str) -> bool:
    normalized = _norm(text)
    return bool(re.search(
        r"\b(?:caus(?:e|es|ed|ing)|lead(?:s|ing)?\s+to|result(?:s|ed|ing)?\s+in|"
        r"due\s+to|attribut(?:e|ed|ing)\s+to|driv(?:e|es|en|ing)|induc(?:e|es|ed|ing))\b",
        normalized,
    ))


def _claim_direction_shape(
    control_key: str,
    text: str,
) -> tuple[str, str] | None:
    normalized = _norm(text)
    response = r"(?:sers|serrs|raman)"

    if control_key == "shell_thickness":
        control = r"(?:ag\s+|silver\s+)?shell\s+thickness"
        positive = bool(
            re.search(
                rf"{response}.{{0,70}}(?:increase\w*|higher|stronger|enhanc\w*).{{0,70}}(?:with|as).{{0,35}}(?:increasing\s+)?{control}",
                normalized,
            )
            or re.search(
                rf"{control}.{{0,55}}(?:increase\w*|thicken\w*).{{0,80}}{response}.{{0,55}}(?:increase\w*|higher|stronger|enhanc\w*)",
                normalized,
            )
            or re.search(
                rf"(?:intensit\w*|enhancement\s+factor).{{0,35}}increase\w*\s+with.{{0,30}}{control}",
                normalized,
            )
        )
        negative = bool(
            re.search(
                rf"{response}.{{0,70}}(?:decrease\w*|lower|weaker).{{0,70}}(?:with|as).{{0,35}}(?:increasing\s+)?{control}",
                normalized,
            )
        )
        if positive and not negative:
            return "positive", ("saturating" if _saturation_marker(text) else "monotonic")
        if negative and not positive:
            return "negative", "monotonic"
        return None

    if control_key == "nanogap_size":
        gap = r"(?:interior\s+)?(?:nano\s*)?gap(?:\s+(?:size|width|distance))?"
        negative = bool(
            re.search(
                rf"{response}.{{0,80}}(?:increase\w*|higher|stronger|enhanc\w*).{{0,80}}(?:as|with).{{0,35}}{gap}.{{0,25}}(?:decrease\w*|smaller|narrower)",
                normalized,
            )
            or re.search(
                rf"(?:smaller|decreasing|decreased).{{0,25}}{gap}.{{0,90}}(?:increase\w*|higher|stronger|enhanc\w*).{{0,40}}{response}",
                normalized,
            )
            or re.search(
                rf"{gap}.{{0,30}}(?:increase\w*|larger|wider).{{0,85}}{response}.{{0,55}}(?:decrease\w*|lower|weaker)",
                normalized,
            )
        )
        positive = bool(
            re.search(
                rf"{response}.{{0,80}}(?:increase\w*|higher|stronger).{{0,80}}(?:as|with).{{0,35}}{gap}.{{0,25}}(?:increase\w*|larger|wider)",
                normalized,
            )
        )
        if negative and not positive:
            return "negative", "monotonic"
        if positive and not negative:
            return "positive", "monotonic"
        return None

    if control_key == "ag_to_au_ratio":
        optimum = bool(
            re.search(
                r"\b(?:strongest|highest|best|optimal|optimum|maximum)\b",
                normalized,
            )
            and re.search(r"\b(?:ratio|au\s*[:/\-]\s*ag)\b", normalized)
        )
        if optimum:
            return "non_monotonic", "single_optimum"
        return None

    return None


def _claim_subjects(
    graph: nx.Graph,
    claim_id: str,
) -> tuple[str, ...]:
    subjects: set[str] = set()
    attrs = graph.nodes[claim_id]
    explicit = str(attrs.get("subject_id", "")).strip()
    if explicit and explicit in graph:
        subjects.add(explicit)
    for node_id in _outgoing(graph, claim_id, "APPLIES_TO"):
        if (
            node_id in graph
            and str(graph.nodes[node_id].get("type", "")) in _SUBJECT_TYPES
        ):
            subjects.add(node_id)
    return tuple(sorted(subjects))


def _requires_verification(attrs: Mapping[str, Any]) -> bool:
    value = attrs.get("requires_verification", False)
    if isinstance(value, bool):
        return value
    if _norm(value) in {"true", "1", "yes"}:
        return True
    return _norm(attrs.get("evidence_status", "")) in {
        "candidate",
        "requires_verification",
        "unconfirmed",
    }


def _claim_trends(source: TrendEvidenceSource) -> list[TrendEvidence]:
    graph = source.graph
    evidence: list[TrendEvidence] = []
    for claim_id, attrs in sorted(graph.nodes(data=True), key=lambda item: str(item[0])):
        if str(attrs.get("type", "")) not in _CLAIM_TYPES:
            continue
        text = _claim_text(attrs)
        if not text:
            continue
        control = _claim_control(text)
        response = _claim_response(text)
        if control is None or response is None:
            continue
        direction_shape = _claim_direction_shape(control[0], text)
        if direction_shape is None:
            continue
        direction, shape = direction_shape
        basis = (
            "reported_correlation"
            if re.search(r"\bcorrelat\w*\b", _norm(text))
            else "reported_directional_claim"
        )
        causal_status = (
            "source_asserted"
            if basis == "reported_directional_claim"
            and _explicit_causal_language(text)
            else "not_asserted"
        )
        trend_id = stable_trend_id(
            paper_id=source.paper_id,
            independent_variable_key=control[0],
            dependent_observable_key=response[0],
            evidence_basis=basis,
            source_node_ids=(str(claim_id),),
        )
        evidence.append(
            TrendEvidence(
                trend_id=trend_id,
                domain_profile_id="sers_au_ag",
                trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
                paper_id=source.paper_id,
                independent_variable_key=control[0],
                independent_variable_label=control[1],
                dependent_observable_key=response[0],
                dependent_observable_label=response[1],
                direction=direction,
                shape=shape,
                evidence_basis=basis,
                causal_status=causal_status,
                varied_dimension=control[0],
                subject_ids=_claim_subjects(graph, str(claim_id)),
                source_expression=text,
                source_expressions=(text,),
                source_claim_ids=(str(claim_id),),
                source_node_ids=(str(claim_id),),
                requires_verification=_requires_verification(attrs),
            )
        )
    return evidence


def extract_sers_au_ag_trend_evidence(
    source: TrendEvidenceSource,
) -> list[TrendEvidence]:
    evidence = [
        *_numeric_trends(source),
        *_claim_trends(source),
    ]
    return sorted(
        evidence,
        key=lambda item: (
            item.paper_id,
            item.independent_variable_key,
            item.dependent_observable_key,
            item.evidence_basis,
            item.trend_id,
        ),
    )


SERS_AU_AG_TREND_ADAPTER = TrendDomainAdapter(
    adapter_id="sers_au_ag",
    domain_profile_id="sers_au_ag",
    semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
    supported_evidence_bases=_SUPPORTED_EVIDENCE_BASES,
    required_inputs=_REQUIRED_INPUTS,
    extract_evidence_fn=extract_sers_au_ag_trend_evidence,
)
