from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Iterable

import networkx as nx

from dac_her.comparison_context import (
    dimension_from_values,
    method_dimension_from_values,
    normalize_comparison_value,
    stable_comparison_id,
)
from dac_her.comparison_domain import (
    ComparisonContext,
    ComparisonDimensionValue,
    ComparisonDomainAdapter,
    ObservableComparisonPolicy,
)
from dac_her.method_context import (
    MethodContext,
    MethodContextSemantics,
    MethodDimensionValue,
)


SERS_COMPARISON_DIMENSIONS = (
    "analyte",
    "reporter",
    "concentration",
    "excitation_wavelength",
    "laser_power",
    "integration_time",
    "raman_peak",
    "measurement_environment",
    "sample_state",
    "substrate_condition",
)

SERS_METHOD_DIMENSIONS = (
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

SERS_METHOD_SEMANTICS = MethodContextSemantics(
    semantics_id="sers_au_ag_method_v4_alpha4b3b321",
    dimensions=SERS_METHOD_DIMENSIONS,
    critical_dimensions=frozenset({
        "analyte",
        "reporter",
        "excitation_wavelength",
    }),
    numeric_ranking_allowed_protocols=frozenset({
        "same_protocol",
    }),
)

SERS_COMPARISON_SEMANTICS_ID = "sers_au_ag_comparison_v7_alpha4b3b321"

# Observable-specific applicability is calibrated only on SERS_1/5/8.
# Exact keys are used intentionally: unseen/held-out observables fail closed
# as `observable_policy_unregistered` rather than being guessed into a family.
_SERS_OBSERVABLE_POLICIES = (
    ObservableComparisonPolicy(
        policy_id="sers_ef_v1",
        family="sers_performance",
        observable_keys=frozenset({"sers_enhancement_factor"}),
        applicable_dimensions=(
            "analyte",
            "reporter",
            "concentration",
            "excitation_wavelength",
            "raman_peak",
            "measurement_environment",
            "sample_state",
            "substrate_condition",
        ),
        ranking_required_dimensions=frozenset({
            "analyte",
            "reporter",
            "concentration",
            "excitation_wavelength",
            "raman_peak",
            "measurement_environment",
            "sample_state",
            "substrate_condition",
        }),
        numeric_ranking_mode="allowed_if_complete",
        ranking_direction="higher_better",
    ),
    ObservableComparisonPolicy(
        policy_id="raman_intensity_v1",
        family="sers_performance",
        observable_keys=frozenset({"raman_intensity"}),
        applicable_dimensions=(
            "analyte",
            "reporter",
            "concentration",
            "excitation_wavelength",
            "laser_power",
            "integration_time",
            "raman_peak",
            "measurement_environment",
            "sample_state",
            "substrate_condition",
        ),
        ranking_required_dimensions=frozenset({
            "analyte",
            "reporter",
            "concentration",
            "excitation_wavelength",
            "laser_power",
            "integration_time",
            "raman_peak",
            "measurement_environment",
            "sample_state",
            "substrate_condition",
        }),
        numeric_ranking_mode="allowed_if_complete",
        ranking_direction="higher_better",
    ),
    ObservableComparisonPolicy(
        policy_id="raman_peak_position_v1",
        family="raman_spectral",
        observable_keys=frozenset({"raman_peak_position"}),
        applicable_dimensions=(
            "analyte",
            "reporter",
            "excitation_wavelength",
            "measurement_environment",
            "sample_state",
            "substrate_condition",
        ),
        ranking_required_dimensions=frozenset(),
        numeric_ranking_mode="disabled",
        ranking_direction="none",
    ),
    ObservableComparisonPolicy(
        policy_id="detection_limit_v1",
        family="analytical_performance",
        observable_keys=frozenset({"detection_limit"}),
        applicable_dimensions=(
            "analyte",
            "reporter",
            "excitation_wavelength",
            "laser_power",
            "integration_time",
            "raman_peak",
            "measurement_environment",
            "sample_state",
            "substrate_condition",
        ),
        ranking_required_dimensions=frozenset(),
        numeric_ranking_mode="disabled",
        ranking_direction="none",
    ),
    ObservableComparisonPolicy(
        policy_id="optical_peak_v1",
        family="optical_spectral",
        observable_keys=frozenset({
            "extinction_peak_wavelength",
            "absorption_band_wavelength",
            "absorption_shoulder_wavelength",
            "lspr_wavelength",
        }),
        applicable_dimensions=(
            "measurement_environment",
            "sample_state",
            "substrate_condition",
        ),
        ranking_required_dimensions=frozenset(),
        numeric_ranking_mode="disabled",
        ranking_direction="none",
    ),
    ObservableComparisonPolicy(
        policy_id="structural_metric_v1",
        family="structural",
        observable_keys=frozenset({
            "xrd_diffraction_peak_position",
            "particle_size",
            "nanogap_size",
            "lattice_plane_spacing",
            "shell_thickness",
            "aspect_ratio",
            "particle_yield",
        }),
        applicable_dimensions=("substrate_condition",),
        ranking_required_dimensions=frozenset(),
        numeric_ranking_mode="disabled",
        ranking_direction="none",
    ),
    ObservableComparisonPolicy(
        policy_id="composition_metric_v1",
        family="composition",
        observable_keys=frozenset({"atomic_fraction"}),
        applicable_dimensions=(),
        ranking_required_dimensions=frozenset(),
        numeric_ranking_mode="disabled",
        ranking_direction="none",
    ),
    ObservableComparisonPolicy(
        policy_id="field_simulation_v1",
        family="simulation",
        observable_keys=frozenset({"local_field_enhancement"}),
        applicable_dimensions=(
            "excitation_wavelength",
        ),
        ranking_required_dimensions=frozenset(),
        numeric_ranking_mode="disabled",
        ranking_direction="none",
    ),
    ObservableComparisonPolicy(
        policy_id="reproducibility_v1",
        family="analytical_quality",
        observable_keys=frozenset({"relative_standard_deviation"}),
        applicable_dimensions=(
            "analyte",
            "reporter",
            "excitation_wavelength",
            "laser_power",
            "integration_time",
            "raman_peak",
            "measurement_environment",
            "sample_state",
            "substrate_condition",
        ),
        ranking_required_dimensions=frozenset(),
        numeric_ranking_mode="disabled",
        ranking_direction="none",
    ),
    ObservableComparisonPolicy(
        policy_id="stability_signal_v1",
        family="stability",
        observable_keys=frozenset({"signal_retention"}),
        applicable_dimensions=(
            "analyte",
            "reporter",
            "concentration",
            "excitation_wavelength",
            "laser_power",
            "integration_time",
            "raman_peak",
            "measurement_environment",
            "sample_state",
            "substrate_condition",
        ),
        ranking_required_dimensions=frozenset(),
        numeric_ranking_mode="disabled",
        ranking_direction="none",
    ),
    ObservableComparisonPolicy(
        policy_id="calibration_statistic_v1",
        family="analytical_calibration",
        observable_keys=frozenset({
            "concentration_sers_intensity_correlation_r2",
            "log_concentration_log_sers_intensity_correlation_coefficient",
        }),
        applicable_dimensions=(
            "analyte",
            "reporter",
            "excitation_wavelength",
            "raman_peak",
            "measurement_environment",
            "sample_state",
            "substrate_condition",
        ),
        ranking_required_dimensions=frozenset(),
        numeric_ranking_mode="disabled",
        ranking_direction="none",
    ),
    ObservableComparisonPolicy(
        policy_id="hybridization_claim_v1",
        family="qualitative_mechanistic",
        observable_keys=frozenset({"hybridization_complex_formation"}),
        applicable_dimensions=(
            "reporter",
            "excitation_wavelength",
        ),
        ranking_required_dimensions=frozenset(),
        numeric_ranking_mode="disabled",
        ranking_direction="none",
    ),
)


_SCIENTIFIC_SUBJECT_TYPES = frozenset({
    "PlasmonicSubstrate",
    "Nanostructure",
    "Metal",
    "Material",
    "Support",
    "StructuralMotif",
    "Morphology",
    "SynthesisMethod",
})

_SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
})

_CONCENTRATION_RE = re.compile(
    r"(?<![A-Za-z0-9.])"
    r"(?P<number>"
    r"(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*(?:x|×)\s*10\s*\^?\s*[+\-−]?\s*\d+|"
    r"[eE]\s*[+\-−]?\s*\d+)?"
    r"|10\s*\^?\s*[+\-−]?\s*\d+"
    r")"
    r"\s*(?P<unit>aM|fM|pM|nM|uM|µM|μM|mM|M)\b"
)
_WAVELENGTH_RE = re.compile(
    r"(?<![\d.])(?P<number>\d+(?:\.\d+)?)\s*nm\b",
    re.I,
)
_POWER_RE = re.compile(
    r"(?<![\d.])(?P<number>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>uW|µW|μW|mW|W)\b",
)
_TIME_RE = re.compile(
    r"(?<![\d.])(?P<number>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>ms|s|sec|secs|second|seconds|min|mins|minute|minutes)\b",
    re.I,
)
_RAMAN_PEAK_RE = re.compile(
    r"(?<![\d.])(?P<number>\d+(?:\.\d+)?)\s*"
    r"(?:cm\s*(?:\^?\s*[-−]\s*1|[-−]\s*1)|cm⁻¹)\b",
    re.I,
)

# Domain-owned aliases are intentionally narrow. Generic role suffixes and
# explicit parenthetical abbreviations may be stripped, but an abbreviation
# such as bare "ATP" is never expanded to a chemical name.
_EXACT_ENTITY_ALIASES = {
    "mb": "methylene blue",
    "methylene blue (mb)": "methylene blue",
}


def _relation(attrs: dict[str, Any]) -> str:
    return str(attrs.get("relation", "")).strip()


def _incoming(
    graph: nx.Graph,
    node_id: str,
    relation: str,
) -> list[str]:
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


def _outgoing(
    graph: nx.Graph,
    node_id: str,
    relation: str,
) -> list[str]:
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


def _node_label(graph: nx.Graph, node_id: str) -> str:
    attrs = graph.nodes[node_id]
    return str(
        attrs.get("label")
        or attrs.get("statement")
        or attrs.get("metric")
        or attrs.get("name")
        or node_id
    )


def _texts(graph: nx.Graph, node_ids: Iterable[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    keys = (
        "label",
        "statement",
        "metric",
        "source_expression",
        "description",
        "condition",
        "conditions",
        "value_text",
        "node_text",
    )
    for node_id in sorted(set(map(str, node_ids))):
        if node_id not in graph:
            continue
        attrs = graph.nodes[node_id]
        for key in keys:
            value = str(attrs.get(key, "")).strip()
            if value:
                rows.append((value, node_id))
    return rows


def _attr_values(
    graph: nx.Graph,
    node_ids: Iterable[str],
    keys: Iterable[str],
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for node_id in sorted(set(map(str, node_ids))):
        if node_id not in graph:
            continue
        attrs = graph.nodes[node_id]
        for key in keys:
            value = str(attrs.get(key, "")).strip()
            if value:
                rows.append((value, node_id))
    return rows


def _ascii_scientific_text(value: Any) -> str:
    text = str(value or "").strip()
    # Convert superscript exponent runs before ordinary character cleanup.
    def convert(match: re.Match[str]) -> str:
        return "^" + match.group(0).translate(_SUPERSCRIPT_MAP)

    text = re.sub(r"[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+", convert, text)
    return (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("μ", "µ")
        .replace("×", "x")
    )


def _parse_number_expression(value: str) -> float | None:
    text = _ascii_scientific_text(value)
    text = re.sub(r"\s+", "", text)

    direct = re.fullmatch(
        r"(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?",
        text,
    )
    if direct:
        try:
            result = float(text)
        except ValueError:
            return None
        return result if math.isfinite(result) else None

    scientific = re.fullmatch(
        r"(?P<mantissa>\d+(?:\.\d+)?|\.\d+)"
        r"x10\^(?P<exponent>[+\-]?\d+)",
        text,
    )
    if scientific:
        result = (
            float(scientific.group("mantissa"))
            * (10.0 ** int(scientific.group("exponent")))
        )
        return result if math.isfinite(result) else None

    power_only = re.fullmatch(r"10\^(?P<exponent>[+\-]?\d+)", text)
    if power_only:
        result = 10.0 ** int(power_only.group("exponent"))
        return result if math.isfinite(result) else None

    return None


def _format_scalar(value: float, unit: str) -> str:
    if value == 0:
        number = "0"
    elif abs(value) < 1e-3 or abs(value) >= 1e4:
        number = f"{value:.12e}"
        mantissa, exponent = number.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exponent = str(int(exponent))
        number = f"{mantissa}e{exponent}"
    else:
        number = f"{value:.12g}"
    return f"{number} {unit}"


def _canonical_entity(value: Any) -> str:
    text = normalize_comparison_value(value)
    if not text:
        return ""
    text = re.sub(
        r"\s+(?:raman\s+reporter|reporter|analyte)$",
        "",
        text,
    ).strip()
    if text in _EXACT_ENTITY_ALIASES:
        return _EXACT_ENTITY_ALIASES[text]

    # A parenthetical abbreviation is safe to discard when a descriptive
    # primary label is present: "methylene blue (MB)" -> "methylene blue".
    match = re.fullmatch(
        r"(?P<primary>.+?)\s*\((?P<abbr>[a-z0-9+\-]+)\)",
        text,
    )
    if match and len(match.group("primary").strip()) >= 4:
        primary = match.group("primary").strip()
        return _EXACT_ENTITY_ALIASES.get(primary, primary)

    return text


def _canonical_concentration(value: Any) -> str:
    text = _ascii_scientific_text(value)
    match = _CONCENTRATION_RE.fullmatch(text.strip())
    if not match:
        return ""

    number = _parse_number_expression(match.group("number"))
    if number is None:
        return ""

    factor = {
        "M": 1.0,
        "mM": 1e-3,
        "uM": 1e-6,
        "µM": 1e-6,
        "nM": 1e-9,
        "pM": 1e-12,
        "fM": 1e-15,
        "aM": 1e-18,
    }[match.group("unit")]
    return _format_scalar(number * factor, "M")


def _canonical_wavelength(value: Any) -> str:
    text = _ascii_scientific_text(value)
    match = _WAVELENGTH_RE.fullmatch(text.strip())
    if not match:
        return ""
    return _format_scalar(float(match.group("number")), "nm")


def _canonical_power(value: Any) -> str:
    text = _ascii_scientific_text(value)
    match = _POWER_RE.fullmatch(text.strip())
    if not match:
        return ""
    number = float(match.group("number"))
    factor = {
        "uW": 1e-3,
        "µW": 1e-3,
        "mW": 1.0,
        "W": 1e3,
    }[match.group("unit")]
    return _format_scalar(number * factor, "mW")


def _canonical_time(value: Any) -> str:
    text = _ascii_scientific_text(value)
    match = _TIME_RE.fullmatch(text.strip())
    if not match:
        return ""
    number = float(match.group("number"))
    unit = match.group("unit").lower()
    if unit == "ms":
        number *= 1e-3
    elif unit in {"min", "mins", "minute", "minutes"}:
        number *= 60.0
    return _format_scalar(number, "s")


def _canonical_raman_peak(value: Any) -> str:
    text = _ascii_scientific_text(value)
    # The regex accepts ASCII forms after scientific-text normalization.
    match = re.fullmatch(
        r"(?P<number>\d+(?:\.\d+)?)\s*cm\s*\^?\s*-\s*1",
        text.strip(),
        re.I,
    )
    if not match:
        return ""
    return _format_scalar(float(match.group("number")), "cm^-1")


def _regex_values(
    rows: Iterable[tuple[str, str]],
    pattern: re.Pattern[str],
    *,
    required_markers: tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for text, node_id in rows:
        normalized_text = normalize_comparison_value(text)
        if required_markers and not any(
            marker in normalized_text for marker in required_markers
        ):
            continue
        for match in pattern.finditer(_ascii_scientific_text(text)):
            found.append((match.group(0), node_id))
    return found


def _entity_dimension(
    graph: nx.Graph,
    node_ids: Iterable[str],
    name: str,
) -> ComparisonDimensionValue:
    values = [
        (_node_label(graph, node_id), node_id)
        for node_id in sorted(set(node_ids))
        if node_id in graph
    ]
    return dimension_from_values(
        name,
        values,
        normalizer=_canonical_entity,
    )


def _producer_subjects(
    graph: nx.Graph,
    producers: Iterable[str],
) -> list[str]:
    subjects: set[str] = set()
    for producer in producers:
        for relation in ("TESTED_IN", "CHARACTERIZED_IN", "SIMULATED_BY"):
            for node_id in _incoming(graph, producer, relation):
                if (
                    node_id in graph
                    and str(graph.nodes[node_id].get("type", ""))
                    in _SCIENTIFIC_SUBJECT_TYPES
                ):
                    subjects.add(node_id)
    return sorted(subjects)


def _observable(graph: nx.Graph, measurement_id: str) -> tuple[str, str]:
    attrs = graph.nodes[measurement_id]
    label = _node_label(graph, measurement_id)
    raw = attrs.get("metric_id") or attrs.get("metric") or label
    key = normalize_comparison_value(raw)
    if not key:
        raise ValueError(
            f"Measurement {measurement_id!r} has no observable identity."
        )
    return key, label


def _numeric_value(
    attrs: dict[str, Any],
    measurement_id: str,
) -> tuple[float | None, str]:
    numeric_raw = str(attrs.get("value_numeric", "")).strip()
    text_raw = str(attrs.get("value_text", "")).strip()
    if numeric_raw and text_raw:
        raise ValueError(
            f"Measurement {measurement_id!r} violates numeric/text XOR."
        )
    if not numeric_raw:
        return None, text_raw
    try:
        return float(numeric_raw), ""
    except ValueError as exc:
        raise ValueError(
            f"Measurement {measurement_id!r} has non-numeric "
            f"value_numeric {numeric_raw!r}."
        ) from exc


def _explicit_scalar_attribute_values(
    graph: nx.Graph,
    node_ids: Iterable[str],
    *,
    generic_keys: tuple[str, ...],
    canonical_unit_keys: dict[str, str],
) -> list[tuple[str, str]]:
    rows = _attr_values(graph, node_ids, generic_keys)
    for node_id in sorted(set(map(str, node_ids))):
        if node_id not in graph:
            continue
        attrs = graph.nodes[node_id]
        for key, unit in canonical_unit_keys.items():
            value = str(attrs.get(key, "")).strip()
            if value:
                rows.append((f"{value} {unit}", node_id))
    return rows


# alpha4b.3b.3.1: conservative method-context harvesting.
#
# Scientific meaning remains domain-owned. Structured Experiment conditions
# and explicit local method language are harvested only when they are clearly
# measurement/protocol metadata. Synthesis precursor conditions and simulation
# media are intentionally excluded from SERS measurement medium.
_CONDITION_WAVELENGTH_NAMES = frozenset({
    "excitation wavelength",
    "laser wavelength",
})
_CONDITION_POWER_NAMES = frozenset({
    "laser power",
})
_CONDITION_TIME_NAMES = frozenset({
    "acquisition time",
    "integration time",
    "exposure time",
    "signal acquisition time",
    "exposure time per pixel",
})
_CONDITION_GENERIC_MEDIUM_NAMES = frozenset({
    "medium",
    "sample medium",
    "solvent",
})
_CONDITION_PREPARATION_MEDIUM_NAMES = frozenset({
    "preparation medium",
    "preparation solvent",
})
_CONDITION_MEASUREMENT_ENVIRONMENT_NAMES = frozenset({
    "measurement environment",
    "measurement medium",
})
_CONDITION_PREPARATION_NAMES = frozenset({
    "sample preparation",
})
_CONDITION_STATE_NAMES = frozenset({
    "substrate state",
    "sample state",
    "state description",
    "substrate condition",
})

_PREPARATION_EVENT_ORDER = (
    "incubation",
    "adsorption",
    "drop_cast",
    "deposition",
    "immobilization",
    "mixing",
    "drying",
)
_PREPARATION_MEDIUM_ORDER = (
    "aqueous",
    "ethanol",
    "methanol",
    "pbs",
)
_MEASUREMENT_ENVIRONMENT_ORDER = (
    "solution",
    "aqueous",
    "ethanol",
    "methanol",
    "pbs",
    "cellular",
    "air",
)
_SAMPLE_STATE_ORDER = (
    "dry",
    "solid",
)
_SUBSTRATE_CONDITION_ORDER = (
    "as_prepared",
    "stored",
    "aged",
    "oxidized",
)


def _producer_is_measurement_experiment(
    graph: nx.Graph,
    node_id: str,
) -> bool:
    if node_id not in graph:
        return False
    attrs = graph.nodes[node_id]
    if str(attrs.get("type", "")) != "Experiment":
        return False
    experiment_type = normalize_comparison_value(
        attrs.get("experiment_type", "")
    )
    method_label = normalize_comparison_value(
        attrs.get("method_label", "")
    )
    if any(
        marker in experiment_type or marker in method_label
        for marker in ("calculation", "simulation", "approximation")
    ):
        return False
    return True


def _structured_conditions(
    graph: nx.Graph,
    node_id: str,
) -> list[dict[str, Any]]:
    if node_id not in graph:
        return []
    raw = str(graph.nodes[node_id].get("conditions_json", "")).strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed conditions_json on node {node_id!r}."
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError(
            f"conditions_json on node {node_id!r} must be a list."
        )
    rows: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError(
                f"conditions_json item on node {node_id!r} must be an object."
            )
        rows.append(dict(item))
    return rows


def _condition_raw_value(condition: dict[str, Any]) -> str:
    value_text = str(condition.get("value_text") or "").strip()
    if value_text:
        return value_text
    numeric = condition.get("value_numeric")
    if numeric is None or str(numeric).strip() == "":
        return ""
    unit = str(condition.get("unit") or "").strip()
    if unit:
        return f"{numeric} {unit}"
    return str(numeric).strip()


def _condition_values(
    graph: nx.Graph,
    node_ids: Iterable[str],
    *,
    accepted_names: frozenset[str],
    measurement_experiments_only: bool = False,
) -> list[tuple[str, str, str]]:
    values: list[tuple[str, str, str]] = []
    for node_id in sorted(set(map(str, node_ids))):
        if node_id not in graph:
            continue
        if (
            measurement_experiments_only
            and not _producer_is_measurement_experiment(graph, node_id)
        ):
            continue
        for condition in _structured_conditions(graph, node_id):
            name = normalize_comparison_value(condition.get("name", ""))
            if name not in accepted_names:
                continue
            raw_value = _condition_raw_value(condition)
            if not raw_value:
                continue
            values.append(
                (raw_value, node_id, "experiment_conditions_json")
            )
    return values


def _local_method_text_rows(
    graph: nx.Graph,
    *,
    measurement_id: str,
    producers: Iterable[str],
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    measurement_experiments = {
        producer_id
        for producer_id in set(map(str, producers))
        if _producer_is_measurement_experiment(graph, producer_id)
    }

    # Free-text method harvesting is only enabled when the Measurement is
    # grounded in a physical Experiment. This prevents phrases such as
    # "water-filled nanogap" in simulations from becoming sample medium.
    source_expression = str(
        graph.nodes[measurement_id].get("source_expression", "")
    ).strip()
    if source_expression and measurement_experiments:
        rows.append(
            (
                source_expression,
                measurement_id,
                "measurement_source_expression",
            )
        )

    for producer_id in sorted(measurement_experiments):
        attrs = graph.nodes[producer_id]
        for key in ("raw_method_name", "description"):
            value = str(attrs.get(key, "")).strip()
            if value:
                rows.append(
                    (value, producer_id, "experiment_method_text")
                )

        for condition in _structured_conditions(graph, producer_id):
            name = normalize_comparison_value(condition.get("name", ""))
            if (
                name not in _CONDITION_PREPARATION_NAMES
                and name not in _CONDITION_STATE_NAMES
                and name not in _CONDITION_GENERIC_MEDIUM_NAMES
                and name not in _CONDITION_PREPARATION_MEDIUM_NAMES
                and name not in _CONDITION_MEASUREMENT_ENVIRONMENT_NAMES
            ):
                continue
            raw_value = _condition_raw_value(condition)
            if raw_value:
                rows.append(
                    (
                        raw_value,
                        producer_id,
                        "experiment_conditions_json",
                    )
                )

    # A measurement-local explicit condition is admissible only when the
    # Measurement is grounded in a physical Experiment.
    if measurement_experiments:
        for condition in _structured_conditions(graph, measurement_id):
            name = normalize_comparison_value(condition.get("name", ""))
            if (
                name not in _CONDITION_PREPARATION_NAMES
                and name not in _CONDITION_STATE_NAMES
                and name not in _CONDITION_GENERIC_MEDIUM_NAMES
                and name not in _CONDITION_PREPARATION_MEDIUM_NAMES
                and name not in _CONDITION_MEASUREMENT_ENVIRONMENT_NAMES
            ):
                continue
            raw_value = _condition_raw_value(condition)
            if raw_value:
                rows.append(
                    (
                        raw_value,
                        measurement_id,
                        "measurement_conditions_json",
                    )
                )
    return rows


def _preparation_tags(text: str) -> tuple[str, ...]:
    normalized = normalize_comparison_value(_ascii_scientific_text(text))
    tags: set[str] = set()
    if re.search(r"\bincubat(?:e|ed|ing|ion)\b", normalized):
        tags.add("incubation")
    if re.search(r"\badsorb(?:ed|ing|tion)?\b", normalized):
        tags.add("adsorption")
    if (
        re.search(r"\bdrop[\s-]?cast(?:ed|ing)?\b", normalized)
        or re.search(r"\bdrop(?:ped)?\s+(?:on|onto)\b", normalized)
    ):
        tags.add("drop_cast")
    if re.search(
        r"\b(?:deposited|deposition|dispersed)\s+(?:on|onto)\b",
        normalized,
    ):
        tags.add("deposition")
    if re.search(r"\bimmobiliz(?:e|ed|ing|ation)\b", normalized):
        tags.add("immobilization")
    if re.search(r"\bmix(?:ed|ing)?\s+with\b", normalized):
        tags.add("mixing")
    if re.search(
        r"\b(?:air[\s-]?dried|dried|drying|allowed\s+to\s+dry)\b",
        normalized,
    ):
        tags.add("drying")
    return tuple(tag for tag in _PREPARATION_EVENT_ORDER if tag in tags)


def _sample_state_tags(text: str) -> tuple[str, ...]:
    normalized = normalize_comparison_value(_ascii_scientific_text(text))
    tags: set[str] = set()
    if re.search(
        r"\b(?:dry|dried|drying|air[\s-]?dried|allowed\s+to\s+dry)\b",
        normalized,
    ):
        tags.add("dry")
    if re.search(r"\bsolid\b", normalized):
        tags.add("solid")
    return tuple(tag for tag in _SAMPLE_STATE_ORDER if tag in tags)


def _substrate_condition_tags(text: str) -> tuple[str, ...]:
    normalized = normalize_comparison_value(_ascii_scientific_text(text))
    tags: set[str] = set()
    if re.search(
        r"\bas[\s-]?(?:prepared|synthesized|obtained)\b",
        normalized,
    ):
        tags.add("as_prepared")
    if re.search(r"\b(?:stored|storage)\b", normalized):
        tags.add("stored")
    if re.search(r"\b(?:aged|aging)\b", normalized):
        tags.add("aged")
    if re.search(r"\b(?:oxidized|oxidation)\b", normalized):
        tags.add("oxidized")
    return tuple(
        tag for tag in _SUBSTRATE_CONDITION_ORDER if tag in tags
    )


def _material_medium_tags(text: str) -> tuple[str, ...]:
    normalized = normalize_comparison_value(_ascii_scientific_text(text))
    tags: set[str] = set()
    if (
        "aqueous" in normalized
        or "pure water" in normalized
        or "deionized water" in normalized
        or re.search(r"\bin\s+water\b", normalized)
    ):
        tags.add("aqueous")
    if "ethanol" in normalized or re.search(r"\betoh\b", normalized):
        tags.add("ethanol")
    if "methanol" in normalized:
        tags.add("methanol")
    if re.search(r"\bpbs\b", normalized):
        tags.add("pbs")
    return tuple(
        tag for tag in _PREPARATION_MEDIUM_ORDER if tag in tags
    )


def _measurement_environment_tags(text: str) -> tuple[str, ...]:
    normalized = normalize_comparison_value(_ascii_scientific_text(text))
    tags: set[str] = set()

    if normalized in set(_MEASUREMENT_ENVIRONMENT_ORDER):
        tags.add(normalized)

    if (
        "solution-based" in normalized
        or "solution-state" in normalized
        or re.search(
            r"\b(?:measured|acquired|collected|recorded|spectra|"
            r"spectroscopy|analysis)\b.{0,60}\bin\s+solution\b",
            normalized,
        )
    ):
        tags.add("solution")

    material_media = set(_material_medium_tags(text))
    if material_media and (
        re.search(
            r"\b(?:measured|acquired|collected|recorded|spectra|"
            r"spectroscopy|analysis)\b",
            normalized,
        )
        and not _preparation_tags(text)
    ):
        tags.update(material_media)

    if (
        re.search(r"\b(?:cells?|cellular|cell culture)\b", normalized)
        and re.search(
            r"\b(?:sers|raman|spectra|mapping|maps?|signal|measured|"
            r"acquired|collected|observed)\b",
            normalized,
        )
    ):
        tags.add("cellular")

    if re.search(
        r"\b(?:measured|acquired|recorded|collected)\s+"
        r"(?:in|under)\s+(?:ambient\s+)?air\b",
        normalized,
    ):
        tags.add("air")

    return tuple(
        tag
        for tag in _MEASUREMENT_ENVIRONMENT_ORDER
        if tag in tags
    )



def _experiment_environment_text_is_measurement_scoped(text: str) -> bool:
    """Accept producer free text only when the environment is syntactically
    tied to the measurement, rather than merely mentioned in a multi-task
    umbrella description.
    """
    normalized = normalize_comparison_value(_ascii_scientific_text(text))
    tags = set(_measurement_environment_tags(text))
    if not tags:
        return False

    # Solution/aqueous/air tags are already emitted only from direct
    # measurement-language patterns in _measurement_environment_tags.
    if tags - {"cellular"}:
        return True

    if "cellular" not in tags:
        return False

    # Strong local evidence: spectra/maps/signals are explicitly obtained
    # in/from/of cells, or the measurement is explicitly performed in a cell.
    if re.search(
        r"\b(?:sers|raman|spectra?|mapping|maps?|signals?)\b"
        r".{0,90}\b(?:in|from|of)\b.{0,70}\bcells?\b",
        normalized,
    ):
        return True
    if re.search(
        r"\b(?:measured|acquired|collected|recorded|observed)\b"
        r".{0,90}\b(?:in|from)\s+(?:an?\s+)?cell\b",
        normalized,
    ):
        return True

    # A concise method label such as "SERS-based target-specific cell
    # imaging" is a direct environment description. Multi-item umbrella
    # prose ("A, B, C, and cell imaging performance") is deliberately not.
    if (
        "cell imaging" in normalized
        and len(normalized) <= 80
        and "," not in normalized
        and " and " not in normalized
    ):
        return True

    return False


def _controlled_event_dimension(
    *,
    name: str,
    rows: Iterable[tuple[str, str, str]],
    tagger,
    allow_composite: bool = True,
) -> MethodDimensionValue:
    tags: set[str] = set()
    source_values: set[str] = set()
    source_node_ids: set[str] = set()
    provenance_scopes: set[str] = set()

    for raw_value, source_node_id, provenance_scope in rows:
        found = tagger(raw_value)
        if not found:
            continue
        tags.update(found)
        source_values.add(str(raw_value).strip())
        if str(source_node_id).strip():
            source_node_ids.add(str(source_node_id))
        if str(provenance_scope).strip():
            provenance_scopes.add(str(provenance_scope))

    if not tags:
        return MethodDimensionValue(name=name, status="unknown")

    if name == "sample_preparation":
        order = _PREPARATION_EVENT_ORDER
    elif name == "preparation_medium":
        order = _PREPARATION_MEDIUM_ORDER
    elif name == "measurement_environment":
        order = _MEASUREMENT_ENVIRONMENT_ORDER
    elif name == "sample_state":
        order = _SAMPLE_STATE_ORDER
    elif name == "substrate_condition":
        order = _SUBSTRATE_CONDITION_ORDER
    else:
        raise ValueError(
            f"Unsupported controlled event dimension: {name!r}"
        )

    normalized_tags = tuple(tag for tag in order if tag in tags)
    common = {
        "source_values": tuple(sorted(source_values)),
        "source_node_ids": tuple(sorted(source_node_ids)),
        "provenance_scopes": tuple(sorted(provenance_scopes)),
    }
    if len(normalized_tags) > 1 and not allow_composite:
        return MethodDimensionValue(
            name=name,
            status="ambiguous",
            **common,
        )

    return MethodDimensionValue(
        name=name,
        status="known",
        normalized_value="+".join(normalized_tags),
        **common,
    )

def _condition_scalar_dimension(
    *,
    name: str,
    graph: nx.Graph,
    producers: Iterable[str],
    measurement_id: str,
    condition_names: frozenset[str],
    normalizer,
    legacy_values: Iterable[tuple[str, str]],
    legacy_scope: str,
    measurement_experiments_only: bool = False,
):
    values: list[tuple[str, str, str]] = [
        (raw, node_id, legacy_scope)
        for raw, node_id in legacy_values
    ]
    values.extend(
        _condition_values(
            graph,
            producers,
            accepted_names=condition_names,
            measurement_experiments_only=measurement_experiments_only,
        )
    )
    values.extend(
        _condition_values(
            graph,
            {measurement_id},
            accepted_names=condition_names,
            measurement_experiments_only=False,
        )
    )
    return method_dimension_from_values(
        name,
        values,
        normalizer=normalizer,
    )



def _method_dimension_from_entities(
    graph: nx.Graph,
    node_ids: Iterable[str],
    name: str,
    provenance_scope: str,
):
    values = [
        (
            _node_label(graph, node_id),
            node_id,
            provenance_scope,
        )
        for node_id in sorted(set(node_ids))
        if node_id in graph
    ]
    return method_dimension_from_values(
        name,
        values,
        normalizer=_canonical_entity,
    )


def _probe_aliases(
    graph: nx.Graph,
    analytes: Iterable[str],
    reporters: Iterable[str],
) -> tuple[str, ...]:
    aliases: set[str] = set()
    for node_id in sorted(set(analytes) | set(reporters)):
        if node_id not in graph:
            continue
        raw = normalize_comparison_value(_node_label(graph, node_id))
        canonical = _canonical_entity(raw)
        if raw:
            aliases.add(raw)
        if canonical:
            aliases.add(canonical)
        match = re.search(r"\(([a-z0-9+\-]+)\)", raw)
        if match:
            aliases.add(match.group(1))
    # Narrow domain-owned reverse aliases only.
    if "methylene blue" in aliases:
        aliases.add("mb")
    return tuple(
        sorted(
            (alias for alias in aliases if alias),
            key=lambda value: (-len(value), value),
        )
    )


def _probe_scoped_concentrations(
    text: str,
    *,
    probe_aliases: tuple[str, ...],
) -> list[str]:
    normalized = normalize_comparison_value(
        _ascii_scientific_text(text)
    )
    if not probe_aliases:
        return []

    found: list[str] = []
    ascii_text = _ascii_scientific_text(text)
    for match in _CONCENTRATION_RE.finditer(ascii_text):
        start = max(0, match.start() - 48)
        end = min(len(ascii_text), match.end() + 48)
        window = normalize_comparison_value(ascii_text[start:end])
        if any(alias in window for alias in probe_aliases):
            found.append(match.group(0))
    return found


def _measurement_local_concentration_values(
    graph: nx.Graph,
    *,
    measurement_id: str,
    producers: Iterable[str],
    groups: Iterable[str],
    analytes: Iterable[str],
    reporters: Iterable[str],
    observable_key: str,
) -> list[tuple[str, str, str]]:
    # Highest priority: explicit measurement-local attributes.
    explicit = _attr_values(
        graph,
        {measurement_id},
        (
            "analyte_concentration",
            "reporter_concentration",
            "measurement_concentration",
            "concentration",
        ),
    )
    if explicit:
        return [
            (value, node_id, "measurement_attribute")
            for value, node_id in explicit
        ]

    aliases = _probe_aliases(graph, analytes, reporters)

    # Output-valued concentration observables (especially LOD) must never
    # recycle their result value as experimental context concentration.
    if observable_key != "detection_limit":
        source_expression = str(
            graph.nodes[measurement_id].get("source_expression", "")
        ).strip()
        local_values = _probe_scoped_concentrations(
            source_expression,
            probe_aliases=aliases,
        )
        if local_values:
            return [
                (
                    value,
                    measurement_id,
                    "measurement_source_expression",
                )
                for value in local_values
            ]

    # Next priority: explicitly named analyte/reporter concentration
    # attributes on producer or group nodes. Generic precursor/synthesis
    # `concentration` attributes are intentionally excluded.
    explicit_context = _attr_values(
        graph,
        set(producers) | set(groups),
        (
            "analyte_concentration",
            "reporter_concentration",
        ),
    )
    if explicit_context:
        return [
            (
                value,
                node_id,
                (
                    "measurement_group_attribute"
                    if node_id in set(groups)
                    else "experiment_attribute"
                ),
            )
            for value, node_id in explicit_context
        ]

    # Last conservative fallback: probe-scoped group text. If a range or
    # several concentrations is stated, the dimension becomes ambiguous.
    group_values: list[tuple[str, str, str]] = []
    for group_id in sorted(set(groups)):
        if group_id not in graph:
            continue
        for text, _node_id in _texts(graph, {group_id}):
            for value in _probe_scoped_concentrations(
                text,
                probe_aliases=aliases,
            ):
                group_values.append(
                    (
                        value,
                        group_id,
                        "measurement_group_text",
                    )
                )
    return group_values


def _method_scalar_dimension(
    *,
    name: str,
    values: Iterable[tuple[str, str]],
    provenance_scope: str,
    normalizer,
):
    return method_dimension_from_values(
        name,
        (
            (value, node_id, provenance_scope)
            for value, node_id in values
        ),
        normalizer=normalizer,
    )


def _extract_one_sers_method_context(
    graph: nx.Graph,
    *,
    paper_id: str,
    measurement_id: str,
) -> MethodContext:
    attrs = graph.nodes[measurement_id]
    observable_key, _observable_label = _observable(
        graph,
        measurement_id,
    )

    producers = sorted(set(
        _incoming(graph, measurement_id, "HAS_MEASUREMENT")
    ))
    groups = _outgoing(
        graph,
        measurement_id,
        "IN_MEASUREMENT_GROUP",
    )
    subjects = {
        node_id
        for node_id in _outgoing(
            graph,
            measurement_id,
            "MEASURED_FOR",
        )
        if node_id in graph
        and str(graph.nodes[node_id].get("type", ""))
        in _SCIENTIFIC_SUBJECT_TYPES
    }
    subjects.update(_producer_subjects(graph, producers))
    subject_ids = tuple(sorted(subjects))

    analytes: set[str] = set()
    reporters: set[str] = set()
    optical_conditions: set[str] = set()
    for producer in producers:
        analytes.update(_outgoing(graph, producer, "USES_ANALYTE"))
        reporters.update(_outgoing(graph, producer, "USES_REPORTER"))
        optical_conditions.update(
            _outgoing(graph, producer, "USES_OPTICAL_CONDITION")
        )
    optical_conditions.update(
        _outgoing(graph, measurement_id, "USES_OPTICAL_CONDITION")
    )

    concentration_values = _measurement_local_concentration_values(
        graph,
        measurement_id=measurement_id,
        producers=producers,
        groups=groups,
        analytes=analytes,
        reporters=reporters,
        observable_key=observable_key,
    )

    optical_nodes = (
        set(optical_conditions)
        | set(producers)
        | {measurement_id}
    )
    optical_text = _texts(graph, optical_nodes)

    wavelength_values = _explicit_scalar_attribute_values(
        graph,
        optical_nodes,
        generic_keys=(
            "excitation_wavelength",
            "wavelength",
            "laser_wavelength",
        ),
        canonical_unit_keys={
            "excitation_wavelength_nm": "nm",
            "wavelength_nm": "nm",
        },
    )
    wavelength_values.extend(
        _regex_values(
            optical_text,
            _WAVELENGTH_RE,
            required_markers=(
                "excitation",
                "excited",
                "laser",
                "wavelength",
                "optical",
            ),
        )
    )
    wavelength_values.extend(
        _regex_values(
            _texts(graph, optical_conditions),
            _WAVELENGTH_RE,
        )
    )
    wavelength_dimension = _condition_scalar_dimension(
        name="excitation_wavelength",
        graph=graph,
        producers=producers,
        measurement_id=measurement_id,
        condition_names=_CONDITION_WAVELENGTH_NAMES,
        normalizer=_canonical_wavelength,
        legacy_values=wavelength_values,
        legacy_scope="optical_context",
    )

    power_values = _explicit_scalar_attribute_values(
        graph,
        optical_nodes,
        generic_keys=("laser_power", "power", "power_density"),
        canonical_unit_keys={"laser_power_mw": "mW"},
    )
    power_values.extend(
        _regex_values(
            optical_text,
            _POWER_RE,
            required_markers=("power", "laser"),
        )
    )
    power_dimension = _condition_scalar_dimension(
        name="laser_power",
        graph=graph,
        producers=producers,
        measurement_id=measurement_id,
        condition_names=_CONDITION_POWER_NAMES,
        normalizer=_canonical_power,
        legacy_values=power_values,
        legacy_scope="optical_context",
    )

    integration_values = _explicit_scalar_attribute_values(
        graph,
        optical_nodes,
        generic_keys=(
            "integration_time",
            "acquisition_time",
            "exposure_time",
        ),
        canonical_unit_keys={},
    )
    integration_values.extend(
        _regex_values(
            optical_text,
            _TIME_RE,
            required_markers=(
                "integration",
                "acquisition",
                "exposure",
            ),
        )
    )
    integration_dimension = _condition_scalar_dimension(
        name="integration_time",
        graph=graph,
        producers=producers,
        measurement_id=measurement_id,
        condition_names=_CONDITION_TIME_NAMES,
        normalizer=_canonical_time,
        legacy_values=integration_values,
        legacy_scope="measurement_protocol",
    )

    physical_producers = {
        producer_id
        for producer_id in producers
        if _producer_is_measurement_experiment(graph, producer_id)
    }
    local_method_rows = _local_method_text_rows(
        graph,
        measurement_id=measurement_id,
        producers=producers,
    )

    preparation_rows: list[tuple[str, str, str]] = []
    if physical_producers:
        preparation_rows.extend(
            (
                raw,
                node_id,
                "measurement_protocol_attribute",
            )
            for raw, node_id in _attr_values(
                graph,
                physical_producers | {measurement_id},
                (
                    "sample_preparation",
                    "preparation_mode",
                    "deposition_method",
                ),
            )
        )
        preparation_rows.extend(local_method_rows)
    sample_preparation_dimension = _controlled_event_dimension(
        name="sample_preparation",
        rows=preparation_rows,
        tagger=_preparation_tags,
        allow_composite=True,
    )
    has_explicit_preparation = (
        sample_preparation_dimension.status == "known"
    )

    preparation_medium_rows: list[tuple[str, str, str]] = []
    measurement_environment_rows: list[
        tuple[str, str, str]
    ] = []
    generic_medium_rows: list[tuple[str, str, str]] = []

    if physical_producers:
        preparation_medium_rows.extend(
            (
                raw,
                node_id,
                "measurement_protocol_attribute",
            )
            for raw, node_id in _attr_values(
                graph,
                physical_producers | {measurement_id},
                (
                    "preparation_medium",
                    "preparation_solvent",
                ),
            )
        )
        measurement_environment_rows.extend(
            (
                raw,
                node_id,
                "measurement_protocol_attribute",
            )
            for raw, node_id in _attr_values(
                graph,
                physical_producers | {measurement_id},
                (
                    "measurement_environment",
                    "measurement_medium",
                ),
            )
        )
        generic_medium_rows.extend(
            (
                raw,
                node_id,
                "measurement_protocol_attribute",
            )
            for raw, node_id in _attr_values(
                graph,
                physical_producers | {measurement_id},
                (
                    "medium",
                    "sample_medium",
                    "solvent",
                ),
            )
        )

        preparation_medium_rows.extend(
            _condition_values(
                graph,
                physical_producers,
                accepted_names=_CONDITION_PREPARATION_MEDIUM_NAMES,
                measurement_experiments_only=True,
            )
        )
        preparation_medium_rows.extend(
            _condition_values(
                graph,
                {measurement_id},
                accepted_names=_CONDITION_PREPARATION_MEDIUM_NAMES,
            )
        )
        measurement_environment_rows.extend(
            _condition_values(
                graph,
                physical_producers,
                accepted_names=_CONDITION_MEASUREMENT_ENVIRONMENT_NAMES,
                measurement_experiments_only=True,
            )
        )
        measurement_environment_rows.extend(
            _condition_values(
                graph,
                {measurement_id},
                accepted_names=_CONDITION_MEASUREMENT_ENVIRONMENT_NAMES,
            )
        )
        generic_medium_rows.extend(
            _condition_values(
                graph,
                physical_producers,
                accepted_names=_CONDITION_GENERIC_MEDIUM_NAMES,
                measurement_experiments_only=True,
            )
        )
        generic_medium_rows.extend(
            _condition_values(
                graph,
                {measurement_id},
                accepted_names=_CONDITION_GENERIC_MEDIUM_NAMES,
            )
        )

        for row in local_method_rows:
            raw, _source_node_id, provenance_scope = row
            if _material_medium_tags(raw) and _preparation_tags(raw):
                preparation_medium_rows.append(row)
            if _measurement_environment_tags(raw):
                if (
                    provenance_scope != "experiment_method_text"
                    or _experiment_environment_text_is_measurement_scoped(raw)
                ):
                    measurement_environment_rows.append(row)

    # A generic medium/solvent field is role-scoped only when the local
    # protocol makes the role explicit enough to avoid mixing preparation
    # solvent with measurement environment.
    if has_explicit_preparation:
        preparation_medium_rows.extend(generic_medium_rows)
    else:
        measurement_environment_rows.extend(generic_medium_rows)

    preparation_medium_dimension = _controlled_event_dimension(
        name="preparation_medium",
        rows=preparation_medium_rows,
        tagger=_material_medium_tags,
        allow_composite=False,
    )
    measurement_environment_dimension = _controlled_event_dimension(
        name="measurement_environment",
        rows=measurement_environment_rows,
        tagger=lambda text: tuple(dict.fromkeys(
            _measurement_environment_tags(text)
            + _material_medium_tags(text)
        )),
        allow_composite=False,
    )

    # Role precision: physical sample state and substrate lifecycle/condition
    # are distinct. A phrase such as "MB solid" must not become a substrate
    # condition, while "as-synthesized nanoparticles" must not become sample
    # physical state.
    state_rows: list[tuple[str, str, str]] = []
    if physical_producers:
        state_rows.extend(
            (
                raw,
                node_id,
                "measurement_protocol_attribute",
            )
            for raw, node_id in _attr_values(
                graph,
                physical_producers | {measurement_id},
                (
                    "substrate_state",
                    "sample_state",
                    "state_description",
                    "substrate_condition",
                ),
            )
        )
        state_rows.extend(local_method_rows)

    sample_state_dimension = _controlled_event_dimension(
        name="sample_state",
        rows=state_rows,
        tagger=_sample_state_tags,
        allow_composite=False,
    )
    substrate_condition_dimension = _controlled_event_dimension(
        name="substrate_condition",
        rows=state_rows,
        tagger=_substrate_condition_tags,
        allow_composite=False,
    )

    dimensions = (
        _method_dimension_from_entities(
            graph,
            analytes,
            "analyte",
            "analyte_relation",
        ),
        _method_dimension_from_entities(
            graph,
            reporters,
            "reporter",
            "reporter_relation",
        ),
        method_dimension_from_values(
            "analyte_concentration",
            concentration_values,
            normalizer=_canonical_concentration,
        ),
        wavelength_dimension,
        power_dimension,
        integration_dimension,
        sample_preparation_dimension,
        preparation_medium_dimension,
        measurement_environment_dimension,
        sample_state_dimension,
        substrate_condition_dimension,
    )

    dimension_sources = {
        source_node_id
        for dimension in dimensions
        for source_node_id in dimension.source_node_ids
    }
    source_node_ids = tuple(sorted(
        set(producers)
        | set(groups)
        | set(analytes)
        | set(reporters)
        | set(optical_conditions)
        | set(subject_ids)
        | {measurement_id}
        | dimension_sources
    ))

    return MethodContext(
        method_context_id=(
            "method_context:"
            + stable_comparison_id(
                SERS_METHOD_SEMANTICS.semantics_id,
                paper_id,
                measurement_id,
            )
        ),
        domain_profile_id="sers_au_ag",
        method_semantics_id=SERS_METHOD_SEMANTICS.semantics_id,
        paper_id=paper_id,
        measurement_id=measurement_id,
        producer_ids=tuple(producers),
        subject_ids=subject_ids,
        dimensions=dimensions,
        source_node_ids=source_node_ids,
    )


def extract_sers_method_contexts(
    graph: nx.Graph,
    paper_id: str,
) -> list[MethodContext]:
    graph_domain = str(graph.graph.get("domain_profile_id", "")).strip()
    if graph_domain and graph_domain != "sers_au_ag":
        raise ValueError(
            "SERS method provider received a different domain graph: "
            f"{graph_domain!r}"
        )

    contexts: list[MethodContext] = []
    for raw_measurement_id, attrs in sorted(
        graph.nodes(data=True),
        key=lambda item: str(item[0]),
    ):
        if str(attrs.get("type", "")) != "Measurement":
            continue
        contexts.append(
            _extract_one_sers_method_context(
                graph,
                paper_id=paper_id,
                measurement_id=str(raw_measurement_id),
            )
        )
    return contexts


def _comparison_dimension_from_method(
    method_context: MethodContext,
    *,
    method_name: str,
    comparison_name: str,
) -> ComparisonDimensionValue:
    dimension = method_context.dimension_map[method_name]
    return ComparisonDimensionValue(
        name=comparison_name,
        status=dimension.status,
        normalized_value=dimension.normalized_value,
        source_values=dimension.source_values,
        source_node_ids=dimension.source_node_ids,
    )


def extract_sers_comparison_contexts(
    graph: nx.Graph,
    paper_id: str,
) -> list[ComparisonContext]:
    graph_domain = str(graph.graph.get("domain_profile_id", "")).strip()
    if graph_domain and graph_domain != "sers_au_ag":
        raise ValueError(
            "SERS comparison provider received a different domain graph: "
            f"{graph_domain!r}"
        )

    method_by_measurement = {
        method.measurement_id: method
        for method in extract_sers_method_contexts(graph, paper_id)
    }
    contexts: list[ComparisonContext] = []

    for raw_measurement_id, attrs in sorted(
        graph.nodes(data=True),
        key=lambda item: str(item[0]),
    ):
        measurement_id = str(raw_measurement_id)
        if str(attrs.get("type", "")) != "Measurement":
            continue

        method_context = method_by_measurement[measurement_id]
        producers = list(method_context.producer_ids)
        groups = _outgoing(
            graph,
            measurement_id,
            "IN_MEASUREMENT_GROUP",
        )
        subject_ids = method_context.subject_ids

        measurement_text = _texts(
            graph,
            {measurement_id} | set(groups),
        )
        raman_peak_values = _explicit_scalar_attribute_values(
            graph,
            {measurement_id} | set(groups),
            generic_keys=("raman_peak", "raman_shift"),
            canonical_unit_keys={
                "raman_peak_cm1": "cm^-1",
                "raman_shift_cm1": "cm^-1",
                "peak_cm1": "cm^-1",
            },
        )
        raman_peak_values.extend(
            _regex_values(
                measurement_text,
                _RAMAN_PEAK_RE,
                required_markers=(
                    "raman",
                    "peak",
                    "band",
                    "shift",
                    "sers",
                ),
            )
        )

        metric_text = normalize_comparison_value(
            str(attrs.get("metric", ""))
            + " "
            + _node_label(graph, measurement_id)
        )
        unit_text = normalize_comparison_value(attrs.get("unit", ""))
        if (
            str(attrs.get("value_numeric", "")).strip()
            and ("cm" in unit_text and "-1" in unit_text)
            and any(
                marker in metric_text
                for marker in ("raman", "peak", "band", "shift", "sers")
            )
        ):
            raman_peak_values.append((
                f"{attrs.get('value_numeric')} {attrs.get('unit')}",
                measurement_id,
            ))

        dimensions = (
            _comparison_dimension_from_method(
                method_context,
                method_name="analyte",
                comparison_name="analyte",
            ),
            _comparison_dimension_from_method(
                method_context,
                method_name="reporter",
                comparison_name="reporter",
            ),
            _comparison_dimension_from_method(
                method_context,
                method_name="analyte_concentration",
                comparison_name="concentration",
            ),
            _comparison_dimension_from_method(
                method_context,
                method_name="excitation_wavelength",
                comparison_name="excitation_wavelength",
            ),
            _comparison_dimension_from_method(
                method_context,
                method_name="laser_power",
                comparison_name="laser_power",
            ),
            _comparison_dimension_from_method(
                method_context,
                method_name="integration_time",
                comparison_name="integration_time",
            ),
            dimension_from_values(
                "raman_peak",
                raman_peak_values,
                normalizer=_canonical_raman_peak,
            ),
            _comparison_dimension_from_method(
                method_context,
                method_name="measurement_environment",
                comparison_name="measurement_environment",
            ),
            _comparison_dimension_from_method(
                method_context,
                method_name="sample_state",
                comparison_name="sample_state",
            ),
            _comparison_dimension_from_method(
                method_context,
                method_name="substrate_condition",
                comparison_name="substrate_condition",
            ),
        )

        observable_key, observable_label = _observable(
            graph,
            measurement_id,
        )
        value_numeric, value_text = _numeric_value(
            dict(attrs),
            measurement_id,
        )

        dimension_sources = {
            source_node_id
            for dimension in dimensions
            for source_node_id in dimension.source_node_ids
        }
        source_node_ids = tuple(sorted(
            set(method_context.source_node_ids)
            | set(groups)
            | {measurement_id}
            | dimension_sources
        ))

        contexts.append(
            ComparisonContext(
                context_id=(
                    "comparison_context:"
                    + stable_comparison_id(
                        SERS_COMPARISON_SEMANTICS_ID,
                        paper_id,
                        measurement_id,
                    )
                ),
                domain_profile_id="sers_au_ag",
                comparison_semantics_id=SERS_COMPARISON_SEMANTICS_ID,
                paper_id=paper_id,
                measurement_id=measurement_id,
                observable_key=observable_key,
                observable_label=observable_label,
                value_numeric=value_numeric,
                value_text=value_text,
                unit=str(attrs.get("unit", "")).strip(),
                source_expression=str(
                    attrs.get("source_expression", "")
                ).strip(),
                subject_ids=subject_ids,
                dimensions=dimensions,
                source_node_ids=source_node_ids,
                method_context_id=method_context.method_context_id,
            )
        )

    return contexts


SERS_AU_AG_COMPARISON_ADAPTER = ComparisonDomainAdapter(
    adapter_id="sers_au_ag",
    domain_profile_id="sers_au_ag",
    semantics_id=SERS_COMPARISON_SEMANTICS_ID,
    dimensions=SERS_COMPARISON_DIMENSIONS,
    required_for_numeric_ranking=frozenset(
        SERS_COMPARISON_DIMENSIONS
    ),
    extract_contexts_fn=extract_sers_comparison_contexts,
    observable_policies=_SERS_OBSERVABLE_POLICIES,
    method_semantics=SERS_METHOD_SEMANTICS,
    extract_method_contexts_fn=extract_sers_method_contexts,
)
