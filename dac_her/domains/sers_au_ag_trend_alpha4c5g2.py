from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import replace
from typing import Any, Iterable, Mapping

from dac_her.domains import sers_au_ag_trend_alpha4c211 as v3
from dac_her.domains.sers_au_ag_trend_alpha4c2121 import (
    SERS_AU_AG_TREND_ADAPTER as _V5_ADAPTER,
)
from dac_her.trend_domain import (
    TrendDomainAdapter,
    TrendEvidence,
    TrendEvidenceSource,
)
from dac_her.trend_evidence import stable_trend_id


SERS_AU_AG_TREND_SEMANTICS_ID = (
    "sers_au_ag_trend_v6_alpha4c5g2"
)

_EXPLICIT_NANOGAP_DIMENSION_RE = re.compile(
    r"\b(?:interior\s+)?(?:nano\s*)?gap\s+"
    r"(?:size|width|distance)s?\b",
    re.I,
)

_COMPARATIVE_GAP_RE = re.compile(
    r"\b(?P<relation>greater|higher|stronger|larger|"
    r"lower|weaker|smaller)\b"
    r".{0,55}?\bfor\s+(?:the\s+)?"
    r"(?P<x1>\d+(?:\.\d+)?)\s*[- ]?\s*"
    r"(?P<u1>nm|µm|um)\b"
    r".{0,45}?\bgap\b"
    r".{0,45}?\bthan\s+(?:for\s+)?(?:the\s+)?"
    r"(?P<x2>\d+(?:\.\d+)?)\s*[- ]?\s*"
    r"(?P<u2>nm|µm|um)\b"
    r".{0,45}?\bgap\b",
    re.I,
)

_LOCAL_EXCITATION_NAMES = frozenset(
    {
        "excitation wavelength",
        "laser wavelength",
    }
)


def _norm(value: object) -> str:
    text = str(value or "").casefold()
    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("μ", "µ")
    )
    return re.sub(r"\s+", " ", text).strip()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _length_nm(
    value: Any,
    unit: Any,
) -> float | None:
    number = _finite(value)
    if number is None:
        return None
    normalized = (
        str(unit or "")
        .strip()
        .replace("μ", "µ")
        .casefold()
    )
    factors = {
        "nm": 1.0,
        "µm": 1000.0,
        "um": 1000.0,
    }
    factor = factors.get(normalized)
    return None if factor is None else number * factor


def _conditions(attrs: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = attrs.get("conditions_json", "")
    if isinstance(raw, list):
        value = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return ()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return ()
    if not isinstance(value, list):
        return ()
    return tuple(
        dict(row)
        for row in value
        if isinstance(row, Mapping)
    )


def _local_excitation_nm(
    graph,
    measurement_id: str,
) -> float | None:
    if measurement_id not in graph:
        return None
    values: set[float] = set()
    for row in _conditions(graph.nodes[measurement_id]):
        name = _norm(row.get("name", ""))
        if name not in _LOCAL_EXCITATION_NAMES:
            continue
        value = _length_nm(
            row.get("value_numeric"),
            row.get("unit"),
        )
        if value is not None:
            values.add(value)
    if len(values) != 1:
        return None
    return next(iter(values))


def _dimension_name(row: Mapping[str, Any]) -> str:
    return str(row.get("name", "")).strip()


def _resolve_excitation_dimension(
    *,
    method_row: Mapping[str, Any],
    local_nm: float,
    measurement_ids: Iterable[str],
) -> dict[str, Any]:
    updated = copy.deepcopy(dict(method_row))
    dimensions = []
    resolved = False
    for raw in updated.get("dimensions", []) or []:
        if not isinstance(raw, Mapping):
            dimensions.append(raw)
            continue
        dim = copy.deepcopy(dict(raw))
        if (
            _dimension_name(dim) == "excitation_wavelength"
            and str(dim.get("status", "")) == "ambiguous"
        ):
            dim["status"] = "known"
            dim["normalized_value"] = f"{local_nm:g} nm"
            dim["alpha4c5g2_resolution"] = {
                "semantics_id": SERS_AU_AG_TREND_SEMANTICS_ID,
                "rule": (
                    "measurement_local_explicit_excitation_precedes_"
                    "broader_optical_context_for_trend_compatibility"
                ),
                "local_value_nm": local_nm,
                "measurement_ids": sorted(
                    {str(value) for value in measurement_ids}
                ),
                "original_status": "ambiguous",
                "original_source_values": list(
                    dim.get("source_values", []) or []
                ),
            }
            resolved = True
        dimensions.append(dim)
    updated["dimensions"] = dimensions
    if resolved:
        updated["alpha4c5g2_locality_resolved"] = True
    return updated


def resolve_measurement_local_method_contexts(
    source: TrendEvidenceSource,
) -> tuple[TrendEvidenceSource, tuple[dict[str, Any], ...]]:
    """
    Resolve only excitation-wavelength ambiguity that is contradicted by
    explicit measurement-local conditions.

    A MethodContext is overridden only when every ComparisonContext using
    that method context has a Measurement with exactly one explicit local
    excitation wavelength and all such measurements agree on the same value.
    The original source rows are never mutated.
    """
    graph = source.graph
    methods = {
        str(row.get("method_context_id", "")): copy.deepcopy(dict(row))
        for row in source.method_context_rows
        if str(row.get("method_context_id", "")).strip()
    }
    usage: dict[str, set[str]] = {}
    for context in source.comparison_context_rows:
        method_id = str(
            context.get("method_context_id", "")
        ).strip()
        measurement_id = str(
            context.get("measurement_id", "")
        ).strip()
        if method_id and measurement_id:
            usage.setdefault(method_id, set()).add(measurement_id)

    audit_rows: list[dict[str, Any]] = []
    for method_id, measurement_ids in sorted(usage.items()):
        method = methods.get(method_id)
        if method is None:
            continue

        excitation_dims = [
            row
            for row in method.get("dimensions", []) or []
            if isinstance(row, Mapping)
            and _dimension_name(row) == "excitation_wavelength"
        ]
        if len(excitation_dims) != 1:
            continue
        dimension = excitation_dims[0]
        if str(dimension.get("status", "")) != "ambiguous":
            continue

        local_values: dict[str, float] = {}
        missing = []
        for measurement_id in sorted(measurement_ids):
            value = _local_excitation_nm(
                graph,
                measurement_id,
            )
            if value is None:
                missing.append(measurement_id)
            else:
                local_values[measurement_id] = value

        if missing:
            audit_rows.append(
                {
                    "method_context_id": method_id,
                    "resolved": False,
                    "reason": "not_all_measurements_have_single_local_excitation",
                    "measurement_ids": sorted(measurement_ids),
                    "missing_or_ambiguous_measurement_ids": missing,
                }
            )
            continue

        distinct = sorted(set(local_values.values()))
        if len(distinct) != 1:
            audit_rows.append(
                {
                    "method_context_id": method_id,
                    "resolved": False,
                    "reason": "local_excitation_values_disagree",
                    "measurement_values_nm": local_values,
                }
            )
            continue

        local_nm = distinct[0]
        methods[method_id] = _resolve_excitation_dimension(
            method_row=method,
            local_nm=local_nm,
            measurement_ids=measurement_ids,
        )
        audit_rows.append(
            {
                "method_context_id": method_id,
                "resolved": True,
                "reason": "measurement_local_explicit_excitation",
                "local_value_nm": local_nm,
                "measurement_ids": sorted(measurement_ids),
                "original_source_values": list(
                    dimension.get("source_values", []) or []
                ),
            }
        )

    resolved_source = TrendEvidenceSource(
        graph=source.graph,
        paper_id=source.paper_id,
        measurement_result_rows=source.measurement_result_rows,
        method_context_rows=tuple(
            methods.get(
                str(row.get("method_context_id", "")),
                copy.deepcopy(dict(row)),
            )
            for row in source.method_context_rows
        ),
        comparison_context_rows=source.comparison_context_rows,
    )
    return resolved_source, tuple(audit_rows)


def _resolved_claim_control(
    text: str,
) -> tuple[str, str] | None:
    # alpha4c211 introduced both nanogap_size and nanogap_presence.
    # An explicit dimensional noun ("gap size/width/distance") must select
    # the quantitative size control rather than fail because both families
    # match the same sentence.
    if _EXPLICIT_NANOGAP_DIMENSION_RE.search(_norm(text)):
        return "nanogap_size", "nanogap size"
    return v3._claim_control(text)


def _comparative_gap_direction(
    text: str,
) -> tuple[str, str] | None:
    normalized = _norm(text)
    match = _COMPARATIVE_GAP_RE.search(normalized)
    if match is None:
        return None

    x1 = _length_nm(
        match.group("x1"),
        match.group("u1"),
    )
    x2 = _length_nm(
        match.group("x2"),
        match.group("u2"),
    )
    if x1 is None or x2 is None or x1 == x2:
        return None

    relation = match.group("relation").casefold()
    first_is_higher = relation in {
        "greater",
        "higher",
        "stronger",
        "larger",
    }
    first_is_lower = relation in {
        "lower",
        "weaker",
        "smaller",
    }
    if first_is_higher == first_is_lower:
        return None

    # Canonical direction is relative to increasing gap size.
    if x1 < x2:
        direction = "negative" if first_is_higher else "positive"
    else:
        direction = "positive" if first_is_higher else "negative"
    return direction, "monotonic"


def _claim_direction_shape(
    text: str,
    control_key: str,
) -> tuple[str, str] | None:
    current = v3._direction_shape(text, control_key)
    if current is not None:
        return current
    if control_key == "nanogap_size":
        return _comparative_gap_direction(text)
    return None


def _supplemental_claims(
    source: TrendEvidenceSource,
    *,
    already_emitted_claim_ids: set[str],
) -> list[TrendEvidence]:
    evidence: list[TrendEvidence] = []
    graph = source.graph

    for claim_id, attrs in sorted(
        graph.nodes(data=True),
        key=lambda item: str(item[0]),
    ):
        claim_id = str(claim_id)
        if claim_id in already_emitted_claim_ids:
            continue
        if str(attrs.get("type", "")) not in v3._CLAIM_TYPES:
            continue

        text = v3.v1._claim_text(attrs)
        if not text:
            continue
        control = _resolved_claim_control(text)
        if control is None:
            continue

        # 5g.2 intentionally fixes only the demonstrated nanogap
        # quantitative-control regression. Other control families remain
        # under the frozen v5 semantics.
        if control[0] != "nanogap_size":
            continue

        response = v3._claim_response(
            text,
            control_key=control[0],
        )
        if response is None:
            continue
        direction_shape = _claim_direction_shape(
            text,
            control[0],
        )
        if direction_shape is None:
            continue
        direction, shape = direction_shape

        trend_id = stable_trend_id(
            paper_id=source.paper_id,
            independent_variable_key=control[0],
            dependent_observable_key=response[0],
            evidence_basis="reported_directional_claim",
            source_node_ids=(claim_id,),
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
                evidence_basis="reported_directional_claim",
                causal_status="not_asserted",
                varied_dimension=control[0],
                subject_ids=v3.v1._claim_subjects(
                    graph,
                    claim_id,
                ),
                source_expression=text,
                source_expressions=(text,),
                source_claim_ids=(claim_id,),
                source_node_ids=(claim_id,),
                requires_verification=v3.v1._requires_verification(
                    attrs
                ),
            )
        )
    return evidence


def extract_sers_au_ag_trend_evidence(
    source: TrendEvidenceSource,
) -> list[TrendEvidence]:
    resolved_source, _locality_audit = (
        resolve_measurement_local_method_contexts(source)
    )

    base = _V5_ADAPTER.extract_evidence(resolved_source)
    updated_base = [
        replace(
            item,
            trend_semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
        )
        for item in base
    ]
    emitted_claim_ids = {
        str(claim_id)
        for item in updated_base
        for claim_id in item.source_claim_ids
    }

    combined = [
        *updated_base,
        *_supplemental_claims(
            resolved_source,
            already_emitted_claim_ids=emitted_claim_ids,
        ),
    ]

    by_id: dict[str, TrendEvidence] = {}
    for item in combined:
        existing = by_id.get(item.trend_id)
        if existing is not None and existing != item:
            raise ValueError(
                "alpha4c5g2 produced conflicting TrendEvidence rows "
                f"for trend_id={item.trend_id!r}."
            )
        by_id[item.trend_id] = item

    return sorted(
        by_id.values(),
        key=lambda item: (
            item.paper_id,
            item.independent_variable_key,
            item.dependent_observable_key,
            item.evidence_basis,
            item.trend_id,
        ),
    )


SERS_AU_AG_TREND_ADAPTER = TrendDomainAdapter(
    adapter_id=_V5_ADAPTER.adapter_id,
    domain_profile_id=_V5_ADAPTER.domain_profile_id,
    semantics_id=SERS_AU_AG_TREND_SEMANTICS_ID,
    supported_evidence_bases=_V5_ADAPTER.supported_evidence_bases,
    required_inputs=_V5_ADAPTER.required_inputs,
    extract_evidence_fn=extract_sers_au_ag_trend_evidence,
)
