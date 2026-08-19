from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Any, Iterable, Mapping

from domains.sers import trend as v1
from domains.sers import trend_alpha4c211 as v3
from domains.sers.trend_alpha4c2121 import (
    SERS_AU_AG_TREND_ADAPTER as _V5_ADAPTER,
)
from domains.sers.trend_alpha4c5g2r1 import (
    _supplemental_nanogap_claims as _V6R1_SUPPLEMENTAL_CLAIMS,
)
from dac_her.trend_domain import (
    TrendDomainAdapter,
    TrendEvidence,
    TrendEvidenceSource,
    TrendSeriesPoint,
)
from dac_her.trend_evidence import stable_trend_id


SERS_AU_AG_TREND_SEMANTICS_ID = (
    "sers_au_ag_trend_v6r2_alpha4c5g2r2"
)

_EXCITATION_DIMENSION = "excitation_wavelength"


def _measurement_local_excitation_nm(
    graph,
    measurement_id: str,
) -> float | None:
    """
    Return exactly one explicit Measurement-local excitation wavelength.

    Only structured conditions attached directly to the Measurement node
    qualify. Broader producer, optical-context, sidecar, or paper-level
    values never satisfy this helper.
    """
    if measurement_id not in graph:
        return None

    values: set[float] = set()
    for condition in v1._structured_conditions(
        graph,
        measurement_id,
    ):
        parsed = v3._control_from_condition(
            condition,
            source_node_id=measurement_id,
            source_scope="measurement_conditions_json",
        )
        if (
            parsed is not None
            and parsed.key == _EXCITATION_DIMENSION
            and parsed.unit == "nm"
        ):
            values.add(float(parsed.value_numeric))

    if len(values) != 1:
        return None
    return next(iter(values))


def _normalized_wavelength_nm(
    value: object,
) -> float | None:
    text = str(value or "").strip().replace("μ", "µ")
    match = re.search(
        r"(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>nm|µm|um)\b",
        text,
        re.I,
    )
    if not match:
        return None
    return v3._length_nm(
        match.group("value"),
        match.group("unit"),
    )


def _candidate_local_methods_compatible(
    *,
    graph,
    rows: Iterable[Mapping[str, Any]],
    varied_control_key: str,
) -> tuple[bool, dict[str, Any]]:
    """
    Candidate-local compatibility override.

    Frozen v3 compatibility remains authoritative unless it fails. A failure
    may be overridden only when its remaining blocker is excitation-wavelength
    ambiguity and every row in THIS numeric Trend candidate has exactly one
    explicit Measurement-local excitation wavelength with the same normalized
    value.

    No MethodContext row is rewritten.
    """
    rows = list(rows)
    methods = [row["method"] for row in rows]

    if v3._methods_compatible(
        methods,
        varied_control_key=varied_control_key,
    ):
        return True, {
            "override_used": False,
            "reason": "frozen_method_compatibility_passed",
        }

    ignored = v3._CONTROL_TO_METHOD_DIMENSION.get(
        varied_control_key,
        "",
    )
    excitation_ambiguous = False

    for name in v3._METHOD_GUARD_DIMENSIONS:
        if name == ignored:
            continue

        known: set[str] = set()
        for row in rows:
            item = v3._method_dimension_map(
                row["method"]
            ).get(name)
            if item is None:
                continue

            status = str(
                item.get("status", "unknown")
            ).strip()

            if status == "ambiguous":
                if name != _EXCITATION_DIMENSION:
                    return False, {
                        "override_used": False,
                        "reason": (
                            "non_excitation_method_dimension_ambiguous"
                        ),
                        "blocking_dimension": name,
                    }
                excitation_ambiguous = True
                continue

            if status == "known":
                value = str(
                    item.get("normalized_value", "")
                ).strip()
                if value:
                    known.add(value)

        if (
            name != _EXCITATION_DIMENSION
            and len(known) > 1
        ):
            return False, {
                "override_used": False,
                "reason": (
                    "non_excitation_method_dimension_conflict"
                ),
                "blocking_dimension": name,
                "known_values": sorted(known),
            }

    if not excitation_ambiguous:
        return False, {
            "override_used": False,
            "reason": (
                "frozen_method_failure_not_due_to_excitation_ambiguity"
            ),
        }

    local_values: dict[str, float] = {}
    for row in rows:
        measurement_id = str(
            row["measurement_id"]
        )
        local_nm = _measurement_local_excitation_nm(
            graph,
            measurement_id,
        )
        if local_nm is None:
            return False, {
                "override_used": False,
                "reason": (
                    "candidate_measurement_missing_single_local_excitation"
                ),
                "measurement_id": measurement_id,
            }
        local_values[measurement_id] = local_nm

    distinct_local = sorted(
        set(local_values.values())
    )
    if len(distinct_local) != 1:
        return False, {
            "override_used": False,
            "reason": (
                "candidate_local_excitation_values_disagree"
            ),
            "local_values_nm": local_values,
        }

    local_nm = distinct_local[0]

    # If a candidate row already has a known excitation MethodContext value,
    # it must agree with the direct Measurement-local value. Unknown values do
    # not authorize anything; ambiguous values are precisely what this narrow
    # override addresses.
    for row in rows:
        item = v3._method_dimension_map(
            row["method"]
        ).get(_EXCITATION_DIMENSION)
        if item is None:
            continue
        if str(
            item.get("status", "unknown")
        ).strip() != "known":
            continue
        known_nm = _normalized_wavelength_nm(
            item.get("normalized_value", "")
        )
        if (
            known_nm is None
            or abs(known_nm - local_nm) > 1e-9
        ):
            return False, {
                "override_used": False,
                "reason": (
                    "known_method_excitation_conflicts_with_local"
                ),
                "measurement_id": str(
                    row["measurement_id"]
                ),
                "known_value": item.get(
                    "normalized_value",
                    "",
                ),
                "local_value_nm": local_nm,
            }

    return True, {
        "override_used": True,
        "reason": (
            "candidate_local_explicit_excitation_agreement"
        ),
        "local_value_nm": local_nm,
        "measurement_ids": sorted(
            local_values
        ),
        "local_values_nm": local_values,
    }


def extract_candidate_local_numeric_trends(
    source: TrendEvidenceSource,
) -> tuple[list[TrendEvidence], list[dict[str, Any]]]:
    """
    Re-run only the frozen v3 numeric grouping and supplement candidates whose
    sole method-compatibility blocker is resolved by explicit, agreeing,
    Measurement-local excitation wavelengths.

    The source and MethodContext rows are never mutated.
    """
    graph = source.graph
    identity_by_rep = v1._identity_by_representative(
        source.measurement_result_rows
    )
    method_by_id = v1._method_by_id(
        source.method_context_rows
    )

    candidates: dict[
        tuple[
            str,
            str,
            str,
            tuple[str, ...],
            str,
            str,
        ],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for context in source.comparison_context_rows:
        measurement_id = str(
            context.get("measurement_id", "")
        ).strip()
        observable_key = str(
            context.get("observable_key", "")
        ).strip()
        dependent_value = v3._finite(
            context.get("value_numeric")
        )

        if (
            not measurement_id
            or observable_key
            not in v3._NUMERIC_RESPONSE_KEYS
            or dependent_value is None
        ):
            continue

        method_row = method_by_id.get(
            str(
                context.get(
                    "method_context_id",
                    "",
                )
            ).strip()
        )
        identity_row = identity_by_rep.get(
            measurement_id
        )
        if (
            method_row is None
            or identity_row is None
        ):
            continue

        lineage = v1._lineage(
            graph,
            measurement_id,
            identity_row,
            method_row,
            context,
        )
        if lineage is None:
            continue
        lineage_kind, lineage_ids = lineage

        dependent_unit = str(
            context.get("unit", "")
        ).strip()

        controls = v3._measurement_controls(
            graph,
            measurement_id,
            identity_row,
            method_row,
            context,
        )
        for control_key, control in controls.items():
            key = (
                control_key,
                observable_key,
                lineage_kind,
                lineage_ids,
                dependent_unit,
                control.unit,
            )
            candidates[key].append(
                {
                    "measurement_id": measurement_id,
                    "identity_id": str(
                        identity_row.get(
                            "identity_id",
                            "",
                        )
                    ),
                    "context": context,
                    "method": method_row,
                    "control": control,
                    "dependent_value": dependent_value,
                }
            )

    evidence: list[TrendEvidence] = []
    audit_rows: list[dict[str, Any]] = []

    for (
        control_key,
        observable_key,
        lineage_kind,
        lineage_ids,
        dependent_unit,
        control_unit,
    ), rows in sorted(
        candidates.items(),
        key=lambda item: str(item[0]),
    ):
        if len(rows) < 2:
            continue

        frozen_compatible = v3._methods_compatible(
            (row["method"] for row in rows),
            varied_control_key=control_key,
        )
        if frozen_compatible:
            # This lane is supplemental only. Frozen-v5 already owns normal
            # compatible candidates.
            continue

        compatible, compatibility = (
            _candidate_local_methods_compatible(
                graph=graph,
                rows=rows,
                varied_control_key=control_key,
            )
        )

        audit = {
            "paper_id": source.paper_id,
            "control_key": control_key,
            "observable_key": observable_key,
            "lineage_kind": lineage_kind,
            "lineage_ids": list(lineage_ids),
            "measurement_ids": sorted(
                {
                    str(row["measurement_id"])
                    for row in rows
                }
            ),
            "method_context_ids": sorted(
                {
                    str(
                        row["method"].get(
                            "method_context_id",
                            "",
                        )
                    )
                    for row in rows
                    if str(
                        row["method"].get(
                            "method_context_id",
                            "",
                        )
                    ).strip()
                }
            ),
            "frozen_method_compatible": False,
            "candidate_local_compatible": compatible,
            **compatibility,
            "emitted": False,
        }

        if not compatible:
            audit_rows.append(audit)
            continue

        x_values = [
            float(row["control"].value_numeric)
            for row in rows
        ]
        if len(x_values) != len(set(x_values)):
            audit["reason_after_override"] = (
                "repeated_control_value"
            )
            audit_rows.append(audit)
            continue

        ordered = sorted(
            rows,
            key=lambda row: float(
                row["control"].value_numeric
            ),
        )
        direction, shape = v1._numeric_direction_shape(
            [
                (
                    float(
                        row["control"].value_numeric
                    ),
                    float(row["dependent_value"]),
                )
                for row in ordered
            ]
        )

        basis = (
            "controlled_numeric_pair"
            if len(ordered) == 2
            else "controlled_numeric_series"
        )
        measurement_ids = tuple(
            sorted(
                {
                    str(row["measurement_id"])
                    for row in ordered
                }
            )
        )
        result_ids = tuple(
            sorted(
                {
                    str(row["identity_id"])
                    for row in ordered
                    if str(
                        row["identity_id"]
                    ).strip()
                }
            )
        )
        method_ids = tuple(
            sorted(
                {
                    str(
                        row["method"].get(
                            "method_context_id",
                            "",
                        )
                    )
                    for row in ordered
                    if str(
                        row["method"].get(
                            "method_context_id",
                            "",
                        )
                    ).strip()
                }
            )
        )
        context_ids = tuple(
            sorted(
                {
                    str(
                        row["context"].get(
                            "context_id",
                            "",
                        )
                    )
                    for row in ordered
                    if str(
                        row["context"].get(
                            "context_id",
                            "",
                        )
                    ).strip()
                }
            )
        )
        subject_ids = tuple(
            sorted(
                {
                    str(subject_id)
                    for row in ordered
                    for subject_id in (
                        row["context"].get(
                            "subject_ids",
                            [],
                        )
                        or []
                    )
                    if str(subject_id).strip()
                }
            )
        )
        source_expressions = tuple(
            sorted(
                {
                    str(
                        row["context"].get(
                            "source_expression",
                            "",
                        )
                    ).strip()
                    for row in ordered
                    if str(
                        row["context"].get(
                            "source_expression",
                            "",
                        )
                    ).strip()
                }
            )
        )

        calculation_ids = v3._calculation_ids(
            graph,
            measurement_ids=measurement_ids,
            lineage_ids=lineage_ids,
            subject_ids=subject_ids,
        )
        source_node_ids = tuple(
            sorted(
                {
                    *measurement_ids,
                    *lineage_ids,
                    *calculation_ids,
                }
            )
        )

        points = tuple(
            TrendSeriesPoint(
                point_id=(
                    f"{source.paper_id}:"
                    f"{row['measurement_id']}"
                ),
                independent_value_numeric=float(
                    row["control"].value_numeric
                ),
                independent_unit=control_unit,
                dependent_value_numeric=float(
                    row["dependent_value"]
                ),
                dependent_unit=dependent_unit,
                source_measurement_result_ids=(
                    str(row["identity_id"]),
                ),
                source_measurement_ids=(
                    str(row["measurement_id"]),
                ),
                source_node_ids=(
                    str(row["measurement_id"]),
                ),
            )
            for row in ordered
        )

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
                trend_semantics_id=(
                    SERS_AU_AG_TREND_SEMANTICS_ID
                ),
                paper_id=source.paper_id,
                independent_variable_key=control_key,
                independent_variable_label=(
                    ordered[0]["control"].label
                ),
                dependent_observable_key=observable_key,
                dependent_observable_label=(
                    str(
                        ordered[0]["context"].get(
                            "observable_label",
                            observable_key,
                        )
                    ).strip()
                    or observable_key
                ),
                direction=direction,
                shape=shape,
                evidence_basis=basis,
                causal_status="not_asserted",
                varied_dimension=control_key,
                subject_ids=subject_ids,
                series_points=points,
                source_expression=(
                    source_expressions[0]
                    if source_expressions
                    else ""
                ),
                source_expressions=source_expressions,
                source_measurement_ids=measurement_ids,
                source_measurement_group_ids=(
                    lineage_ids
                    if lineage_kind
                    == "measurement_group"
                    else ()
                ),
                source_experiment_ids=(
                    lineage_ids
                    if lineage_kind == "experiment"
                    else ()
                ),
                source_calculation_ids=calculation_ids,
                source_measurement_result_ids=result_ids,
                source_method_context_ids=method_ids,
                source_comparison_context_ids=context_ids,
                source_node_ids=source_node_ids,
            )
        )

        audit["emitted"] = True
        audit["trend_id"] = trend_id
        audit["direction"] = direction
        audit["shape"] = shape
        audit_rows.append(audit)

    return evidence, audit_rows


def extract_sers_au_ag_trend_evidence(
    source: TrendEvidenceSource,
) -> list[TrendEvidence]:
    # Frozen v5 runs on the unmodified source.
    base = _V5_ADAPTER.extract_evidence(source)
    updated_base = [
        replace(
            item,
            trend_semantics_id=(
                SERS_AU_AG_TREND_SEMANTICS_ID
            ),
        )
        for item in base
    ]

    emitted_claim_ids = {
        str(claim_id)
        for item in updated_base
        for claim_id in item.source_claim_ids
    }

    claim_supplement = [
        replace(
            item,
            trend_semantics_id=(
                SERS_AU_AG_TREND_SEMANTICS_ID
            ),
        )
        for item in _V6R1_SUPPLEMENTAL_CLAIMS(
            source,
            already_emitted_claim_ids=(
                emitted_claim_ids
            ),
        )
    ]

    numeric_supplement, _audit = (
        extract_candidate_local_numeric_trends(
            source
        )
    )

    combined = [
        *updated_base,
        *claim_supplement,
        *numeric_supplement,
    ]

    by_id: dict[str, TrendEvidence] = {}
    for item in combined:
        existing = by_id.get(item.trend_id)
        if existing is not None:
            if existing == item:
                continue
            raise ValueError(
                "alpha4c5g2r2 produced conflicting "
                "TrendEvidence rows for "
                f"trend_id={item.trend_id!r}."
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
    supported_evidence_bases=(
        _V5_ADAPTER.supported_evidence_bases
    ),
    required_inputs=_V5_ADAPTER.required_inputs,
    extract_evidence_fn=(
        extract_sers_au_ag_trend_evidence
    ),
)
