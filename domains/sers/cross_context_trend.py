from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from dac_her.cross_context_trend import (
    CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
    CrossContextTrendAdapter,
    CrossContextTrendSource,
    TrendContextDimension,
    TrendContextProfile,
    stable_trend_context_profile_id,
    stable_trend_relation_id,
)


SERS_AU_AG_TREND_CONTEXT_SEMANTICS_ID = (
    "sers_au_ag_trend_context_v1_alpha4c3b"
)

SERS_TREND_CONTEXT_DIMENSIONS = (
    "analyte",
    "reporter",
    "analyte_concentration",
    "excitation_wavelength",
    "laser_power",
    "integration_time",
    "raman_peak",
    "sample_preparation",
    "preparation_medium",
    "measurement_environment",
    "sample_state",
    "substrate_condition",
)

# Trend controls that are themselves one of the context dimensions. Only these
# exact controls are masked as varied_control. Structural/composition/synthesis
# controls intentionally do not mask substrate_condition or other context.
_CONTROL_TO_VARIED_CONTEXT_DIMENSION = {
    "analyte_concentration": "analyte_concentration",
    "concentration": "analyte_concentration",
    "excitation_wavelength": "excitation_wavelength",
    "laser_power": "laser_power",
    "integration_time": "integration_time",
}

_METHOD_DIMENSIONS = frozenset({
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
})
_COMPARISON_DIMENSIONS = frozenset({"raman_peak"})


def _text(value: object) -> str:
    return str(value or "").strip()


def _tuple_strings(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(
        sorted({
            str(value).strip()
            for value in values
            if str(value).strip()
        })
    )


def _row_dimensions(
    row: Mapping[str, Any],
    *,
    row_label: str,
) -> dict[str, Mapping[str, Any]]:
    raw = row.get("dimensions", [])
    if not isinstance(raw, list):
        raise ValueError(
            f"{row_label} dimensions must be a list."
        )
    result: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError(
                f"{row_label} dimension must be an object."
            )
        name = _text(item.get("name"))
        if not name:
            raise ValueError(
                f"{row_label} dimension is missing name."
            )
        if name in result:
            raise ValueError(
                f"{row_label} has duplicate dimension {name!r}."
            )
        result[name] = item
    return result


def _build_sidecar_indexes(
    source: CrossContextTrendSource,
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    dict[tuple[str, str], Mapping[str, Any]],
    dict[tuple[str, str], tuple[Mapping[str, Any], ...]],
    dict[tuple[str, str], tuple[Mapping[str, Any], ...]],
]:
    comparison_by_measurement: dict[
        tuple[str, str], Mapping[str, Any]
    ] = {}
    method_by_measurement: dict[
        tuple[str, str], Mapping[str, Any]
    ] = {}

    comparison_mentions: dict[
        tuple[str, str], list[Mapping[str, Any]]
    ] = {}
    method_mentions: dict[
        tuple[str, str], list[Mapping[str, Any]]
    ] = {}

    for row in source.comparison_context_rows:
        paper_id = _text(row.get("paper_id"))
        measurement_id = _text(row.get("measurement_id"))
        context_id = _text(row.get("context_id"))
        if not paper_id or not measurement_id or not context_id:
            raise ValueError(
                "ComparisonContext row is missing paper/measurement/context ID."
            )
        key = (paper_id, measurement_id)
        if key in comparison_by_measurement:
            raise ValueError(
                "Duplicate ComparisonContext for paper/measurement: "
                f"{key!r}."
            )
        comparison_by_measurement[key] = row

        mentions = set(
            _tuple_strings(row.get("source_node_ids", []))
        )
        mentions.add(measurement_id)
        for mention_id in mentions:
            comparison_mentions.setdefault(
                (paper_id, mention_id),
                [],
            ).append(row)

    for row in source.method_context_rows:
        paper_id = _text(row.get("paper_id"))
        measurement_id = _text(row.get("measurement_id"))
        context_id = _text(row.get("method_context_id"))
        if not paper_id or not measurement_id or not context_id:
            raise ValueError(
                "MethodContext row is missing paper/measurement/context ID."
            )
        key = (paper_id, measurement_id)
        if key in method_by_measurement:
            raise ValueError(
                "Duplicate MethodContext for paper/measurement: "
                f"{key!r}."
            )
        method_by_measurement[key] = row

        mentions = set(
            _tuple_strings(row.get("source_node_ids", []))
        )
        mentions.add(measurement_id)
        for mention_id in mentions:
            method_mentions.setdefault(
                (paper_id, mention_id),
                [],
            ).append(row)

    return (
        comparison_by_measurement,
        method_by_measurement,
        {
            key: tuple(rows)
            for key, rows in comparison_mentions.items()
        },
        {
            key: tuple(rows)
            for key, rows in method_mentions.items()
        },
    )


def _resolve_direct_sidecars(
    *,
    paper_id: str,
    source_measurement_ids: tuple[str, ...],
    comparison_mentions: Mapping[
        tuple[str, str],
        tuple[Mapping[str, Any], ...],
    ],
    method_mentions: Mapping[
        tuple[str, str],
        tuple[Mapping[str, Any], ...],
    ],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    if not source_measurement_ids:
        return (), ()

    comparisons: dict[str, Mapping[str, Any]] = {}
    methods: dict[str, Mapping[str, Any]] = {}

    for source_measurement_id in source_measurement_ids:
        cands = comparison_mentions.get(
            (paper_id, source_measurement_id),
            (),
        )
        mcands = method_mentions.get(
            (paper_id, source_measurement_id),
            (),
        )
        if len(cands) != 1:
            raise ValueError(
                "Direct source Measurement must resolve to exactly one "
                "ComparisonContext through measurement identity provenance: "
                f"{paper_id}:{source_measurement_id}:"
                f"{len(cands)} candidates."
            )
        if len(mcands) != 1:
            raise ValueError(
                "Direct source Measurement must resolve to exactly one "
                "MethodContext through measurement identity provenance: "
                f"{paper_id}:{source_measurement_id}:"
                f"{len(mcands)} candidates."
            )

        comparison = cands[0]
        method = mcands[0]
        comparison_id = _text(comparison.get("context_id"))
        method_id = _text(method.get("method_context_id"))
        linked_method_id = _text(
            comparison.get("method_context_id")
        )
        if linked_method_id and linked_method_id != method_id:
            raise ValueError(
                "ComparisonContext/MethodContext linkage mismatch for "
                f"{paper_id}:{source_measurement_id}: "
                f"{linked_method_id!r} != {method_id!r}."
            )
        if (
            _text(comparison.get("measurement_id"))
            != _text(method.get("measurement_id"))
        ):
            raise ValueError(
                "Resolved ComparisonContext and MethodContext use "
                "different representative Measurement IDs."
            )

        comparisons[comparison_id] = comparison
        methods[method_id] = method

    return (
        tuple(
            comparisons[key]
            for key in sorted(comparisons)
        ),
        tuple(
            methods[key]
            for key in sorted(methods)
        ),
    )


def _source_dimension_rows(
    *,
    dimension_name: str,
    comparisons: tuple[Mapping[str, Any], ...],
    methods: tuple[Mapping[str, Any], ...],
) -> list[
    tuple[
        Mapping[str, Any],
        str,
        str,
    ]
]:
    rows: list[
        tuple[Mapping[str, Any], str, str]
    ] = []

    if dimension_name in _METHOD_DIMENSIONS:
        for method in methods:
            method_id = _text(
                method.get("method_context_id")
            )
            dimensions = _row_dimensions(
                method,
                row_label=f"MethodContext {method_id}",
            )
            if dimension_name not in dimensions:
                raise ValueError(
                    "MethodContext dimension contract drift: "
                    f"{method_id}:{dimension_name}."
                )
            rows.append((
                dimensions[dimension_name],
                "method_context",
                method_id,
            ))
        return rows

    if dimension_name in _COMPARISON_DIMENSIONS:
        for comparison in comparisons:
            context_id = _text(
                comparison.get("context_id")
            )
            dimensions = _row_dimensions(
                comparison,
                row_label=f"ComparisonContext {context_id}",
            )
            if dimension_name not in dimensions:
                raise ValueError(
                    "ComparisonContext dimension contract drift: "
                    f"{context_id}:{dimension_name}."
                )
            rows.append((
                dimensions[dimension_name],
                "comparison_context",
                context_id,
            ))
        return rows

    raise ValueError(
        f"Unsupported SERS trend-context dimension: {dimension_name!r}."
    )


def _aggregate_dimension(
    *,
    dimension_name: str,
    source_rows: list[
        tuple[
            Mapping[str, Any],
            str,
            str,
        ]
    ],
) -> TrendContextDimension:
    if not source_rows:
        return TrendContextDimension(
            name=dimension_name,
            status="unknown",
        )

    statuses: list[str] = []
    known_values: set[str] = set()
    source_values: set[str] = set()
    source_node_ids: set[str] = set()
    provenance_scopes: set[str] = set()

    for dimension, source_kind, source_id in source_rows:
        status = _text(dimension.get("status"))
        if status not in {"known", "unknown", "ambiguous"}:
            raise ValueError(
                "Source context dimension has unknown status: "
                f"{dimension_name}:{status!r}."
            )
        statuses.append(status)

        normalized = _text(
            dimension.get("normalized_value")
        )
        if status == "known":
            if not normalized:
                raise ValueError(
                    "Known source context dimension is missing "
                    f"normalized_value: {dimension_name}."
                )
            known_values.add(normalized)

        source_values.update(
            _tuple_strings(
                dimension.get("source_values", [])
            )
        )
        source_node_ids.update(
            _tuple_strings(
                dimension.get("source_node_ids", [])
            )
        )

        provenance_scopes.add(
            f"{source_kind}:{source_id}"
        )
        provenance_scopes.add(
            "direct_measurement_provenance"
        )
        for scope in _tuple_strings(
            dimension.get("provenance_scopes", [])
        ):
            provenance_scopes.add(
                f"{source_kind}_scope:{scope}"
            )

    common = {
        "source_values": tuple(sorted(source_values)),
        "source_node_ids": tuple(
            sorted(source_node_ids)
        ),
        "provenance_scopes": tuple(
            sorted(provenance_scopes)
        ),
    }

    if "ambiguous" in statuses:
        return TrendContextDimension(
            name=dimension_name,
            status="ambiguous",
            **common,
        )

    # A mixture of known and unknown rows is not complete enough to assert
    # one trend-wide context value. Preserve the partial evidence but fail
    # closed to unknown rather than silently propagating the known subset.
    if "unknown" in statuses:
        return TrendContextDimension(
            name=dimension_name,
            status="unknown",
            **common,
        )

    if len(known_values) == 1:
        return TrendContextDimension(
            name=dimension_name,
            status="known",
            normalized_value=next(iter(known_values)),
            **common,
        )

    if len(known_values) > 1:
        return TrendContextDimension(
            name=dimension_name,
            status="ambiguous",
            **common,
        )

    return TrendContextDimension(
        name=dimension_name,
        status="unknown",
        **common,
    )


def _project_one(
    *,
    result,
    comparisons: tuple[Mapping[str, Any], ...],
    methods: tuple[Mapping[str, Any], ...],
) -> TrendContextProfile:
    varied_dimension = _CONTROL_TO_VARIED_CONTEXT_DIMENSION.get(
        result.independent_variable_key
    )

    dimensions: list[TrendContextDimension] = []
    for name in SERS_TREND_CONTEXT_DIMENSIONS:
        if name == varied_dimension:
            dimensions.append(
                TrendContextDimension(
                    name=name,
                    status="varied_control",
                    provenance_scopes=(
                        "trend_independent_variable",
                    ),
                )
            )
            continue

        rows = _source_dimension_rows(
            dimension_name=name,
            comparisons=comparisons,
            methods=methods,
        )
        dimensions.append(
            _aggregate_dimension(
                dimension_name=name,
                source_rows=rows,
            )
        )

    comparison_ids = _tuple_strings(
        row.get("context_id")
        for row in comparisons
    )
    method_ids = _tuple_strings(
        row.get("method_context_id")
        for row in methods
    )

    context_source_nodes = {
        source_node_id
        for row in (*comparisons, *methods)
        for source_node_id in _tuple_strings(
            row.get("source_node_ids", [])
        )
    }
    source_node_ids = tuple(sorted(
        set(result.source_node_ids)
        | context_source_nodes
    ))

    relation_id = stable_trend_relation_id(
        independent_variable_key=
            result.independent_variable_key,
        dependent_observable_key=
            result.dependent_observable_key,
        control_family=result.control_family,
        observable_semantics=result.observable_semantics,
    )

    return TrendContextProfile(
        context_profile_id=
            stable_trend_context_profile_id(
                context_semantics_id=
                    SERS_AU_AG_TREND_CONTEXT_SEMANTICS_ID,
                local_result_id=result.result_id,
            ),
        domain_profile_id=result.domain_profile_id,
        contract_semantics_id=
            CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
        context_semantics_id=
            SERS_AU_AG_TREND_CONTEXT_SEMANTICS_ID,
        local_result_id=result.result_id,
        paper_id=result.paper_id,
        relation_id=relation_id,
        independent_variable_key=
            result.independent_variable_key,
        dependent_observable_key=
            result.dependent_observable_key,
        control_family=result.control_family,
        observable_semantics=result.observable_semantics,
        result_lane=result.result_lane,
        direction=result.direction,
        shape=result.shape,
        evidence_kinds=result.evidence_kinds,
        member_trend_ids=result.member_trend_ids,
        dimensions=tuple(dimensions),
        source_comparison_context_ids=comparison_ids,
        source_method_context_ids=method_ids,
        source_claim_ids=result.source_claim_ids,
        source_measurement_ids=
            result.source_measurement_ids,
        source_measurement_result_ids=
            result.source_measurement_result_ids,
        source_calculation_ids=
            result.source_calculation_ids,
        source_node_ids=source_node_ids,
    )


def project_sers_au_ag_trend_contexts(
    source: CrossContextTrendSource,
) -> list[TrendContextProfile]:
    (
        _comparison_by_measurement,
        _method_by_measurement,
        comparison_mentions,
        method_mentions,
    ) = _build_sidecar_indexes(source)

    profiles: list[TrendContextProfile] = []
    for result in sorted(
        source.local_results,
        key=lambda row: (
            row.paper_id,
            row.result_id,
        ),
    ):
        if result.domain_profile_id != "sers_au_ag":
            raise ValueError(
                "SERS context projector received a different "
                f"domain result: {result.domain_profile_id!r}."
            )

        comparisons, methods = (
            _resolve_direct_sidecars(
                paper_id=result.paper_id,
                source_measurement_ids=
                    result.source_measurement_ids,
                comparison_mentions=comparison_mentions,
                method_mentions=method_mentions,
            )
        )
        profiles.append(
            _project_one(
                result=result,
                comparisons=comparisons,
                methods=methods,
            )
        )
    return profiles


@dataclass(frozen=True)
class SersTrendContextProjectionAudit:
    context_semantics_id: str
    local_result_count: int
    profile_count: int
    direct_measurement_profile_count: int
    no_direct_measurement_profile_count: int
    profiles_with_known_context: int
    profiles_with_ambiguous_context: int
    varied_control_profile_count: int
    dimension_status_counts: dict[str, dict[str, int]]
    paper_global_leakage_count: int
    unresolved_direct_measurement_count: int
    issues: tuple[str, ...]
    structural_gate: bool

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["issues"] = list(self.issues)
        return row


def audit_sers_au_ag_trend_context_projection(
    *,
    source: CrossContextTrendSource,
    profiles: list[TrendContextProfile],
) -> SersTrendContextProjectionAudit:
    issues: list[str] = []
    result_by_id = {
        result.result_id: result
        for result in source.local_results
    }
    profile_by_result = {
        profile.local_result_id: profile
        for profile in profiles
    }

    if len(profile_by_result) != len(profiles):
        issues.append(
            "duplicate_profile_for_local_result"
        )
    if set(profile_by_result) != set(result_by_id):
        issues.append(
            "context_profile_coverage_mismatch"
        )

    dimension_status_counts: dict[
        str, Counter[str]
    ] = {
        name: Counter()
        for name in SERS_TREND_CONTEXT_DIMENSIONS
    }
    direct_count = 0
    no_direct_count = 0
    known_profile_count = 0
    ambiguous_profile_count = 0
    varied_profile_count = 0
    leakage_count = 0
    unresolved_direct_count = 0

    (
        _comparison_by_measurement,
        _method_by_measurement,
        comparison_mentions,
        method_mentions,
    ) = _build_sidecar_indexes(source)

    for result_id, result in result_by_id.items():
        profile = profile_by_result.get(result_id)
        if profile is None:
            continue

        if not set(result.source_node_ids).issubset(
            set(profile.source_node_ids)
        ):
            issues.append(
                f"profile_drops_result_provenance:{result_id}"
            )

        dim_map = profile.dimension_map
        expected_varied = (
            _CONTROL_TO_VARIED_CONTEXT_DIMENSION.get(
                result.independent_variable_key
            )
        )
        observed_varied = {
            name
            for name, dimension in dim_map.items()
            if dimension.status == "varied_control"
        }
        expected_varied_set = (
            {expected_varied}
            if expected_varied
            else set()
        )
        if observed_varied != expected_varied_set:
            issues.append(
                f"varied_control_mask_mismatch:{result_id}"
            )
        if observed_varied:
            varied_profile_count += 1

        statuses = {
            dimension.status
            for dimension in profile.dimensions
        }
        if "known" in statuses:
            known_profile_count += 1
        if "ambiguous" in statuses:
            ambiguous_profile_count += 1

        for dimension in profile.dimensions:
            dimension_status_counts[
                dimension.name
            ][dimension.status] += 1

        if result.source_measurement_ids:
            direct_count += 1
            try:
                comparisons, methods = (
                    _resolve_direct_sidecars(
                        paper_id=result.paper_id,
                        source_measurement_ids=
                            result.source_measurement_ids,
                        comparison_mentions=
                            comparison_mentions,
                        method_mentions=method_mentions,
                    )
                )
            except ValueError:
                unresolved_direct_count += 1
                issues.append(
                    f"unresolved_direct_measurement_context:{result_id}"
                )
                continue

            expected_comparison_ids = {
                _text(row.get("context_id"))
                for row in comparisons
            }
            expected_method_ids = {
                _text(row.get("method_context_id"))
                for row in methods
            }
            if (
                set(profile.source_comparison_context_ids)
                != expected_comparison_ids
            ):
                issues.append(
                    f"comparison_context_membership_mismatch:{result_id}"
                )
            if (
                set(profile.source_method_context_ids)
                != expected_method_ids
            ):
                issues.append(
                    f"method_context_membership_mismatch:{result_id}"
                )
        else:
            no_direct_count += 1
            # Critical no-paper-global-leakage invariant: without an explicit
            # source Measurement, no sidecar context may be projected. The
            # only non-unknown state allowed is the trend's own varied control.
            if (
                profile.source_comparison_context_ids
                or profile.source_method_context_ids
            ):
                leakage_count += 1
                issues.append(
                    f"paper_global_context_leakage:{result_id}"
                )
            leaked_dimensions = [
                dimension.name
                for dimension in profile.dimensions
                if dimension.status
                not in {
                    "unknown",
                    "varied_control",
                    "not_applicable",
                }
            ]
            if leaked_dimensions:
                leakage_count += 1
                issues.append(
                    "paper_global_dimension_leakage:"
                    f"{result_id}:"
                    f"{','.join(sorted(leaked_dimensions))}"
                )

    normalized_status_counts = {
        name: dict(sorted(counts.items()))
        for name, counts in sorted(
            dimension_status_counts.items()
        )
    }
    unique_issues = tuple(sorted(set(issues)))
    return SersTrendContextProjectionAudit(
        context_semantics_id=
            SERS_AU_AG_TREND_CONTEXT_SEMANTICS_ID,
        local_result_count=len(source.local_results),
        profile_count=len(profiles),
        direct_measurement_profile_count=direct_count,
        no_direct_measurement_profile_count=no_direct_count,
        profiles_with_known_context=known_profile_count,
        profiles_with_ambiguous_context=
            ambiguous_profile_count,
        varied_control_profile_count=
            varied_profile_count,
        dimension_status_counts=
            normalized_status_counts,
        paper_global_leakage_count=leakage_count,
        unresolved_direct_measurement_count=
            unresolved_direct_count,
        issues=unique_issues,
        structural_gate=not unique_issues,
    )


SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER = CrossContextTrendAdapter(
    adapter_id="sers_au_ag",
    domain_profile_id="sers_au_ag",
    context_semantics_id=
        SERS_AU_AG_TREND_CONTEXT_SEMANTICS_ID,
    context_dimensions=
        SERS_TREND_CONTEXT_DIMENSIONS,
    required_inputs=frozenset({
        "paper_local_trend_results",
        "comparison_context",
        "method_context",
    }),
    project_contexts_fn=
        project_sers_au_ag_trend_contexts,
)
