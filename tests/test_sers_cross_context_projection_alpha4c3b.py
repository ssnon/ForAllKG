from __future__ import annotations

import pytest

from dac_her.cross_context_trend import (
    CrossContextTrendSource,
)
from domains.sers.cross_context_trend import (
    SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER,
    SERS_AU_AG_TREND_CONTEXT_SEMANTICS_ID,
    audit_sers_au_ag_trend_context_projection,
)
from dac_her.trend_precision import PaperLocalTrendResult


METHOD_DIMENSIONS = (
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


def _dimension(
    name: str,
    *,
    value: str | None = None,
    ambiguous: tuple[str, ...] = (),
):
    if ambiguous:
        return {
            "name": name,
            "status": "ambiguous",
            "normalized_value": "",
            "source_values": list(ambiguous),
            "source_node_ids": [
                f"node:{name}:ambiguous"
            ],
            "provenance_scopes": [
                "synthetic_method_scope"
            ],
        }
    if value is None:
        return {
            "name": name,
            "status": "unknown",
            "normalized_value": "",
            "source_values": [],
            "source_node_ids": [],
            "provenance_scopes": [],
        }
    return {
        "name": name,
        "status": "known",
        "normalized_value": value,
        "source_values": [value],
        "source_node_ids": [f"node:{name}:{value}"],
        "provenance_scopes": [
            "synthetic_method_scope"
        ],
    }


def _method_row(
    paper_id: str,
    measurement_id: str,
    *,
    source_mentions: tuple[str, ...] = (),
    overrides: dict[str, dict] | None = None,
):
    overrides = overrides or {}
    dimensions = [
        overrides.get(name, _dimension(name))
        for name in METHOD_DIMENSIONS
    ]
    return {
        "method_context_id":
            f"method:{paper_id}:{measurement_id}",
        "domain_profile_id": "sers_au_ag",
        "method_semantics_id":
            "sers_au_ag_method_v4_alpha4b3b321",
        "paper_id": paper_id,
        "measurement_id": measurement_id,
        "producer_ids": [],
        "subject_ids": ["substrate"],
        "dimensions": dimensions,
        "source_node_ids": [
            measurement_id,
            *source_mentions,
        ],
    }


def _comparison_row(
    paper_id: str,
    measurement_id: str,
    *,
    source_mentions: tuple[str, ...] = (),
    raman_peak: str | None = None,
):
    return {
        "context_id":
            f"comparison:{paper_id}:{measurement_id}",
        "domain_profile_id": "sers_au_ag",
        "comparison_semantics_id":
            "sers_au_ag_comparison_v7_alpha4b3b321",
        "paper_id": paper_id,
        "measurement_id": measurement_id,
        "observable_key": "raman_intensity",
        "observable_label": "Raman intensity",
        "value_numeric": None,
        "value_text": "",
        "unit": "",
        "source_expression": "",
        "subject_ids": ["substrate"],
        "dimensions": [
            _dimension(
                "raman_peak",
                value=raman_peak,
            )
        ],
        "source_node_ids": [
            measurement_id,
            *source_mentions,
        ],
        "method_context_id":
            f"method:{paper_id}:{measurement_id}",
    }


def _result(
    *,
    result_id: str = "result:1",
    paper_id: str = "P1",
    independent_variable_key: str = "shell_thickness",
    source_measurement_ids: tuple[str, ...] = (),
    result_lane: str = "claim",
    evidence_kinds: tuple[str, ...] = ("reported_claim",),
):
    return PaperLocalTrendResult(
        result_id=result_id,
        paper_id=paper_id,
        domain_profile_id="sers_au_ag",
        trend_semantics_id=
            "sers_au_ag_trend_v5_alpha4c2121",
        precision_semantics_id=
            "sers_au_ag_trend_precision_v5_alpha4c21211",
        result_lane=result_lane,
        independent_variable_key=independent_variable_key,
        dependent_observable_key="raman_intensity",
        direction="positive",
        shape="monotonic",
        control_family="structural",
        observable_semantics="measured_signal_intensity",
        member_trend_ids=(f"{result_id}:member",),
        evidence_kinds=evidence_kinds,
        source_measurement_ids=source_measurement_ids,
        source_node_ids=(f"{result_id}:source",),
    )


def _source(
    results,
    comparisons,
    methods,
):
    return CrossContextTrendSource(
        local_results=tuple(results),
        comparison_context_rows=tuple(comparisons),
        method_context_rows=tuple(methods),
    )


def test_direct_measurement_projects_method_and_comparison_dimensions():
    result = _result(
        source_measurement_ids=("m1",),
    )
    method = _method_row(
        "P1",
        "m1",
        overrides={
            "analyte": _dimension(
                "analyte",
                value="atp",
            ),
            "excitation_wavelength": _dimension(
                "excitation_wavelength",
                value="532 nm",
            ),
        },
    )
    comparison = _comparison_row(
        "P1",
        "m1",
        raman_peak="1078 cm^-1",
    )
    source = _source(
        [result],
        [comparison],
        [method],
    )

    profiles = (
        SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER
        .project_contexts(source)
    )
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.context_semantics_id == (
        SERS_AU_AG_TREND_CONTEXT_SEMANTICS_ID
    )
    assert profile.dimension_map["analyte"].status == "known"
    assert (
        profile.dimension_map["analyte"].normalized_value
        == "atp"
    )
    assert (
        profile.dimension_map["excitation_wavelength"]
        .normalized_value
        == "532 nm"
    )
    assert (
        profile.dimension_map["raman_peak"].normalized_value
        == "1078 cm^-1"
    )
    assert profile.source_comparison_context_ids == (
        "comparison:P1:m1",
    )
    assert profile.source_method_context_ids == (
        "method:P1:m1",
    )


def test_identity_source_mention_resolves_to_representative_context():
    result = _result(
        source_measurement_ids=("m_source_alias",),
    )
    method = _method_row(
        "P1",
        "m_representative",
        source_mentions=("m_source_alias",),
        overrides={
            "analyte": _dimension(
                "analyte",
                value="atp",
            ),
        },
    )
    comparison = _comparison_row(
        "P1",
        "m_representative",
        source_mentions=("m_source_alias",),
    )
    source = _source(
        [result],
        [comparison],
        [method],
    )

    profile = (
        SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER
        .project_contexts(source)[0]
    )
    assert profile.source_comparison_context_ids == (
        "comparison:P1:m_representative",
    )
    assert profile.source_method_context_ids == (
        "method:P1:m_representative",
    )


def test_claim_without_direct_measurement_does_not_consume_paper_context():
    result = _result(
        source_measurement_ids=(),
    )
    method = _method_row(
        "P1",
        "m1",
        overrides={
            "analyte": _dimension(
                "analyte",
                value="atp",
            ),
            "excitation_wavelength": _dimension(
                "excitation_wavelength",
                value="532 nm",
            ),
        },
    )
    comparison = _comparison_row(
        "P1",
        "m1",
        raman_peak="1078 cm^-1",
    )
    source = _source(
        [result],
        [comparison],
        [method],
    )

    profile = (
        SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER
        .project_contexts(source)[0]
    )
    assert profile.source_comparison_context_ids == ()
    assert profile.source_method_context_ids == ()
    assert all(
        dimension.status == "unknown"
        for dimension in profile.dimensions
    )

    audit = audit_sers_au_ag_trend_context_projection(
        source=source,
        profiles=[profile],
    )
    assert audit.structural_gate is True
    assert audit.paper_global_leakage_count == 0


def test_analyte_concentration_is_varied_control_not_mismatch_context():
    result = _result(
        independent_variable_key="analyte_concentration",
        source_measurement_ids=("m1", "m2"),
        result_lane="numeric",
        evidence_kinds=("experimental_numeric",),
    )
    methods = [
        _method_row(
            "P1",
            "m1",
            overrides={
                "analyte_concentration": _dimension(
                    "analyte_concentration",
                    value="1e-9 M",
                ),
            },
        ),
        _method_row(
            "P1",
            "m2",
            overrides={
                "analyte_concentration": _dimension(
                    "analyte_concentration",
                    value="1e-6 M",
                ),
            },
        ),
    ]
    comparisons = [
        _comparison_row("P1", "m1"),
        _comparison_row("P1", "m2"),
    ]
    source = _source(
        [result],
        comparisons,
        methods,
    )

    profile = (
        SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER
        .project_contexts(source)[0]
    )
    assert (
        profile.dimension_map["analyte_concentration"].status
        == "varied_control"
    )
    audit = audit_sers_au_ag_trend_context_projection(
        source=source,
        profiles=[profile],
    )
    assert audit.structural_gate is True


def test_spectral_detuning_does_not_mask_fixed_excitation_wavelength():
    result = _result(
        independent_variable_key="spr_excitation_detuning",
        source_measurement_ids=("m1",),
    )
    method = _method_row(
        "P1",
        "m1",
        overrides={
            "excitation_wavelength": _dimension(
                "excitation_wavelength",
                value="532 nm",
            ),
        },
    )
    comparison = _comparison_row("P1", "m1")
    source = _source(
        [result],
        [comparison],
        [method],
    )

    profile = (
        SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER
        .project_contexts(source)[0]
    )
    excitation = profile.dimension_map[
        "excitation_wavelength"
    ]
    assert excitation.status == "known"
    assert excitation.normalized_value == "532 nm"


def test_conflicting_known_values_across_direct_measurements_become_ambiguous():
    result = _result(
        source_measurement_ids=("m1", "m2"),
        result_lane="numeric",
        evidence_kinds=("experimental_numeric",),
    )
    methods = [
        _method_row(
            "P1",
            "m1",
            overrides={
                "excitation_wavelength": _dimension(
                    "excitation_wavelength",
                    value="532 nm",
                ),
            },
        ),
        _method_row(
            "P1",
            "m2",
            overrides={
                "excitation_wavelength": _dimension(
                    "excitation_wavelength",
                    value="633 nm",
                ),
            },
        ),
    ]
    comparisons = [
        _comparison_row("P1", "m1"),
        _comparison_row("P1", "m2"),
    ]
    source = _source(
        [result],
        comparisons,
        methods,
    )

    profile = (
        SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER
        .project_contexts(source)[0]
    )
    assert (
        profile.dimension_map["excitation_wavelength"].status
        == "ambiguous"
    )


def test_partial_known_unknown_coverage_fails_closed_to_unknown():
    result = _result(
        source_measurement_ids=("m1", "m2"),
        result_lane="numeric",
        evidence_kinds=("experimental_numeric",),
    )
    methods = [
        _method_row(
            "P1",
            "m1",
            overrides={
                "excitation_wavelength": _dimension(
                    "excitation_wavelength",
                    value="532 nm",
                ),
            },
        ),
        _method_row(
            "P1",
            "m2",
        ),
    ]
    comparisons = [
        _comparison_row("P1", "m1"),
        _comparison_row("P1", "m2"),
    ]
    source = _source(
        [result],
        comparisons,
        methods,
    )

    profile = (
        SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER
        .project_contexts(source)[0]
    )
    dimension = profile.dimension_map[
        "excitation_wavelength"
    ]
    assert dimension.status == "unknown"
    assert "532 nm" in dimension.source_values


def test_unresolvable_direct_measurement_fails_closed():
    result = _result(
        source_measurement_ids=("missing_measurement",),
    )
    method = _method_row("P1", "m1")
    comparison = _comparison_row("P1", "m1")
    source = _source(
        [result],
        [comparison],
        [method],
    )

    with pytest.raises(
        ValueError,
        match="resolve to exactly one ComparisonContext",
    ):
        (
            SERS_AU_AG_CROSS_CONTEXT_TREND_ADAPTER
            .project_contexts(source)
        )
