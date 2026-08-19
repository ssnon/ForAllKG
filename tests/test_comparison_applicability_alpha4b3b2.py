from __future__ import annotations

from dac_her.comparison_context import compare_contexts, dimension_from_values
from dac_her.comparison_domain import ComparisonContext
from domains.sers.comparison import (
    SERS_AU_AG_COMPARISON_ADAPTER,
)


def _context(
    context_id: str,
    paper_id: str,
    observable_key: str,
    values: dict[str, str | None],
    *,
    numeric: float | None = 1.0,
    unit: str = "a.u.",
) -> ComparisonContext:
    dimensions = []
    for name in SERS_AU_AG_COMPARISON_ADAPTER.dimensions:
        value = values.get(name)
        if value is None:
            dimensions.append(dimension_from_values(name, []))
        else:
            dimensions.append(
                dimension_from_values(name, [(value, context_id)])
            )
    return ComparisonContext(
        context_id=context_id,
        domain_profile_id="sers_au_ag",
        comparison_semantics_id=(
            SERS_AU_AG_COMPARISON_ADAPTER.semantics_id
        ),
        paper_id=paper_id,
        measurement_id=f"m::{context_id}",
        observable_key=observable_key,
        observable_label=observable_key,
        value_numeric=numeric,
        value_text="" if numeric is not None else "text",
        unit=unit,
        source_expression="source",
        subject_ids=(),
        dimensions=tuple(dimensions),
        source_node_ids=(),
    )


def test_alpha4b3b2_raman_peak_position_does_not_self_compare_peak_dimension():
    left = _context(
        "a",
        "P1",
        "raman_peak_position",
        {
            "analyte": "methylene blue",
            "raman_peak": "1624 cm^-1",
        },
    )
    right = _context(
        "b",
        "P2",
        "raman_peak_position",
        {
            "analyte": "methylene blue",
            "raman_peak": "1575 cm^-1",
        },
    )
    result = compare_contexts(
        left,
        right,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    assert result.observable_policy_id == "raman_peak_position_v1"
    assert "raman_peak" not in result.applicable_dimensions
    assert result.compatibility == "partially_compatible"
    assert result.matched_dimensions == ("analyte",)
    assert result.mismatched_dimensions == ()
    assert result.numeric_ranking_allowed is False


def test_alpha4b3b2_optical_peak_ignores_irrelevant_raman_context():
    left = _context(
        "a",
        "P1",
        "absorption_band_wavelength",
        {
            "analyte": "methylene blue",
            "raman_peak": "1624 cm^-1",
        },
        unit="nm",
    )
    right = _context(
        "b",
        "P2",
        "absorption_band_wavelength",
        {
            "analyte": "atp",
            "raman_peak": "1097 cm^-1",
        },
        unit="nm",
    )
    result = compare_contexts(
        left,
        right,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    assert result.observable_family == "optical_spectral"
    assert result.applicable_dimensions == (
        "measurement_environment",
        "sample_state",
        "substrate_condition",
    )
    assert result.compatibility == "unknown"
    assert result.mismatched_dimensions == ()


def test_alpha4b3b2_unregistered_observable_fails_closed():
    left = _context(
        "a",
        "P1",
        "unregistered_uv_vis_spectrum",
        {},
        numeric=None,
        unit="",
    )
    right = _context(
        "b",
        "P2",
        "unregistered_uv_vis_spectrum",
        {},
        numeric=None,
        unit="",
    )
    result = compare_contexts(
        left,
        right,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    assert result.observable_policy_id == "unregistered"
    assert result.compatibility == "unknown"
    assert result.numeric_ranking_allowed is False
    assert result.reasons == ("observable_policy_unregistered",)


def test_alpha4b3b2_ef_ranking_remains_fail_closed_when_required_context_missing():
    shared = {
        "analyte": "methylene blue",
        "reporter": "methylene blue",
        "concentration": "1e-7 M",
        "raman_peak": "1624 cm^-1",
        # excitation wavelength / medium / substrate state intentionally absent
    }
    left = _context(
        "a",
        "P1",
        "sers_enhancement_factor",
        shared,
    )
    right = _context(
        "b",
        "P2",
        "sers_enhancement_factor",
        shared,
    )
    result = compare_contexts(
        left,
        right,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    assert result.compatibility == "partially_compatible"
    assert result.numeric_ranking_allowed is False
    assert "ranking_required_context_not_fully_matched" in result.reasons


def test_alpha4b3b2_ef_can_rank_only_when_every_required_dimension_matches():
    complete = {
        "analyte": "methylene blue",
        "reporter": "methylene blue",
        "concentration": "1e-7 M",
        "excitation_wavelength": "785 nm",
        "raman_peak": "1624 cm^-1",
        "measurement_environment": "aqueous",
        "sample_state": "dry",
        "substrate_condition": "as_prepared",
    }
    left = _context(
        "a",
        "P1",
        "sers_enhancement_factor",
        complete,
        numeric=1e6,
        unit="dimensionless",
    )
    right = _context(
        "b",
        "P2",
        "sers_enhancement_factor",
        complete,
        numeric=1e7,
        unit="dimensionless",
    )
    result = compare_contexts(
        left,
        right,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    assert result.compatibility == "compatible"
    assert result.numeric_ranking_allowed is True
    assert result.ranking_direction == "higher_better"


def test_alpha4b3b2_descriptive_structural_metric_never_ranks():
    left = _context(
        "a",
        "P1",
        "particle_size",
        {"substrate_condition": "as_prepared"},
        numeric=50.0,
        unit="nm",
    )
    right = _context(
        "b",
        "P2",
        "particle_size",
        {"substrate_condition": "as_prepared"},
        numeric=100.0,
        unit="nm",
    )
    result = compare_contexts(
        left,
        right,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    assert result.compatibility == "compatible"
    assert result.observable_family == "structural"
    assert result.numeric_ranking_allowed is False
    assert "numeric_ranking_disabled_for_observable" in result.reasons
