from __future__ import annotations

import networkx as nx

from campaigns.sers_alpha4_epoch.support.trend_yield_diagnostic import (
    broad_claim_candidate,
    classify_paper,
    methods_compatible,
)


def test_broad_claim_candidate_is_diagnostic_not_admission():
    row = broad_claim_candidate(
        paper_id="P1",
        claim_id="c1",
        attrs={
            "type": "ObservationClaim",
            "statement": (
                "SERS intensity increased as shell thickness "
                "was increased."
            ),
        },
        admitted_claim_ids=set(),
    )
    assert row is not None
    assert row["admitted_by_current_trend"] is False


def test_non_directional_sers_claim_is_not_candidate():
    row = broad_claim_candidate(
        paper_id="P1",
        claim_id="c1",
        attrs={
            "type": "ObservationClaim",
            "statement": "SERS spectra were measured.",
        },
        admitted_claim_ids=set(),
    )
    assert row is None


def test_method_compatibility_ignores_varied_dimension():
    rows = [
        {
            "dimensions": [
                {
                    "name": "analyte_concentration",
                    "status": "known",
                    "normalized_value": "1 nM",
                },
                {
                    "name": "excitation_wavelength",
                    "status": "known",
                    "normalized_value": "785 nm",
                },
            ]
        },
        {
            "dimensions": [
                {
                    "name": "analyte_concentration",
                    "status": "known",
                    "normalized_value": "10 nM",
                },
                {
                    "name": "excitation_wavelength",
                    "status": "known",
                    "normalized_value": "785 nm",
                },
            ]
        },
    ]
    ok, reasons = methods_compatible(
        rows,
        varied_method_dimension="analyte_concentration",
    )
    assert ok is True
    assert reasons == []


def test_method_compatibility_reports_nonvaried_conflict():
    rows = [
        {
            "dimensions": [
                {
                    "name": "excitation_wavelength",
                    "status": "known",
                    "normalized_value": "633 nm",
                }
            ]
        },
        {
            "dimensions": [
                {
                    "name": "excitation_wavelength",
                    "status": "known",
                    "normalized_value": "785 nm",
                }
            ]
        },
    ]
    ok, reasons = methods_compatible(rows)
    assert ok is False
    assert (
        "conflicting_method_dimension:excitation_wavelength"
        in reasons
    )


def test_primary_class_prefers_current_yield():
    row = classify_paper(
        actual_precision_count=1,
        actual_trend_count=1,
        claim_candidates=[
            {"admitted_by_current_trend": False}
        ],
        numeric_candidates=[
            {"admitted_by_current_trend": False}
        ],
    )
    assert row["primary_class"] == "D_current_trend_yield"


def test_primary_class_preserves_bc_overlap():
    row = classify_paper(
        actual_precision_count=0,
        actual_trend_count=0,
        claim_candidates=[
            {"admitted_by_current_trend": False}
        ],
        numeric_candidates=[
            {"admitted_by_current_trend": False}
        ],
    )
    assert row["primary_class"] == "BC_claim_and_numeric_miss"


def test_no_candidate_is_a():
    row = classify_paper(
        actual_precision_count=0,
        actual_trend_count=0,
        claim_candidates=[],
        numeric_candidates=[],
    )
    assert (
        row["primary_class"]
        == "A_no_broad_directional_candidate"
    )
