from __future__ import annotations

from scripts.sers_alpha4c211_regression_checks import (
    checks_gate,
    seen_regression_checks,
)


def test_seen_regression_check_rejects_calculated_ef_as_formal():
    evidence = [{
        "trend_id": "t",
        "paper_id": "Kiwook_SERS_2",
        "evidence_basis": "controlled_numeric_pair",
        "dependent_observable_key": "sers_enhancement_factor",
        "source_calculation_ids": ["calc"],
    }]
    annotations = [{
        "trend_id": "t",
        "evidence_kind": "calculated_numeric",
        "observable_semantics": "formal_sers_enhancement_factor",
    }]
    checks = seen_regression_checks(evidence, annotations, [])
    assert checks[
        "no_calculated_numeric_promoted_to_formal_empirical_ef"
    ]["pass"] is False
    assert checks_gate(checks) is False
