from pathlib import Path

from scripts.discovery.build_rvg_treatment_case_audit import (
    stable_id,
)


def test_stable_id_is_deterministic():
    a = stable_id("x", {"b": 2, "a": 1})
    b = stable_id("x", {"a": 1, "b": 2})
    assert a == b


def test_treatment_projection_preserves_e1_literals_and_drops_compiled_ids():
    # Regression intent: treatment rows must remain valid AdaptiveYieldCaseAudit
    # candidates. Treatment identity belongs in the paired mapping, not in new
    # condition/record_kind literals, and compiled prediction/falsifier IDs
    # must not be projected into the E1 candidate schema.
    source = Path(
        "scripts/discovery/build_rvg_treatment_case_audit.py"
    ).read_text(encoding="utf-8")

    assert '"condition": control["condition"]' in source
    assert '"record_kind": control["record_kind"]' in source
    assert '"record_kind": "RELATION_VALIDITY_AWARE_TREATMENT"' not in source
    assert '"condition": "RVG_TREATMENT"' not in source
    assert '"observation_id"' not in source
    assert '"criterion_id"' not in source
