from pathlib import Path


def test_prospective_cohort_runner_emits_materialization_survival_sidecar():
    source = Path("scripts/discovery/run_prospective_validation_cohort.py").read_text(
        encoding="utf-8"
    )
    assert "scripts.discovery.run_materialization_survival_audit" in source
    assert "materialization_survival.cohort.audit.json" in source
    assert "Materialization survival audit failed; continuing primary cohort semantics" in source
