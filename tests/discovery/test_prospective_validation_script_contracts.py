from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_standard_verification_never_runs_n10_and_is_capability_aware():
    source = _text(
        "scripts/discovery/run_standard_portfolio_verification_shadow.py"
    )
    assert "resolve_optional_feasibility_adapter" in source
    assert '"n10_run": False' in source
    assert "SKIPPED_UNSUPPORTED_DOMAIN" in source
    assert "run_external_novelty" in source
    assert "run_nonobviousness_full_shadow" in source
    assert "run_novelty_refinement" not in source


def test_arm_materialization_is_explicitly_no_quality_ranking():
    source = _text(
        "scripts/discovery/run_prospective_arm_materialization.py"
    )
    assert "DETERMINISTIC_FAMILY_BALANCED_NO_QUALITY_RANKING" in _text(
        "pipeline_core/discovery/prospective_validation.py"
    )
    assert "INSPIRATION ONLY" in source
    assert "eligible_as_premise" in source
    assert "PRODUCTION_SELECTION_AUTHORITY=False" in source


def test_development_cases_are_explicitly_frozen_and_heldout_is_separate():
    source = _text(
        "scripts/discovery/run_prospective_validation_cohort.py"
    )
    assert "sers_raman_orientation_direct_ho_reference" in source
    assert "dac_her_low_overpotential_comparison" in source
    assert "frozen_before_run" in source
    assert "HELD_OUT" in source


def test_verification_harness_is_checkpointed_and_provider_budget_aware():
    source = _text(
        "scripts/discovery/run_standard_portfolio_verification_shadow.py"
    )
    assert "verification.checkpoint.json" in source
    assert "PAUSED_PROVIDER_BUDGET_EXHAUSTED" in source
    assert "OPENALEX_API_KEY" in source
    assert "rate-limit?api_key=" in source
    assert "Reusing completed semantic stage" in source
    assert "Reusing completed N9 full shadow" in source


def test_cohort_harness_has_resume_partial_audit_and_heldout_source_lock():
    source = _text(
        "scripts/discovery/run_prospective_validation_cohort.py"
    )
    assert "held_out_manifest.lock.json" in source
    assert "NOT_STARTED_PROVIDER_BUDGET_PAUSE" in source
    assert "_case_needs_resume" in source
    assert "planned_case_count" in source
    assert "PAUSED_PROVIDER_BUDGET_EXHAUSTED" in source
