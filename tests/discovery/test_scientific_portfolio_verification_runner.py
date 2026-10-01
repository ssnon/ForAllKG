from pathlib import Path


def test_verification_runner_reuses_existing_downstream_modules_and_never_runs_n10():
    text = Path("scripts/discovery/run_scientific_portfolio_verification_shadow.py").read_text(encoding="utf-8")
    for module in [
        "scripts.discovery.run_hypothesis_semantic_critic",
        "scripts.discovery.run_external_novelty",
        "scripts.discovery.build_nonobviousness_shadow",
        "scripts.discovery.run_nonobviousness_full_shadow",
        "scripts.discovery.run_feasibility_e2e",
    ]:
        assert module in text
    assert '"n10_run": False' in text
    assert '"production_selection_authority": False' in text


def test_verification_runner_is_capability_aware_for_feasibility():
    text = Path(
        "scripts/discovery/run_scientific_portfolio_verification_shadow.py"
    ).read_text(encoding="utf-8")
    assert "resolve_optional_feasibility_adapter" in text
    assert "SKIPPED_UNSUPPORTED_DOMAIN" in text
