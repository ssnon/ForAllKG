from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).parents[2]


def test_atomic_prompt_explicitly_requires_novelty_bearing_specification():
    source = (
        _repo_root()
        / "pipeline_core"
        / "discovery"
        / "reframing"
        / "atomic_cross_lane_synthesis.py"
    ).read_text(encoding="utf-8")
    assert (
        'at least one atomic specification with novelty_selection_role exactly '
        '"NOVELTY_BEARING"'
    ) in source
    assert "TESTING_PREDICTION alone is not sufficient" in source


def test_atomic_runner_retries_schema_contract_failures():
    source = (
        _repo_root()
        / "scripts"
        / "discovery"
        / "run_atomic_cross_lane_scientific_synthesis.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--parse-retries", type=int, default=3)' in source


def test_prospective_wrapper_can_resume_exact_candidate_contract():
    source = (
        _repo_root()
        / "scripts"
        / "discovery"
        / "run_atomic_scientific_verifier_prospective_e2e.py"
    ).read_text(encoding="utf-8")
    assert '"--resume-existing-candidate-contract"' in source
    assert 'pre_manifest.get("status") != "complete_candidate_contract_only"' in source
    assert 'manifest["candidate_contract_reused_read_only"] = True' in source
    assert '"--parse-retries"' in source
    assert "str(args.atomic_parse_retries)" in source


def test_prospective_wrapper_can_resume_exact_frozen_atomic_cohort():
    source = (
        _repo_root()
        / "scripts"
        / "discovery"
        / "run_atomic_scientific_verifier_prospective_e2e.py"
    ).read_text(encoding="utf-8")
    assert '"--resume-frozen-atomic-cohort"' in source
    assert 'stored_freeze != recomputed_freeze' in source
    assert 'manifest["resumed_from_frozen_atomic_cohort"] = True' in source
    assert 'manifest["frozen_atomic_artifacts_reused_read_only"] = True' in source
