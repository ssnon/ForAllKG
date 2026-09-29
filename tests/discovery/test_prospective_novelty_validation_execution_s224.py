import hashlib

from pipeline_core.discovery.prospective_novelty_validation_cohort import (
    ProspectiveNoveltyValidationCohortSpec,
    build_prospective_novelty_validation_cohort_freeze,
)
from pipeline_core.discovery.prospective_novelty_validation_execution import (
    build_prospective_novelty_execution_plan,
)


def _freeze():
    spec = ProspectiveNoveltyValidationCohortSpec.model_validate(
        {
            "cohort_name": "pilot",
            "cases": [
                {
                    "case_id": "A",
                    "case_role": "DIRECT_HO_REFERENCE",
                    "domain_profile_id": "sers_au_ag",
                    "corpus_id": "sers500_final_v2",
                    "source": "orientation",
                    "target": "Raman intensity",
                    "question": "q",
                    "evaluation_focus": [
                        "conceptual_first_gap_level"
                    ],
                    "research_value_capability_expected": False,
                }
            ],
        }
    )

    return build_prospective_novelty_validation_cohort_freeze(
        spec=spec,
        source_spec_sha256=hashlib.sha256(b"spec").hexdigest(),
    )


def test_execution_plan_is_deterministic_and_keeps_shadow_authority():
    freeze = _freeze()

    left = build_prospective_novelty_execution_plan(
        freeze=freeze,
        source_freeze_file_sha256="a" * 64,
        run_root="/tmp/pilot",
        model_name="model-a",
        critic_model_name="model-b",
        base_url="https://example.invalid/v1",
        api_key_env="KEY",
        results_per_query=12,
    )

    right = build_prospective_novelty_execution_plan(
        freeze=freeze,
        source_freeze_file_sha256="a" * 64,
        run_root="/tmp/pilot",
        model_name="model-a",
        critic_model_name="model-b",
        base_url="https://example.invalid/v1",
        api_key_env="KEY",
        results_per_query=12,
    )

    assert left == right
    assert left.plan_sha256 == right.plan_sha256
    assert left.scientific_case_selection_changed is False
    assert left.result_conditioned_route_changes_allowed is False
    assert left.overwrite_existing_case_runs_allowed is False
    assert left.production_selection_authority is False

    argv = left.cases[0].command_argv

    assert "--direct-relationpattern-task-shadow" in argv
    assert "--direct-higher-order-shadow" in argv
    assert "--direct-higher-order-downstream-shadow" in argv
    assert "--research-value-shadow" in argv
    assert "--overwrite-run" not in argv


def test_execution_plan_never_stores_a_secret_value(monkeypatch):
    freeze = _freeze()

    secret = "sk-test-secret-value"
    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        secret,
    )

    plan = build_prospective_novelty_execution_plan(
        freeze=freeze,
        source_freeze_file_sha256="a" * 64,
        run_root="/tmp/pilot",
        model_name="m",
        critic_model_name="c",
        base_url=None,
        api_key_env="OPENROUTER_API_KEY",
        results_per_query=12,
    )

    raw = plan.model_dump_json()

    assert plan.api_key_env == "OPENROUTER_API_KEY"
    assert "OPENROUTER_API_KEY" in raw
    assert secret not in raw

def test_execution_plan_freezes_domain_data_root_and_emits_cli(tmp_path):
    freeze = _freeze()

    data_root = tmp_path / "data_sers"
    graph = (
        data_root
        / "corpus"
        / "sers500_final_v2"
        / "mechanism"
        / "navigation"
        / "graph.graphml"
    )
    graph.parent.mkdir(parents=True)
    graph.write_text("<graphml/>", encoding="utf-8")

    plan = build_prospective_novelty_execution_plan(
        freeze=freeze,
        source_freeze_file_sha256="a" * 64,
        run_root="/tmp/pilot",
        model_name="m",
        critic_model_name="c",
        base_url=None,
        api_key_env="OPENROUTER_API_KEY",
        results_per_query=12,
        domain_data_roots={
            "sers_au_ag": str(data_root),
        },
    )

    resolved = str(data_root.resolve())
    assert plan.domain_data_roots == {
        "sers_au_ag": resolved,
    }

    argv = plan.cases[0].command_argv
    index = argv.index("--data-root")
    assert argv[index + 1] == resolved


def test_execution_plan_rejects_wrong_domain_data_root(tmp_path):
    freeze = _freeze()

    try:
        build_prospective_novelty_execution_plan(
            freeze=freeze,
            source_freeze_file_sha256="a" * 64,
            run_root="/tmp/pilot",
            model_name="m",
            critic_model_name="c",
            base_url=None,
            api_key_env="OPENROUTER_API_KEY",
            results_per_query=12,
            domain_data_roots={
                "sers_au_ag": str(tmp_path / "missing"),
            },
        )
    except ValueError as exc:
        assert "does not contain the frozen case graph" in str(exc)
    else:
        raise AssertionError("wrong domain data root should fail closed")
