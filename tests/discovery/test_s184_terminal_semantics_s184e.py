from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import scripts.discovery.run_discovery_axis_hypothesis_maker as maker
import pipeline_core.discovery.prospective_canonical_validation_v10 as v10
from tests.discovery.test_prospective_canonical_validation_v10_s184a import (
    _plan,
)


def _empty_outcome():
    portfolio = SimpleNamespace(
        portfolio_id="fixture-empty-portfolio",
        hypotheses=[],
    )
    report = SimpleNamespace(
        policy_version="discovery-axis-synthesis-policy-v2",
        lineages=[],
        attempts=[],
        attempted_axis_count=0,
        accepted_hypothesis_count=0,
        external_novelty_status="not_assessed",
    )
    return SimpleNamespace(
        final_draft=SimpleNamespace(hypotheses=[]),
        portfolio=portfolio,
        report=report,
        internal_novelty_report=SimpleNamespace(cards=[]),
        axis_prompts=(),
        inference_reviews=(),
        inference_review_history=(),
        context_reviews=(),
        context_review_history=(),
    )


def _empty_diversity():
    return SimpleNamespace(
        used_statement_count=0,
        eligible_statement_count=0,
        eligible_statement_coverage=0.0,
        shared_core_statement_count=0,
        exact_premise_set_duplicate_group_count=0,
        mean_pairwise_statement_jaccard=0.0,
        max_pairwise_statement_jaccard=0.0,
    )


def test_zero_axis_plan_is_normal_terminal_and_reaches_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    dual_path = tmp_path / "dual.json"
    dual_path.write_text("{}\n", encoding="utf-8")

    args = SimpleNamespace(
        dual_context=dual_path,
        task_source="source",
        task_target="target",
        task_question="question",
        evidence_family_decomposition_report=None,
        index_dir=tmp_path,
        model="fixture-model",
        base_url="https://example.invalid/v1",
        api_key_env="FIXTURE_KEY",
        instructor_mode="JSON",
        temperature=0.0,
        parse_retries=1,
        timeout=1.0,
        header=[],
        device=None,
        max_axes=5,
        min_exploration_score=0.05,
        min_candidate_unit_score=0.30,
        max_reaction_switch_penalty=0.50,
        allow_non_candidate_axes=False,
        max_compile_repairs=1,
        max_fidelity_repairs=1,
        max_inference_repairs=2,
        max_novelty_repairs=1,
        inference_critic_model=None,
        context_critic_model=None,
        context_graph=None,
        context_grounded_graph=None,
        context_axis_graph=None,
        external_axis_bundle=None,
        axis_plan_input=None,
        output_prefix=tmp_path / "alpha4",
        save_prompts=False,
        dry_run_plan=False,
    )
    dual = SimpleNamespace(
        dual_context_id="dual:fixture",
        dual_context_sha256="a" * 64,
        grounded_context=SimpleNamespace(
            corpus_id="fixture-corpus",
            domain_profile_id="fixture-profile",
        ),
        discovery_bundle=SimpleNamespace(inspirations=[object()]),
    )
    plan = SimpleNamespace(
        plan_id="discovery_axis_plan:empty",
        axes=[],
    )

    monkeypatch.setattr(maker, "parse_args", lambda: args)
    monkeypatch.setattr(
        maker,
        "DualHypothesisContext",
        SimpleNamespace(model_validate_json=lambda payload: dual),
    )
    monkeypatch.setattr(
        maker,
        "_require_axis_source_availability",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        maker,
        "_resolve_axis_plan",
        lambda **kwargs: (plan, "planned_from_dual"),
    )

    writes = []
    monkeypatch.setattr(
        maker,
        "_write_json",
        lambda path, value: writes.append(Path(path)),
    )
    monkeypatch.setattr(
        maker,
        "NodeMapper",
        SimpleNamespace(from_directory=lambda *args, **kwargs: object()),
    )
    monkeypatch.setattr(
        maker,
        "InstructorOpenAICompatibleHypothesisBackend",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        maker,
        "InstructorOpenAICompatibleAxisInferenceBackend",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        maker,
        "DiscoveryAxisInferenceCritic",
        lambda backend: object(),
    )
    monkeypatch.setattr(
        maker,
        "available_context_review_profiles",
        lambda: [],
    )

    class FakeRuntime:
        called = False

        def __init__(self, *args, **kwargs):
            pass

        def run(self, observed_dual, observed_plan):
            assert observed_dual is dual
            assert observed_plan is plan
            FakeRuntime.called = True
            return _empty_outcome()

    class FakeDiversityAssessor:
        def assess(self, context, portfolio):
            assert context is dual.grounded_context
            assert portfolio.hypotheses == []
            return _empty_diversity()

    monkeypatch.setattr(
        maker,
        "DiscoveryAxisSynthesisRuntime",
        FakeRuntime,
    )
    monkeypatch.setattr(
        maker,
        "HypothesisEvidenceDiversityAssessor",
        FakeDiversityAssessor,
    )

    assert maker.main() == 0
    assert FakeRuntime.called is True
    assert any(str(path).endswith(".portfolio.json") for path in writes)
    assert any(str(path).endswith(".lineage.json") for path in writes)
    assert any(str(path).endswith(".inference.json") for path in writes)


def _collector_freeze(tmp_path):
    plan = _plan(tmp_path)
    return v10.build_prospective_canonical_validation_collector_freeze_v10(
        execution_plan=plan,
        execution_plan_file_sha256="f" * 64,
        collector_repository_head_sha="1" * 40,
        repository_worktree_dirty=False,
        validation_output_path=tmp_path / "canonical.validation.json",
    )


def _write_campaign_tokens(freeze) -> None:
    for case in freeze.cases:
        path = Path(case.campaign_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(case.case_id + "\n", encoding="utf-8")


def _campaign(*, final_status: str, initial_active: bool):
    status_by_stage = {
        "initial_vpre": "EXECUTED" if initial_active else "SKIPPED",
        "primary_router": "SKIPPED",
        "regeneration_semantic_reentry": "SKIPPED",
        "downstream_handoff": "SKIPPED",
        "relational_binding_bridge": "SKIPPED",
        "vpost_shadow": "SKIPPED",
    }
    return SimpleNamespace(
        final_status=final_status,
        stages=[
            SimpleNamespace(stage_name=name, status=status)
            for name, status in status_by_stage.items()
        ],
    )


def test_v10_collector_terminal_before_initial_vpre_sets_vpost_zeros(
    tmp_path,
    monkeypatch,
) -> None:
    freeze = _collector_freeze(tmp_path)
    _write_campaign_tokens(freeze)
    campaigns = {
        case.case_id: _campaign(
            final_status="ZERO_HYPOTHESES_AFTER_ALPHA4",
            initial_active=False,
        )
        for case in freeze.cases
    }
    monkeypatch.setattr(
        v10,
        "PreN10ProspectiveCampaignReportV1",
        SimpleNamespace(
            model_validate_json=lambda payload: campaigns[payload.strip()]
        ),
    )

    report = v10.collect_prospective_canonical_validation_v10(freeze)

    assert report.terminal_before_initial_vpre_case_count == 5
    assert report.pre_n10_terminal_case_count == 0
    assert all(row.vpost_reached is False for row in report.cases)
    assert all(row.vpost_legacy_flag_count == 0 for row in report.cases)
    assert all(row.vpost_completed_count == 0 for row in report.cases)


def test_v10_collector_pre_n10_terminal_sets_vpost_zeros(
    tmp_path,
    monkeypatch,
) -> None:
    freeze = _collector_freeze(tmp_path)
    _write_campaign_tokens(freeze)
    campaigns = {
        case.case_id: _campaign(
            final_status="PRE_N10_NO_EXTERNAL_ELIGIBLE_LINEAGE",
            initial_active=True,
        )
        for case in freeze.cases
    }
    monkeypatch.setattr(
        v10,
        "PreN10ProspectiveCampaignReportV1",
        SimpleNamespace(
            model_validate_json=lambda payload: campaigns[payload.strip()]
        ),
    )
    monkeypatch.setattr(
        v10,
        "_verify_v2_chain",
        lambda **kwargs: (None, None, None),
    )

    report = v10.collect_prospective_canonical_validation_v10(freeze)

    assert report.terminal_before_initial_vpre_case_count == 0
    assert report.pre_n10_terminal_case_count == 5
    assert all(row.vpost_reached is False for row in report.cases)
    assert all(row.vpost_legacy_flag_count == 0 for row in report.cases)
    assert all(row.vpost_completed_count == 0 for row in report.cases)
