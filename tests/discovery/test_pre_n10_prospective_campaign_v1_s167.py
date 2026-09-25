from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_core.discovery.pre_n10_initial_semantic_gate_v1 import (
    execute_pre_n10_initial_semantic_gate_v1,
)
from pipeline_core.discovery.pre_n10_prospective_campaign_v1 import (
    CAMPAIGN_STAGE_ORDER,
    PreN10ProspectiveCampaignReportV1,
    PreN10ProspectiveCampaignStageRecordV1,
    build_pre_n10_prospective_campaign_plan_v1,
    execute_pre_n10_prospective_campaign_v1,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    build_regeneration_unit_v2_freeze,
)
from tests.discovery.test_pre_n10_initial_semantic_gate_v1_s160 import (
    _portfolio,
    _run,
    _write,
)
from tests.discovery.test_pre_n10_regeneration_v1_s154 import (
    _context,
)


def _inputs(tmp_path: Path, *, semantic_terminal: bool = True):
    portfolio = _portfolio()
    context = _context().model_copy(
        update={
            "context_id": portfolio.source_context_id,
            "context_sha256": portfolio.source_context_sha256,
            "domain_profile_id": portfolio.domain_profile_id,
        }
    )
    portfolio_path = tmp_path / "portfolio.json"
    context_path = tmp_path / "context.json"
    _write(portfolio_path, portfolio)
    _write(context_path, context)

    run = _run(
        portfolio,
        accepted=not semantic_terminal,
        hard_gate_passed=not semantic_terminal,
        review_id=None,
        failure_stage=("hard_gate" if semantic_terminal else "none"),
    )
    run_path = tmp_path / "semantic.run.json"
    _write(run_path, run)

    freeze = build_regeneration_unit_v2_freeze(
        repository_head_sha="2" * 40,
        repository_tracked_worktree_dirty=False,
    )
    freeze_path = tmp_path / "regeneration.freeze.json"
    _write(freeze_path, freeze)

    provider_path = tmp_path / "provider.json"
    provider_path.write_text('{"fixture":"provider"}\n', encoding="utf-8")
    return (
        portfolio_path,
        run_path,
        context_path,
        freeze_path,
        provider_path,
    )


def _plan(tmp_path: Path):
    (
        portfolio,
        run,
        context,
        freeze,
        provider,
    ) = _inputs(tmp_path, semantic_terminal=True)
    return build_pre_n10_prospective_campaign_plan_v1(
        portfolio_path=portfolio,
        semantic_run_path=run,
        semantic_review_path=None,
        hypothesis_context_path=context,
        regeneration_unit_freeze_path=freeze,
        provider_plan_path=provider,
        output_root=tmp_path / "campaign",
        decomposition_model="decomp",
        primary_model="primary",
        specification_repair_model="repair",
        specification_audit_model="audit",
        source_alignment_model="align",
        regeneration_model="regen",
        semantic_critic_model="critic",
        external_n10_model="external",
        vpost_model="vpost",
        api_key_env="FIXTURE_KEY",
    )


def test_campaign_plan_fingerprints_inputs_and_freezes_stage_order(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    assert plan.stage_order == list(CAMPAIGN_STAGE_ORDER)
    assert plan.partial_stage_resume_allowed is False
    assert plan.second_regeneration_allowed is False
    assert plan.api_key_env == "FIXTURE_KEY"
    assert not hasattr(plan, "api_key")
    assert Path(plan.portfolio.path).is_file()
    assert len(plan.portfolio.sha256) == 64


def test_campaign_plan_rejects_portfolio_context_mismatch(tmp_path: Path) -> None:
    (
        portfolio,
        run,
        context,
        freeze,
        provider,
    ) = _inputs(tmp_path, semantic_terminal=True)
    payload = json.loads(context.read_text(encoding="utf-8"))
    payload["context_id"] = "context:other"
    context.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="portfolio/context ID mismatch"):
        build_pre_n10_prospective_campaign_plan_v1(
            portfolio_path=portfolio,
            semantic_run_path=run,
            semantic_review_path=None,
            hypothesis_context_path=context,
            regeneration_unit_freeze_path=freeze,
            provider_plan_path=provider,
            output_root=tmp_path / "campaign",
            decomposition_model="m",
            primary_model="m",
            specification_repair_model="m",
            specification_audit_model="m",
            source_alignment_model="m",
            regeneration_model="m",
            semantic_critic_model="m",
            external_n10_model="m",
            vpost_model="m",
        )


def test_semantic_terminal_campaign_executes_only_initial_gate(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    calls = []

    def runner(stage_name: str, argv: list[str]) -> None:
        calls.append((stage_name, list(argv)))
        assert stage_name == "initial_semantic_gate"
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        portfolio = Path(argv[argv.index("--portfolio") + 1])
        semantic_run = Path(argv[argv.index("--semantic-run") + 1])
        execute_pre_n10_initial_semantic_gate_v1(
            portfolio_path=portfolio,
            semantic_run_path=semantic_run,
            semantic_review_path=None,
            output_root=output_dir,
        )

    report = execute_pre_n10_prospective_campaign_v1(
        plan=plan,
        runner=runner,
    )
    assert report.final_status == "INITIAL_SEMANTIC_TERMINAL"
    assert len(calls) == 1
    assert report.stages[0].status == "EXECUTED"
    assert all(
        row.status == "SKIPPED_TERMINAL"
        for row in report.stages[1:]
    )


def test_completed_terminal_campaign_replay_invokes_no_runner(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    def first_runner(stage_name: str, argv: list[str]) -> None:
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        execute_pre_n10_initial_semantic_gate_v1(
            portfolio_path=Path(argv[argv.index("--portfolio") + 1]),
            semantic_run_path=Path(argv[argv.index("--semantic-run") + 1]),
            semantic_review_path=None,
            output_root=output_dir,
        )

    first = execute_pre_n10_prospective_campaign_v1(
        plan=plan,
        runner=first_runner,
    )

    def forbidden_runner(_stage_name: str, _argv: list[str]) -> None:
        raise AssertionError("completed campaign replay must not execute a stage")

    second = execute_pre_n10_prospective_campaign_v1(
        plan=plan,
        runner=forbidden_runner,
    )
    assert second.model_dump(mode="json") == first.model_dump(mode="json")


def test_partial_stage_output_fails_closed(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    partial_root = Path(plan.output_root) / "00_initial_semantic_gate"
    partial_root.mkdir(parents=True, exist_ok=True)
    (partial_root / "semantic.disposition.json").write_text(
        '{"partial":true}\n',
        encoding="utf-8",
    )

    def forbidden_runner(_stage_name: str, _argv: list[str]) -> None:
        raise AssertionError("partial stage must fail before execution")

    with pytest.raises(ValueError, match="partial write-once campaign stage"):
        execute_pre_n10_prospective_campaign_v1(
            plan=plan,
            runner=forbidden_runner,
        )


def test_campaign_stage_record_rejects_skipped_authority_artifact() -> None:
    with pytest.raises(ValueError, match="skipped campaign stage cannot carry"):
        PreN10ProspectiveCampaignStageRecordV1(
            stage_index=1,
            stage_name="initial_semantic_gate",
            status="SKIPPED_TERMINAL",
            authority_artifact_id="report:bad",
            authority_artifact_sha256="a" * 64,
            reason="fixture",
        )


def test_campaign_report_requires_complete_stage_accounting() -> None:
    with pytest.raises(ValueError, match="must account for every stage"):
        PreN10ProspectiveCampaignReportV1(
            report_id="pre_n10_prospective_campaign_report_v1:" + "0" * 20,
            report_sha256="0" * 64,
            source_plan_id="plan:1",
            source_plan_sha256="1" * 64,
            final_status="INITIAL_SEMANTIC_TERMINAL",
            stages=[],
            stage_count=0,
            initial_semantic_status="SEMANTIC_HARD_GATE_FAILED",
            primary_regeneration_fallback_count=0,
            handoff_external_eligible_count=0,
            n10_certified_count=0,
            n10_unresolved_count=0,
            n10_rejected_count=0,
            binding_ready_lineage_count=0,
            vpost_completed_count=0,
            scientific_certification_decision_counts={},
        )
