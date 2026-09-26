from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_core.discovery.pre_n10_prospective_campaign_v1 import (
    CAMPAIGN_STAGE_ORDER,
    build_pre_n10_prospective_campaign_plan_v1,
    execute_pre_n10_prospective_campaign_v1,
)
from pipeline_core.discovery.prospective_atomic_admissibility_comparison_collector_v5 import (
    is_terminal_before_initial_vpre_status,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    build_regeneration_unit_v2_freeze,
)
from tests.discovery.test_pre_n10_initial_semantic_gate_v1_s160 import (
    _portfolio,
    _write,
)
from tests.discovery.test_pre_n10_regeneration_v1_s154 import (
    _context,
)


def _zero_inputs(tmp_path: Path):
    base = _portfolio()
    portfolio = base.model_copy(
        update={
            "hypotheses": [],
            "abstention_reason": "no hypotheses survived alpha4",
        }
    )
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

    freeze = build_regeneration_unit_v2_freeze(
        repository_head_sha="2" * 40,
        repository_tracked_worktree_dirty=False,
    )
    freeze_path = tmp_path / "regeneration.freeze.json"
    _write(freeze_path, freeze)

    provider_path = tmp_path / "provider.json"
    provider_path.write_text(
        '{"fixture":"provider"}\n',
        encoding="utf-8",
    )

    return (
        portfolio_path,
        tmp_path / "semantic.run.json",
        context_path,
        freeze_path,
        provider_path,
    )


def _zero_plan(tmp_path: Path):
    portfolio, run, context, freeze, provider = _zero_inputs(tmp_path)
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


def test_zero_hypothesis_plan_does_not_require_semantic_run(
    tmp_path: Path,
) -> None:
    plan = _zero_plan(tmp_path)

    assert plan.semantic_run is None
    assert plan.semantic_review is None
    assert plan.portfolio is not None


def test_zero_hypothesis_campaign_is_explicit_terminal_without_runner(
    tmp_path: Path,
) -> None:
    plan = _zero_plan(tmp_path)

    def forbidden_runner(_stage_name: str, _argv: list[str]) -> None:
        raise AssertionError(
            "zero-hypothesis terminal must not execute downstream stages"
        )

    report = execute_pre_n10_prospective_campaign_v1(
        plan=plan,
        runner=forbidden_runner,
    )

    assert report.final_status == "ZERO_HYPOTHESES_AFTER_ALPHA4"
    assert report.initial_semantic_status == (
        "NOT_RUN_ZERO_HYPOTHESES_AFTER_ALPHA4"
    )
    assert len(report.stages) == len(CAMPAIGN_STAGE_ORDER)
    assert all(row.status == "SKIPPED_TERMINAL" for row in report.stages)
    assert (
        Path(plan.output_root) / "campaign.report.json"
    ).is_file()


def test_zero_hypothesis_terminal_replay_invokes_no_runner(
    tmp_path: Path,
) -> None:
    plan = _zero_plan(tmp_path)

    def forbidden_runner(_stage_name: str, _argv: list[str]) -> None:
        raise AssertionError(
            "zero-hypothesis terminal/replay must not execute runner"
        )

    first = execute_pre_n10_prospective_campaign_v1(
        plan=plan,
        runner=forbidden_runner,
    )
    second = execute_pre_n10_prospective_campaign_v1(
        plan=plan,
        runner=forbidden_runner,
    )

    assert second.model_dump(mode="json") == first.model_dump(mode="json")


def test_nonempty_portfolio_still_requires_semantic_run(
    tmp_path: Path,
) -> None:
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

    freeze = build_regeneration_unit_v2_freeze(
        repository_head_sha="2" * 40,
        repository_tracked_worktree_dirty=False,
    )
    freeze_path = tmp_path / "regeneration.freeze.json"
    _write(freeze_path, freeze)

    provider_path = tmp_path / "provider.json"
    provider_path.write_text(
        '{"fixture":"provider"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing campaign artifact"):
        build_pre_n10_prospective_campaign_plan_v1(
            portfolio_path=portfolio_path,
            semantic_run_path=tmp_path / "missing.semantic.run.json",
            semantic_review_path=None,
            hypothesis_context_path=context_path,
            regeneration_unit_freeze_path=freeze_path,
            provider_plan_path=provider_path,
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


def test_collector_classifies_zero_hypothesis_as_terminal_before_vpre() -> None:
    assert is_terminal_before_initial_vpre_status(
        "INITIAL_SEMANTIC_TERMINAL"
    )
    assert is_terminal_before_initial_vpre_status(
        "ZERO_HYPOTHESES_AFTER_ALPHA4"
    )
    assert not is_terminal_before_initial_vpre_status(
        "PRE_N10_NO_EXTERNAL_ELIGIBLE_LINEAGE"
    )
