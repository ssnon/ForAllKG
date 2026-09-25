from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.hypothesis_semantic_runtime import (
    HypothesisSemanticCriticRuntime,
)
from pipeline_core.discovery.pre_n10_downstream_handoff_v1 import (
    build_pre_n10_downstream_handoff_v1,
)
from pipeline_core.discovery.pre_n10_initial_semantic_gate_v1 import (
    execute_pre_n10_initial_semantic_gate_v1,
)
from pipeline_core.discovery.pre_n10_primary_router_v1 import (
    execute_pre_n10_primary_router_v1,
)
from pipeline_core.discovery.pre_n10_regeneration_reentry_v2 import (
    execute_pre_n10_regeneration_reentry_v2,
)
from pipeline_core.discovery.pre_n10_regeneration_router_adapter_v1 import (
    execute_pre_n10_regeneration_from_router_v1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    build_pre_n10_scientific_contract_v1,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    build_regeneration_unit_v2_freeze,
)
from tests.discovery.test_pre_n10_initial_semantic_gate_v1_s160 import (
    _review,
    _run,
)
from tests.discovery.test_pre_n10_primary_router_v1_s159 import (
    _mixed_plan,
)
from tests.discovery.test_pre_n10_regeneration_reentry_v1_s155 import (
    _DecompositionBackend,
    _GenerationBackend as RegenerationBackend,
    _SemanticBackend,
    _context as regeneration_context,
    _draft as regeneration_draft,
)
from tests.discovery.test_pre_n10_specification_repair_primary_v1_s157 import (
    FakeRepairBackend,
    _plan as specification_plan,
    _portfolio as specification_portfolio,
)


def _write(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _semantic_gate(tmp_path: Path, portfolio_path: Path):
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    review = _review(portfolio)
    run = _run(
        portfolio,
        accepted=True,
        hard_gate_passed=True,
        review_id=review.review_id,
        failure_stage="none",
    )
    run_path = tmp_path / "semantic.run.json"
    review_path = tmp_path / "semantic.review.json"
    _write(run_path, run)
    _write(review_path, review)
    gate, _ = execute_pre_n10_initial_semantic_gate_v1(
        portfolio_path=portfolio_path,
        semantic_run_path=run_path,
        semantic_review_path=review_path,
        output_root=tmp_path / "semantic_gate",
    )
    return gate


def _initial_setup(tmp_path: Path, *, fallback: bool):
    portfolio = specification_portfolio()
    plan = _mixed_plan() if fallback else specification_plan(missing_bridge=True)
    portfolio_path = tmp_path / "portfolio.json"
    plan_path = tmp_path / "claims_queries.json"
    _write(portfolio_path, portfolio)
    _write(plan_path, plan)
    gate = _semantic_gate(tmp_path, portfolio_path)
    contract = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        claim_decomposition_request_count=1,
    )
    repair_backend = FakeRepairBackend(audit_passes=True)
    routed_plan, _post, primary = execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=tmp_path / "router",
        specification_repair_backend_factory=(
            (lambda _hid, _root: repair_backend) if not fallback else None
        ),
    )
    routed_path = tmp_path / "router" / "post_primary.claims_queries.json"
    assert routed_path.is_file()
    assert routed_plan.plan_id == json.loads(routed_path.read_text())["plan_id"]
    return portfolio_path, routed_path, gate, primary


def test_initial_ready_lineage_is_normalized_and_authorized(tmp_path: Path) -> None:
    portfolio_path, routed_path, gate, primary = _initial_setup(
        tmp_path, fallback=False
    )

    report = build_pre_n10_downstream_handoff_v1(
        initial_semantic_gate=gate,
        initial_portfolio_path=portfolio_path,
        post_primary_query_plan_path=routed_path,
        primary_router_report=primary,
        output_root=tmp_path / "handoff",
    )

    assert report.initial_ready_lineage_count == 1
    assert report.regeneration_fallback_source_count == 0
    assert report.regenerated_ready_lineage_count == 0
    assert report.blocked_after_regeneration_count == 0
    assert report.eligible_for_external_novelty_count == 1
    row = report.lineages[0]
    assert row.origin == "INITIAL_PRIMARY_READY"
    assert row.semantic_disposition == "PASS"
    assert row.pre_n10_status == "READY_FOR_N10"
    assert row.eligible_for_external_novelty is True
    assert Path(row.portfolio_path).is_file()
    assert Path(row.query_plan_path).is_file()
    assert Path(row.contract_report_path).is_file()
    assert report.external_novelty_performed is False


def _regenerated_reentry(tmp_path: Path, *, semantic_pass: bool):
    portfolio_path, routed_path, gate, primary = _initial_setup(
        tmp_path, fallback=True
    )
    backend = RegenerationBackend(regeneration_draft())
    freeze = build_regeneration_unit_v2_freeze(
        repository_head_sha="2" * 40,
        repository_tracked_worktree_dirty=False,
    )
    regeneration, _ = execute_pre_n10_regeneration_from_router_v1(
        context=regeneration_context(),
        primary_router_report=primary,
        unit_freeze=freeze,
        backend_factory=lambda _source, _dir: backend,
        output_root=tmp_path / "regen",
    )
    regenerated_path = Path(regeneration.lineages[0].regenerated_portfolio_path)
    regenerated = HypothesisPortfolio.model_validate_json(
        regenerated_path.read_text(encoding="utf-8")
    )
    semantic_backend = _SemanticBackend(regenerated, accept=semantic_pass)
    decomposition_backend = _DecompositionBackend(malformed=False)
    reentry, _ = execute_pre_n10_regeneration_reentry_v2(
        context=regeneration_context(),
        regeneration_report=regeneration,
        semantic_runner_factory=lambda _source, _dir: (
            HypothesisSemanticCriticRuntime(semantic_backend)
        ),
        decomposition_backend_factory=lambda _source, _dir: decomposition_backend,
        output_root=tmp_path / "reentry",
    )
    return (
        portfolio_path,
        routed_path,
        gate,
        primary,
        regeneration,
        reentry,
    )


def test_regenerated_ready_lineage_is_only_fallback_handoff(tmp_path: Path) -> None:
    (
        portfolio_path,
        routed_path,
        gate,
        primary,
        regeneration,
        reentry,
    ) = _regenerated_reentry(tmp_path, semantic_pass=True)

    report = build_pre_n10_downstream_handoff_v1(
        initial_semantic_gate=gate,
        initial_portfolio_path=portfolio_path,
        post_primary_query_plan_path=routed_path,
        primary_router_report=primary,
        regeneration_report=regeneration,
        regeneration_reentry_report=reentry,
        output_root=tmp_path / "handoff",
    )

    assert report.initial_ready_lineage_count == 0
    assert report.regeneration_fallback_source_count == 1
    assert report.regenerated_ready_lineage_count == 1
    assert report.blocked_after_regeneration_count == 0
    assert report.eligible_for_external_novelty_count == 1
    row = report.lineages[0]
    assert row.origin == "REGENERATED_REENTRY_READY"
    assert row.source_hypothesis_id == "hypothesis:h1"
    assert row.semantic_disposition == "PASS"
    assert row.pre_n10_status == "READY_FOR_N10"


def test_semantic_failed_regeneration_is_recorded_but_not_eligible(tmp_path: Path) -> None:
    (
        portfolio_path,
        routed_path,
        gate,
        primary,
        regeneration,
        reentry,
    ) = _regenerated_reentry(tmp_path, semantic_pass=False)

    report = build_pre_n10_downstream_handoff_v1(
        initial_semantic_gate=gate,
        initial_portfolio_path=portfolio_path,
        post_primary_query_plan_path=routed_path,
        primary_router_report=primary,
        regeneration_report=regeneration,
        regeneration_reentry_report=reentry,
        output_root=tmp_path / "handoff",
    )

    assert report.regeneration_fallback_source_count == 1
    assert report.regenerated_ready_lineage_count == 0
    assert report.blocked_after_regeneration_count == 1
    assert report.eligible_for_external_novelty_count == 0
    assert report.lineages == []
    blocked = report.blocked_regeneration_lineages[0]
    assert blocked.final_status == "SEMANTIC_INTERVENTION_REQUIRED"
    assert blocked.eligible_for_external_novelty is False
    assert blocked.second_regeneration_allowed is False


def test_fallback_population_cannot_skip_regeneration_reentry(tmp_path: Path) -> None:
    portfolio_path, routed_path, gate, primary = _initial_setup(
        tmp_path, fallback=True
    )

    with pytest.raises(ValueError, match="requires regeneration and re-entry"):
        build_pre_n10_downstream_handoff_v1(
            initial_semantic_gate=gate,
            initial_portfolio_path=portfolio_path,
            post_primary_query_plan_path=routed_path,
            primary_router_report=primary,
            output_root=tmp_path / "handoff",
        )


def test_handoff_is_write_once_and_replay_exact(tmp_path: Path) -> None:
    portfolio_path, routed_path, gate, primary = _initial_setup(
        tmp_path, fallback=False
    )
    root = tmp_path / "handoff"

    first = build_pre_n10_downstream_handoff_v1(
        initial_semantic_gate=gate,
        initial_portfolio_path=portfolio_path,
        post_primary_query_plan_path=routed_path,
        primary_router_report=primary,
        output_root=root,
    )
    second = build_pre_n10_downstream_handoff_v1(
        initial_semantic_gate=gate,
        initial_portfolio_path=portfolio_path,
        post_primary_query_plan_path=routed_path,
        primary_router_report=primary,
        output_root=root,
    )

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
