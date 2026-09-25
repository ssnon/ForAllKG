from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
)
from pipeline_core.discovery.pre_n10_primary_router_v1 import (
    execute_pre_n10_primary_router_v1,
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
from tests.discovery.test_pre_n10_decomposition_primary_v1_s158 import (
    _plan as decomposition_plan,
    _portfolio as decomposition_portfolio,
)
from tests.discovery.test_pre_n10_regeneration_v1_s154 import (
    _Backend as RegenerationBackend,
    _context as regeneration_context,
    _draft as regeneration_draft,
)
from tests.discovery.test_pre_n10_source_alignment_primary_v1_s153 import (
    FakeAuditBackend,
    _plan as source_alignment_plan,
    _portfolio as source_alignment_portfolio,
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


def _setup(tmp_path: Path, portfolio, plan):
    portfolio_path = tmp_path / "portfolio.json"
    plan_path = tmp_path / "claims_queries.json"
    _write(portfolio_path, portfolio)
    _write(plan_path, plan)
    contract = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        claim_decomposition_request_count=1,
    )
    return portfolio_path, plan_path, contract


def test_router_executes_specification_primary_and_recovers(tmp_path: Path) -> None:
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        specification_portfolio(),
        specification_plan(missing_bridge=True),
    )
    backend = FakeRepairBackend(audit_passes=True)

    routed, post, report = execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=tmp_path / "router",
        specification_repair_backend_factory=lambda _hid, _root: backend,
    )

    assert post.disposition == "READY_FOR_N10"
    assert report.route_counts == {"SPECIFICATION_REPAIR": 1}
    assert report.primary_intervention_count == 1
    assert report.recovered_for_n10_count == 1
    assert report.regeneration_fallback_required_count == 0
    assert routed.claims[0].claims[0].required_bridge
    assert backend.generation_calls == 1
    assert backend.audit_calls == 1


def test_router_executes_source_alignment_primary_and_recovers(tmp_path: Path) -> None:
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        source_alignment_portfolio(),
        source_alignment_plan(),
    )
    backend = FakeAuditBackend(passes=True)

    _routed, post, report = execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=tmp_path / "router",
        source_alignment_audit_backend_factory=lambda _hid, _root: backend,
    )

    assert post.disposition == "READY_FOR_N10"
    assert report.route_counts == {"SOURCE_ALIGNMENT": 1}
    assert report.primary_intervention_count == 1
    assert report.recovered_for_n10_count == 1
    assert report.regeneration_fallback_required_count == 0
    assert backend.calls == 1


def test_router_executes_deterministic_decomposition_and_recovers(tmp_path: Path) -> None:
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        decomposition_portfolio(),
        decomposition_plan(available=True),
    )

    routed, post, report = execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=tmp_path / "router",
    )

    assert post.disposition == "READY_FOR_N10"
    assert report.route_counts == {"ATOMIC_DECOMPOSITION": 1}
    assert report.primary_intervention_count == 1
    assert report.recovered_for_n10_count == 1
    assert report.regeneration_fallback_required_count == 0
    assert [row.claim_id for row in routed.claims[0].claims] == [
        "claim:a",
        "claim:b",
    ]
    assert [row.claim_id for row in routed.queries] == [
        "claim:a",
        "claim:b",
    ]


def _mixed_plan() -> LiteratureQueryPlan:
    spec = specification_plan(missing_bridge=True).claims[0].claims[0]
    source = source_alignment_plan().claims[0].claims[0].model_copy(
        update={
            "claim_id": "claim:2",
            "claim_rank": 2,
        }
    )
    return LiteratureQueryPlan(
        plan_id="literature_query_plan:mixed",
        plan_sha256="d" * 64,
        source_portfolio_id="hypothesis_portfolio:p1",
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:h1",
                title="Mixed contract failures",
                claims=[spec, source],
            )
        ],
    )


def test_mixed_route_is_not_chained_and_goes_directly_to_regeneration(
    tmp_path: Path,
) -> None:
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        specification_portfolio(),
        _mixed_plan(),
    )
    hints = {
        row.router_hint for row in contract.hypotheses[0].claims
        if row.contract_status == "NOT_READY_FOR_N10_CONTRACT"
    }
    assert hints == {
        "SPECIFICATION_REPAIR_REVIEW",
        "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW",
    }

    routed, post, report = execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=tmp_path / "router",
    )

    row = report.hypotheses[0]
    assert row.route == "MIXED_ROUTE_REGENERATION_REQUIRED"
    assert row.dominant_router_hint == (
        "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"
    )
    assert row.primary_intervention_performed is False
    assert row.regeneration_fallback_required is True
    assert report.primary_intervention_count == 0
    assert report.mixed_route_fallback_count == 1
    assert report.regeneration_fallback_required_count == 1
    assert post.disposition == "INTERVENTION_REQUIRED"
    assert routed.model_dump(mode="json") == _mixed_plan().model_dump(mode="json")


def test_router_fallback_feeds_exactly_one_existing_regeneration_unit(
    tmp_path: Path,
) -> None:
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        specification_portfolio(),
        _mixed_plan(),
    )
    _routed, _post, primary = execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=tmp_path / "router",
    )

    backend = RegenerationBackend(regeneration_draft())
    freeze = build_regeneration_unit_v2_freeze(
        repository_head_sha="2" * 40,
        repository_tracked_worktree_dirty=False,
    )
    report, raw = execute_pre_n10_regeneration_from_router_v1(
        context=regeneration_context(),
        primary_router_report=primary,
        unit_freeze=freeze,
        backend_factory=lambda _source, _dir: backend,
        output_root=tmp_path / "regen",
    )

    assert report.regeneration_required_count == 1
    assert report.generation_call_count == 1
    assert report.repair_call_count == 0
    assert backend.generate_calls == 1
    assert backend.repair_calls == 0
    assert set(raw) == {"hypothesis:h1"}
    assert report.source_primary_report_id == primary.report_id
    assert report.previous_hypothesis_text_consumed is False
    assert report.n10_outcome_consumed is False


def test_router_is_write_once_and_replay_exact(tmp_path: Path) -> None:
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        decomposition_portfolio(),
        decomposition_plan(available=True),
    )
    root = tmp_path / "router"

    first = execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=root,
    )
    second = execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=root,
    )

    assert first[0].model_dump(mode="json") == second[0].model_dump(mode="json")
    assert first[1].model_dump(mode="json") == second[1].model_dump(mode="json")
    assert first[2].model_dump(mode="json") == second[2].model_dump(mode="json")
