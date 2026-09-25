from __future__ import annotations

import json
from pathlib import Path

import pytest

import pipeline_core.discovery.pre_n10_regeneration_router_runtime_v1 as runtime
from pipeline_core.discovery.pre_n10_primary_router_v1 import (
    execute_pre_n10_primary_router_v1,
)
from pipeline_core.discovery.pre_n10_regeneration_router_runtime_v1 import (
    execute_pre_n10_regeneration_router_runtime_v1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    build_pre_n10_scientific_contract_v1,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    build_regeneration_unit_v2_freeze,
)
from tests.discovery.test_pre_n10_primary_router_v1_s159 import (
    _mixed_plan,
)
from tests.discovery.test_pre_n10_regeneration_v1_s154 import (
    _Backend as FakeRegenerationBackend,
    _context,
    _draft,
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


def _freeze_path(tmp_path: Path) -> Path:
    freeze = build_regeneration_unit_v2_freeze(
        repository_head_sha="2" * 40,
        repository_tracked_worktree_dirty=False,
    )
    path = tmp_path / "regeneration.freeze.json"
    _write(path, freeze)
    return path


def _fallback_router(tmp_path: Path) -> Path:
    portfolio = specification_portfolio()
    plan = _mixed_plan()
    portfolio_path = tmp_path / "portfolio.json"
    query_path = tmp_path / "claims_queries.json"
    _write(portfolio_path, portfolio)
    _write(query_path, plan)
    contract = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_path,
        query_plan_path=query_path,
        claim_decomposition_request_count=1,
    )
    _routed, _post, primary = execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=query_path,
        contract_report=contract,
        output_root=tmp_path / "router",
    )
    assert primary.regeneration_fallback_required_count == 1
    path = tmp_path / "primary_router.report.json"
    _write(path, primary)
    return path


def _ready_router(tmp_path: Path) -> Path:
    portfolio = specification_portfolio()
    plan = specification_plan(missing_bridge=True)
    portfolio_path = tmp_path / "portfolio.json"
    query_path = tmp_path / "claims_queries.json"
    _write(portfolio_path, portfolio)
    _write(query_path, plan)
    contract = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_path,
        query_plan_path=query_path,
        claim_decomposition_request_count=1,
    )
    backend = FakeRepairBackend(audit_passes=True)
    _routed, _post, primary = execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=query_path,
        contract_report=contract,
        output_root=tmp_path / "router",
        specification_repair_backend_factory=lambda _hid, _root: backend,
    )
    assert primary.regeneration_fallback_required_count == 0
    path = tmp_path / "primary_router.report.json"
    _write(path, primary)
    return path


def test_router_runtime_executes_exactly_one_frozen_regeneration_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_path = tmp_path / "context.json"
    _write(context_path, _context())
    primary_path = _fallback_router(tmp_path)
    freeze_path = _freeze_path(tmp_path)

    fake = FakeRegenerationBackend(_draft())
    constructed = []

    def backend_ctor(**kwargs):
        constructed.append(kwargs)
        return fake

    monkeypatch.setattr(
        runtime,
        "InstructorOpenAICompatibleHypothesisBackend",
        backend_ctor,
    )

    report, raw = execute_pre_n10_regeneration_router_runtime_v1(
        context_path=context_path,
        primary_router_report_path=primary_path,
        regeneration_unit_freeze_path=freeze_path,
        output_root=tmp_path / "regeneration",
        model="regen-model",
        api_key_env="FIXTURE_KEY",
    )

    assert report.regeneration_required_count == 1
    assert report.generation_call_count == 1
    assert report.repair_call_count == 0
    assert report.generated_and_compiled_count == 1
    assert len(raw) == 1
    assert fake.generate_calls == 1
    assert fake.repair_calls == 0
    assert len(constructed) == 1
    assert constructed[0]["model"] == "regen-model"
    assert constructed[0]["api_key_env"] == "FIXTURE_KEY"
    assert report.source_primary_report_id == json.loads(
        primary_path.read_text(encoding="utf-8")
    )["report_id"]


def test_router_runtime_no_fallback_fails_before_backend_construction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_path = tmp_path / "context.json"
    _write(context_path, _context())
    primary_path = _ready_router(tmp_path)
    freeze_path = _freeze_path(tmp_path)

    def backend_ctor(**_kwargs):
        raise AssertionError("regeneration backend must not be constructed")

    monkeypatch.setattr(
        runtime,
        "InstructorOpenAICompatibleHypothesisBackend",
        backend_ctor,
    )

    with pytest.raises(ValueError, match="at least one fallback lineage"):
        execute_pre_n10_regeneration_router_runtime_v1(
            context_path=context_path,
            primary_router_report_path=primary_path,
            regeneration_unit_freeze_path=freeze_path,
            output_root=tmp_path / "regeneration",
            model="regen-model",
        )


def test_router_runtime_requires_nonempty_model_before_execution(
    tmp_path: Path,
) -> None:
    context_path = tmp_path / "context.json"
    _write(context_path, _context())

    with pytest.raises(ValueError, match="requires --model"):
        execute_pre_n10_regeneration_router_runtime_v1(
            context_path=context_path,
            primary_router_report_path=_fallback_router(tmp_path),
            regeneration_unit_freeze_path=_freeze_path(tmp_path),
            output_root=tmp_path / "regeneration",
            model=" ",
        )


def test_router_runtime_missing_freeze_fails_closed(tmp_path: Path) -> None:
    context_path = tmp_path / "context.json"
    _write(context_path, _context())

    with pytest.raises(ValueError, match="regeneration-unit freeze"):
        execute_pre_n10_regeneration_router_runtime_v1(
            context_path=context_path,
            primary_router_report_path=_fallback_router(tmp_path),
            regeneration_unit_freeze_path=tmp_path / "missing.freeze.json",
            output_root=tmp_path / "regeneration",
            model="regen-model",
        )
