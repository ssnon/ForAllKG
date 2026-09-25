from __future__ import annotations

import json
from pathlib import Path

import pytest

import pipeline_core.discovery.pre_n10_primary_router_runtime_v1 as runtime
from pipeline_core.discovery.pre_n10_primary_router_runtime_v1 import (
    execute_pre_n10_primary_router_runtime_v1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    build_pre_n10_scientific_contract_v1,
)
from tests.discovery.test_pre_n10_decomposition_primary_v1_s158 import (
    _plan as decomposition_plan,
    _portfolio as decomposition_portfolio,
)
from tests.discovery.test_pre_n10_primary_router_v1_s159 import (
    _mixed_plan,
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
    query_path = tmp_path / "claims_queries.json"
    contract_path = tmp_path / "contract.json"
    _write(portfolio_path, portfolio)
    _write(query_path, plan)
    contract = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_path,
        query_plan_path=query_path,
        claim_decomposition_request_count=1,
    )
    _write(contract_path, contract)
    return portfolio_path, query_path, contract_path


def _fail_constructor(name: str):
    def _ctor(**_kwargs):
        raise AssertionError(name + " backend must not be constructed")
    return _ctor


def test_runtime_specification_route_constructs_only_repair_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    portfolio, query, contract = _setup(
        tmp_path,
        specification_portfolio(),
        specification_plan(missing_bridge=True),
    )
    fake = FakeRepairBackend(audit_passes=True)
    captured = {}

    def repair_ctor(**kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(
        runtime,
        "InstructorSpecificationRepairBackend",
        repair_ctor,
    )
    monkeypatch.setattr(
        runtime,
        "InstructorSourceAlignmentAuditBackend",
        _fail_constructor("source-alignment"),
    )

    _routed, post, report = execute_pre_n10_primary_router_runtime_v1(
        portfolio_path=portfolio,
        query_plan_path=query,
        contract_report_path=contract,
        output_root=tmp_path / "router",
        model="default-model",
        specification_repair_model="repair-model",
        specification_audit_model="audit-model",
        api_key_env="FIXTURE_KEY",
    )

    assert post.disposition == "READY_FOR_N10"
    assert report.route_counts == {"SPECIFICATION_REPAIR": 1}
    assert fake.generation_calls == 1
    assert fake.audit_calls == 1
    assert captured["repair_model"] == "repair-model"
    assert captured["audit_model"] == "audit-model"
    assert captured["api_key_env"] == "FIXTURE_KEY"


def test_runtime_source_alignment_constructs_only_alignment_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    portfolio, query, contract = _setup(
        tmp_path,
        source_alignment_portfolio(),
        source_alignment_plan(),
    )
    fake = FakeAuditBackend(passes=True)
    captured = {}

    def alignment_ctor(**kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(
        runtime,
        "InstructorSpecificationRepairBackend",
        _fail_constructor("specification"),
    )
    monkeypatch.setattr(
        runtime,
        "InstructorSourceAlignmentAuditBackend",
        alignment_ctor,
    )

    _routed, post, report = execute_pre_n10_primary_router_runtime_v1(
        portfolio_path=portfolio,
        query_plan_path=query,
        contract_report_path=contract,
        output_root=tmp_path / "router",
        model="default-model",
        source_alignment_model="alignment-model",
        base_url="https://example.invalid/v1",
    )

    assert post.disposition == "READY_FOR_N10"
    assert report.route_counts == {"SOURCE_ALIGNMENT": 1}
    assert fake.calls == 1
    assert captured["model"] == "alignment-model"
    assert captured["base_url"] == "https://example.invalid/v1"


def test_runtime_source_alignment_fails_only_when_route_needs_missing_base_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    portfolio, query, contract = _setup(
        tmp_path,
        source_alignment_portfolio(),
        source_alignment_plan(),
    )
    monkeypatch.setattr(
        runtime,
        "InstructorSpecificationRepairBackend",
        _fail_constructor("specification"),
    )

    with pytest.raises(ValueError, match="requires --base-url"):
        execute_pre_n10_primary_router_runtime_v1(
            portfolio_path=portfolio,
            query_plan_path=query,
            contract_report_path=contract,
            output_root=tmp_path / "router",
            model="fixture-model",
        )


def test_runtime_decomposition_route_initializes_no_llm_primary_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    portfolio, query, contract = _setup(
        tmp_path,
        decomposition_portfolio(),
        decomposition_plan(available=True),
    )
    monkeypatch.setattr(
        runtime,
        "InstructorSpecificationRepairBackend",
        _fail_constructor("specification"),
    )
    monkeypatch.setattr(
        runtime,
        "InstructorSourceAlignmentAuditBackend",
        _fail_constructor("source-alignment"),
    )

    routed, post, report = execute_pre_n10_primary_router_runtime_v1(
        portfolio_path=portfolio,
        query_plan_path=query,
        contract_report_path=contract,
        output_root=tmp_path / "router",
        model="fixture-model",
    )

    assert post.disposition == "READY_FOR_N10"
    assert report.route_counts == {"ATOMIC_DECOMPOSITION": 1}
    assert [row.claim_id for row in routed.claims[0].claims] == [
        "claim:a",
        "claim:b",
    ]


def test_runtime_mixed_route_initializes_no_primary_backend_and_falls_back(
    tmp_path: Path,
    monkeypatch,
) -> None:
    portfolio, query, contract = _setup(
        tmp_path,
        specification_portfolio(),
        _mixed_plan(),
    )
    monkeypatch.setattr(
        runtime,
        "InstructorSpecificationRepairBackend",
        _fail_constructor("specification"),
    )
    monkeypatch.setattr(
        runtime,
        "InstructorSourceAlignmentAuditBackend",
        _fail_constructor("source-alignment"),
    )

    _routed, post, report = execute_pre_n10_primary_router_runtime_v1(
        portfolio_path=portfolio,
        query_plan_path=query,
        contract_report_path=contract,
        output_root=tmp_path / "router",
        model="fixture-model",
    )

    assert post.disposition == "INTERVENTION_REQUIRED"
    assert report.route_counts == {"MIXED_ROUTE_REGENERATION_REQUIRED": 1}
    assert report.primary_intervention_count == 0
    assert report.regeneration_fallback_required_count == 1
