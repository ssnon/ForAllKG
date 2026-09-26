from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.atomic_scientific_source_provenance import (
    build_atomic_scientific_source_binding_bundle,
    build_atomic_scientific_source_binding_record,
)
from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.pre_n10_canonical_source_reference_v1 import (
    build_pre_n10_canonical_source_reference_report_v1,
)
from pipeline_core.discovery.pre_n10_primary_router_v1 import (
    execute_pre_n10_primary_router_v1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    build_pre_n10_scientific_contract_v1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v2 import (
    PreN10ScientificContractReportV2,
    build_pre_n10_scientific_contract_v2,
)
from pipeline_core.discovery.pre_n10_v2_child_execution_adapter import (
    PreN10V2ChildExecutionContractAdapterV1,
)
from tests.discovery.test_pre_n10_source_alignment_primary_v1_s153 import (
    FakeAuditBackend,
    _plan as source_alignment_plan,
    _portfolio as source_alignment_portfolio,
)
from tests.discovery.test_pre_n10_specification_repair_primary_v1_s157 import (
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


def _materialize_authorities(
    tmp_path: Path,
    *,
    portfolio,
    plan,
    stale_ids: bool,
):
    portfolio_path = tmp_path / "portfolio.json"
    plan_path = tmp_path / "claims_queries.json"
    bundle_path = tmp_path / "source_binding.bundle.json"
    canonical_path = tmp_path / "canonical_source.report.json"

    _write(portfolio_path, portfolio)
    _write(plan_path, plan)

    cards = {row.hypothesis_id: row for row in portfolio.hypotheses}
    records = []
    for group in plan.claims:
        card = cards[group.hypothesis_id]
        prediction = card.predicted_observations[0]
        falsifier = card.falsification_criteria[0]
        for claim in group.claims:
            records.append(
                build_atomic_scientific_source_binding_record(
                    hypothesis_id=claim.hypothesis_id,
                    claim_local_id="local:" + claim.claim_id,
                    claim=claim,
                    binding=NoveltyClaimSemanticFidelityBindingDraft(
                        proposition_basis=claim.text,
                        relation_endpoint_anchors=list(
                            claim.relation_nucleus_terms[:2]
                        ),
                        prediction_observation_id=(
                            "prediction:stale"
                            if stale_ids
                            else prediction.observation_id
                        ),
                        falsification_criterion_id=(
                            "falsifier:stale"
                            if stale_ids
                            else falsifier.criterion_id
                        ),
                    ),
                )
            )

    bundle = build_atomic_scientific_source_binding_bundle(
        source_portfolio_id=portfolio.portfolio_id,
        query_plan=plan,
        records=records,
    )
    _write(bundle_path, bundle)

    canonical = build_pre_n10_canonical_source_reference_report_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        source_binding_bundle_path=bundle_path,
    )
    _write(canonical_path, canonical)

    v1 = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        claim_decomposition_request_count=1,
    )
    v2 = build_pre_n10_scientific_contract_v2(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        canonical_source_reference_path=canonical_path,
        claim_decomposition_request_count=1,
    )
    return portfolio_path, plan_path, bundle, v1, v2


def test_v2_ready_exact_text_mismatch_bypasses_legacy_alignment(
    tmp_path: Path,
) -> None:
    portfolio = source_alignment_portfolio()
    plan = source_alignment_plan()
    portfolio_path, plan_path, bundle, v1, v2 = _materialize_authorities(
        tmp_path,
        portfolio=portfolio,
        plan=plan,
        stale_ids=False,
    )

    assert v1.disposition == "INTERVENTION_REQUIRED"
    assert v2.disposition == "READY_FOR_N10"

    routed, post_authority, report = execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=v1,
        output_root=tmp_path / "router",
        source_binding_bundle=bundle,
        authority_contract_v2=v2,
    )

    assert report.route_counts == {"PASSTHROUGH_READY": 1}
    assert report.primary_intervention_count == 0
    assert report.recovered_for_n10_count == 1
    assert report.source_contract_report_id == v2.report_id
    assert isinstance(post_authority, PreN10ScientificContractReportV2)
    assert post_authority.disposition == "READY_FOR_N10"
    assert report.post_contract_report_id == post_authority.report_id
    assert routed.model_dump(mode="json") == plan.model_dump(mode="json")


def test_v2_invalid_stable_ids_drive_alignment_even_when_v1_ready(
    tmp_path: Path,
) -> None:
    portfolio = specification_portfolio()
    plan = specification_plan(missing_bridge=False)
    portfolio_path, plan_path, bundle, v1, v2 = _materialize_authorities(
        tmp_path,
        portfolio=portfolio,
        plan=plan,
        stale_ids=True,
    )

    assert v1.disposition == "READY_FOR_N10"
    assert v2.disposition == "INTERVENTION_REQUIRED"
    assert v2.hypotheses[0].claims[0].router_hint == (
        "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"
    )

    root = tmp_path / "router"
    _routed, post_authority, report = execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=v1,
        output_root=root,
        source_alignment_audit_backend_factory=lambda _hid, _root: (
            FakeAuditBackend(passes=True)
        ),
        source_binding_bundle=bundle,
        authority_contract_v2=v2,
    )

    assert report.route_counts == {"SOURCE_ALIGNMENT": 1}
    assert report.primary_intervention_count == 1
    assert report.recovered_for_n10_count == 1
    assert isinstance(post_authority, PreN10ScientificContractReportV2)
    assert post_authority.disposition == "READY_FOR_N10"

    adapter_path = (
        root
        / "primary"
        / "hypothesis_h1"
        / "contract.before_primary.v2_execution_adapter.json"
    )
    adapter = PreN10V2ChildExecutionContractAdapterV1.model_validate_json(
        adapter_path.read_text(encoding="utf-8")
    )
    assert adapter.source_authority_report_id == v2.report_id
    assert adapter.stable_source_ids_are_authority is True
    assert adapter.exact_text_reconstruction_used_for_authority is False
    assert adapter.scientific_authority is False


def test_v2_authority_requires_matching_canonical_bundle(
    tmp_path: Path,
) -> None:
    portfolio = source_alignment_portfolio()
    plan = source_alignment_plan()
    portfolio_path, plan_path, _bundle, v1, v2 = _materialize_authorities(
        tmp_path,
        portfolio=portfolio,
        plan=plan,
        stale_ids=False,
    )

    try:
        execute_pre_n10_primary_router_v1(
            portfolio_path=portfolio_path,
            query_plan_path=plan_path,
            contract_report=v1,
            output_root=tmp_path / "router",
            authority_contract_v2=v2,
        )
    except ValueError as exc:
        assert "requires canonical source-binding bundle" in str(exc)
    else:
        raise AssertionError("V2 authority without bundle must fail closed")
