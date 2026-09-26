from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.atomic_scientific_source_provenance import (
    AtomicScientificSourceBindingBundle,
    build_atomic_scientific_source_binding_bundle,
    build_atomic_scientific_source_binding_record,
)
from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.pre_n10_canonical_source_reference_v1 import (
    PreN10CanonicalSourceReferenceReportV1,
    build_pre_n10_canonical_source_reference_report_v1,
)
from pipeline_core.discovery.pre_n10_downstream_handoff_v1 import (
    build_pre_n10_downstream_handoff_v1,
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
from tests.discovery.test_pre_n10_downstream_handoff_v1_s161 import (
    _regenerated_reentry,
    _semantic_gate,
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


def _canonical_initial(tmp_path: Path):
    portfolio = specification_portfolio()
    plan = specification_plan(missing_bridge=True)
    portfolio_path = tmp_path / "portfolio.json"
    plan_path = tmp_path / "claims_queries.json"
    bundle_path = tmp_path / "source_binding.bundle.json"
    canonical_path = tmp_path / "canonical.report.json"

    _write(portfolio_path, portfolio)
    _write(plan_path, plan)
    gate = _semantic_gate(tmp_path, portfolio_path)

    card = portfolio.hypotheses[0]
    claim = plan.claims[0].claims[0]
    binding = NoveltyClaimSemanticFidelityBindingDraft(
        proposition_basis=claim.text,
        relation_endpoint_anchors=list(claim.relation_nucleus_terms[:2]),
        prediction_observation_id=(
            card.predicted_observations[0].observation_id
        ),
        falsification_criterion_id=(
            card.falsification_criteria[0].criterion_id
        ),
    )
    record = build_atomic_scientific_source_binding_record(
        hypothesis_id=claim.hypothesis_id,
        claim_local_id="local:" + claim.claim_id,
        claim=claim,
        binding=binding,
    )
    bundle = build_atomic_scientific_source_binding_bundle(
        source_portfolio_id=portfolio.portfolio_id,
        query_plan=plan,
        records=[record],
    )
    _write(bundle_path, bundle)
    canonical = build_pre_n10_canonical_source_reference_report_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        source_binding_bundle_path=bundle_path,
    )
    _write(canonical_path, canonical)
    v2 = build_pre_n10_scientific_contract_v2(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        canonical_source_reference_path=canonical_path,
        claim_decomposition_request_count=1,
    )
    v1 = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        claim_decomposition_request_count=1,
    )

    root = tmp_path / "router"
    routed, post_v2, primary = execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=v1,
        output_root=root,
        specification_repair_backend_factory=lambda _hid, _root: (
            FakeRepairBackend(audit_passes=True)
        ),
        source_binding_bundle=bundle,
        authority_contract_v2=v2,
    )
    assert post_v2.disposition == "READY_FOR_N10"
    assert routed.plan_id == primary.post_primary_query_plan_id

    return (
        portfolio_path,
        root / "post_primary.claims_queries.json",
        gate,
        primary,
        root / "atomic_source_binding.after_primary.bundle.json",
        root / "canonical_source_reference.after_primary.report.json",
        root / "contract_v2.after_primary_router.json",
    )


def test_initial_handoff_rebinds_canonical_provenance_to_subset(
    tmp_path: Path,
) -> None:
    (
        portfolio_path,
        routed_path,
        gate,
        primary,
        bundle_path,
        canonical_path,
        contract_v2_path,
    ) = _canonical_initial(tmp_path)

    report = build_pre_n10_downstream_handoff_v1(
        initial_semantic_gate=gate,
        initial_portfolio_path=portfolio_path,
        post_primary_query_plan_path=routed_path,
        primary_router_report=primary,
        initial_source_binding_bundle_path=bundle_path,
        initial_canonical_source_reference_path=canonical_path,
        initial_contract_v2_path=contract_v2_path,
        output_root=tmp_path / "handoff",
    )

    row = report.lineages[0]
    assert row.origin == "INITIAL_PRIMARY_READY"
    assert row.stable_source_ids_used_for_pre_n10_authority is True
    assert (
        row.exact_text_reconstruction_used_for_pre_n10_authority
        is False
    )

    subset_bundle = AtomicScientificSourceBindingBundle.model_validate_json(
        Path(str(row.source_binding_bundle_path)).read_text(
            encoding="utf-8"
        )
    )
    subset_canonical = PreN10CanonicalSourceReferenceReportV1.model_validate_json(
        Path(str(row.canonical_source_reference_path)).read_text(
            encoding="utf-8"
        )
    )
    subset_v2 = PreN10ScientificContractReportV2.model_validate_json(
        Path(row.contract_report_path).read_text(encoding="utf-8")
    )

    assert subset_bundle.source_portfolio_id == row.portfolio_id
    assert subset_bundle.source_query_plan_id == row.query_plan_id
    assert subset_canonical.source_binding_bundle_id == subset_bundle.bundle_id
    assert subset_v2.source_binding_bundle_id == subset_bundle.bundle_id
    assert subset_v2.disposition == "READY_FOR_N10"
    assert row.contract_report_id == subset_v2.report_id


def test_regenerated_handoff_uses_reentry_v2_canonical_authority(
    tmp_path: Path,
) -> None:
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

    row = report.lineages[0]
    assert row.origin == "REGENERATED_REENTRY_READY"
    assert row.stable_source_ids_used_for_pre_n10_authority is True
    assert (
        row.exact_text_reconstruction_used_for_pre_n10_authority
        is False
    )
    contract_v2 = PreN10ScientificContractReportV2.model_validate_json(
        Path(row.contract_report_path).read_text(encoding="utf-8")
    )
    assert contract_v2.disposition == "READY_FOR_N10"
    assert row.contract_report_id == contract_v2.report_id


def test_initial_canonical_inputs_are_all_or_none(tmp_path: Path) -> None:
    (
        portfolio_path,
        routed_path,
        gate,
        primary,
        bundle_path,
        _canonical_path,
        _contract_v2_path,
    ) = _canonical_initial(tmp_path)

    try:
        build_pre_n10_downstream_handoff_v1(
            initial_semantic_gate=gate,
            initial_portfolio_path=portfolio_path,
            post_primary_query_plan_path=routed_path,
            primary_router_report=primary,
            initial_source_binding_bundle_path=bundle_path,
            output_root=tmp_path / "handoff",
        )
    except ValueError as exc:
        assert "must be supplied all-or-none" in str(exc)
    else:
        raise AssertionError(
            "partial initial canonical chain must fail closed"
        )
