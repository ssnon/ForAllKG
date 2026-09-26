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
)
from pipeline_core.discovery.pre_n10_primary_router_v1 import (
    execute_pre_n10_primary_router_v1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    build_pre_n10_scientific_contract_v1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v2 import (
    PreN10ScientificContractReportV2,
)
from tests.discovery.test_pre_n10_decomposition_primary_v1_s158 import (
    _plan as decomposition_plan,
    _portfolio as decomposition_portfolio,
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


def _bundle_for(
    *,
    portfolio,
    plan,
    stale_ids: bool = False,
) -> AtomicScientificSourceBindingBundle:
    cards = {
        row.hypothesis_id: row
        for row in portfolio.hypotheses
    }
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
    return build_atomic_scientific_source_binding_bundle(
        source_portfolio_id=portfolio.portfolio_id,
        query_plan=plan,
        records=records,
    )


def _load_post(root: Path):
    bundle = AtomicScientificSourceBindingBundle.model_validate_json(
        (
            root / "atomic_source_binding.after_primary.bundle.json"
        ).read_text(encoding="utf-8")
    )
    canonical = PreN10CanonicalSourceReferenceReportV1.model_validate_json(
        (
            root
            / "canonical_source_reference.after_primary.report.json"
        ).read_text(encoding="utf-8")
    )
    v2 = PreN10ScientificContractReportV2.model_validate_json(
        (
            root / "contract_v2.after_primary_router.json"
        ).read_text(encoding="utf-8")
    )
    return bundle, canonical, v2


def test_specification_repair_projects_ids_and_materializes_post_v2(
    tmp_path: Path,
) -> None:
    portfolio = specification_portfolio()
    plan = specification_plan(missing_bridge=True)
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        portfolio,
        plan,
    )
    bundle = _bundle_for(portfolio=portfolio, plan=plan)
    before = bundle.records[0]

    root = tmp_path / "router"
    execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=root,
        specification_repair_backend_factory=lambda _hid, _root: (
            FakeRepairBackend(audit_passes=True)
        ),
        source_binding_bundle=bundle,
    )

    post_bundle, canonical, v2 = _load_post(root)
    after = post_bundle.records[0]

    assert after.prediction_observation_id == (
        before.prediction_observation_id
    )
    assert after.falsification_criterion_id == (
        before.falsification_criterion_id
    )
    assert after.source_claim_sha256 != before.source_claim_sha256
    assert canonical.stable_ready_count == 1
    assert v2.disposition == "READY_FOR_N10"


def test_source_alignment_installs_audited_selected_source_ids(
    tmp_path: Path,
) -> None:
    portfolio = source_alignment_portfolio()
    plan = source_alignment_plan()
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        portfolio,
        plan,
    )
    bundle = _bundle_for(
        portfolio=portfolio,
        plan=plan,
        stale_ids=True,
    )
    card = portfolio.hypotheses[0]

    root = tmp_path / "router"
    execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=root,
        source_alignment_audit_backend_factory=lambda _hid, _root: (
            FakeAuditBackend(passes=True)
        ),
        source_binding_bundle=bundle,
    )

    post_bundle, canonical, v2 = _load_post(root)
    row = post_bundle.records[0]

    assert row.prediction_observation_id == (
        card.predicted_observations[0].observation_id
    )
    assert row.falsification_criterion_id == (
        card.falsification_criteria[0].criterion_id
    )
    assert canonical.stable_ready_count == 1
    assert v2.disposition == "READY_FOR_N10"


def test_decomposition_drops_removed_composite_provenance(
    tmp_path: Path,
) -> None:
    portfolio = decomposition_portfolio()
    plan = decomposition_plan(available=True)
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        portfolio,
        plan,
    )
    bundle = _bundle_for(portfolio=portfolio, plan=plan)

    root = tmp_path / "router"
    routed, _post, _report = execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=root,
        source_binding_bundle=bundle,
    )

    post_bundle, canonical, v2 = _load_post(root)
    routed_ids = [
        claim.claim_id
        for group in routed.claims
        for claim in group.claims
    ]

    assert [row.claim_id for row in post_bundle.records] == routed_ids
    assert "claim:c" not in routed_ids
    assert post_bundle.claim_count == 2
    assert canonical.stable_ready_count == 2
    assert v2.disposition == "READY_FOR_N10"


def test_router_without_canonical_bundle_emits_no_post_canonical_sidecars(
    tmp_path: Path,
) -> None:
    portfolio = decomposition_portfolio()
    plan = decomposition_plan(available=True)
    portfolio_path, plan_path, contract = _setup(
        tmp_path,
        portfolio,
        plan,
    )
    root = tmp_path / "router"

    execute_pre_n10_primary_router_v1(
        portfolio_path=portfolio_path,
        query_plan_path=plan_path,
        contract_report=contract,
        output_root=root,
    )

    assert not (
        root / "atomic_source_binding.after_primary.bundle.json"
    ).exists()
    assert not (
        root / "canonical_source_reference.after_primary.report.json"
    ).exists()
    assert not (
        root / "contract_v2.after_primary_router.json"
    ).exists()
