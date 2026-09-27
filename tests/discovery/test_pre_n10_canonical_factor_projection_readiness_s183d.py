from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.pre_n10_downstream_handoff_v1 import (
    build_pre_n10_downstream_handoff_v1,
)
from pipeline_core.discovery.pre_n10_relational_binding_bridge_v1 import (
    build_pre_n10_relational_binding_bridge_v1,
)
from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    classify_router_hint,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    assess_claim_binding_readiness,
)
from tests.discovery.test_pre_n10_downstream_canonical_authority_s181c2 import (
    _canonical_initial,
)
from tests.discovery.test_pre_n10_relational_binding_bridge_v1_s163 import (
    _shadow,
)


def _claim():
    from tests.discovery.test_pre_n10_specification_repair_primary_v1_s157 import (
        _plan,
    )

    return _plan(missing_bridge=False).claims[0].claims[0]


def _canonical_handoff(tmp_path: Path):
    (
        portfolio_path,
        routed_path,
        gate,
        primary,
        bundle_path,
        canonical_path,
        contract_v2_path,
    ) = _canonical_initial(tmp_path)
    return build_pre_n10_downstream_handoff_v1(
        initial_semantic_gate=gate,
        initial_portfolio_path=portfolio_path,
        post_primary_query_plan_path=routed_path,
        primary_router_report=primary,
        initial_source_binding_bundle_path=bundle_path,
        initial_canonical_source_reference_path=canonical_path,
        initial_contract_v2_path=contract_v2_path,
        output_root=tmp_path / "handoff",
    )


def test_complete_identity_endpoint_overlap_is_not_factor_projectable() -> None:
    claim = _claim()
    result = assess_claim_binding_readiness(
        claim=claim,
        candidate_hypothesis_id=claim.hypothesis_id,
        final_hypothesis_id=claim.hypothesis_id,
        relation_endpoint_anchors=[
            "spacing disorder",
            "spatial SERS intensity variance",
        ],
    )

    assert result.binding_status == (
        "INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION"
    )
    assert result.reason_codes == [
        "prior_art_identity_overlaps_relation_endpoint:0:0"
    ]


def test_distinct_canonical_endpoints_remain_binding_ready() -> None:
    claim = _claim()
    result = assess_claim_binding_readiness(
        claim=claim,
        candidate_hypothesis_id=claim.hypothesis_id,
        final_hypothesis_id=claim.hypothesis_id,
        relation_endpoint_anchors=[
            "disorder",
            "spatial SERS intensity variance",
        ],
    )

    assert result.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING"
    assert result.reason_codes == []


def test_overlap_reason_routes_to_decomposition_not_spec_repair() -> None:
    assert classify_router_hint(
        binding_reason_codes=[
            "prior_art_identity_overlaps_relation_endpoint:0:0"
        ],
        source_reason_codes=[],
    ) == "DECOMPOSE_OR_REGENERATE_REVIEW"


def test_canonical_bridge_threads_source_bound_endpoints_into_readiness(
    tmp_path: Path,
    monkeypatch,
) -> None:
    handoff = _canonical_handoff(tmp_path)
    external_path, _ = _shadow(
        tmp_path,
        handoff,
        selection="CONDITIONAL",
    )

    import pipeline_core.discovery.pre_n10_relational_binding_bridge_v1 as bridge_module

    original = bridge_module.assess_claim_binding_readiness
    observed: list[list[str] | None] = []

    def wrapped(**kwargs):
        anchors = kwargs.get("relation_endpoint_anchors")
        observed.append(
            list(anchors) if anchors is not None else None
        )
        return original(**kwargs)

    monkeypatch.setattr(
        bridge_module,
        "assess_claim_binding_readiness",
        wrapped,
    )

    report = build_pre_n10_relational_binding_bridge_v1(
        handoff=handoff,
        external_n10_report_path=external_path,
        output_root=tmp_path / "bridge",
    )

    assert report.stable_source_ids_used_for_relational_input is True
    assert observed == [
        ["disorder", "spatial SERS intensity variance"]
    ]
    assert report.binding_ready_lineage_count == 1
    assert report.lineages[0].binding_status == (
        "READY_FOR_LITERAL_ENDPOINT_BINDING"
    )


def test_legacy_bridge_does_not_invent_canonical_endpoint_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tests.discovery.test_pre_n10_external_n10_shadow_v1_s162 import (
        _initial_handoff,
    )

    handoff = _initial_handoff(tmp_path)
    external_path, _ = _shadow(
        tmp_path,
        handoff,
        selection="CONDITIONAL",
    )

    import pipeline_core.discovery.pre_n10_relational_binding_bridge_v1 as bridge_module

    original = bridge_module.assess_claim_binding_readiness
    observed: list[list[str] | None] = []

    def wrapped(**kwargs):
        anchors = kwargs.get("relation_endpoint_anchors")
        observed.append(
            list(anchors) if anchors is not None else None
        )
        return original(**kwargs)

    monkeypatch.setattr(
        bridge_module,
        "assess_claim_binding_readiness",
        wrapped,
    )

    report = build_pre_n10_relational_binding_bridge_v1(
        handoff=handoff,
        external_n10_report_path=external_path,
        output_root=tmp_path / "legacy_bridge",
    )

    assert report.stable_source_ids_used_for_relational_input is False
    assert observed == [None]
