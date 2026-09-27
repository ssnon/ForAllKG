from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.atomic_scientific_specification_bundle import (
    AtomicScientificSpecificationBundle,
)
from pipeline_core.discovery.pre_n10_downstream_handoff_v1 import (
    build_pre_n10_downstream_handoff_v1,
)
from pipeline_core.discovery.pre_n10_relational_binding_bridge_v1 import (
    build_pre_n10_relational_binding_bridge_v1,
)
from pipeline_core.discovery.pre_n10_vpost_shadow_v1 import (
    compile_pre_n10_vpost_shadow_plan_v1,
)
from tests.discovery.test_pre_n10_downstream_canonical_authority_s181c2 import (
    _canonical_initial,
)
from tests.discovery.test_pre_n10_relational_binding_bridge_v1_s163 import (
    _shadow,
)
from tests.discovery.test_pre_n10_vpost_shadow_v1_s164 import (
    _provider,
)


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


def test_bridge_materializes_canonical_spec_bundle_from_stable_ids(
    tmp_path: Path,
) -> None:
    handoff = _canonical_handoff(tmp_path)
    external_path, _ = _shadow(
        tmp_path,
        handoff,
        selection="CONDITIONAL",
    )
    bridge = build_pre_n10_relational_binding_bridge_v1(
        handoff=handoff,
        external_n10_report_path=external_path,
        output_root=tmp_path / "bridge",
    )

    assert bridge.stable_source_ids_used_for_relational_input is True
    assert (
        bridge.exact_text_reconstruction_required_for_relational_input
        is False
    )
    assert bridge.canonical_spec_bundle_path is not None
    bundle_path = Path(bridge.canonical_spec_bundle_path)
    assert bundle_path.is_file()

    bundle = AtomicScientificSpecificationBundle.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    assert bundle.bundle_id == bridge.canonical_spec_bundle_id
    assert bundle.bundle_sha256 == bridge.canonical_spec_bundle_sha256
    assert bundle.source_report_id == handoff.report_id
    assert bundle.source_contract == handoff.schema_version
    assert bundle.hypothesis_count == 1
    assert bundle.atomic_specification_count == 1

    spec = bundle.hypotheses[0].specifications[0]
    hrow = handoff.lineages[0]
    assert bundle.hypotheses[0].hypothesis_id == (
        hrow.downstream_hypothesis_id
    )
    assert spec.prediction_observation_id
    assert spec.falsification_criterion_id
    assert spec.observable


def test_vpost_auto_consumes_bridge_canonical_bundle(
    tmp_path: Path,
) -> None:
    handoff = _canonical_handoff(tmp_path)
    external_path, _ = _shadow(
        tmp_path,
        handoff,
        selection="CONDITIONAL",
    )
    bridge = build_pre_n10_relational_binding_bridge_v1(
        handoff=handoff,
        external_n10_report_path=external_path,
        output_root=tmp_path / "bridge",
    )

    plan = compile_pre_n10_vpost_shadow_plan_v1(
        bridge=bridge,
        provider_plan_path=_provider(tmp_path / "providers.json"),
        model="fixture-model",
        output_root=tmp_path / "vpost",
    )

    assert plan.canonical_spec_bundle_path == (
        bridge.canonical_spec_bundle_path
    )
    assert plan.canonical_spec_bundle_file_sha256 == (
        bridge.canonical_spec_bundle_file_sha256
    )
    row = plan.lineages[0]
    verifier = row.stages[1]
    assert "--canonical-spec-bundle" in verifier.argv
    index = verifier.argv.index("--canonical-spec-bundle")
    assert verifier.argv[index + 1] == bridge.canonical_spec_bundle_path
    sha_index = verifier.argv.index(
        "--canonical-spec-bundle-sha256"
    )
    assert verifier.argv[sha_index + 1] == (
        bridge.canonical_spec_bundle_file_sha256
    )


def test_legacy_bridge_remains_without_canonical_spec_bundle(
    tmp_path: Path,
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
    bridge = build_pre_n10_relational_binding_bridge_v1(
        handoff=handoff,
        external_n10_report_path=external_path,
        output_root=tmp_path / "bridge",
    )

    assert bridge.stable_source_ids_used_for_relational_input is False
    assert (
        bridge.exact_text_reconstruction_required_for_relational_input
        is True
    )
    assert bridge.canonical_spec_bundle_path is None
