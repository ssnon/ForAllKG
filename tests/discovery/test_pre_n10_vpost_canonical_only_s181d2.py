from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pipeline_core.discovery.pre_n10_relational_binding_bridge_v1 import (
    build_pre_n10_relational_binding_bridge_v1,
)
from pipeline_core.discovery.pre_n10_vpost_shadow_v1 import (
    compile_pre_n10_vpost_shadow_plan_v1,
)
from scripts.discovery.build_relational_atomic_relation_ir_shadow import (
    main as build_relation_ir_main,
)
from tests.discovery.test_pre_n10_external_n10_shadow_v1_s162 import (
    _initial_handoff,
)
from tests.discovery.test_pre_n10_relational_binding_bridge_v1_s163 import (
    _shadow,
)
from tests.discovery.test_pre_n10_vpost_canonical_bridge_s181d1 import (
    _canonical_handoff,
)
from tests.discovery.test_pre_n10_vpost_shadow_v1_s164 import (
    _provider,
)
from tests.discovery.test_relational_atomic_projection_s107 import (
    _fixture as relational_fixture,
    _write,
)


def test_fresh_canonical_vpost_never_threads_legacy_exact_text_flag(
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

    assert plan.source_binding_mode == "CANONICAL_STABLE_ID"
    assert plan.canonical_spec_bundle_path is not None
    verifier = plan.lineages[0].stages[1]
    assert "--canonical-spec-bundle" in verifier.argv
    assert (
        "--allow-legacy-exact-text-source-binding"
        not in verifier.argv
    )


def test_legacy_bridge_freezes_explicit_exact_text_compatibility(
    tmp_path: Path,
) -> None:
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

    plan = compile_pre_n10_vpost_shadow_plan_v1(
        bridge=bridge,
        provider_plan_path=_provider(tmp_path / "providers.json"),
        model="fixture-model",
        output_root=tmp_path / "vpost",
    )

    assert plan.source_binding_mode == (
        "LEGACY_EXACT_TEXT_COMPATIBILITY"
    )
    assert plan.canonical_spec_bundle_path is None
    verifier = plan.lineages[0].stages[1]
    assert "--canonical-spec-bundle" not in verifier.argv
    assert (
        "--allow-legacy-exact-text-source-binding"
        in verifier.argv
    )


def test_relation_ir_cli_rejects_implicit_legacy_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, endpoint = relational_fixture(tmp_path)
    plan_path = tmp_path / "binding.plan.json"
    endpoint_path = tmp_path / "endpoint.report.json"
    _write(plan_path, plan)
    _write(endpoint_path, endpoint)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_relational_atomic_relation_ir_shadow.py",
            "--plan",
            str(plan_path),
            "--endpoint-report",
            str(endpoint_path),
            "--domain-profile",
            "sers_au_ag",
            "--projection-output",
            str(tmp_path / "projection.json"),
            "--relation-ir-output",
            str(tmp_path / "relation_ir.json"),
        ],
    )

    with pytest.raises(
        ValueError,
        match="unless legacy exact-text compatibility is explicitly enabled",
    ):
        build_relation_ir_main()
