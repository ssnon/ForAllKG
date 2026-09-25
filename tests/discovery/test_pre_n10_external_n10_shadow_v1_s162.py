from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_core.discovery.pre_n10_downstream_handoff_v1 import (
    build_pre_n10_downstream_handoff_v1,
)
from pipeline_core.discovery.pre_n10_external_n10_shadow_v1 import (
    classify_pre_n10_n10_production_gate_v1,
    compile_pre_n10_external_n10_shadow_plan_v1,
)
from tests.discovery.test_pre_n10_downstream_handoff_v1_s161 import (
    _initial_setup,
    _regenerated_reentry,
    regeneration_context,
)


def _write(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _provider_plan(path: Path) -> Path:
    path.write_text('{"fixture":"provider-plan"}\n', encoding="utf-8")
    return path


def _initial_handoff(tmp_path: Path):
    portfolio_path, routed_path, gate, primary = _initial_setup(
        tmp_path, fallback=False
    )
    return build_pre_n10_downstream_handoff_v1(
        initial_semantic_gate=gate,
        initial_portfolio_path=portfolio_path,
        post_primary_query_plan_path=routed_path,
        primary_router_report=primary,
        output_root=tmp_path / "handoff",
    )


def _regenerated_handoff(tmp_path: Path):
    (
        portfolio_path,
        routed_path,
        gate,
        primary,
        regeneration,
        reentry,
    ) = _regenerated_reentry(tmp_path, semantic_pass=True)
    return build_pre_n10_downstream_handoff_v1(
        initial_semantic_gate=gate,
        initial_portfolio_path=portfolio_path,
        post_primary_query_plan_path=routed_path,
        primary_router_report=primary,
        regeneration_report=regeneration,
        regeneration_reentry_report=reentry,
        output_root=tmp_path / "handoff",
    )


def test_initial_lineage_reuses_frozen_query_plan_and_original_scope(tmp_path: Path):
    handoff = _initial_handoff(tmp_path)
    context = regeneration_context().model_copy(
        update={"domain_profile_id": "sers_au_ag"}
    )
    context_path = tmp_path / "context.json"
    _write(context_path, context)
    provider = _provider_plan(tmp_path / "providers.json")

    plan = compile_pre_n10_external_n10_shadow_plan_v1(
        handoff=handoff,
        context_path=context_path,
        provider_plan_path=provider,
        model="fixture-model",
        output_root=tmp_path / "downstream",
    )

    assert plan.lineage_count == 1
    row = plan.lineages[0]
    assert row.origin == "INITIAL_PRIMARY_READY"
    assert row.expected_n10_authority_scope == "alpha6_original_fallback"
    assert row.n10_production_module == (
        "scripts.discovery.build_nonobviousness_production_gate_v2"
    )
    assert "--reuse-query-plan" in row.stages[0].argv
    reuse_index = row.stages[0].argv.index("--reuse-query-plan")
    assert row.stages[0].argv[reuse_index + 1] == row.query_plan_path
    assert row.stages[2].dynamic_max_ready_claims_from_intake is True


def test_regenerated_lineage_uses_post_generation_scope(tmp_path: Path):
    handoff = _regenerated_handoff(tmp_path)
    context_path = tmp_path / "context.json"
    _write(context_path, regeneration_context())
    provider = _provider_plan(tmp_path / "providers.json")

    plan = compile_pre_n10_external_n10_shadow_plan_v1(
        handoff=handoff,
        context_path=context_path,
        provider_plan_path=provider,
        model="fixture-model",
        output_root=tmp_path / "downstream",
    )

    row = plan.lineages[0]
    assert row.origin == "REGENERATED_REENTRY_READY"
    assert row.expected_n10_authority_scope == (
        "alpha6_post_generation_candidate"
    )
    assert row.n10_production_module == (
        "scripts.discovery.build_nonobviousness_post_generation_production_gate_v2"
    )


def _gate(*, scope: str, selection: str, positive: bool, fallback: bool):
    return {
        "schema_version": "scientific-novelty-fallback-gate-v2",
        "production_authority": True,
        "authority_scope": scope,
        "authority_source": "n10_role_aware_nonobviousness_v2",
        "positive_authority_requires": (
            "ELIGIBLE_AND_ROLE_AWARE_POSITIVE_NONOBVIOUSNESS"
        ),
        "conditional_is_positive": False,
        "absence_is_novelty": False,
        "candidate_semantics_preserved": True,
        "source_portfolio_id": "portfolio:1",
        "source_query_plan_id": "plan:1",
        "gates": [
            {
                "hypothesis_id": "hypothesis:1",
                "selection_class": selection,
                "positive_nonobviousness_authority": positive,
                "fallback_allowed": fallback,
                "action": "fixture",
            }
        ],
    }


def test_common_certification_mapping_preserves_frozen_n10_semantics():
    status, _ = classify_pre_n10_n10_production_gate_v1(
        origin="INITIAL_PRIMARY_READY",
        downstream_hypothesis_id="hypothesis:1",
        portfolio_id="portfolio:1",
        query_plan_id="plan:1",
        gate=_gate(
            scope="alpha6_original_fallback",
            selection="ELIGIBLE",
            positive=True,
            fallback=True,
        ),
    )
    assert status == "NOVELTY_CERTIFIED"

    status, _ = classify_pre_n10_n10_production_gate_v1(
        origin="REGENERATED_REENTRY_READY",
        downstream_hypothesis_id="hypothesis:1",
        portfolio_id="portfolio:1",
        query_plan_id="plan:1",
        gate=_gate(
            scope="alpha6_post_generation_candidate",
            selection="CONDITIONAL",
            positive=False,
            fallback=False,
        ),
    )
    assert status == "NOVELTY_UNRESOLVED"

    status, _ = classify_pre_n10_n10_production_gate_v1(
        origin="REGENERATED_REENTRY_READY",
        downstream_hypothesis_id="hypothesis:1",
        portfolio_id="portfolio:1",
        query_plan_id="plan:1",
        gate=_gate(
            scope="alpha6_post_generation_candidate",
            selection="INELIGIBLE",
            positive=False,
            fallback=False,
        ),
    )
    assert status == "NOVELTY_REJECTED"


def test_n10_authority_scope_cannot_cross_initial_regenerated_boundary():
    with pytest.raises(ValueError, match="scope/origin"):
        classify_pre_n10_n10_production_gate_v1(
            origin="INITIAL_PRIMARY_READY",
            downstream_hypothesis_id="hypothesis:1",
            portfolio_id="portfolio:1",
            query_plan_id="plan:1",
            gate=_gate(
                scope="alpha6_post_generation_candidate",
                selection="ELIGIBLE",
                positive=True,
                fallback=True,
            ),
        )


def test_shadow_plan_rejects_query_plan_mutation_after_handoff(tmp_path: Path):
    handoff = _initial_handoff(tmp_path)
    row = handoff.lineages[0]
    query_path = Path(row.query_plan_path)
    query_path.write_text(query_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    context = regeneration_context().model_copy(
        update={"domain_profile_id": "sers_au_ag"}
    )
    context_path = tmp_path / "context.json"
    _write(context_path, context)
    provider = _provider_plan(tmp_path / "providers.json")

    with pytest.raises(ValueError, match="query plan changed after freeze"):
        compile_pre_n10_external_n10_shadow_plan_v1(
            handoff=handoff,
            context_path=context_path,
            provider_plan_path=provider,
            model="fixture-model",
            output_root=tmp_path / "downstream",
        )
