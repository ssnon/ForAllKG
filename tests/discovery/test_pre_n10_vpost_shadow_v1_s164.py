from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_core.discovery.pre_n10_relational_binding_bridge_v1 import (
    build_pre_n10_relational_binding_bridge_v1,
)
from pipeline_core.discovery.pre_n10_vpost_shadow_v1 import (
    build_pre_n10_vpost_lineage_result_v1,
    build_pre_n10_vpost_shadow_report_v1,
    compile_pre_n10_vpost_shadow_plan_v1,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_endpoint_binding import (
    LiteralEndpointBindingBatchDraft,
    LiteralEndpointBindingDraft,
    compile_endpoint_bindings,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    RelationalVerifierStageRecord,
    build_relational_scientific_verifier_input_freeze,
    build_relational_scientific_verifier_run_manifest,
    fingerprint_file,
)
from tests.discovery.test_pre_n10_external_n10_shadow_v1_s162 import (
    _initial_handoff,
    _regenerated_handoff,
)
from tests.discovery.test_pre_n10_relational_binding_bridge_v1_s163 import (
    _shadow,
)


_VERIFIER_STAGES = (
    "relational_atomic_projection_and_relation_ir",
    "relational_atomic_identity_and_factor_projection",
    "supporting_projection_retrieval",
    "counterevidence_projection_retrieval",
    "semantic_second_pass_resolution",
    "exhaustive_relation_adjudication",
    "claim_evidence_graph",
    "structural_claim_centrality",
    "hypothesis_evidence_aggregation",
    "positive_nonobviousness_basis",
    "positive_nonobviousness_adjudication",
    "external_novelty_lineage_projection",
    "scientific_certification_gate",
)


def _write(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _provider(path: Path) -> Path:
    path.write_text('{"fixture":"provider"}\n', encoding="utf-8")
    return path


def _bridge(tmp_path: Path, *, regenerated: bool, selection: str):
    handoff = (
        _regenerated_handoff(tmp_path)
        if regenerated
        else _initial_handoff(tmp_path)
    )
    external_report_path, _ = _shadow(
        tmp_path,
        handoff,
        selection=selection,
    )
    bridge = build_pre_n10_relational_binding_bridge_v1(
        handoff=handoff,
        external_n10_report_path=external_report_path,
        output_root=tmp_path / "bridge",
    )
    return bridge


@pytest.mark.parametrize("selection", ["ELIGIBLE", "CONDITIONAL", "INELIGIBLE"])
def test_n10_status_does_not_filter_vpost_execution_plan(
    tmp_path: Path,
    selection: str,
) -> None:
    bridge = _bridge(tmp_path, regenerated=False, selection=selection)
    plan = compile_pre_n10_vpost_shadow_plan_v1(
        bridge=bridge,
        provider_plan_path=_provider(tmp_path / "providers.json"),
        model="fixture-model",
        output_root=tmp_path / "vpost",
    )

    assert plan.execution_required_lineage_count == 1
    row = plan.lineages[0]
    assert row.execution_required is True
    assert [stage.stage for stage in row.stages] == [
        "literal_endpoint_binding",
        "relational_scientific_verifier",
    ]
    assert row.n10_status_filters_vpost_reachability is False
    assert "--source-external-report" in row.stages[1].argv
    assert row.source_external_report_path in row.stages[1].argv
    assert "--final-hypothesis-id" in row.stages[1].argv
    assert row.downstream_hypothesis_id in row.stages[1].argv


def test_regenerated_vpost_plan_preserves_exact_downstream_identity(
    tmp_path: Path,
) -> None:
    bridge = _bridge(tmp_path, regenerated=True, selection="CONDITIONAL")
    plan = compile_pre_n10_vpost_shadow_plan_v1(
        bridge=bridge,
        provider_plan_path=_provider(tmp_path / "providers.json"),
        model="fixture-model",
        output_root=tmp_path / "vpost",
        save_prompts=True,
    )
    row = plan.lineages[0]
    binding = RelationalAtomicBindingPlan.model_validate_json(
        Path(row.binding_plan_path).read_text(encoding="utf-8")
    )
    hypothesis = binding.hypotheses[0]
    assert hypothesis.candidate_hypothesis_id == row.downstream_hypothesis_id
    assert hypothesis.final_hypothesis_id == row.downstream_hypothesis_id
    assert row.endpoint_prompt_path is not None
    assert "--save-prompts" in row.stages[1].argv


def _materialize_completed_outputs(plan_row) -> None:
    binding = RelationalAtomicBindingPlan.model_validate_json(
        Path(plan_row.binding_plan_path).read_text(encoding="utf-8")
    )
    claim = binding.hypotheses[0].claims[0]
    # Endpoint annotations must project the complete branch identity away.
    # The fixture identity is "spacing disorder", so use the literal
    # non-identity scientific spans that remain in claim/required_bridge.
    endpoints = [
        "disorder",
        "spatial SERS intensity variance",
    ]
    endpoint = compile_endpoint_bindings(
        plan=binding,
        draft=LiteralEndpointBindingBatchDraft(
            bindings=[
                LiteralEndpointBindingDraft(
                    claim_id=claim.claim_id,
                    relation_endpoint_anchors=endpoints,
                )
            ]
        ),
        backend_name="fixture",
        model_name="fixture-model",
        llm_calls_performed=1,
    )
    endpoint_path = Path(plan_row.endpoint_report_path)
    _write(endpoint_path, endpoint)

    source_portfolio_path = Path(binding.source_alpha6_candidate_portfolio)
    provider_path = Path(
        plan_row.stages[1].argv[
            plan_row.stages[1].argv.index("--provider-plan") + 1
        ]
    )
    external_path = Path(plan_row.source_external_report_path)
    freeze = build_relational_scientific_verifier_input_freeze(
        repository_head_sha="1" * 40,
        repository_worktree_dirty=False,
        final_hypothesis_id=plan_row.downstream_hypothesis_id,
        candidate_hypothesis_id=plan_row.downstream_hypothesis_id,
        binding_plan_id=binding.plan_id,
        endpoint_binding_report_id=endpoint.report_id,
        source_external_novelty_report_id="external_novelty_report:fixture",
        final_alpha6_portfolio_id=(
            json.loads(source_portfolio_path.read_text())["portfolio_id"]
        ),
        domain_profile_id=str(plan_row.domain_profile_id),
        model_name="fixture-model",
        support_results_per_query=12,
        second_pass_results_per_query=16,
        max_review_works_per_claim=20,
        max_exhaustive_rounds=3,
        max_second_pass_queries_per_claim=8,
        max_resolution_queries=24,
        max_alias_query_variants_per_projection=4,
        max_lower_order_factor_order=1,
        max_source_alias_variants_per_projection=2,
        input_artifacts=[
            fingerprint_file(Path(plan_row.binding_plan_path)),
            fingerprint_file(endpoint_path),
            fingerprint_file(provider_path),
            fingerprint_file(external_path),
            fingerprint_file(source_portfolio_path),
        ],
    )
    _write(Path(plan_row.verifier_input_freeze_path), freeze)

    stage_records = [
        RelationalVerifierStageRecord(
            stage_index=index,
            stage_name=name,
            argv=["-m", "fixture"],
            output_artifacts=[],
        )
        for index, name in enumerate(_VERIFIER_STAGES, start=1)
    ]
    manifest = build_relational_scientific_verifier_run_manifest(
        freeze=freeze,
        stage_records=stage_records,
        certification_report_id="scientific_certification_gate:fixture",
        certification_decision="UNRESOLVED",
        bounded_closure_state="BOUNDED_CLOSURE_COMPLETE",
        bounded_external_distinctness_state="UNRESOLVED",
        positive_nonobviousness_authority_state="NOT_AUTHORIZED",
        fatal_blocker_state="NO_FATAL_BLOCKER",
    )
    _write(Path(plan_row.verifier_manifest_path), manifest)
    _write(
        Path(plan_row.verifier_certification_path),
        {
            "report_id": "scientific_certification_gate:fixture",
            "decisions": [
                {
                    "hypothesis_id": plan_row.downstream_hypothesis_id,
                    "decision": "UNRESOLVED",
                    "bounded_closure_state": "BOUNDED_CLOSURE_COMPLETE",
                    "bounded_external_distinctness_state": "UNRESOLVED",
                    "positive_nonobviousness_authority_state": "NOT_AUTHORIZED",
                    "fatal_blocker_state": "NO_FATAL_BLOCKER",
                }
            ],
        },
    )


def test_completed_vpost_result_requires_endpoint_freeze_manifest_and_certification(
    tmp_path: Path,
) -> None:
    bridge = _bridge(tmp_path, regenerated=False, selection="CONDITIONAL")
    execution = compile_pre_n10_vpost_shadow_plan_v1(
        bridge=bridge,
        provider_plan_path=_provider(tmp_path / "providers.json"),
        model="fixture-model",
        output_root=tmp_path / "vpost",
    )
    row_plan = execution.lineages[0]
    _materialize_completed_outputs(row_plan)

    result = build_pre_n10_vpost_lineage_result_v1(plan=row_plan)
    assert result.status == "VPOST_COMPLETED"
    assert result.endpoint_llm_calls == 1
    assert result.certification_decision == "UNRESOLVED"
    assert result.endpoint_binding_stage_invocations == 1
    assert result.relational_verifier_stage_invocations == 1

    report = build_pre_n10_vpost_shadow_report_v1(
        execution_plan=execution,
        bridge=bridge,
        lineages=[result],
    )
    assert report.completed_count == 1
    assert report.certification_decision_counts == {"UNRESOLVED": 1}
    assert report.external_novelty_reassessed is False
    assert report.verifier_result_consumed_by_production is False


def test_vpost_result_rejects_incomplete_verifier_stage_contract(
    tmp_path: Path,
) -> None:
    bridge = _bridge(tmp_path, regenerated=False, selection="CONDITIONAL")
    execution = compile_pre_n10_vpost_shadow_plan_v1(
        bridge=bridge,
        provider_plan_path=_provider(tmp_path / "providers.json"),
        model="fixture-model",
        output_root=tmp_path / "vpost",
    )
    row_plan = execution.lineages[0]
    _materialize_completed_outputs(row_plan)

    manifest_path = Path(row_plan.verifier_manifest_path)
    payload = json.loads(manifest_path.read_text())
    payload["stage_records"] = payload["stage_records"][:-1]
    payload["stage_count"] = len(payload["stage_records"])
    # Rebuild a valid manifest with an intentionally incomplete stage list.
    freeze = json.loads(Path(row_plan.verifier_input_freeze_path).read_text())
    from pipeline_core.discovery.relational_scientific_verifier_shadow import (
        RelationalScientificVerifierInputFreeze,
    )
    freeze_model = RelationalScientificVerifierInputFreeze.model_validate(freeze)
    incomplete_records = [
        RelationalVerifierStageRecord.model_validate(row)
        for row in payload["stage_records"]
    ]
    incomplete = build_relational_scientific_verifier_run_manifest(
        freeze=freeze_model,
        stage_records=incomplete_records,
        certification_report_id="scientific_certification_gate:fixture",
        certification_decision="UNRESOLVED",
        bounded_closure_state="BOUNDED_CLOSURE_COMPLETE",
        bounded_external_distinctness_state="UNRESOLVED",
        positive_nonobviousness_authority_state="NOT_AUTHORIZED",
        fatal_blocker_state="NO_FATAL_BLOCKER",
    )
    _write(manifest_path, incomplete)

    with pytest.raises(ValueError, match="13-stage"):
        build_pre_n10_vpost_lineage_result_v1(plan=row_plan)


def test_vpost_plan_rejects_binding_plan_mutation_after_bridge(
    tmp_path: Path,
) -> None:
    bridge = _bridge(tmp_path, regenerated=False, selection="CONDITIONAL")
    binding_path = Path(bridge.lineages[0].binding_plan_path)
    binding_path.write_text(binding_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after bridge"):
        compile_pre_n10_vpost_shadow_plan_v1(
            bridge=bridge,
            provider_plan_path=_provider(tmp_path / "providers.json"),
            model="fixture-model",
            output_root=tmp_path / "vpost",
        )
