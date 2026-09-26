from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pipeline_core.discovery.atomic_scientific_specification import (
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.atomic_scientific_specification_bundle import (
    build_atomic_scientific_specification_bundle,
)
from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
)
from pipeline_core.discovery.scientific_verifier_shadow_companion import (
    build_scientific_verifier_shadow_companion_plan,
    resolve_scientific_verifier_shadow_inputs,
    validate_scientific_verifier_shadow_lineage,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, object]:
    run = tmp_path / "run"
    run.mkdir()
    context = {
        "schema_version": "hypothesis-context-v1",
        "context_id": "context:1",
        "context_sha256": "sha-context",
        "task_id": "task:1",
        "domain_profile_id": "sers",
    }
    candidate = {
        "schema_version": "production-facing-scientific-candidate-portfolio-v1",
        "portfolio_id": "candidate:1",
        "source_context_id": "context:1",
        "source_context_sha256": "sha-context",
        "source_task_id": "task:1",
        "domain_profile_id": "sers",
    }
    atomic_report = {
        "schema_version": "atomic-cross-lane-scientific-synthesis-report-v1",
        "report_id": "atomic-report:1",
        "source_candidate_portfolio_id": "candidate:1",
        "source_context_id": "context:1",
        "source_task_id": "task:1",
        "hypotheses": [
            {
                "hypothesis_id": "hypothesis:1",
                "atomic_specifications": [
                    {"claim_id": "claim:1"},
                    {"claim_id": "claim:2"},
                ],
            }
        ],
    }
    atomic_portfolio = {
        "schema_version": "hypothesis-portfolio-v1",
        "portfolio_id": "atomic-portfolio:1",
        "source_context_id": "context:1",
        "source_context_sha256": "sha-context",
        "domain_profile_id": "sers",
        "hypotheses": [{"hypothesis_id": "hypothesis:1"}],
    }
    external = {
        "schema_version": "external-novelty-report-v1",
        "report_id": "external:1",
        "source_portfolio_id": "atomic-portfolio:1",
        "source_prior_art_packet_id": "packet:1",
        "cards": [{"hypothesis_id": "hypothesis:1"}],
    }

    _write(run / "hypothesis.context.json", context)
    _write(run / "scientific_pre_n10_candidate_portfolio.json", candidate)
    _write(run / "scientific_atomic_cross_lane.report.json", atomic_report)
    _write(run / "scientific_atomic_cross_lane.portfolio.json", atomic_portfolio)
    _write(run / "scientific_atomic_n10_external.report.json", external)
    _write(run / "scientific_atomic_n10_external.provider_plan.json", {"schema_version": "provider-plan-v1"})
    _write(
        run / "scientific_atomic_n10_e2e_manifest.json",
        {
            "schema_version": "scientific-atomic-n10-e2e-manifest-v1",
            "status": "complete",
            "source_portfolio": str((run / "scientific_atomic_cross_lane.portfolio.json").resolve()),
            "source_hypothesis_count": 1,
            "precomputed_atomic_query_plan_reused": True,
            "external_novelty_llm_redecomposition_performed": False,
            "authority_mode": "certification_only",
        },
    )
    inputs = resolve_scientific_verifier_shadow_inputs(run_dir=run)
    return run, inputs


def test_exact_completed_atomic_lineage_is_accepted(tmp_path: Path) -> None:
    run, inputs = _fixture(tmp_path)
    lineage = validate_scientific_verifier_shadow_lineage(inputs)
    assert lineage.atomic_portfolio_id == "atomic-portfolio:1"
    assert lineage.atomic_hypothesis_ids == ["hypothesis:1"]
    assert lineage.atomic_claim_ids == ["claim:1", "claim:2"]
    assert lineage.atomic_n10_authority_mode == "certification_only"
    assert lineage.production_selection_changed is False

    plan = build_scientific_verifier_shadow_companion_plan(inputs=inputs)
    assert plan.stage_count == 15
    assert plan.stage_names[0] == "grounded_identity_annotation"
    assert plan.stage_names[-1] == "scientific_certification_gate"
    assert plan.outputs.output_dir == str((run / "scientific_verifier_shadow").resolve())
    assert plan.semantic_second_pass_precedes_final_adjudication is True


def test_candidate_to_atomic_lineage_mismatch_fails_closed(tmp_path: Path) -> None:
    run, inputs = _fixture(tmp_path)
    path = run / "scientific_atomic_cross_lane.report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_candidate_portfolio_id"] = "candidate:stale"
    _write(path, payload)
    with pytest.raises(ValueError, match="source_candidate_portfolio_id mismatch"):
        validate_scientific_verifier_shadow_lineage(inputs)


def test_external_hypothesis_set_mismatch_fails_closed(tmp_path: Path) -> None:
    run, inputs = _fixture(tmp_path)
    path = run / "scientific_atomic_n10_external.report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cards"] = [{"hypothesis_id": "hypothesis:stale"}]
    _write(path, payload)
    with pytest.raises(ValueError, match="hypothesis ID set mismatch"):
        validate_scientific_verifier_shadow_lineage(inputs)


def test_atomic_n10_must_be_complete_and_exact_plan_reuse(tmp_path: Path) -> None:
    run, inputs = _fixture(tmp_path)
    path = run / "scientific_atomic_n10_e2e_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "running"
    _write(path, payload)
    with pytest.raises(ValueError, match="requires completed atomic N10 E2E"):
        validate_scientific_verifier_shadow_lineage(inputs)

    legacy = validate_scientific_verifier_shadow_lineage(
        inputs,
        allow_legacy_atomic_n10_artifacts=True,
    )
    assert legacy.atomic_n10_completion_verified is False
    assert legacy.legacy_atomic_n10_artifact_mode is True
    assert legacy.completed_atomic_n10_required is False
    assert legacy.manifest_source_portfolio_binding_verified is True
    assert legacy.manifest_source_hypothesis_count_verified is True
    assert legacy.prospective_validation_input_contract_satisfied is False

    payload["status"] = "complete"
    payload["precomputed_atomic_query_plan_reused"] = False
    _write(path, payload)
    with pytest.raises(ValueError, match="precomputed atomic query plan"):
        validate_scientific_verifier_shadow_lineage(inputs)


def test_n10_closure_detail_root_is_not_required(tmp_path: Path) -> None:
    run, inputs = _fixture(tmp_path)
    assert not (run / "scientific_atomic_n10_full.details").exists()
    lineage = validate_scientific_verifier_shadow_lineage(inputs)
    assert lineage.grounded_identity_annotation_independent_of_n10_closure is True


def test_custom_output_directory_is_isolated_from_run(tmp_path: Path) -> None:
    _, inputs = _fixture(tmp_path)
    output = tmp_path / "verifier-output"
    plan = build_scientific_verifier_shadow_companion_plan(
        inputs=inputs,
        output_dir=output,
    )
    assert plan.outputs.output_dir == str(output.resolve())
    assert Path(plan.outputs.certification_report).parent == output.resolve()
    assert plan.verifier_result_consumed_by_production is False
    assert plan.canonical_graph_mutated is False


def test_dry_run_materializes_frozen_stage_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = _fixture(tmp_path)
    bundle = build_atomic_scientific_specification_bundle(
        source_report_id="atomic-report:1",
        source_contract=(
            "atomic-cross-lane-scientific-synthesis-report-v1"
        ),
        hypotheses=[
            (
                "hypothesis:1",
                [
                    _bundle_spec("claim:1", "1"),
                    _bundle_spec("claim:2", "2"),
                ],
            )
        ],
    )
    bundle_path = run / "canonical.bundle.json"
    _write(bundle_path, bundle)

    from scripts.discovery.run_scientific_verifier_shadow_companion import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scientific_verifier_shadow_companion",
            "--run-dir",
            str(run),
            "--canonical-spec-bundle",
            str(bundle_path),
            "--model",
            "test-model",
            "--dry-run",
        ],
    )
    assert main() == 0
    manifest_path = (
        run
        / "scientific_verifier_shadow"
        / "scientific_verifier_shadow_e2e_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "planned"
    assert manifest["execution_performed"] is False
    assert len(manifest["planned_stages"]) == 15
    names = [row["name"] for row in manifest["planned_stages"]]
    assert names.index("semantic_second_pass_resolution") < names.index(
        "exhaustive_relation_adjudication"
    )
    grounded = manifest["planned_stages"][0]["argv"]
    relation_ir_stage = manifest["planned_stages"][1]["argv"]
    assert "--canonical-spec-bundle" in relation_ir_stage
    assert (
        manifest["canonical_spec_bundle_is_relation_ir_authority"]
        is True
    )
    assert (
        manifest["atomic_report_relation_ir_compatibility_mode"]
        is False
    )
    assert "--annotation-only" in grounded
    assert "--detail-root" not in grounded
    assert manifest["grounded_identity_annotation_independent_of_n10_closure"] is True
    assert manifest["verifier_result_consumed_by_production"] is False



def test_legacy_mode_autoresolves_exact_external_atomic_lineage_and_ignores_stale_manifest_binding(
    tmp_path: Path,
) -> None:
    run, _ = _fixture(tmp_path)

    old_portfolio_path = run / "scientific_atomic_cross_lane.portfolio.json"
    old_report_path = run / "scientific_atomic_cross_lane.report.json"
    old_portfolio = json.loads(old_portfolio_path.read_text(encoding="utf-8"))
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))

    current_portfolio = dict(old_portfolio)
    current_portfolio["portfolio_id"] = "atomic-portfolio:v2"
    _write(run / "scientific_atomic_cross_lane_v2.portfolio.json", current_portfolio)

    current_report = dict(old_report)
    current_report["report_id"] = "atomic-report:v2"
    _write(run / "scientific_atomic_cross_lane_v2.report.json", current_report)

    external_path = run / "scientific_atomic_n10_external.report.json"
    external = json.loads(external_path.read_text(encoding="utf-8"))
    external["source_portfolio_id"] = "atomic-portfolio:v2"
    _write(external_path, external)

    manifest_path = run / "scientific_atomic_n10_e2e_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "failed"
    # Keep the historical manifest deliberately bound to the old portfolio.
    manifest["source_portfolio"] = str(old_portfolio_path.resolve())
    _write(manifest_path, manifest)

    inputs = resolve_scientific_verifier_shadow_inputs(
        run_dir=run,
        allow_legacy_atomic_n10_artifacts=True,
    )
    assert Path(inputs.atomic_portfolio).name == (
        "scientific_atomic_cross_lane_v2.portfolio.json"
    )
    assert Path(inputs.atomic_report).name == (
        "scientific_atomic_cross_lane_v2.report.json"
    )

    lineage = validate_scientific_verifier_shadow_lineage(
        inputs,
        allow_legacy_atomic_n10_artifacts=True,
    )
    assert lineage.atomic_portfolio_id == "atomic-portfolio:v2"
    assert lineage.atomic_report_id == "atomic-report:v2"
    assert lineage.manifest_source_portfolio_binding_verified is False
    assert lineage.manifest_source_hypothesis_count_verified is True
    assert lineage.prospective_validation_input_contract_satisfied is False


def test_dry_run_legacy_artifact_mode_is_explicitly_nonprospective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, _ = _fixture(tmp_path)
    manifest_path = run / "scientific_atomic_n10_e2e_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    _write(manifest_path, payload)

    from scripts.discovery.run_scientific_verifier_shadow_companion import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scientific_verifier_shadow_companion",
            "--run-dir",
            str(run),
            "--model",
            "test-model",
            "--allow-legacy-atomic-n10-artifacts",
            "--allow-atomic-report-compatibility",
            "--dry-run",
        ],
    )
    assert main() == 0
    verifier_manifest = json.loads(
        (
            run
            / "scientific_verifier_shadow"
            / "scientific_verifier_shadow_e2e_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert verifier_manifest["atomic_n10_completion_verified"] is False
    assert verifier_manifest["legacy_atomic_n10_artifact_mode"] is True
    assert verifier_manifest["legacy_atomic_artifact_autoresolution_used"] is True
    assert verifier_manifest["legacy_mode_is_historical_smoke_only"] is True
    assert (
        verifier_manifest["atomic_report_relation_ir_compatibility_mode"]
        is True
    )
    assert (
        verifier_manifest[
            "atomic_report_relation_ir_compatibility_explicitly_allowed"
        ]
        is True
    )
    assert (
        verifier_manifest["prospective_validation_input_contract_satisfied"]
        is False
    )
    assert verifier_manifest["verifier_result_consumed_by_production"] is False

def _bundle_spec(claim_id: str, suffix: str) -> CompiledAtomicSpecification:
    return CompiledAtomicSpecification(
        local_id="atomic:" + suffix,
        claim_id=claim_id,
        kind="mechanistic_link",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="Factor X affects response " + suffix + ".",
        rationale="fixture",
        source_candidate_ids=["candidate:1"],
        premise_statement_ids=["statement:1"],
        gap_statement_ids=[],
        prior_art_identity_terms=["Factor X"],
        relation_endpoint_anchors=[
            "Factor X",
            "response " + suffix,
        ],
        scope_qualifier_spans=[],
        directional_qualifier_spans=[],
        relation_nucleus_terms=[
            "Factor X",
            "response " + suffix,
        ],
        distinguishing_terms=[],
        required_bridge="Factor X affects response " + suffix + ".",
        observable="response " + suffix,
        predicted_observation="response changes " + suffix,
        falsification_condition="response does not change " + suffix,
        prediction_observation_id="prediction:" + suffix,
        falsification_criterion_id="falsifier:" + suffix,
        search_concepts=["Factor X", "response " + suffix],
        search_queries=["Factor X response " + suffix],
        scientific_structure=NoveltyClaimScientificStructure(),
        scientific_structure_reason_codes=[],
    )


def test_companion_verifies_explicit_canonical_bundle_lineage(
    tmp_path: Path,
) -> None:
    run, _ = _fixture(tmp_path)
    bundle = build_atomic_scientific_specification_bundle(
        source_report_id="atomic-report:1",
        source_contract=(
            "atomic-cross-lane-scientific-synthesis-report-v1"
        ),
        hypotheses=[
            (
                "hypothesis:1",
                [
                    _bundle_spec("claim:1", "1"),
                    _bundle_spec("claim:2", "2"),
                ],
            )
        ],
    )
    bundle_path = run / "canonical.bundle.json"
    _write(bundle_path, bundle)

    inputs = resolve_scientific_verifier_shadow_inputs(
        run_dir=run,
        canonical_spec_bundle=bundle_path,
    )
    lineage = validate_scientific_verifier_shadow_lineage(inputs)

    assert lineage.canonical_spec_bundle_id == bundle.bundle_id
    assert (
        lineage.canonical_spec_bundle_sha256
        == bundle.bundle_sha256
    )
    assert lineage.canonical_spec_bundle_lineage_verified is True
    assert lineage.canonical_spec_bundle_is_relation_ir_authority is True


def test_companion_rejects_canonical_bundle_claim_population_drift(
    tmp_path: Path,
) -> None:
    run, _ = _fixture(tmp_path)
    bundle = build_atomic_scientific_specification_bundle(
        source_report_id="atomic-report:1",
        source_contract=(
            "atomic-cross-lane-scientific-synthesis-report-v1"
        ),
        hypotheses=[
            (
                "hypothesis:1",
                [_bundle_spec("claim:1", "1")],
            )
        ],
    )
    bundle_path = run / "canonical.bundle.json"
    _write(bundle_path, bundle)

    inputs = resolve_scientific_verifier_shadow_inputs(
        run_dir=run,
        canonical_spec_bundle=bundle_path,
    )
    with pytest.raises(
        ValueError,
        match="canonical bundle/atomic claim ID set mismatch",
    ):
        validate_scientific_verifier_shadow_lineage(inputs)

def test_fresh_companion_rejects_missing_canonical_bundle_without_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, _ = _fixture(tmp_path)
    from scripts.discovery.run_scientific_verifier_shadow_companion import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scientific_verifier_shadow_companion",
            "--run-dir",
            str(run),
            "--model",
            "test-model",
            "--dry-run",
        ],
    )
    with pytest.raises(
        ValueError,
        match="fresh scientific verifier requires --canonical-spec-bundle",
    ):
        main()


def test_completed_historical_companion_can_explicitly_opt_into_report_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, _ = _fixture(tmp_path)
    from scripts.discovery.run_scientific_verifier_shadow_companion import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scientific_verifier_shadow_companion",
            "--run-dir",
            str(run),
            "--model",
            "test-model",
            "--allow-atomic-report-compatibility",
            "--dry-run",
        ],
    )
    assert main() == 0

    manifest = json.loads(
        (
            run
            / "scientific_verifier_shadow"
            / "scientific_verifier_shadow_e2e_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["atomic_n10_completion_verified"] is True
    assert manifest["atomic_report_relation_ir_compatibility_mode"] is True
    assert (
        manifest[
            "atomic_report_relation_ir_compatibility_explicitly_allowed"
        ]
        is True
    )
    relation_ir_stage = manifest["planned_stages"][1]["argv"]
    assert "--allow-atomic-report-compatibility" in relation_ir_stage
    assert "--canonical-spec-bundle" not in relation_ir_stage
