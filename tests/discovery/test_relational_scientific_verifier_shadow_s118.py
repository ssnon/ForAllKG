from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    RelationalScientificVerifierRunManifest,
    RelationalVerifierStageRecord,
    assert_fingerprints_unchanged,
    build_relational_scientific_verifier_input_freeze,
    build_relational_scientific_verifier_run_manifest,
    fingerprint_file,
    write_json_exclusive,
)


def _freeze(tmp_path: Path):
    plan = tmp_path / "plan.json"
    endpoint = tmp_path / "endpoint.json"
    provider = tmp_path / "provider.json"
    external = tmp_path / "external.json"
    portfolio = tmp_path / "portfolio.json"

    for index, path in enumerate(
        [plan, endpoint, provider, external, portfolio],
        start=1,
    ):
        path.write_text(
            json.dumps({"index": index}) + "\n",
            encoding="utf-8",
        )

    fingerprints = [
        fingerprint_file(path)
        for path in [plan, endpoint, provider, external, portfolio]
    ]
    freeze = build_relational_scientific_verifier_input_freeze(
        repository_head_sha="a" * 40,
        repository_worktree_dirty=False,
        final_hypothesis_id="hypothesis:final",
        candidate_hypothesis_id="hypothesis:candidate",
        binding_plan_id="relational_atomic_binding_plan:test",
        endpoint_binding_report_id="endpoint:test",
        source_external_novelty_report_id="external:test",
        final_alpha6_portfolio_id="portfolio:test",
        domain_profile_id="sers_au_ag",
        model_name="test-model",
        support_results_per_query=12,
        second_pass_results_per_query=16,
        max_review_works_per_claim=20,
        max_exhaustive_rounds=3,
        max_second_pass_queries_per_claim=8,
        max_resolution_queries=24,
        max_alias_query_variants_per_projection=4,
        max_lower_order_factor_order=1,
        max_source_alias_variants_per_projection=2,
        input_artifacts=fingerprints,
    )
    return freeze, fingerprints


def test_input_freeze_is_deterministic_and_hash_validated(
    tmp_path: Path,
) -> None:
    freeze1, _ = _freeze(tmp_path)
    freeze2, _ = _freeze(tmp_path)

    assert freeze1.freeze_id == freeze2.freeze_id
    assert freeze1.freeze_sha256 == freeze2.freeze_sha256
    assert freeze1.source_population_frozen_before_verifier is True
    assert freeze1.endpoint_binding_frozen_before_verifier is True
    assert freeze1.verifier_result_observed_before_freeze is False


def test_write_json_exclusive_refuses_manifest_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "immutable.json"
    write_json_exclusive(path, {"value": 1})

    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"value": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "value": 1
    }


def test_frozen_input_mutation_is_detected(tmp_path: Path) -> None:
    _, fingerprints = _freeze(tmp_path)
    target = Path(fingerprints[0].path)
    target.write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="frozen verifier input changed during execution",
    ):
        assert_fingerprints_unchanged(fingerprints)


def test_completed_manifest_binds_stage_output_hashes(
    tmp_path: Path,
) -> None:
    freeze, _ = _freeze(tmp_path)
    out = tmp_path / "certification.json"
    out.write_text('{"decision":"UNRESOLVED"}\n', encoding="utf-8")

    stage = RelationalVerifierStageRecord(
        stage_index=1,
        stage_name="scientific_certification_gate",
        argv=["-m", "example"],
        output_artifacts=[fingerprint_file(out)],
    )
    manifest = build_relational_scientific_verifier_run_manifest(
        freeze=freeze,
        stage_records=[stage],
        certification_report_id="scientific_certification_gate:test",
        certification_decision="UNRESOLVED",
        bounded_closure_state="PARTIAL_REVIEW_COVERAGE",
        bounded_external_distinctness_state=(
            "INSUFFICIENT_EXTERNAL_SEARCH_EVIDENCE"
        ),
        positive_nonobviousness_authority_state="NOT_AUTHORIZED",
        fatal_blocker_state="NONE",
    )

    assert manifest.stage_count == 1
    assert manifest.input_artifacts_unchanged_after_execution is True
    assert manifest.manifest_write_once is True
    assert (
        manifest.stage_records[0].output_artifacts[0].sha256
        == fingerprint_file(out).sha256
    )


def test_manifest_rejects_content_mutation(tmp_path: Path) -> None:
    freeze, _ = _freeze(tmp_path)
    out = tmp_path / "certification.json"
    out.write_text("{}\n", encoding="utf-8")
    stage = RelationalVerifierStageRecord(
        stage_index=1,
        stage_name="scientific_certification_gate",
        argv=[],
        output_artifacts=[fingerprint_file(out)],
    )
    manifest = build_relational_scientific_verifier_run_manifest(
        freeze=freeze,
        stage_records=[stage],
        certification_report_id="report:1",
        certification_decision="UNRESOLVED",
        bounded_closure_state="PARTIAL_REVIEW_COVERAGE",
        bounded_external_distinctness_state="INSUFFICIENT",
        positive_nonobviousness_authority_state="NOT_AUTHORIZED",
        fatal_blocker_state="NONE",
    )
    payload = manifest.model_dump(mode="json")
    payload["certification_decision"] = "CERTIFIED"

    with pytest.raises(
        ValidationError,
        match="run manifest SHA mismatch",
    ):
        RelationalScientificVerifierRunManifest.model_validate(payload)
