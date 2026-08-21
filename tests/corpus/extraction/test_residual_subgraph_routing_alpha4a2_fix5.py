from __future__ import annotations

from domains.dac_her.micro_reextract_prompts import build_micro_reextract_prompt
from pipeline_core.corpus.recovery_policy import (
    RecoveryAction,
    decide_recovery,
    has_coupled_claim_subgraph_residual,
    has_dominant_undefined_endpoint_cluster,
)
from pipeline_core.runtime.validation_issues import (
    IssueCode,
    IssueStage,
    ValidationReport,
    issue,
)


def _decision(
    report: ValidationReport,
    *,
    micro_attempts: int,
    post_patch_attempts: int = 0,
):
    return decide_recovery(
        report=report,
        normalization_attempted=True,
        patch_attempts=2,
        micro_reextract_attempts=micro_attempts,
        post_micro_patch_attempts=post_patch_attempts,
        split_depth=3,
        source_tokens=260,
        max_patch_attempts=2,
        max_micro_reextract_attempts=1,
        max_post_micro_patch_attempts=1,
        micro_reextract_max_source_tokens=900,
        max_split_depth=3,
        min_rechunk_source_tokens=400,
        isolated_rechunk_threshold=3,
        issue_family_rechunk_threshold=3,
        undefined_endpoint_rechunk_threshold=4,
    )


def _dominant_missing_node_report() -> ValidationReport:
    missing = "seed_mediated_ag_growth"
    issues = [
        issue(
            code=IssueCode.UNDEFINED_EDGE_TARGET,
            stage=IssueStage.STRUCTURAL,
            message="missing target",
            edge_index=7,
            source_id="substrate_sio2_au_ag",
            target_id=missing,
            relation="PREPARED_BY",
        ),
        issue(
            code=IssueCode.UNDEFINED_EDGE_SOURCE,
            stage=IssueStage.STRUCTURAL,
            message="missing source",
            edge_index=8,
            source_id=missing,
            target_id="gold_iii_chloride",
            relation="USES_PRECURSOR",
        ),
        issue(
            code=IssueCode.UNDEFINED_EDGE_SOURCE,
            stage=IssueStage.STRUCTURAL,
            message="missing source",
            edge_index=9,
            source_id=missing,
            target_id="thpc",
            relation="USES_MATERIAL",
        ),
        issue(
            code=IssueCode.MISSING_MEASUREMENT_PRODUCER,
            stage=IssueStage.MEASUREMENT,
            message="measurement producer missing",
            node_id="measurement_silica_core_size",
            node_collection="measurements",
        ),
        issue(
            code=IssueCode.MISSING_MEASUREMENT_PRODUCER,
            stage=IssueStage.MEASUREMENT,
            message="measurement producer missing",
            node_id="measurement_au_seed_size",
            node_collection="measurements",
        ),
    ]
    return ValidationReport.from_issues(issues)


def _claim_subgraph_report() -> ValidationReport:
    return ValidationReport.from_issues([
        issue(
            code=IssueCode.CLAIM_MISSING_APPLICATION_TARGET,
            stage=IssueStage.CLAIM,
            message="missing APPLIES_TO",
            node_id="observation_sers_components_importance",
            node_collection="observation_claims",
        ),
        issue(
            code=IssueCode.MECHANISM_MISSING_SUPPORT,
            stage=IssueStage.CLAIM,
            message="mechanism has no support",
            node_id="mechanism_sers_em_lspr",
            node_collection="mechanism_claims",
        ),
        issue(
            code=IssueCode.RELATION_SOURCE_TYPE_MISMATCH,
            stage=IssueStage.RELATION,
            message="bad SUPPORTS_CLAIM source",
            edge_index=8,
            source_id="substrate",
            target_id="observation_sers_components_importance",
            relation="SUPPORTS_CLAIM",
        ),
        issue(
            code=IssueCode.RELATION_TARGET_TYPE_MISMATCH,
            stage=IssueStage.RELATION,
            message="bad INTERPRETED_AS target",
            edge_index=10,
            source_id="mechanism_sers_em_lspr",
            target_id="observation_sers_components_importance",
            relation="INTERPRETED_AS",
        ),
    ])


def test_dominant_missing_node_cluster_survives_mixed_residual():
    report = _dominant_missing_node_report()
    assert has_dominant_undefined_endpoint_cluster(report)


def test_single_dangling_edge_is_not_dominant_cluster():
    report = ValidationReport.from_issues([
        issue(
            code=IssueCode.UNDEFINED_EDGE_SOURCE,
            stage=IssueStage.STRUCTURAL,
            message="missing source",
            edge_index=1,
            source_id="x",
            target_id="y",
            relation="USES_MATERIAL",
        ),
        issue(
            code=IssueCode.MISSING_MEASUREMENT_PRODUCER,
            stage=IssueStage.MEASUREMENT,
            message="missing producer",
            node_id="m1",
            node_collection="measurements",
        ),
        issue(
            code=IssueCode.MISSING_MEASUREMENT_PRODUCER,
            stage=IssueStage.MEASUREMENT,
            message="missing producer",
            node_id="m2",
            node_collection="measurements",
        ),
    ])
    assert not has_dominant_undefined_endpoint_cluster(report)


def test_dominant_missing_node_routes_to_second_micro():
    report = _dominant_missing_node_report()
    decision = _decision(report, micro_attempts=1)
    assert decision.action == RecoveryAction.MICRO_REEXTRACT
    assert "dominant missing-node" in decision.reason.lower()


def test_dominant_missing_node_retry_is_bounded():
    report = _dominant_missing_node_report()
    decision = _decision(report, micro_attempts=2)
    assert decision.action == RecoveryAction.SEMANTIC_PATCH


def test_coupled_claim_subgraph_is_detected():
    assert has_coupled_claim_subgraph_residual(_claim_subgraph_report())


def test_relation_mismatch_alone_is_not_claim_subgraph():
    report = ValidationReport.from_issues([
        issue(
            code=IssueCode.RELATION_SOURCE_TYPE_MISMATCH,
            stage=IssueStage.RELATION,
            message="generic mismatch",
            edge_index=0,
            source_id="x",
            target_id="y",
            relation="HAS_COMPONENT",
        )
    ])
    assert not has_coupled_claim_subgraph_residual(report)


def test_claim_subgraph_routes_to_second_micro():
    report = _claim_subgraph_report()
    decision = _decision(report, micro_attempts=1)
    assert decision.action == RecoveryAction.MICRO_REEXTRACT
    assert "coupled claim subgraph" in decision.reason.lower()


def test_claim_subgraph_retry_is_bounded():
    report = _claim_subgraph_report()
    decision = _decision(report, micro_attempts=2)
    assert decision.action == RecoveryAction.SEMANTIC_PATCH


def test_micro_prompt_contains_claim_subgraph_rebuild_hint():
    report = _claim_subgraph_report()
    prompt = build_micro_reextract_prompt(
        paper_id="P",
        chunk_id="P:main:c",
        document_id="main",
        document_role="main",
        section="whole",
        page_ids=[],
        asset_ids=[],
        core_text="SERS performance and electromagnetic interpretation were discussed.",
        left_context="",
        right_context="",
        asset_context="",
        graph_payload={
            "entities": [],
            "experiments": [],
            "calculations": [],
            "measurements": [],
            "measurement_groups": [],
            "observation_claims": [],
            "mechanism_claims": [],
            "edges": [],
        },
        report=report,
    )

    assert "COUPLED_CLAIM_SUBGRAPH_RECOVERY_HINTS" in prompt
    assert "SUPPORTS_CLAIM" in prompt
    assert "INTERPRETED_AS" in prompt
    assert "APPLIES_TO" in prompt
    assert "rebuild" in prompt.lower()
