from __future__ import annotations

from dac_her.micro_reextract_prompts import (
    build_micro_reextract_prompt,
)
from dac_her.recovery_policy import (
    RecoveryAction,
    decide_recovery,
    has_common_undefined_endpoint_cluster,
    requires_complete_reextract,
)
from dac_her.semantic_patch_prompts import (
    build_patch_rejection_feedback,
)
from dac_her.validation_issues import (
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
        source_tokens=200,
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


def _mixed_direction_missing_node_report() -> ValidationReport:
    missing = "synthesis_atp_modification"
    return ValidationReport.from_issues([
        issue(
            code=IssueCode.UNDEFINED_EDGE_TARGET,
            stage=IssueStage.STRUCTURAL,
            message="missing target",
            edge_index=4,
            source_id="substrate",
            target_id=missing,
            relation="PREPARED_BY",
        ),
        issue(
            code=IssueCode.UNDEFINED_EDGE_SOURCE,
            stage=IssueStage.STRUCTURAL,
            message="missing source",
            edge_index=5,
            source_id=missing,
            target_id="reporter",
            relation="USES_REPORTER",
        ),
        issue(
            code=IssueCode.UNDEFINED_EDGE_SOURCE,
            stage=IssueStage.STRUCTURAL,
            message="missing source",
            edge_index=6,
            source_id=missing,
            target_id="ethanol",
            relation="USES_MATERIAL",
        ),
    ])


def test_common_endpoint_cluster_aggregates_source_and_target_directions():
    report = _mixed_direction_missing_node_report()
    assert has_common_undefined_endpoint_cluster(report)

    prompt = build_micro_reextract_prompt(
        paper_id="P",
        chunk_id="P:main:c",
        document_id="main",
        document_role="main",
        section="whole",
        page_ids=[],
        asset_ids=[],
        core_text="ATP modification was performed in ethanol.",
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

    assert "synthesis_atp_modification" in prompt
    assert "3 invalid incident edge(s)" in prompt
    assert "PREPARED_BY" in prompt
    assert "USES_REPORTER" in prompt
    assert "USES_MATERIAL" in prompt


def test_claim_like_entity_requires_complete_reextract():
    report = ValidationReport.from_issues([
        issue(
            code=IssueCode.CLAIM_LIKE_ENTITY,
            stage=IssueStage.STRUCTURAL,
            message="claim-like entity",
            node_id="claim_gap_field_enhancement",
            node_collection="entities",
        )
    ])
    assert requires_complete_reextract(report)


def test_claim_like_residual_routes_to_bounded_second_micro():
    report = ValidationReport.from_issues([
        issue(
            code=IssueCode.CLAIM_LIKE_ENTITY,
            stage=IssueStage.STRUCTURAL,
            message="claim-like entity",
            node_id="claim_gap_field_enhancement",
            node_collection="entities",
        )
    ])

    first_retry = _decision(report, micro_attempts=1)
    assert first_retry.action == RecoveryAction.MICRO_REEXTRACT
    assert "semantic patch cannot represent" in first_retry.reason.lower()

    bounded = _decision(report, micro_attempts=2)
    assert bounded.action == RecoveryAction.SEMANTIC_PATCH


def test_replace_edge_schema_feedback_is_explicit():
    error = ValueError(
        "Operation 'replace_edge' requires non-null fields: "
        "['edge', 'edge_index', 'expected_source', "
        "'expected_relation', 'expected_target']."
    )
    feedback = build_patch_rejection_feedback(error)

    assert "replace_edge" in feedback
    assert "edge_index" in feedback
    assert "expected_source" in feedback
    assert "expected_relation" in feedback
    assert "expected_target" in feedback
    assert "complete replacement KGEdge" in feedback
    assert "every unrelated operation-specific field is null" in feedback


def test_non_shape_patch_error_falls_back_without_loss():
    error = ValueError("arbitrary patch rejection")
    feedback = build_patch_rejection_feedback(error)
    assert "arbitrary patch rejection" in feedback
