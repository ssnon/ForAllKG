from __future__ import annotations

from dac_her.recovery_policy import RecoveryAction, decide_recovery
from dac_her.validation_issues import (
    IssueCode,
    IssueStage,
    ValidationReport,
    issue,
)


def _decision(report):
    return decide_recovery(
        report=report,
        normalization_attempted=True,
        patch_attempts=2,
        micro_reextract_attempts=1,
        post_micro_patch_attempts=0,
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


def test_pure_common_cluster_keeps_specific_reason():
    report = ValidationReport.from_issues([
        issue(
            code=IssueCode.UNDEFINED_EDGE_SOURCE,
            stage=IssueStage.STRUCTURAL,
            message="missing",
            edge_index=i,
            source_id="missing_method",
            target_id=f"target_{i}",
            relation="USES_MATERIAL",
        )
        for i in range(3)
    ])
    decision = _decision(report)
    assert decision.action == RecoveryAction.MICRO_REEXTRACT
    assert "common undefined endpoint" in decision.reason.lower()


def test_mixed_cluster_uses_dominant_reason():
    missing = "missing_method"
    report = ValidationReport.from_issues([
        issue(
            code=IssueCode.UNDEFINED_EDGE_TARGET,
            stage=IssueStage.STRUCTURAL,
            message="missing",
            edge_index=0,
            source_id="substrate",
            target_id=missing,
            relation="PREPARED_BY",
        ),
        issue(
            code=IssueCode.UNDEFINED_EDGE_SOURCE,
            stage=IssueStage.STRUCTURAL,
            message="missing",
            edge_index=1,
            source_id=missing,
            target_id="precursor",
            relation="USES_PRECURSOR",
        ),
        issue(
            code=IssueCode.UNDEFINED_EDGE_SOURCE,
            stage=IssueStage.STRUCTURAL,
            message="missing",
            edge_index=2,
            source_id=missing,
            target_id="material",
            relation="USES_MATERIAL",
        ),
        issue(
            code=IssueCode.MISSING_MEASUREMENT_PRODUCER,
            stage=IssueStage.MEASUREMENT,
            message="producer",
            node_id="m1",
            node_collection="measurements",
        ),
        issue(
            code=IssueCode.MISSING_MEASUREMENT_PRODUCER,
            stage=IssueStage.MEASUREMENT,
            message="producer",
            node_id="m2",
            node_collection="measurements",
        ),
    ])
    decision = _decision(report)
    assert decision.action == RecoveryAction.MICRO_REEXTRACT
    assert "dominant missing-node" in decision.reason.lower()
