from __future__ import annotations

from dac_her.corpus_acquisition.access_contracts import (
    SourceAcquisitionPolicy,
    SourceArtifact,
)


def test_policy_defers_supplementary_discovery():
    policy = SourceAcquisitionPolicy(policy_id="p")
    assert policy.supplementary_discovery == "deferred_to_m3_1"


def test_source_artifact_cannot_claim_positive_evidence_promotion():
    artifact = SourceArtifact(
        artifact_id="a",
        work_id="w",
        role="main",
        status="not_attempted",
    )
    assert artifact.positive_evidence_promotion_performed is False
