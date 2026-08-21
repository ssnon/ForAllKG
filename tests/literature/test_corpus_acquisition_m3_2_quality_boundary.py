from __future__ import annotations

from pipeline_core.literature.acquisition.backfill_contracts import (
    AcquisitionAwareBackfillPolicy,
)


def test_backfill_policy_cannot_auto_use_manual_review():
    import pytest
    with pytest.raises(Exception):
        AcquisitionAwareBackfillPolicy(
            policy_id="x",
            required_quality_status="manual_review",
        )


def test_oa_hint_is_not_scientific_score_policy():
    policy = AcquisitionAwareBackfillPolicy(policy_id="x")
    assert policy.required_quality_status == "pass"
    assert policy.oa_hint_tiebreak_only is True
