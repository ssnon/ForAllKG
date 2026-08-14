from __future__ import annotations

import copy

import pytest

from scripts.run_sers_alpha4b4a11_holdout import (
    FrozenContractViolation,
    detection_limit_ranking_violations,
    validate_campaign_id,
    validate_protocol,
)


def _protocol():
    return {
        "holdout_execution_state": "enabled",
        "calibration_papers": ["A", "B", "C"],
        "holdout_papers": ["D", "E", "F"],
        "required_campaign_id_prefix": "sers_alpha4b4a11_",
        "retired_epoch": {
            "old_holdout_campaign_must_not_resume": True,
            "known_retired_campaign_ids": [
                "sers_alpha4b4_holdout_v1_real_campaign"
            ],
        },
        "holdout_input_refreeze": {
            "required": True,
        },
        "acceptance_policy": {
            "minimum_numeric_ranking_allowed": None,
            "minimum_same_protocol_pairs": None,
            "maximum_unknown_contexts": None,
            "maximum_different_protocol_pairs": None,
            "minimum_metric_definition_known": None,
        },
    }


def test_refrozen_protocol_accepts_heterogeneity_without_count_targets():
    validate_protocol(_protocol())


def test_old_campaign_prefix_is_rejected():
    with pytest.raises(FrozenContractViolation):
        validate_campaign_id(
            _protocol(),
            "sers_alpha4b4_holdout_v1_real_campaign",
        )


def test_new_epoch_campaign_prefix_is_required():
    validate_campaign_id(
        _protocol(),
        "sers_alpha4b4a11_holdout_real_v1",
    )


def test_distribution_targets_cannot_be_reintroduced():
    protocol = copy.deepcopy(_protocol())
    protocol["acceptance_policy"]["maximum_unknown_contexts"] = 0
    with pytest.raises(ValueError):
        validate_protocol(protocol)


def test_detection_limit_numeric_ranking_is_blocking_invariant():
    rows = [
        {
            "observable_key": "detection_limit",
            "numeric_ranking_allowed": False,
        },
        {
            "observable_key": "sers_enhancement_factor",
            "numeric_ranking_allowed": True,
        },
    ]
    assert detection_limit_ranking_violations(rows) == []

    rows[0]["numeric_ranking_allowed"] = True
    assert len(detection_limit_ranking_violations(rows)) == 1
