from __future__ import annotations

import copy

import pytest

from campaigns.sers_alpha4_epoch.holdout.trend_holdout import (
    TREND_HOLDOUT_SELECTION_ALGORITHM,
    TREND_HOLDOUT_SPLIT_SEMANTICS_ID,
    build_trend_holdout_split,
    validate_protocol_split,
)


ALL = ['Kiwook_SERS_1', 'Kiwook_SERS_2', 'Kiwook_SERS_3', 'Kiwook_SERS_4', 'Kiwook_SERS_5', 'Kiwook_SERS_6', 'Kiwook_SERS_7', 'Kiwook_SERS_8', 'Kiwook_SERS_9', 'Kiwook_SERS_10', 'Kiwook_SERS_11', 'Kiwook_SERS_12', 'Kiwook_SERS_13', 'Kiwook_SERS_14', 'Kiwook_SERS_15', 'Kiwook_SERS_16', 'Kiwook_SERS_17', 'Kiwook_SERS_18', 'Kiwook_SERS_19', 'Kiwook_SERS_20', 'Kiwook_SERS_21', 'Kiwook_SERS_22', 'Kiwook_SERS_23', 'Kiwook_SERS_24', 'Kiwook_SERS_25', 'Kiwook_SERS_26', 'Kiwook_SERS_27', 'Kiwook_SERS_28', 'Kiwook_SERS_29', 'Kiwook_SERS_30', 'Kiwook_SERS_31', 'Kiwook_SERS_32', 'Kiwook_SERS_33', 'Kiwook_SERS_34', 'Kiwook_SERS_35', 'Kiwook_SERS_36', 'Kiwook_SERS_37', 'Kiwook_SERS_38']
CAL = ['Kiwook_SERS_1', 'Kiwook_SERS_5', 'Kiwook_SERS_8']
SEEN = ['Kiwook_SERS_2', 'Kiwook_SERS_6', 'Kiwook_SERS_10']
NAMESPACE = 'sers-alpha4c4-v1'
EXPECTED_HOLDOUT = ['Kiwook_SERS_16', 'Kiwook_SERS_35', 'Kiwook_SERS_34', 'Kiwook_SERS_19', 'Kiwook_SERS_13', 'Kiwook_SERS_37', 'Kiwook_SERS_30', 'Kiwook_SERS_25', 'Kiwook_SERS_4', 'Kiwook_SERS_18']
EXPECTED_RESERVE = ['Kiwook_SERS_26', 'Kiwook_SERS_15', 'Kiwook_SERS_14', 'Kiwook_SERS_11', 'Kiwook_SERS_38', 'Kiwook_SERS_20', 'Kiwook_SERS_24', 'Kiwook_SERS_23', 'Kiwook_SERS_9', 'Kiwook_SERS_36', 'Kiwook_SERS_33', 'Kiwook_SERS_32', 'Kiwook_SERS_29', 'Kiwook_SERS_27', 'Kiwook_SERS_12', 'Kiwook_SERS_3', 'Kiwook_SERS_22', 'Kiwook_SERS_7', 'Kiwook_SERS_31', 'Kiwook_SERS_17', 'Kiwook_SERS_21', 'Kiwook_SERS_28']


def _split():
    return build_trend_holdout_split(
        all_paper_ids=ALL,
        development_calibration=CAL,
        development_seen_regression=SEEN,
        namespace=NAMESPACE,
        holdout_count=10,
    )


def test_curated_split_is_exact_and_deterministic():
    split = _split()
    assert len(split.candidate_papers) == 32
    assert list(split.holdout_papers) == EXPECTED_HOLDOUT
    assert list(split.reserved_future_papers) == EXPECTED_RESERVE
    assert len(split.holdout_papers) == 10
    assert len(split.reserved_future_papers) == 22


def test_development_holdout_reserve_are_disjoint():
    split = _split()
    development = set(CAL) | set(SEEN)
    assert not development & set(split.holdout_papers)
    assert not development & set(split.reserved_future_papers)
    assert not (
        set(split.holdout_papers)
        & set(split.reserved_future_papers)
    )
    assert (
        development
        | set(split.holdout_papers)
        | set(split.reserved_future_papers)
        == set(ALL)
    )


def test_rank_depends_only_on_paper_id_and_namespace():
    first = _split()
    second = build_trend_holdout_split(
        all_paper_ids=list(ALL),
        development_calibration=list(CAL),
        development_seen_regression=list(SEEN),
        namespace=NAMESPACE,
        holdout_count=10,
    )
    assert first.ranked_candidates == second.ranked_candidates
    assert first.split_sha256 == second.split_sha256


def test_protocol_validator_rejects_count_targets():
    split = _split()
    protocol = {
        "selection": {
            "split_semantics_id":
                TREND_HOLDOUT_SPLIT_SEMANTICS_ID,
            "algorithm":
                TREND_HOLDOUT_SELECTION_ALGORITHM,
            "selection_inputs": ["paper_id"],
            "scientific_content_inspected_for_split": False,
            "trend_outputs_inspected_for_split": False,
            "namespace": NAMESPACE,
            "holdout_count": 10,
            "split_sha256": split.split_sha256,
        },
        "papers": {
            "curated_corpus": list(split.all_paper_ids),
            "development_calibration":
                list(split.development_calibration),
            "development_seen_regression":
                list(split.development_seen_regression),
            "candidate_papers": list(split.candidate_papers),
            "ranked_candidates": list(split.ranked_candidates),
            "frozen_holdout": list(split.holdout_papers),
            "reserved_future":
                list(split.reserved_future_papers),
        },
        "alpha4c4b_acceptance_policy": {
            "count_thresholds_used": False,
            "minimum_trend_evidence_count": None,
            "minimum_cross_paper_pair_count": None,
            "minimum_repeated_count": None,
            "minimum_reversed_count": None,
            "minimum_context_specific_count": None,
            "maximum_insufficient_count": None,
        },
    }
    validate_protocol_split(protocol)

    bad = copy.deepcopy(protocol)
    bad["alpha4c4b_acceptance_policy"][
        "minimum_trend_evidence_count"
    ] = 1
    with pytest.raises(ValueError):
        validate_protocol_split(bad)
