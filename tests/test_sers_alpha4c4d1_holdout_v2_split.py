from __future__ import annotations

import hashlib

from campaigns.sers_alpha4_epoch.holdout.trend_holdout import rank_candidate_papers


CANDIDATES = ['Kiwook_SERS_26', 'Kiwook_SERS_15', 'Kiwook_SERS_14', 'Kiwook_SERS_11', 'Kiwook_SERS_38', 'Kiwook_SERS_20', 'Kiwook_SERS_24', 'Kiwook_SERS_23', 'Kiwook_SERS_9', 'Kiwook_SERS_36', 'Kiwook_SERS_33', 'Kiwook_SERS_32', 'Kiwook_SERS_29', 'Kiwook_SERS_27', 'Kiwook_SERS_12', 'Kiwook_SERS_3', 'Kiwook_SERS_22', 'Kiwook_SERS_7', 'Kiwook_SERS_31', 'Kiwook_SERS_17', 'Kiwook_SERS_21', 'Kiwook_SERS_28']
NAMESPACE = 'sers-alpha4c4-v2'
EXPECTED_HOLDOUT = ['Kiwook_SERS_21', 'Kiwook_SERS_38', 'Kiwook_SERS_12', 'Kiwook_SERS_28', 'Kiwook_SERS_17', 'Kiwook_SERS_22', 'Kiwook_SERS_23', 'Kiwook_SERS_11']
EXPECTED_RESERVE = ['Kiwook_SERS_36', 'Kiwook_SERS_32', 'Kiwook_SERS_7', 'Kiwook_SERS_20', 'Kiwook_SERS_3', 'Kiwook_SERS_15', 'Kiwook_SERS_24', 'Kiwook_SERS_29', 'Kiwook_SERS_33', 'Kiwook_SERS_27', 'Kiwook_SERS_26', 'Kiwook_SERS_31', 'Kiwook_SERS_14', 'Kiwook_SERS_9']


def test_v2_split_is_paper_id_only_and_deterministic():
    ranked = list(rank_candidate_papers(
        CANDIDATES,
        namespace=NAMESPACE,
    ))
    assert [r["paper_id"] for r in ranked[:8]] == EXPECTED_HOLDOUT
    assert [r["paper_id"] for r in ranked[8:]] == EXPECTED_RESERVE


def test_v2_and_v3_partition_untouched_22():
    assert len(EXPECTED_HOLDOUT) == 8
    assert len(EXPECTED_RESERVE) == 14
    assert not set(EXPECTED_HOLDOUT) & set(EXPECTED_RESERVE)
    assert set(EXPECTED_HOLDOUT) | set(EXPECTED_RESERVE) == set(CANDIDATES)


def test_namespace_is_new_epoch():
    assert NAMESPACE == "sers-alpha4c4-v2"
    assert NAMESPACE != "sers-alpha4c4-v1"
