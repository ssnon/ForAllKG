from __future__ import annotations

from domains.dac_her.relation_constraints import DAC_LEGACY_STRICT_RELATION_CONSTRAINTS as DAC_HER_STRICT_RELATION_CONSTRAINTS
from pipeline_core.corpus.graph.legacy_dac_relation_policy import (
    LEGACY_DAC_RELATION_ENDPOINT_POLICY,
    LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION,
)


def _legacy_policy_rows():
    return {
        (
            item.relation,
            item.source_types,
            item.target_types,
        )
        for item in LEGACY_DAC_RELATION_ENDPOINT_POLICY
    }


def _active_dac_rows():
    return {
        (
            item.relation,
            item.source_types,
            item.target_types,
        )
        for item in DAC_HER_STRICT_RELATION_CONSTRAINTS
    }


def test_legacy_policy_has_exactly_fifteen_unique_relations():
    relations = [
        item.relation
        for item in LEGACY_DAC_RELATION_ENDPOINT_POLICY
    ]

    assert len(relations) == 15
    assert len(set(relations)) == 15


def test_legacy_policy_matches_active_dac_endpoint_contract_exactly():
    assert _legacy_policy_rows() == _active_dac_rows()


def test_legacy_policy_mapping_is_complete():
    assert set(
        LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION
    ) == {
        item.relation
        for item in LEGACY_DAC_RELATION_ENDPOINT_POLICY
    }


def test_legacy_policy_mapping_preserves_object_identity():
    for item in LEGACY_DAC_RELATION_ENDPOINT_POLICY:
        assert (
            LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION[
                item.relation
            ]
            is item
        )
