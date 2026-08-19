from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from campaigns.sers_alpha4_epoch.alpha4.alpha4c5f2_reserve import (
    BLIND_SPLIT_SEMANTICS_ID,
    DEVELOPMENT_COUNT,
    EXPECTED_POOL_SIZE,
    RESERVE_A_COUNT,
    RESERVE_B_COUNT,
    make_blind_split,
    validate_blind_split,
)
from campaigns.sers_alpha4_epoch.alpha4.cli.run_sers_alpha4c5f2_reserve import (
    ReadinessLockedRunner,
)


def _pool(ids: list[str]) -> dict:
    return {
        "pool_id": "pool::test",
        "manifest_sha256": "a" * 64,
        "paper_ids": sorted(ids),
    }


def test_blind_split_is_deterministic_and_id_only():
    ids = [f"PAPER_{i:03d}" for i in range(EXPECTED_POOL_SIZE)]
    a = make_blind_split(_pool(ids))
    b = make_blind_split(_pool(list(reversed(ids))))
    assert a == b
    assert a["semantics_id"] == BLIND_SPLIT_SEMANTICS_ID
    assert a["split_input_fields"] == ["paper_id"]
    assert a["scientific_fields_used"] is False
    assert a["llm_calls_at_split"] == 0


def test_blind_split_counts_and_partition_disjointness():
    ids = [f"PAPER_{i:03d}" for i in range(EXPECTED_POOL_SIZE)]
    split = make_blind_split(_pool(ids))
    assert len(split["development"]) == DEVELOPMENT_COUNT
    assert len(split["reserve_a"]) == RESERVE_A_COUNT
    assert len(split["reserve_b"]) == RESERVE_B_COUNT

    dev = set(split["development"])
    a = set(split["reserve_a"])
    b = set(split["reserve_b"])
    assert not (dev & a)
    assert not (dev & b)
    assert not (a & b)
    assert dev | a | b == set(ids)
    assert split["reserve_b_sealed_for_future_confirmation"] is True


def test_blind_split_validator_rejects_manual_reserve_edit():
    ids = [f"PAPER_{i:03d}" for i in range(EXPECTED_POOL_SIZE)]
    pool = _pool(ids)
    split = make_blind_split(pool)
    split["reserve_a"][0], split["reserve_b"][0] = (
        split["reserve_b"][0],
        split["reserve_a"][0],
    )
    with pytest.raises(ValueError):
        validate_blind_split(pool=pool, split=split)


def test_runner_consumption_boundary_is_guarded_not_direct():
    source = inspect.getsource(ReadinessLockedRunner.execute)
    assert "guarded_write_consumption_marker(" in source
    assert "write_json(self.marker" not in source
    assert "self._freeze_canonical_sources()" in source
    assert source.index("guarded_write_consumption_marker(") < source.index(
        "self._freeze_canonical_sources()"
    )


def test_runner_reuses_frozen_5f_scientific_sequence_after_guard():
    source = inspect.getsource(ReadinessLockedRunner.execute)
    ordered = [
        "self._freeze_canonical_sources()",
        "self._build_evidence_substrate()",
        "self._build_explorer_context()",
        "self._build_trend_input(",
        "self._run_maker_and_evaluate(",
    ]
    positions = [source.index(token) for token in ordered]
    assert positions == sorted(positions)


def test_reserve_b_never_enters_runner_execution_contract():
    source = inspect.getsource(ReadinessLockedRunner.execute)
    assert "reserve_b" not in source.lower() or (
        "Reserve B remains sealed" in source
    )
