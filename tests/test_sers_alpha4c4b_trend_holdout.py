from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_sers_alpha4c4b_trend_holdout import (
    FrozenTrendHoldoutError,
    _require_equal,
)
from dac_her.trend_holdout import validate_protocol_split


def test_runner_exposes_no_arbitrary_paper_list_cli():
    source = Path(
        "scripts/run_sers_alpha4c4b_trend_holdout.py"
    ).read_text(encoding="utf-8")
    # The runner legitimately passes --paper-id to the frozen projection
    # builder internally. What is forbidden is exposing a user CLI argument
    # that can replace the alpha4c.4a frozen paper set.
    assert 'parser.add_argument("--paper-ids"' not in source
    assert 'parser.add_argument("--paper-id"' not in source


def test_runner_is_explicitly_llm_and_bridge_free():
    source = Path(
        "scripts/run_sers_alpha4c4b_trend_holdout.py"
    ).read_text(encoding="utf-8")
    assert "scripts.extract_paper" not in source
    assert "scripts.extract_bridge_graph" not in source
    assert '"bridge_used": False' in source
    assert '"llm_calls_performed_by_runner": False' in source


def test_protocol_reuses_exact_alpha4c4a_holdout():
    run_protocol = json.loads(
        Path(
            "configs/heldout/"
            "sers_alpha4c4b_trend_holdout_run.json"
        ).read_text(encoding="utf-8")
    )
    split_protocol = json.loads(
        Path(
            "configs/heldout/"
            "sers_alpha4c4_trend_holdout.json"
        ).read_text(encoding="utf-8")
    )
    split = validate_protocol_split(split_protocol)
    assert run_protocol["holdout_papers"] == list(
        split.holdout_papers
    )
    assert (
        run_protocol["source_split_protocol"]["split_sha256"]
        == split.split_sha256
    )
    assert run_protocol["arbitrary_paper_override_allowed"] is False


def test_acceptance_contains_no_distribution_targets():
    protocol = json.loads(
        Path(
            "configs/heldout/"
            "sers_alpha4c4b_trend_holdout_run.json"
        ).read_text(encoding="utf-8")
    )
    policy = protocol["acceptance_policy"]
    assert policy["count_thresholds_used"] is False
    for key in (
        "minimum_trend_evidence_count",
        "minimum_cross_paper_pair_count",
        "minimum_repeated_count",
        "minimum_reversed_count",
        "minimum_context_specific_count",
        "maximum_insufficient_count",
    ):
        assert policy[key] is None


def test_evidence_mode_and_no_bridge_are_frozen():
    protocol = json.loads(
        Path(
            "configs/heldout/"
            "sers_alpha4c4b_trend_holdout_run.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["mode"] == "evidence"
    assert protocol["bridge_used"] is False
    assert protocol["llm_calls_allowed"] is False
    assert protocol["canonical_input_policy"][
        "automatic_strict_extraction"
    ] is False


def test_require_equal_fails_closed():
    with pytest.raises(FrozenTrendHoldoutError):
        _require_equal("x", "observed", "expected")
