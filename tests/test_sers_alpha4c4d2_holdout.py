from __future__ import annotations

import json
from pathlib import Path

from dac_her.alpha4c4d2_holdout_support import quality_snapshot


def test_protocol_freezes_exact_v2_split():
    p = json.loads(
        Path(
            "configs/heldout/"
            "sers_alpha4c4d2_trend_holdout_v2_run.json"
        ).read_text(encoding="utf-8")
    )
    assert p["source_split_sha256"] == '6eebae74732070408e920154ba898841c879a8e97cb866a8971cb9feef526966'
    assert p["holdout_papers"] == ['Kiwook_SERS_21', 'Kiwook_SERS_38', 'Kiwook_SERS_12', 'Kiwook_SERS_28', 'Kiwook_SERS_17', 'Kiwook_SERS_22', 'Kiwook_SERS_23', 'Kiwook_SERS_11']
    assert p["frozen_semantics"]["metric_definition"] == 'sers_au_ag_metric_definition_v3_alpha4c4c1'


def test_zero_yield_policy_is_frozen():
    p = json.loads(
        Path(
            "configs/heldout/"
            "sers_alpha4c4d2_trend_holdout_v2_run.json"
        ).read_text(encoding="utf-8")
    )
    policy = p["holdout_acceptance_policy"]
    assert policy["count_thresholds_used"] is False
    assert policy["zero_trend_evidence_valid"] is True
    assert policy["zero_local_results_valid"] is True
    assert (
        policy["zero_local_results_terminal_status"]
        == "not_applicable_zero_local_results"
    )
    assert (
        policy["cross_context_builder_called_when_zero_local_results"]
        is False
    )


def test_input_preparation_never_uses_active_complete_flag_as_gate():
    p = json.loads(
        Path(
            "configs/heldout/"
            "sers_alpha4c4d2_trend_holdout_v2_run.json"
        ).read_text(encoding="utf-8")
    )
    policy = p["input_preparation_policy"]
    assert policy["active_chunks_complete_flag_is_diagnostic_only"] is True
    assert policy["partial_critical_allowed_with_allow_incomplete"] is True
    assert policy["rejected_allowed"] is False
    assert policy["new_llm_extraction_allowed"] is False
