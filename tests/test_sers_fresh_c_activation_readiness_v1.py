from __future__ import annotations

import json
from pathlib import Path

from dac_her.fresh_c_activation import (
    EXPECTED_BROAD_QUERIES,
    EXPECTED_C01A_FREEZE_COMMIT,
    EXPECTED_PROVIDERS,
    FRESH_C_TARGET_COUNT,
    MAX_RAW_METADATA_ROWS,
    PROVIDER_QUERY_EXECUTIONS,
    RESULTS_PER_QUERY,
    extract_historical_identities,
    make_search_budget,
    make_target_count_policy,
)


def test_search_budget_is_frozen_to_existing_provider_clamp():
    budget = make_search_budget()
    assert budget.providers == EXPECTED_PROVIDERS
    assert budget.broad_queries == EXPECTED_BROAD_QUERIES
    assert budget.results_per_query == RESULTS_PER_QUERY == 100
    assert budget.provider_query_executions == (
        PROVIDER_QUERY_EXECUTIONS
    ) == 8
    assert budget.max_raw_metadata_rows == MAX_RAW_METADATA_ROWS == 800
    assert budget.budget_is_scientific_acceptance_threshold is False
    assert budget.expansion_after_observing_results_allowed is False
    assert (
        budget.insufficient_candidate_behavior
        == "fail_closed_new_protocol_epoch_required"
    )


def test_target_count_matches_preexisting_reserve_cardinality():
    policy = make_target_count_policy()
    assert policy.target_acquired_papers == FRESH_C_TARGET_COUNT == 25
    assert (
        policy.basis
        == "match_preexisting_reserve_a_and_reserve_b_cardinality_25"
    )
    assert policy.target_is_scientific_acceptance_threshold is False
    assert policy.target_must_not_change_after_live_discovery is True
    assert (
        policy.inaccessible_candidate_behavior
        == "continue_next_identity_in_frozen_blind_order"
    )
    assert (
        policy.insufficient_acquired_papers_behavior
        == "fail_closed_new_protocol_epoch_required"
    )


def test_historical_identity_extraction_prefers_doi(tmp_path: Path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            {
                "title": (
                    "A long historical paper title that would otherwise "
                    "be hashed"
                ),
                "doi": "https://doi.org/10.1234/ABC.Def",
            }
        ),
        encoding="utf-8",
    )
    identities = extract_historical_identities(path)
    assert "doi:10.1234/abc.def" in identities


def test_historical_identity_extraction_has_title_fallback(
    tmp_path: Path,
):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            {
                "title": (
                    "A sufficiently long historical paper title "
                    "without any DOI"
                )
            }
        ),
        encoding="utf-8",
    )
    identities = extract_historical_identities(path)
    assert len(identities) == 1
    only = next(iter(identities))
    assert only.startswith("title_sha256:")
    assert "historical paper title" not in only


def test_historical_sweep_retains_no_scientific_text(tmp_path: Path):
    path = tmp_path / "history.json"
    secret = (
        "This sentence represents scientific content that must never "
        "be retained in the identity ledger."
    )
    path.write_text(
        json.dumps(
            {
                "title": "A sufficiently long bibliographic paper title",
                "abstract": secret,
            }
        ),
        encoding="utf-8",
    )
    identities = extract_historical_identities(path)
    assert identities
    serialized = json.dumps(sorted(identities))
    assert secret not in serialized


def test_doi_regex_catches_embedded_historical_reference(
    tmp_path: Path,
):
    path = tmp_path / "report.txt"
    path.write_text(
        "Historical evidence DOI 10.1021/acs.jpcc.0c07701 was used.",
        encoding="utf-8",
    )
    identities = extract_historical_identities(path)
    assert "doi:10.1021/acs.jpcc.0c07701" in identities


def test_c01a_freeze_commit_is_pinned():
    assert (
        EXPECTED_C01A_FREEZE_COMMIT
        == "6f02c92b84a7e36e335b79f812b1e8803645fe12"
    )
