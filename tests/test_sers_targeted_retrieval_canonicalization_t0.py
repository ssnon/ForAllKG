from dac_her.sers_targeted_retrieval_t0_offline_validation import (
    build_t0_offline_report,
)


def test_t0_shared_canonicalization_offline_contract() -> None:
    report = build_t0_offline_report()
    assert report["structural_outcome"].endswith("_PASS")
    assert all(report["checks"].values())
    assert report["targeted_retrieval_called"] is False
    assert report["network_calls"] == 0
    assert report["llm_calls"] == 0
    assert report["fresh_reserve_c_consumed"] is False


def test_t0_collision_scenario_counts_are_conservative() -> None:
    report = build_t0_offline_report()
    assert report["scenario_counts"] == {
        "same_doi": 1,
        "distinct_doi_same_title": 2,
        "doi_less_one_family": 1,
        "doi_less_two_families": 3,
        "supplementary_family": 1,
    }
