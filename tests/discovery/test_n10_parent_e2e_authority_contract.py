from pathlib import Path

from scripts.discovery.run_dac_discovery_e2e import (
    _post_generation_n10_consumer_contract,
    _post_generation_n10_output_contract,
)


def test_hard_filter_contract_preserves_legacy_single_portfolio():
    run = Path("/tmp/n10-parent-hard")
    contract = _post_generation_n10_output_contract(run, "hard_filter")
    assert contract["downstream_portfolio"] == (
        run / "novelty_refinement_a6.n10.portfolio.json"
    )
    assert contract["legacy_output_portfolio"] == contract["downstream_portfolio"]
    assert contract["certification_report"] is None
    assert contract["certified_novelty_portfolio"] is None


def test_certification_only_contract_has_three_authority_artifacts():
    run = Path("/tmp/n10-parent-cert")
    contract = _post_generation_n10_output_contract(
        run,
        "certification_only",
    )
    assert contract["scientific_candidate_portfolio"] == (
        run / "novelty_refinement_a6.n10.candidate.portfolio.json"
    )
    assert contract["certification_report"] == (
        run / "novelty_refinement_a6.n10.certification.json"
    )
    assert contract["certified_novelty_portfolio"] == (
        run / "novelty_refinement_a6.n10.certified.portfolio.json"
    )
    assert contract["downstream_portfolio"] == contract[
        "scientific_candidate_portfolio"
    ]


def test_certification_only_consumers_use_candidate_authority():
    consumers = _post_generation_n10_consumer_contract("certification_only")
    assert consumers["stage12_semantic"] == "scientific_candidate_portfolio"
    assert consumers["stage13_feasibility"] == "scientific_candidate_portfolio"
    assert consumers["demo_viewer"] == (
        "scientific_candidate_portfolio+certification_report"
    )
    assert consumers["novelty_certified_export"] == "certified_novelty_portfolio"


def test_hard_filter_consumer_contract_remains_legacy():
    consumers = _post_generation_n10_consumer_contract("hard_filter")
    assert consumers["stage12_semantic"] == "legacy_n10_filtered_portfolio"
    assert consumers["stage13_feasibility"] == "legacy_n10_filtered_portfolio"
    assert consumers["demo_viewer"] == "legacy_alpha6_portfolio"
