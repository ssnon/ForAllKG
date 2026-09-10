from pipeline_core.discovery.nonobviousness_post_generation import (
    POST_GENERATION_N10_AUTHORITY_MODE_CERTIFICATION_ONLY,
    POST_GENERATION_N10_AUTHORITY_MODE_HARD_FILTER,
    POST_GENERATION_N10_AUTHORITY_MODES,
    POST_GENERATION_N10_CERTIFICATION_OUTPUT_CONTRACT,
    certified_novelty_portfolio_id,
    novelty_certification_report_id,
)


def test_authority_modes_are_explicit_and_backward_compatible():
    assert POST_GENERATION_N10_AUTHORITY_MODES == (
        POST_GENERATION_N10_AUTHORITY_MODE_HARD_FILTER,
        POST_GENERATION_N10_AUTHORITY_MODE_CERTIFICATION_ONLY,
    )


def test_certification_output_contract_names_three_distinct_artifacts():
    assert POST_GENERATION_N10_CERTIFICATION_OUTPUT_CONTRACT == {
        "candidate_portfolio": "--output-candidate-portfolio",
        "certification_report": "--output-certification-report",
        "certified_portfolio": "--output-certified-portfolio",
    }


def test_empty_certified_portfolio_identity_is_source_scoped():
    common = dict(
        domain_profile_id="dac_her",
        source_context_sha256="context-sha",
        certified_hypothesis_ids=(),
        abstention_reason="No certified candidates.",
    )
    b_id = certified_novelty_portfolio_id(
        source_portfolio_id="portfolio:B",
        **common,
    )
    c_id = certified_novelty_portfolio_id(
        source_portfolio_id="portfolio:C",
        **common,
    )
    assert b_id != c_id


def test_certification_report_identity_is_source_scoped():
    b_id = novelty_certification_report_id(
        source_portfolio_id="portfolio:B",
        certified_portfolio_id="certified:B",
        refinement_report_id="report:B",
        decision_signatures=("candidate:1::CONDITIONAL",),
    )
    c_id = novelty_certification_report_id(
        source_portfolio_id="portfolio:C",
        certified_portfolio_id="certified:C",
        refinement_report_id="report:C",
        decision_signatures=("candidate:1::CONDITIONAL",),
    )
    assert b_id != c_id
