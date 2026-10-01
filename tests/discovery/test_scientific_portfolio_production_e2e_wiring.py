from pathlib import Path


def test_scientific_portfolio_production_mode_is_opt_in_and_final_authoritative():
    text = Path("scripts/discovery/run_dac_discovery_e2e.py").read_text(
        encoding="utf-8"
    )
    assert "--scientific-portfolio-production-enforce" in text
    assert "--production-enforce" in text
    assert "scientific_portfolio_production" in text
    assert "SCIENTIFIC_PORTFOLIO_CANDIDATE_N10_CERTIFICATION" in text
    assert "refined_portfolio = scientific_portfolio_production_candidate" in text


def test_scientific_portfolio_verification_can_promote_n10_v2():
    text = Path(
        "scripts/discovery/run_scientific_portfolio_verification_shadow.py"
    ).read_text(encoding="utf-8")
    assert "--production-enforce" in text
    assert "build_nonobviousness_production_gate_v2_candidate" in text
    assert "build_nonobviousness_production_gate_v2" in text
    assert "bind_scientific_portfolio_production" in text


def test_n10_is_certification_not_candidate_survival_authority():
    text = Path("scripts/discovery/run_dac_discovery_e2e.py").read_text(encoding="utf-8")
    assert '"n10_candidate_survival_authority": False' in text
    assert '"conditional_candidates_retained": True' in text
