import pytest

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.scientific_portfolio_production import (
    bind_scientific_portfolio_production,
    certify_scientific_portfolio_novelty,
)


def _card(hid: str) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hid,
        domain_profile_id="dac_her",
        source_context_id="ctx",
        source_context_sha256="csha",
        source_report_id="report",
        source_report_sha256="rsha",
        title=hid,
        hypothesis_statement=hid + " statement",
        hypothesis_type="mechanistic_extension",
        premise_statement_ids=["s1"],
        inferential_bridge="bridge",
        predicted_observations=[
            PredictedObservation(
                observation_id=hid + ":p",
                observable="response",
                expected_direction="qualitative_change",
                rationale="test",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id=hid + ":f",
                observable="response",
                falsifying_outcome="no difference",
            )
        ],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )


def _portfolio() -> HypothesisPortfolio:
    return HypothesisPortfolio(
        portfolio_id="portfolio:p",
        domain_profile_id="dac_her",
        source_context_id="ctx",
        source_context_sha256="csha",
        source_report_id="report",
        source_report_sha256="rsha",
        hypotheses=[_card("hyp:a"), _card("hyp:b"), _card("hyp:c")],
    )


def _gate() -> dict:
    rows = [
        ("hyp:a", "ELIGIBLE", True, True),
        ("hyp:b", "CONDITIONAL", False, False),
        ("hyp:c", "INELIGIBLE", False, False),
    ]
    return {
        "schema_version": "scientific-novelty-fallback-gate-v2",
        "production_authority": True,
        "authority_scope": "alpha6_original_fallback",
        "authority_source": "n10_role_aware_nonobviousness_v2",
        "source_portfolio_id": "portfolio:p",
        "source_query_plan_id": "plan:q",
        "gate_count": 3,
        "fallback_allowed_count": 1,
        "fallback_blocked_count": 2,
        "selection_counts": {
            "ELIGIBLE": 1,
            "CONDITIONAL": 1,
            "INELIGIBLE": 1,
        },
        "gates": [
            {
                "hypothesis_id": hid,
                "selection_class": cls,
                "fallback_allowed": allowed,
                "positive_nonobviousness_authority": positive,
                "reason_codes": [cls.lower()],
                "blocking_claim_ids": [],
                "unresolved_claim_ids": [],
                "resolution_requirements": [],
            }
            for hid, cls, allowed, positive in rows
        ],
    }


def test_certification_only_preserves_candidates_and_splits_certified_subset():
    report, candidate, certified = certify_scientific_portfolio_novelty(
        portfolio=_portfolio(),
        n10_production_gate=_gate(),
    )
    assert len(candidate.hypotheses) == 3
    assert [x.hypothesis_id for x in certified.hypotheses] == ["hyp:a"]
    assert report.certified_count == 1
    assert report.conditional_count == 1
    assert report.ineligible_count == 1
    assert report.n10_candidate_survival_authority is False
    assert report.conditional_candidates_retained is True
    assert report.scientific_reaggregation_performed is False


def test_zero_certified_still_preserves_candidates():
    gate = _gate()
    for row in gate["gates"]:
        row["selection_class"] = "CONDITIONAL"
        row["fallback_allowed"] = False
        row["positive_nonobviousness_authority"] = False

    report, candidate, certified = certify_scientific_portfolio_novelty(
        portfolio=_portfolio(),
        n10_production_gate=gate,
    )
    assert len(candidate.hypotheses) == 3
    assert certified.hypotheses == []
    assert report.certified_count == 0
    assert report.conditional_count == 3


def test_hard_filter_still_available():
    report, output = bind_scientific_portfolio_production(
        portfolio=_portfolio(),
        n10_production_gate=_gate(),
    )
    assert [x.hypothesis_id for x in output.hypotheses] == ["hyp:a"]
    assert report.production_blocked_count == 2


def test_gate_must_cover_portfolio_exactly():
    gate = _gate()
    gate["gates"] = gate["gates"][:2]
    gate["gate_count"] = 2
    with pytest.raises(ValueError, match="cover source portfolio exactly"):
        certify_scientific_portfolio_novelty(
            portfolio=_portfolio(),
            n10_production_gate=gate,
        )
