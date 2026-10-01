from __future__ import annotations

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from scripts.discovery.run_scientific_portfolio_materialization_fidelity_ablation import (
    FIDELITY_SYSTEM_ADDENDUM,
    _premise_stats,
)


def _card(hid: str, premise_ids: list[str]) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hid,
        domain_profile_id="sers_au_ag",
        source_context_id="ctx",
        source_context_sha256="sha",
        source_report_id="report",
        source_report_sha256="rsha",
        title=hid,
        hypothesis_statement="statement",
        hypothesis_type="mechanistic_extension",
        premise_statement_ids=premise_ids,
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
                falsifying_outcome="no change",
            )
        ],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=len(premise_ids),
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=len(premise_ids),
            synthesis_premise_count=0,
        ),
    )


def test_premise_stats_detects_collapse():
    p = HypothesisPortfolio(
        portfolio_id="p",
        domain_profile_id="sers_au_ag",
        source_context_id="ctx",
        source_context_sha256="sha",
        source_report_id="report",
        source_report_sha256="rsha",
        hypotheses=[
            _card("h1", ["a", "b"]),
            _card("h2", ["a", "b"]),
            _card("h3", ["a", "b", "c"]),
        ],
    )
    stats = _premise_stats(p)
    assert stats["distinct_premise_set_count"] == 2
    assert stats["largest_premise_set_share"] == 2 / 3
    assert stats["shared_core_premise_ids"] == ["a", "b"]


def test_fidelity_policy_preserves_conjecture_without_promoting_evidence():
    assert "do NOT need to already" in FIDELITY_SYSTEM_ADDENDUM
    assert "Do NOT weaken" in FIDELITY_SYSTEM_ADDENDUM
    assert "ABSTAIN" in FIDELITY_SYSTEM_ADDENDUM
    assert "never becomes a" in FIDELITY_SYSTEM_ADDENDUM
