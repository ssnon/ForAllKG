from __future__ import annotations

import pytest

from dac_her.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisContext,
    HypothesisEvidenceProfile,
    HypothesisEvidenceStatement,
    HypothesisPortfolio,
    PredictedObservation,
)
from dac_her.hypothesis_evidence_diversity import (
    HypothesisEvidenceDiversityAssessor,
)


def _statement(statement_id: str, papers: list[str]):
    return HypothesisEvidenceStatement(
        statement_id=statement_id,
        text=f"evidence {statement_id}",
        epistemic_role="reported",
        claim_kind="observation",
        paper_ids=papers,
        eligible_as_premise=True,
    )


def _context() -> HypothesisContext:
    return HypothesisContext(
        context_id="ctx:test",
        context_sha256="a" * 64,
        source_packet_id="packet:test",
        source_packet_sha256="b" * 64,
        source_report_id="report:test",
        source_report_sha256="c" * 64,
        task_id="task:test",
        question="test question",
        corpus_id="test-corpus",
        domain_profile_id="dac_her",
        evidence_statements=[
            _statement("A", ["P1"]),
            _statement("B", ["P2", "P3", "P4"]),
            _statement("C", ["P5"]),
            _statement("D", ["P6"]),
            _statement("E", ["P7"]),
            _statement("F", ["P8"]),
        ],
    )


def _card(hypothesis_id: str, premises: list[str]) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hypothesis_id,
        domain_profile_id="dac_her",
        source_context_id="ctx:test",
        source_context_sha256="a" * 64,
        source_report_id="report:test",
        source_report_sha256="c" * 64,
        title=hypothesis_id,
        hypothesis_statement=f"statement {hypothesis_id}",
        hypothesis_type="mechanistic_extension",
        premise_statement_ids=premises,
        inferential_bridge="proposed bridge",
        predicted_observations=[
            PredictedObservation(
                observation_id=f"pred:{hypothesis_id}",
                observable="observable",
                expected_direction="shift",
                rationale="rationale",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id=f"false:{hypothesis_id}",
                observable="observable",
                falsifying_outcome="no shift",
            )
        ],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=len(premises),
            gap_count=0,
            source_paper_count=0,
            candidate_premise_count=0,
            reported_premise_count=len(premises),
            synthesis_premise_count=0,
        ),
    )


def _portfolio() -> HypothesisPortfolio:
    return HypothesisPortfolio(
        portfolio_id="portfolio:test",
        domain_profile_id="dac_her",
        source_context_id="ctx:test",
        source_context_sha256="a" * 64,
        source_report_id="report:test",
        source_report_sha256="c" * 64,
        hypotheses=[
            _card("H1", ["A", "B", "C"]),
            _card("H2", ["A", "B", "C", "D"]),
            _card("H3", ["A", "B", "C", "E"]),
            _card("H4", ["A", "B", "C", "E"]),
        ],
    )


def test_sd1_q3_like_overlap_metrics():
    report = HypothesisEvidenceDiversityAssessor().assess(
        _context(),
        _portfolio(),
    )

    assert report.diagnostic_only is True
    assert report.scientific_selection_changed is False
    assert report.eligible_statement_count == 6
    assert report.used_statement_count == 5
    assert report.eligible_statement_coverage == pytest.approx(5 / 6)
    assert report.unused_eligible_statement_ids == ["F"]
    assert report.shared_core_statement_ids == ["A", "B", "C"]
    assert report.distinct_premise_set_count == 3
    assert report.exact_premise_set_duplicate_group_count == 1
    assert report.exact_premise_set_groups[0].hypothesis_ids == ["H3", "H4"]
    assert report.max_pairwise_statement_jaccard == pytest.approx(1.0)
    assert report.mean_pairwise_statement_jaccard == pytest.approx(
        (0.75 + 0.75 + 0.75 + 0.6 + 0.6 + 1.0) / 6
    )

    by_id = {row.hypothesis_id: row for row in report.cards}
    assert by_id["H2"].portfolio_unique_premise_statement_ids == ["D"]
    assert by_id["H3"].exact_premise_set_duplicate is True
    assert by_id["H4"].exact_premise_set_duplicate is True
    assert by_id["H3"].most_overlapping_hypothesis_ids == ["H4"]

    usage = {row.statement_id: row for row in report.statement_usage}
    assert usage["B"].paper_count == 3
    assert usage["B"].hypothesis_usage_count == 4
    assert report.multi_paper_used_statement_count == 1
    assert report.mean_papers_per_used_statement == pytest.approx(7 / 5)


def test_sd1_single_hypothesis_has_no_artificial_shared_core():
    portfolio = HypothesisPortfolio(
        portfolio_id="portfolio:single",
        domain_profile_id="dac_her",
        source_context_id="ctx:test",
        source_context_sha256="a" * 64,
        source_report_id="report:test",
        source_report_sha256="c" * 64,
        hypotheses=[_card("H1", ["A", "B"])],
    )
    report = HypothesisEvidenceDiversityAssessor().assess(
        _context(),
        portfolio,
    )
    assert report.shared_core_statement_ids == []
    assert report.pairwise_overlaps == []
    assert report.mean_pairwise_statement_jaccard == 0.0
    assert report.max_pairwise_statement_jaccard == 0.0


def test_sd1_fails_closed_on_context_lineage_mismatch():
    portfolio = _portfolio().model_copy(
        update={"source_context_id": "ctx:other"}
    )
    with pytest.raises(ValueError, match="portfolio/context ID mismatch"):
        HypothesisEvidenceDiversityAssessor().assess(
            _context(),
            portfolio,
        )
