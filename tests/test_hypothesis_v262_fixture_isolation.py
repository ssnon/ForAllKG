from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from scripts.build_hypothesis_v262_fixtures import build_cases


TARGET_CASES = {
    "canonical_valid",
    "adv_candidate_overclaim",
    "adv_alignment_causalization",
    "adv_causal_strengthening",
    "adv_redundancy",
}


def _load_portfolio(tmp_path: Path, case_id: str) -> HypothesisPortfolio:
    path = tmp_path / "cases" / case_id / "portfolio.json"
    return HypothesisPortfolio.model_validate_json(path.read_text(encoding="utf-8"))


def test_semantic_target_fixtures_use_informative_falsifiers(tmp_path: Path):
    build_cases(tmp_path)
    generic = "The predicted qualitative response is not observed."

    for case_id in TARGET_CASES:
        portfolio = _load_portfolio(tmp_path, case_id)
        assert portfolio.hypotheses
        for card in portfolio.hypotheses:
            assert card.falsification_criteria
            for criterion in card.falsification_criteria:
                assert criterion.falsifying_outcome != generic, (
                    f"{case_id} still uses the generic falsifier: "
                    f"{criterion.falsifying_outcome!r}"
                )
                assert len(criterion.falsifying_outcome.split()) >= 7, (
                    f"{case_id} has an under-specified falsifier: "
                    f"{criterion.falsifying_outcome!r}"
                )


def test_redundancy_fixture_is_individually_linked_but_pairwise_redundant(tmp_path: Path):
    build_cases(tmp_path)
    portfolio = _load_portfolio(tmp_path, "adv_redundancy")
    assert len(portfolio.hypotheses) == 2

    left, right = portfolio.hypotheses
    assert left.hypothesis_statement == right.hypothesis_statement
    assert left.inferential_bridge == right.inferential_bridge
    assert [row.observable for row in left.predicted_observations] == [
        row.observable for row in right.predicted_observations
    ]
    assert [row.rationale for row in left.predicted_observations] == [
        row.rationale for row in right.predicted_observations
    ]
    assert [row.observable for row in left.falsification_criteria] == [
        row.observable for row in right.falsification_criteria
    ]
    assert [row.falsifying_outcome for row in left.falsification_criteria] == [
        row.falsifying_outcome for row in right.falsification_criteria
    ]
