from __future__ import annotations

from pipeline_core.discovery.higher_order_competing_explanations import (
    generate_competing_explanations,
)
from pipeline_core.discovery.higher_order_tension_extractor import (
    ScientificTensionCandidate,
    ScientificTensionCandidateSet,
)


def _tension(
    *,
    tid: str,
    tension_type: str,
    candidate: bool = False,
):
    return ScientificTensionCandidate(
        tension_id=tid,
        tension_type=tension_type,
        source_arm_indices=[1],
        source_context_ids=["ctx:1"],
        basis_premise_ids=["premise:1"],
        basis_relation_texts=[
            "SERS enhancement factor --VARIES_WITH--> air exposure time"
        ],
        trigger_codes=["TEST"],
        requested_source="nanostructure shape",
        requested_target=(
            "electromagnetic hotspot location and intensity"
        ),
        tension_statement="unresolved test tension",
        rationale="test",
        candidate_inspiration_involved=candidate,
    )


def _set(*rows):
    return ScientificTensionCandidateSet(
        candidate_count=len(rows),
        type_counts={},
        candidate_inspiration_candidate_count=sum(
            row.candidate_inspiration_involved
            for row in rows
        ),
        candidates=list(rows),
    )


def test_proxy_decoupling_generates_two_competing_explanations():
    result = generate_competing_explanations(
        _set(
            _tension(
                tid="t:proxy",
                tension_type="proxy_decoupling",
                candidate=True,
            )
        )
    )

    assert result.pair_count == 1
    pair = result.pairs[0]
    assert [x.explanation_role for x in pair.explanations] == ["A", "B"]
    assert {
        x.explanation_type for x in pair.explanations
    } == {
        "target_state_change",
        "anchor_or_proxy_only_change",
    }
    assert pair.candidate_inspiration_involved is True


def test_tradeoff_generates_intrinsic_vs_condition_dependent_pair():
    result = generate_competing_explanations(
        _set(
            _tension(
                tid="t:tradeoff",
                tension_type="tradeoff_pareto",
            )
        )
    )

    assert {
        x.explanation_type
        for x in result.pairs[0].explanations
    } == {
        "intrinsic_tradeoff",
        "condition_or_proxy_dependent_tradeoff",
    }


def test_contrast_does_not_become_conflicting_literature_claim():
    result = generate_competing_explanations(
        _set(
            _tension(
                tid="t:contrast",
                tension_type="contrasting_relations",
            )
        )
    )

    text = " ".join(
        x.explanation_statement
        for x in result.pairs[0].explanations
    ).lower()

    assert "conflicting literature" not in text
    assert result.novelty_authority is False


def test_competing_explanations_remain_non_authorizing():
    result = generate_competing_explanations(
        _set(
            _tension(
                tid="t:proxy",
                tension_type="proxy_decoupling",
                candidate=True,
            ),
            _tension(
                tid="t:tradeoff",
                tension_type="tradeoff_pareto",
            ),
        )
    )

    assert result.explanation_selection_performed is False
    assert result.production_selection_authority is False
    assert result.positive_premise_authority is False
    assert result.gap_authority is False
    assert result.novelty_authority is False
    assert result.discriminating_hypothesis_generation_authorized is False

    for pair in result.pairs:
        assert pair.explanation_selection_authority is False
        assert pair.discriminating_hypothesis_generation_authorized is False
