from __future__ import annotations

from dac_her.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from dac_her.ig1_grounded_bridge import (
    IG1Blueprint,
    IG1DiscriminativeTest,
    IG1GroundedEndpoint,
    IG1NovelBridge,
    blueprint_premise_ids,
    draft_conformance_issues,
)


def _blueprint() -> IG1Blueprint:
    return IG1Blueprint(
        axis_id="axis:test",
        endpoint_a=IG1GroundedEndpoint(
            endpoint_id="endpoint_a",
            anchor_statement_id="stmt:a",
            grounded_excerpt="grounded A",
            supporting_statement_ids=["stmt:a"],
            scientific_role="structural endpoint",
        ),
        endpoint_b=IG1GroundedEndpoint(
            endpoint_id="endpoint_b",
            anchor_statement_id="stmt:b",
            grounded_excerpt="grounded B",
            supporting_statement_ids=["stmt:b"],
            scientific_role="electronic endpoint",
        ),
        novel_bridge=IG1NovelBridge(
            subject_endpoint_id="endpoint_a",
            relation="separation modulates the tradeoff",
            object_endpoint_id="endpoint_b",
            bridge_kind="tradeoff",
            axis_inspiration_summary="axis inspiration",
        ),
        discriminative_test=IG1DiscriminativeTest(
            observable="delta adsorption energy across separation",
            expected_direction="shift",
            falsifying_outcome="no systematic change",
        ),
    )


def _draft(
    *,
    premises=None,
    relation=True,
    predictions=1,
    falsifiers=1,
) -> HypothesisPortfolioDraft:
    bp = _blueprint()
    phrase = bp.novel_bridge.relation
    statement = (
        f"A and B: {phrase}"
        if relation
        else "A and B are related"
    )
    bridge = (
        f"Test one edge: {phrase}"
        if relation
        else "Test an unspecified edge"
    )
    return HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id="h1",
                title="test",
                hypothesis_statement=statement,
                hypothesis_type="mechanistic_extension",
                premise_statement_ids=(
                    premises
                    if premises is not None
                    else ["stmt:a", "stmt:b"]
                ),
                inferential_bridge=bridge,
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id=f"p{i}",
                        observable=(
                            "delta adsorption energy across separation"
                        ),
                        expected_direction="shift",
                        rationale="tests bridge",
                    )
                    for i in range(predictions)
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id=f"f{i}",
                        observable=(
                            "delta adsorption energy across separation"
                        ),
                        falsifying_outcome="no systematic change",
                    )
                    for i in range(falsifiers)
                ],
            )
        ],
        abstention_reason=None,
    )


def test_ig1_blueprint_has_exactly_one_structural_bridge_object():
    bp = _blueprint()
    assert bp.novel_bridge is not None
    assert bp.novel_bridge.bridge_id == "novel_bridge_1"
    assert bp.novel_bridge.reported_fact is False
    assert bp.novel_bridge.evidence_boundary_acknowledged is True


def test_ig1_blueprint_premise_union_is_deterministic():
    assert blueprint_premise_ids(_blueprint()) == [
        "stmt:a",
        "stmt:b",
    ]


def test_ig1_conforming_draft_passes():
    issues = draft_conformance_issues(
        _draft(),
        _blueprint(),
    )
    assert issues == []


def test_ig1_extra_safe_premise_fails():
    issues = draft_conformance_issues(
        _draft(
            premises=[
                "stmt:a",
                "stmt:b",
                "stmt:safe",
            ]
        ),
        _blueprint(),
    )
    assert "premise_set_mismatch" in {
        row.code for row in issues
    }


def test_ig1_missing_exact_relation_fails():
    issues = draft_conformance_issues(
        _draft(relation=False),
        _blueprint(),
    )
    codes = {row.code for row in issues}
    assert "novel_relation_missing_from_hypothesis" in codes
    assert "novel_relation_missing_from_bridge" in codes


def test_ig1_multiple_predictions_and_falsifiers_fail():
    issues = draft_conformance_issues(
        _draft(
            predictions=2,
            falsifiers=2,
        ),
        _blueprint(),
    )
    codes = {row.code for row in issues}
    assert "prediction_cardinality" in codes
    assert "falsifier_cardinality" in codes


def test_ig1_blueprint_abstention_is_strict():
    bp = IG1Blueprint(
        axis_id="axis:test",
        abstain=True,
        abstention_reason="cannot form one bounded bridge",
    )
    assert bp.endpoint_a is None
    assert bp.novel_bridge is None
