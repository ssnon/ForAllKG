from __future__ import annotations

import pytest

from pipeline_core.discovery.hypothesis_compiler import HypothesisCompileError, HypothesisCompiler
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from tests._hypothesis_v260_fixtures import make_context


def proposal(*, premises=None, gaps=None):
    return HypothesisProposalDraft(
        local_id="h1",
        title="Coordination/charge coupling hypothesis",
        hypothesis_statement="Coordination-dependent charge redistribution may mediate changes in hydrogen adsorption energetics.",
        hypothesis_type="descriptor_mediation",
        premise_statement_ids=premises or ["s:reported", "s:candidate"],
        gap_statement_ids=gaps or ["s:gap"],
        inferential_bridge="The reported coordination effect and tentative electronic connection motivate a testable mediation hypothesis.",
        predicted_observations=[
            PredictedObservationDraft(
                local_id="p1",
                observable="hydrogen adsorption energetics",
                expected_direction="shift",
                rationale="A mediation mechanism predicts a coordinated shift in the adsorption descriptor.",
            )
        ],
        falsification_criteria=[
            FalsificationCriterionDraft(
                local_id="f1",
                observable="hydrogen adsorption energetics",
                falsifying_outcome="The adsorption energetics change independently of the proposed electronic redistribution.",
            )
        ],
        assumptions=["Metal identity and reaction regime remain comparable."],
    )


def test_compiler_derives_candidate_and_cross_paper_metadata():
    context = make_context()
    draft = HypothesisPortfolioDraft(hypotheses=[proposal()])
    portfolio = HypothesisCompiler().compile(context, draft)
    card = portfolio.hypotheses[0]

    assert card.candidate_dependency == "supporting"
    assert card.cross_paper_synthesis is True
    assert card.source_paper_ids == ["Kiwook_1", "Kiwook_2"]
    assert card.gap_paper_ids == ["Kiwook_10"]
    assert card.status == "hypothesized"
    assert card.novelty_status == "not_assessed"
    assert card.evidence_profile.candidate_premise_count == 1


def test_compiler_rejects_unresolved_statement_as_positive_premise():
    context = make_context()
    draft = HypothesisPortfolioDraft(hypotheses=[proposal(premises=["s:gap"], gaps=[])])
    with pytest.raises(HypothesisCompileError) as exc:
        HypothesisCompiler().compile(context, draft)
    assert any(x.code == "INELIGIBLE_POSITIVE_PREMISE" for x in exc.value.issues)


def test_compiler_allows_explicit_abstention():
    context = make_context()
    draft = HypothesisPortfolioDraft(
        hypotheses=[],
        abstention_reason="The supplied evidence is insufficient for a falsifiable hypothesis.",
    )
    portfolio = HypothesisCompiler().compile(context, draft)
    assert portfolio.hypotheses == []
    assert portfolio.abstention_reason


def test_compiler_rejects_alignment_dependent_positive_premise():
    from pipeline_core.discovery.hypothesis_contracts import HypothesisEvidenceStatement

    context = make_context()
    alignment_statement = HypothesisEvidenceStatement(
        statement_id="s:alignment",
        text="Two papers are connected through a registry alignment route.",
        epistemic_role="evidence_synthesis",
        claim_kind="association",
        paper_ids=["Kiwook_1", "Kiwook_2"],
        scientific_support_node_ids=["n:reported"],
        support_path_ids=["path:alignment"],
        alignment_path_ids=["path:alignment"],
        eligible_as_premise=False,
        premise_restrictions=["alignment_path_not_scientific_premise"],
    )
    context = context.model_copy(
        update={"evidence_statements": context.evidence_statements + [alignment_statement]}
    )
    draft = HypothesisPortfolioDraft(hypotheses=[proposal(premises=["s:alignment"], gaps=[])])
    with pytest.raises(HypothesisCompileError) as exc:
        HypothesisCompiler().compile(context, draft)
    assert any(x.code == "INELIGIBLE_POSITIVE_PREMISE" for x in exc.value.issues)
