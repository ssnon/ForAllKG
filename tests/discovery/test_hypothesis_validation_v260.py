from __future__ import annotations

from pipeline_core.discovery.hypothesis_compiler import HypothesisCompiler
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolioDraft
from pipeline_core.discovery.hypothesis_validation import HypothesisValidator
from tests.support._hypothesis_v260_fixtures import make_context
from tests.discovery.test_hypothesis_compiler_v260 import proposal


def _valid():
    context = make_context()
    portfolio = HypothesisCompiler().compile(
        context,
        HypothesisPortfolioDraft(hypotheses=[proposal()]),
    )
    return context, portfolio


def test_valid_grounded_hypothesis_passes():
    context, portfolio = _valid()
    result = HypothesisValidator().validate(context, portfolio)
    assert result.passes, result.model_dump()


def test_external_novelty_claim_is_rejected():
    context, portfolio = _valid()
    card = portfolio.hypotheses[0].model_copy(
        update={"hypothesis_statement": "This novel mechanism couples coordination to adsorption."}
    )
    modified = portfolio.model_copy(update={"hypotheses": [card]})
    result = HypothesisValidator().validate(context, modified)
    assert any(x.code == "EXTERNAL_NOVELTY_CLAIM" for x in result.issues)


def test_unsupported_number_is_rejected():
    context, portfolio = _valid()
    card = portfolio.hypotheses[0].model_copy(
        update={"hypothesis_statement": "The proposed mechanism should improve activity by 37%."}
    )
    modified = portfolio.model_copy(update={"hypotheses": [card]})
    result = HypothesisValidator().validate(context, modified)
    assert any(x.code == "UNSUPPORTED_NUMERIC_PREDICTION" for x in result.issues)


def test_partial_paper_absence_claim_is_rejected():
    context = make_context()
    draft = HypothesisPortfolioDraft(hypotheses=[proposal(premises=["s:k10"], gaps=["s:gap"])])
    portfolio = HypothesisCompiler().compile(context, draft)
    card = portfolio.hypotheses[0].model_copy(
        update={"hypothesis_statement": "Kiwook_10 does not report hydrogen spillover, so another mechanism must operate."}
    )
    modified = portfolio.model_copy(update={"hypotheses": [card]})
    result = HypothesisValidator().validate(context, modified)
    assert any(x.code == "PARTIAL_PAPER_ABSENCE_CLAIM" for x in result.issues)


def test_protocol_leakage_is_rejected():
    context, portfolio = _valid()
    card = portfolio.hypotheses[0].model_copy(
        update={"inferential_bridge": "Anneal the catalyst at 500 C for 2 h to test this mechanism."}
    )
    modified = portfolio.model_copy(update={"hypotheses": [card]})
    result = HypothesisValidator().validate(context, modified)
    codes = {x.code for x in result.issues}
    assert "EXPERIMENT_PROTOCOL_LEAKAGE" in codes
    assert "UNSUPPORTED_NUMERIC_PREDICTION" in codes
