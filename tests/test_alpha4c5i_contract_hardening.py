from __future__ import annotations

from types import SimpleNamespace

import pytest

import dac_her.hypothesis_trend_contract_hardened_compiler as compiler_mod
from dac_her.hypothesis_trend_compiler import (
    TrendHypothesisCompileError,
)
from dac_her.hypothesis_trend_contract_hardened_compiler import (
    ContractHardenedTrendHypothesisCompiler,
)
from dac_her.hypothesis_trend_contract_hardened_contracts import (
    ContractHardenedFalsificationCriterionDraft,
    ContractHardenedPredictedObservationDraft,
    ContractHardenedTrendHypothesisPortfolioDraft,
    ContractHardenedTrendHypothesisProposalDraft,
)
from dac_her.hypothesis_trend_contract_hardened_renderer import (
    render_canonical_trend_clause,
)
from dac_her.hypothesis_trend_contracts import TrendReferenceDraft
from dac_her.hypothesis_trend_directional_validator import (
    _uses_decrease_frame,
)


def _view(
    *,
    view_id: str = "view_neg",
    iv: str = "feature_width",
    dv: str = "signal_response",
    direction: str = "negative",
    shape: str = "monotonic",
    status: str = "insufficient",
):
    return SimpleNamespace(
        view_id=view_id,
        independent_variable_key=iv,
        dependent_observable_key=dv,
        directions=[direction],
        shapes=[shape],
        cross_context_status=status,
    )


def _source(*views):
    statements = [
        SimpleNamespace(
            statement_id="premise_1",
            paper_ids=["SYNTH_PAPER_01"],
        ),
        SimpleNamespace(
            statement_id="gap_1",
            paper_ids=["SYNTH_PAPER_02"],
        ),
    ]
    return SimpleNamespace(
        input_sha256="b" * 64,
        trend_views=list(views),
        grounded_context=SimpleNamespace(
            evidence_statements=statements,
        ),
    )


def _draft(
    *,
    mechanism: str = "A local mechanism couples the variables.",
    rationale: str = "The mechanism should be testable.",
    falsifier: str = "The predicted response fails to occur.",
    view_id: str = "view_neg",
):
    return ContractHardenedTrendHypothesisPortfolioDraft(
        hypotheses=[
            ContractHardenedTrendHypothesisProposalDraft(
                local_id="h1",
                title="Synthetic mechanism",
                hypothesis_type="mechanistic_extension",
                premise_statement_ids=[],
                gap_statement_ids=[],
                trend_references=[
                    TrendReferenceDraft(
                        view_id=view_id,
                        use_role="positive_empirical_support",
                    )
                ],
                mechanistic_proposal=mechanism,
                inferential_bridge="We propose a bounded mechanism.",
                predicted_observations=[
                    ContractHardenedPredictedObservationDraft(
                        local_id="p1",
                        prediction_kind="trend_bound",
                        trend_view_ids=[view_id],
                        mechanistic_rationale=rationale,
                    )
                ],
                falsification_criteria=[
                    ContractHardenedFalsificationCriterionDraft(
                        local_id="f1",
                        prediction_local_id="p1",
                        falsifying_outcome=falsifier,
                    )
                ],
                assumptions=[],
            )
        ],
        abstention_reason=None,
    )


class _CaptureDirectionalCompiler:
    semantics_id = "synthetic_directional_compiler"

    def __init__(self):
        self.translated = None

    def compile(self, source, draft):
        self.translated = draft
        return SimpleNamespace(portfolio_id="synthetic_portfolio")


@pytest.fixture(autouse=True)
def _skip_source_verifier(monkeypatch):
    monkeypatch.setattr(
        compiler_mod,
        "verify_trend_aware_input_sources",
        lambda source: None,
    )


def test_negative_direction_is_compiler_owned():
    source = _source(_view())
    downstream = _CaptureDirectionalCompiler()
    compiler = ContractHardenedTrendHypothesisCompiler(
        directional_compiler=downstream
    )

    compiler.compile(source, _draft())

    translated = downstream.translated
    prediction = translated.hypotheses[0].predicted_observations[0]
    binding = prediction.trend_direction_bindings[0]

    assert prediction.observable == "signal_response"
    assert prediction.expected_direction == "decrease"
    assert binding.independent_change == "increase"
    assert binding.dependent_change == "decrease"
    assert "As feature width increases" in prediction.rationale
    assert "signal response is predicted to decrease" in prediction.rationale


def test_falsifier_observable_is_bound_to_prediction():
    source = _source(_view())
    downstream = _CaptureDirectionalCompiler()
    compiler = ContractHardenedTrendHypothesisCompiler(
        directional_compiler=downstream
    )

    compiler.compile(source, _draft())
    translated = downstream.translated
    falsifier = translated.hypotheses[0].falsification_criteria[0]

    assert falsifier.observable == "signal_response"


def test_llm_directional_restatement_is_rejected():
    source = _source(_view())
    compiler = ContractHardenedTrendHypothesisCompiler(
        directional_compiler=_CaptureDirectionalCompiler()
    )
    bad = _draft(
        mechanism=(
            "Smaller feature width should strengthen the local mechanism."
        )
    )

    with pytest.raises(TrendHypothesisCompileError) as exc:
        compiler.compile(source, bad)

    codes = {row.code for row in exc.value.issues}
    assert "LLM_DIRECTIONAL_AUTHORITY_VIOLATION" in codes


def test_llm_paper_absence_claim_is_rejected():
    source = _source(_view())
    compiler = ContractHardenedTrendHypothesisCompiler(
        directional_compiler=_CaptureDirectionalCompiler()
    )
    bad = _draft(
        rationale=(
            "The study does not report the relevant control, so the "
            "mechanism fills that gap."
        )
    )

    with pytest.raises(TrendHypothesisCompileError) as exc:
        compiler.compile(source, bad)

    codes = {row.code for row in exc.value.issues}
    assert "LLM_PAPER_ABSENCE_AUTHORITY_VIOLATION" in codes


def test_missing_positive_trend_binding_is_rejected():
    source = _source(_view())
    draft = _draft()
    payload = draft.model_dump(mode="json")
    hypothesis = payload["hypotheses"][0]
    hypothesis["predicted_observations"] = [
        {
            "local_id": "p1",
            "prediction_kind": "exploratory",
            "trend_view_ids": [],
            "exploratory_observable": "other_response",
            "exploratory_expected_direction": "unspecified",
            "mechanistic_rationale": "A separate exploratory readout.",
        }
    ]
    repaired = ContractHardenedTrendHypothesisPortfolioDraft.model_validate(
        payload
    )
    compiler = ContractHardenedTrendHypothesisCompiler(
        directional_compiler=_CaptureDirectionalCompiler()
    )

    with pytest.raises(TrendHypothesisCompileError) as exc:
        compiler.compile(source, repaired)

    codes = {row.code for row in exc.value.issues}
    assert "MISSING_HARDENED_TREND_BINDING" in codes


def test_incompatible_binding_group_is_rejected():
    source = _source(
        _view(view_id="v1", direction="negative"),
        _view(view_id="v2", direction="positive"),
    )
    draft = _draft(view_id="v1")
    payload = draft.model_dump(mode="json")
    hypothesis = payload["hypotheses"][0]
    hypothesis["trend_references"].append(
        {
            "view_id": "v2",
            "use_role": "positive_empirical_support",
        }
    )
    hypothesis["predicted_observations"][0][
        "trend_view_ids"
    ] = ["v1", "v2"]
    combined = ContractHardenedTrendHypothesisPortfolioDraft.model_validate(
        payload
    )
    compiler = ContractHardenedTrendHypothesisCompiler(
        directional_compiler=_CaptureDirectionalCompiler()
    )

    with pytest.raises(TrendHypothesisCompileError) as exc:
        compiler.compile(source, combined)

    codes = {row.code for row in exc.value.issues}
    assert "INCOMPATIBLE_HARDENED_TREND_BINDING_GROUP" in codes


@pytest.mark.parametrize(
    ("change", "shapes", "needle"),
    [
        ("increase", ["monotonic"], "predicted to increase"),
        ("decrease", ["monotonic"], "predicted to decrease"),
        ("unchanged", ["monotonic"], "remain unchanged"),
        ("non_monotonic", ["single_optimum"], "single-optimum"),
        ("non_monotonic", ["u_shaped"], "U-shaped"),
        ("non_monotonic", ["inverted_u"], "inverted-U"),
        ("non_monotonic", ["threshold"], "threshold-type"),
        ("non_monotonic", ["saturating"], "saturating"),
        ("non_monotonic", ["unspecified"], "non-monotonic"),
        ("unspecified", [], "remains unspecified"),
    ],
)
def test_renderer_preserves_canonical_increase_frame(
    change,
    shapes,
    needle,
):
    text = render_canonical_trend_clause(
        independent_variable_key="feature_width",
        dependent_observable_key="signal_response",
        dependent_change=change,
        shapes=shapes,
    )
    assert text.startswith("As feature width increases")
    assert needle in text


def test_negative_canonical_render_does_not_trip_frozen_5d1_window():
    text = render_canonical_trend_clause(
        independent_variable_key="particle_size",
        dependent_observable_key="EF",
        dependent_change="decrease",
        shapes=["monotonic"],
    )
    assert _uses_decrease_frame(text, "particle_size") is False


def test_test_fixture_contains_no_real_reserve_identity():
    from pathlib import Path
    source_text = Path(__file__).read_text(encoding="utf-8")
    forbidden_prefix = "SERS" + "_API_"
    assert forbidden_prefix not in source_text
