from pipeline_core.discovery.feasibility.experimental_contracts import (
    ExperimentalRealizabilityReport,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.research_value_shadow import (
    assess_research_value_card,
)
from pipeline_core.runtime.validation_contracts import (
    ValidationSpecification,
)


def _card():
    return HypothesisCard.model_construct(
        hypothesis_id="hypothesis:1",
        hypothesis_type="mechanistic_extension",
        predicted_observations=[
            PredictedObservation(
                observation_id="obs:1",
                observable="mode ratio",
                expected_direction="shift",
                rationale="test",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:1",
                observable="mode ratio",
                falsifying_outcome="no shift",
            )
        ],
    )


def _spec():
    return ValidationSpecification(
        specification_id="spec:1",
        hypothesis_id="hypothesis:1",
        source_scope_id="scope:1",
        validation_strategy="mechanism_validation",
        requires_candidate_concretization=False,
        primary_observables=["mode ratio"],
        required_comparisons=[
            "compare the proposed mechanism against at least one plausible alternative pathway"
        ],
        success_patterns=[
            "mode ratio: expected_direction=shift"
        ],
        falsification_patterns=[
            "mode ratio: no shift"
        ],
    )


def _experimental():
    return ExperimentalRealizabilityReport(
        report_id="experimental:1",
        source_intake_id="intake:1",
        source_intake_sha256="a" * 64,
        source_physics_report_id="physics:1",
        source_scope_id="scope:1",
        source_validation_specification_id="spec:1",
        hypothesis_id="hypothesis:1",
        disposition="experimentally_plausible",
        relative_cost_burden="moderate",
        relative_effort_burden="moderate",
    )


def test_research_value_is_independent_of_novelty_and_selection():
    result = assess_research_value_card(
        card=_card(),
        specification=_spec(),
        experimental=_experimental(),
    )

    assert result.mechanistic_discrimination.signal == "STRONG"
    assert result.two_sided_outcome_informativeness.signal == "STRONG"
    assert result.observable_decisiveness.signal == "STRONG"
    assert result.information_gain_proxy.signal == "STRONG"
    assert result.experimental_resolvability.signal == "STRONG"
    assert result.value_argument_class == "VALUE_ARGUMENT_SUPPORTED"

    assert result.novelty_signal_consumed is False
    assert result.external_novelty_used_as_value_evidence is False
    assert result.conceptual_knownness_used_as_value_evidence is False
    assert result.research_value_selection_authority is False
    assert result.production_selection_authority is False


def test_high_cost_is_descriptive_not_automatic_value_rejection():
    experimental = _experimental().model_copy(
        update={
            "relative_cost_burden": "very_high",
            "relative_effort_burden": "high",
        }
    )

    result = assess_research_value_card(
        card=_card(),
        specification=_spec(),
        experimental=experimental,
    )

    assert result.value_argument_class == "VALUE_ARGUMENT_SUPPORTED"
    assert "HIGH_RESOURCE_COST_BURDEN" in result.reason_codes
    assert "HIGH_RESOURCE_EFFORT_BURDEN" in result.reason_codes
    assert result.production_selection_authority is False
