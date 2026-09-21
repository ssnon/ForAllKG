from __future__ import annotations

from pipeline_core.discovery.higher_order_competing_explanations import (
    generate_competing_explanations,
)
from pipeline_core.discovery.higher_order_discriminating_experiments import (
    generate_discriminating_experiments,
)
from pipeline_core.discovery.higher_order_experiment_critic import (
    critique_discriminating_experiment,
    critique_discriminating_experiments,
)
from pipeline_core.discovery.higher_order_tension_extractor import (
    ScientificTensionCandidate,
    ScientificTensionCandidateSet,
)


def _tension(
    *,
    tid: str,
    tension_type: str,
    relation: str,
    modifier: str | None = None,
    anchor: str | None = None,
    candidate: bool = False,
):
    return ScientificTensionCandidate(
        tension_id=tid,
        tension_type=tension_type,
        source_arm_indices=[1],
        source_context_ids=["ctx:1"],
        basis_premise_ids=["premise:1"],
        basis_relation_texts=[relation],
        trigger_codes=["TEST"],
        requested_source="nanostructure shape",
        requested_target=(
            "electromagnetic hotspot location and intensity"
        ),
        tension_statement="unresolved test tension",
        rationale="test",
        modifier_text=modifier,
        modifier_anchor_text=anchor,
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


def _experiments(*tensions):
    tension_set = _set(*tensions)
    explanations = generate_competing_explanations(tension_set)
    return generate_discriminating_experiments(
        explanations=explanations,
        tensions=tension_set,
    )


def _codes(row):
    return {issue.code for issue in row.issues}


def test_proxy_requires_measurement_independence_witness_diagnostic():
    experiments = _experiments(
        _tension(
            tid="t:proxy",
            tension_type="proxy_decoupling",
            relation=(
                "SERS enhancement factor --VARIES_WITH--> "
                "air exposure time"
            ),
            modifier="air exposure time",
            anchor="SERS enhancement factor",
            candidate=True,
        )
    )
    row = critique_discriminating_experiment(
        experiments.experiments[0]
    )

    assert "MEASUREMENT_INDEPENDENCE_UNRESOLVED" in _codes(row)
    assert "PREDICTION_SEPARATION_UNRESOLVED" not in _codes(row)
    assert "BASIS_ALIGNMENT_UNRESOLVED" not in _codes(row)


def test_tradeoff_generic_alternative_control_is_underspecified():
    experiments = _experiments(
        _tension(
            tid="t:trade",
            tension_type="tradeoff_pareto",
            relation=(
                "hotspot intensity --IMPOSES_TRADEOFF--> hotspot size"
            ),
        )
    )
    row = critique_discriminating_experiment(
        experiments.experiments[0]
    )

    assert "INTERVENTION_UNDERSPECIFIED" in _codes(row)
    assert "BASIS_ALIGNMENT_UNRESOLVED" not in _codes(row)


def test_contrast_template_is_structurally_discriminating():
    experiments = _experiments(
        _tension(
            tid="t:contrast",
            tension_type="contrasting_relations",
            relation=(
                "electromagnetic enhancement --CONTRASTS_WITH--> "
                "nanoparticle material used for decoration"
            ),
        )
    )
    row = critique_discriminating_experiment(
        experiments.experiments[0]
    )

    assert "PREDICTION_SEPARATION_UNRESOLVED" not in _codes(row)
    assert "OBSERVABLE_SET_DEGENERATE" not in _codes(row)
    assert "INTERVENTION_UNDERSPECIFIED" not in _codes(row)
    assert "BASIS_ALIGNMENT_UNRESOLVED" not in _codes(row)


def test_identical_prediction_pair_is_flagged_without_authority():
    experiments = _experiments(
        _tension(
            tid="t:contrast",
            tension_type="contrasting_relations",
            relation=(
                "electromagnetic enhancement --CONTRASTS_WITH--> "
                "nanoparticle material used for decoration"
            ),
        )
    )
    base = experiments.experiments[0]
    same = base.predictions[0].predicted_pattern
    tampered = base.model_copy(
        update={
            "predictions": [
                base.predictions[0],
                base.predictions[1].model_copy(
                    update={"predicted_pattern": same}
                ),
            ]
        }
    )
    row = critique_discriminating_experiment(tampered)

    assert "PREDICTION_SEPARATION_UNRESOLVED" in _codes(row)

    report = critique_discriminating_experiments(experiments)
    assert report.experiment_selection_performed is False
    assert report.rejection_authority is False
    assert report.production_selection_authority is False
    assert report.positive_premise_authority is False
    assert report.gap_authority is False
    assert report.novelty_authority is False
