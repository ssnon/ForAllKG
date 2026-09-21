from __future__ import annotations

from pipeline_core.discovery.higher_order_competing_explanations import (
    generate_competing_explanations,
)
from pipeline_core.discovery.higher_order_discriminating_experiments import (
    generate_discriminating_experiments,
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


def _run(tensions):
    explanations = generate_competing_explanations(tensions)
    return generate_discriminating_experiments(
        explanations=explanations,
        tensions=tensions,
    )


def test_proxy_experiment_independently_measures_target_and_anchor():
    tensions = _set(
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
    result = _run(tensions)
    row = result.experiments[0]

    assert row.experiment_mode == "matched_modifier_intervention"
    assert "air exposure time" in row.intervention_or_sweep
    assert "SERS enhancement factor" in row.observables
    assert (
        "electromagnetic hotspot location and intensity"
        in row.observables
    )
    assert [x.explanation_role for x in row.predictions] == ["A", "B"]


def test_tradeoff_experiment_uses_both_recorded_quantities():
    tensions = _set(
        _tension(
            tid="t:trade",
            tension_type="tradeoff_pareto",
            relation="hotspot intensity --IMPOSES_TRADEOFF--> hotspot size",
        )
    )
    result = _run(tensions)
    row = result.experiments[0]

    assert row.experiment_mode == "matched_joint_tradeoff_sweep"
    assert "hotspot intensity" in row.observables
    assert "hotspot size" in row.observables


def test_contrast_experiment_varies_contrasted_factor():
    tensions = _set(
        _tension(
            tid="t:contrast",
            tension_type="contrasting_relations",
            relation=(
                "electromagnetic enhancement --CONTRASTS_WITH--> "
                "nanoparticle material used for decoration"
            ),
        )
    )
    result = _run(tensions)
    row = result.experiments[0]

    assert row.experiment_mode == "matched_contrast_test"
    assert (
        "nanoparticle material used for decoration"
        in row.intervention_or_sweep
    )


def test_discriminating_experiments_remain_shadow_non_authorizing():
    tensions = _set(
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
    result = _run(tensions)

    assert result.deterministic_generation is True
    assert result.llm_generation_used is False
    assert result.explanation_selection_performed is False
    assert result.production_selection_authority is False
    assert result.positive_premise_authority is False
    assert result.gap_authority is False
    assert result.novelty_authority is False
    assert result.external_novelty_review_bypass_authorized is False
