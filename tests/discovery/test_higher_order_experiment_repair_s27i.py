from __future__ import annotations

from pipeline_core.discovery.higher_order_competing_explanations import (
    generate_competing_explanations,
)
from pipeline_core.discovery.higher_order_discriminating_experiments import (
    generate_discriminating_experiments,
)
from pipeline_core.discovery.higher_order_experiment_critic import (
    critique_discriminating_experiments,
)
from pipeline_core.discovery.higher_order_experiment_repair import (
    repair_discriminating_experiments,
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


def _pipeline(*tensions):
    tension_set = _set(*tensions)
    explanations = generate_competing_explanations(tension_set)
    experiments = generate_discriminating_experiments(
        explanations=explanations,
        tensions=tension_set,
    )
    critique = critique_discriminating_experiments(experiments)
    repairs = repair_discriminating_experiments(
        experiments=experiments,
        critique=critique,
    )
    repaired_critique = critique_discriminating_experiments(
        repairs.repaired_experiments
    )
    return experiments, critique, repairs, repaired_critique


def test_tradeoff_underspecification_is_repaired_deterministically():
    experiments, _, repairs, repaired_critique = _pipeline(
        _tension(
            tid="t:trade",
            tension_type="tradeoff_pareto",
            relation=(
                "hotspot intensity --IMPOSES_TRADEOFF--> hotspot size"
            ),
            candidate=True,
        )
    )

    original = experiments.experiments[0]
    repaired = repairs.repaired_experiments.experiments[0]
    record = repairs.records[0]

    assert "or a task-relevant structural control" in (
        original.intervention_or_sweep
    )
    assert "or a task-relevant structural control" not in (
        repaired.intervention_or_sweep
    )
    assert repaired.intervention_or_sweep.startswith(
        "Perform a matched sweep over nanostructure shape"
    )
    assert repaired.experiment_id != original.experiment_id
    assert record.changed is True
    assert record.actions[0].disposition == (
        "applied_deterministic_repair"
    )
    assert repaired_critique.flagged_experiment_count == 0


def test_measurement_independence_is_deferred_not_invented():
    experiments, _, repairs, repaired_critique = _pipeline(
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

    original = experiments.experiments[0]
    repaired = repairs.repaired_experiments.experiments[0]
    record = repairs.records[0]

    assert repaired == original
    assert record.changed is False
    assert record.actions[0].disposition == (
        "deferred_requires_witness"
    )
    assert "measurement-method" in record.actions[0].required_witness
    assert repaired_critique.flagged_experiment_count == 1


def test_clean_experiment_is_preserved_exactly():
    experiments, critique, repairs, _ = _pipeline(
        _tension(
            tid="t:contrast",
            tension_type="contrasting_relations",
            relation=(
                "electromagnetic enhancement --CONTRASTS_WITH--> "
                "nanoparticle material used for decoration"
            ),
        )
    )

    assert critique.flagged_experiment_count == 0
    assert repairs.repaired_experiments.experiments[0] == (
        experiments.experiments[0]
    )
    assert repairs.records[0].actions[0].disposition == (
        "no_repair_needed"
    )


def test_repair_layer_remains_shadow_non_authorizing():
    _, _, repairs, _ = _pipeline(
        _tension(
            tid="t:trade",
            tension_type="tradeoff_pareto",
            relation=(
                "hotspot intensity --IMPOSES_TRADEOFF--> hotspot size"
            ),
        ),
        _tension(
            tid="t:proxy",
            tension_type="proxy_decoupling",
            relation=(
                "SERS enhancement factor --VARIES_WITH--> "
                "air exposure time"
            ),
            modifier="air exposure time",
            anchor="SERS enhancement factor",
        ),
    )

    assert repairs.deterministic_repair_only is True
    assert repairs.llm_repair_used is False
    assert repairs.original_artifact_mutated is False
    assert repairs.scientific_claim_created is False
    assert repairs.experiment_selection_performed is False
    assert repairs.rejection_authority is False
    assert repairs.production_selection_authority is False
    assert repairs.positive_premise_authority is False
    assert repairs.gap_authority is False
    assert repairs.novelty_authority is False
    assert repairs.external_novelty_review_bypass_authorized is False
