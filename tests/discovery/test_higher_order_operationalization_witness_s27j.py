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
from pipeline_core.discovery.higher_order_operationalization_witness import (
    build_operationalization_witness_requirements,
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


def _requirements(*tensions):
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
    return build_operationalization_witness_requirements(repairs)


def test_proxy_deferred_issue_becomes_witness_requirement():
    result = _requirements(
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

    assert result.requirement_count == 1
    row = result.requirements[0]
    assert row.requested_target_observable == (
        "electromagnetic hotspot location and intensity"
    )
    assert row.anchor_observable == "SERS enhancement factor"
    assert row.distinct_measurement_provider_required is True
    assert row.measurement_independence_verified is False


def test_tradeoff_repair_does_not_create_measurement_requirement():
    result = _requirements(
        _tension(
            tid="t:trade",
            tension_type="tradeoff_pareto",
            relation=(
                "hotspot intensity --IMPOSES_TRADEOFF--> hotspot size"
            ),
        )
    )

    assert result.requirement_count == 0


def test_two_proxy_cases_create_two_independent_requirements():
    result = _requirements(
        _tension(
            tid="t:air",
            tension_type="proxy_decoupling",
            relation=(
                "SERS enhancement factor --VARIES_WITH--> "
                "air exposure time"
            ),
            modifier="air exposure time",
            anchor="SERS enhancement factor",
            candidate=True,
        ),
        _tension(
            tid="t:pt",
            tension_type="proxy_decoupling",
            relation=(
                "field enhancement --VARIES_WITH--> "
                "Pt portion in AuPt nanoparticles"
            ),
            modifier="Pt portion in AuPt nanoparticles",
            anchor="field enhancement",
        ),
    )

    assert result.requirement_count == 2
    assert result.candidate_inspiration_requirement_count == 1
    assert {
        row.anchor_observable
        for row in result.requirements
    } == {
        "SERS enhancement factor",
        "field enhancement",
    }


def test_requirement_layer_does_not_claim_witness_or_authority():
    result = _requirements(
        _tension(
            tid="t:proxy",
            tension_type="proxy_decoupling",
            relation=(
                "SERS enhancement factor --VARIES_WITH--> "
                "air exposure time"
            ),
            modifier="air exposure time",
            anchor="SERS enhancement factor",
        )
    )

    assert result.corpus_lookup_performed is False
    assert result.measurement_independence_verified_count == 0
    assert result.witness_candidate_count == 0
    assert result.experiment_selection_performed is False
    assert result.rejection_authority is False
    assert result.production_selection_authority is False
    assert result.positive_premise_authority is False
    assert result.gap_authority is False
    assert result.novelty_authority is False
    assert result.external_novelty_review_bypass_authorized is False
