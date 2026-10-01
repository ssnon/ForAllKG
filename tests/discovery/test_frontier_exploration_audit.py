from __future__ import annotations

import pytest

from pipeline_core.discovery.frontier_exploration_audit import (
    build_frontier_exploration_audit,
    build_frontier_exploration_cohort_audit,
)
from pipeline_core.discovery.frontier_idea_population import (
    FrontierCompetingExplanationSignature,
    FrontierIdea,
    FrontierIdeaPopulation,
    FrontierIdeaSourceLineage,
    FrontierRelationSignature,
    FrontierTensionSignature,
    FrontierTopologySignature,
)


def _lineage(
    source_kind: str,
    object_id: str,
    *,
    contexts=(),
    candidate=False,
    external=False,
):
    return FrontierIdeaSourceLineage(
        source_kind=source_kind,
        source_artifact=f"/{source_kind}.json",
        source_artifact_sha256=f"sha:{source_kind}",
        source_object_id=object_id,
        source_context_ids=list(contexts),
        external_literature_lineage=external,
        candidate_or_unverified_lineage=candidate,
    )


def _axis(
    idea_id: str,
    source_kind: str,
    subject: str,
    relation: str,
    obj: str,
    *,
    candidate_unit_id: str = "",
):
    return FrontierIdea(
        idea_id=idea_id,
        idea_form="RELATION_AXIS",
        source_kind=source_kind,
        source_context_id="context:1",
        source_context_sha256="context-sha",
        source_lineage=[
            _lineage(
                source_kind,
                f"axis:{idea_id}",
                candidate=bool(candidate_unit_id),
                external=source_kind == "OPEN_WORLD_AXIS",
            )
        ],
        rendered_scientific_intent=f"{subject} {relation} {obj}",
        task_relation_mode="SUBORDINATE",
        relation_signature=FrontierRelationSignature(
            subject=subject,
            relation=relation,
            object=obj,
            candidate_unit_id=candidate_unit_id,
        ),
        exact_scientific_signature=f"sig:{idea_id}",
    )


def _topology(
    idea_id: str,
    source_kind: str,
    topology_id: str,
    *,
    backbone: list[str],
    modifier: str,
    modifier_relation: str,
    candidate_unit_id: str = "",
):
    return FrontierIdea(
        idea_id=idea_id,
        idea_form="HIGHER_ORDER_TOPOLOGY",
        source_kind=source_kind,
        source_context_id="context:1",
        source_context_sha256="context-sha",
        source_lineage=[
            _lineage(
                source_kind,
                topology_id,
                candidate=bool(candidate_unit_id),
            )
        ],
        rendered_scientific_intent=f"S -> T with {modifier}",
        task_relation_mode="SUBORDINATE",
        topology_signature=FrontierTopologySignature(
            task_source="S",
            task_target="T",
            modifier_text=modifier,
            modifier_anchor_role="mediator",
            modifier_anchor_text="M",
            modifier_component_id=f"component:{idea_id}",
            modifier_candidate_unit_id=candidate_unit_id,
            modifier_relation_text=modifier_relation,
            backbone_component_ids=[f"b:{idea_id}"],
            backbone_relation_texts=backbone,
        ),
        exact_scientific_signature=f"sig:{idea_id}",
    )


def _tension():
    return FrontierIdea(
        idea_id="idea:tension",
        idea_form="TENSION_SEED",
        source_kind="TENSION_DERIVED",
        source_context_id="context:1",
        source_context_sha256="context-sha",
        source_lineage=[
            _lineage(
                "TENSION_DERIVED",
                "tension:1",
                contexts=["ho-context:1"],
            )
        ],
        rendered_scientific_intent="Target change or proxy-only change?",
        task_relation_mode="UNKNOWN",
        tension_signature=FrontierTensionSignature(
            tension_type="proxy_decoupling",
            requested_source="S",
            requested_target="T",
            tension_statement="Target change or proxy-only change?",
            basis_relation_texts=["M --VARIES_WITH--> modifier-a"],
        ),
        exact_scientific_signature="sig:tension",
    )


def _explanation(role: str):
    return FrontierIdea(
        idea_id=f"idea:explanation:{role}",
        idea_form="COMPETING_EXPLANATION_SEED",
        source_kind="TENSION_DERIVED",
        source_context_id="context:1",
        source_context_sha256="context-sha",
        source_lineage=[
            _lineage(
                "TENSION_DERIVED",
                f"explanation:{role}",
                contexts=["ho-context:1"],
            )
        ],
        rendered_scientific_intent=f"Explanation {role}",
        task_relation_mode="UNKNOWN",
        competing_explanation_signature=(
            FrontierCompetingExplanationSignature(
                tension_id="tension:1",
                tension_type="proxy_decoupling",
                explanation_type=(
                    "target_state_change"
                    if role == "A"
                    else "anchor_or_proxy_only_change"
                ),
                explanation_role=role,
                requested_source="S",
                requested_target="T",
                explanation_statement=f"Explanation {role}",
                discriminator_requirement="Measure both.",
                basis_relation_texts=["M --VARIES_WITH--> modifier-a"],
            )
        ),
        exact_scientific_signature=f"sig:explanation:{role}",
    )


def _population():
    ideas = [
        _axis(
            "idea:kg:1",
            "KG_AXIS",
            "S",
            "AFFECTS",
            "M",
            candidate_unit_id="candidate:kg:1",
        ),
        _axis(
            "idea:kg:2",
            "KG_AXIS",
            "X",
            "VARIES_WITH",
            "Y",
        ),
        _axis(
            "idea:ow:1",
            "OPEN_WORLD_AXIS",
            "oxidation",
            "MODULATES",
            "adsorption",
        ),
        _topology(
            "idea:ho:1",
            "HIGHER_ORDER",
            "topology:1",
            backbone=["S --AFFECTS--> M", "M --PROMOTES--> T"],
            modifier="modifier-a",
            modifier_relation="M --VARIES_WITH--> modifier-a",
            candidate_unit_id="candidate:modifier:a",
        ),
        _topology(
            "idea:ho:2",
            "HIGHER_ORDER",
            "topology:2",
            backbone=["S --AFFECTS--> M", "M --PROMOTES--> T"],
            modifier="modifier-b",
            modifier_relation="M --VARIES_WITH--> modifier-b",
        ),
        _topology(
            "idea:ho:3",
            "HIGHER_ORDER",
            "topology:3",
            backbone=["S --AFFECTS--> M", "M --PROMOTES--> T"],
            modifier="modifier-c",
            modifier_relation="M --VARIES_WITH--> modifier-c",
        ),
        _topology(
            "idea:direct:1",
            "DIRECT_HIGHER_ORDER",
            "direct-topology:1",
            backbone=["S --CORRELATES_WITH--> T"],
            modifier="modifier-d",
            modifier_relation="T --VARIES_WITH--> modifier-d",
        ),
        _tension(),
        _explanation("A"),
        _explanation("B"),
    ]
    return FrontierIdeaPopulation(
        population_id="population:1",
        population_sha256="population-sha",
        source_context_id="context:1",
        source_context_sha256="context-sha",
        research_question="How does S affect T?",
        task_source="S",
        task_target="T",
        ideas=ideas,
        total_idea_count=len(ideas),
        idea_count_by_source_kind={
            "KG_AXIS": 2,
            "OPEN_WORLD_AXIS": 1,
            "HIGHER_ORDER": 3,
            "DIRECT_HIGHER_ORDER": 1,
            "TENSION_DERIVED": 3,
        },
        idea_count_by_idea_form={
            "RELATION_AXIS": 3,
            "HIGHER_ORDER_TOPOLOGY": 4,
            "TENSION_SEED": 1,
            "COMPETING_EXPLANATION_SEED": 2,
        },
        task_relation_mode_counts={"SUBORDINATE": 7, "UNKNOWN": 3},
        exact_duplicate_groups=[],
        cross_source_exact_duplicate_group_count=0,
        overlap_diagnostics=[],
        cross_source_overlap_pair_count=0,
        structural_overlap_diagnostics=[],
        cross_source_structural_overlap_pair_count=0,
        external_literature_lineage_idea_count=1,
        candidate_or_unverified_lineage_idea_count=2,
    )


def test_factorization_separates_raw_topologies_from_families_and_lineage():
    population = _population()
    audit = build_frontier_exploration_audit(
        population=population,
        higher_order_contexts=[
            {
                "context_id": "ho-context:1",
                "higher_order_topology_id": "topology:1",
            }
        ],
        higher_order_generation_report={
            "higher_order_topology_count": 3,
            "strict_backbone_count": 2,
            "strict_two_component_backbone_count": 0,
            "strict_three_component_backbone_count": 2,
            "eligible_modifier_count": 10,
            "eligible_candidate_modifier_count": 1,
            "higher_order_context_count": 3,
            "generation_context_budget": 3,
            "generation_candidate_modifier_context_count": 1,
            "generation_unique_candidate_modifier_count": 1,
            "selected_context_count": 3,
            "proposed_count": 3,
        },
        higher_order_topology_cap=3,
        higher_order_topology_cap_source="TEST",
    )

    assert audit.raw_total_idea_count == 10
    assert audit.primitive_layer.unique_primitive_family_count == 3
    assert audit.primitive_layer.kg_axis_primitive_reused_in_topology_count == 1
    assert audit.primitive_layer.open_world_axis_primitive_reused_in_topology_count == 0
    assert audit.primitive_layer.open_world_exclusive_primitive_family_count == 1
    assert audit.primitive_layer.open_world_uncomposed_primitive_family_count == 1

    assert audit.topology_layer.topology_idea_count == 4
    assert audit.topology_layer.unique_backbone_family_count == 2
    assert audit.topology_layer.unique_modifier_family_count == 4
    assert audit.topology_layer.largest_backbone_family_size == 3
    assert audit.topology_layer.largest_backbone_family_share == pytest.approx(0.75)
    assert audit.topology_layer.candidate_modifier_topology_count == 1

    assert audit.interpretive_layer.tension_seed_count == 1
    assert audit.interpretive_layer.competing_explanation_pair_count == 1
    assert audit.interpretive_layer.exact_context_linked_tension_count == 1
    assert audit.interpretive_layer.exact_full_chain_linked_explanation_seed_count == 2
    assert audit.interpretive_layer.max_verified_structural_derivation_depth == 3

    assert audit.capacity.higher_order_topology_cap_reached is True
    assert audit.capacity.higher_order_topology_cap_utilization == pytest.approx(1.0)
    assert audit.production_selection_authority is False
    assert audit.stage8_input_changed is False


def test_generation_report_must_match_frontier_general_ho_count():
    with pytest.raises(ValueError, match="topology count does not match"):
        build_frontier_exploration_audit(
            population=_population(),
            higher_order_generation_report={
                "higher_order_topology_count": 999,
            },
            higher_order_topology_cap=512,
            higher_order_topology_cap_source="TEST",
        )


def test_cohort_audit_aggregates_without_ranking_or_winner_selection():
    audit = build_frontier_exploration_audit(
        population=_population(),
        higher_order_contexts=[
            {
                "context_id": "ho-context:1",
                "higher_order_topology_id": "topology:1",
            }
        ],
        higher_order_generation_report={
            "higher_order_topology_count": 3,
        },
        higher_order_topology_cap=3,
        higher_order_topology_cap_source="TEST",
    )
    cohort = build_frontier_exploration_cohort_audit(
        audits={"CASE_A": audit, "CASE_B": audit}
    )

    assert cohort.case_count == 2
    assert cohort.case_count_with_topology_cap_reached == 2
    assert cohort.case_count_with_open_world_exclusive_primitives == 2
    assert cohort.case_count_with_open_world_topology_reuse == 0
    assert cohort.case_count_with_interpretive_branches == 2
    assert cohort.case_count_with_candidate_modifier_topologies == 2
    assert cohort.case_count_with_verified_depth_3 == 2
    assert cohort.cross_case_winner_selected is False
    assert cohort.production_selection_authority is False


def test_explanation_without_population_tension_fails_closed():
    population = _population()
    population = population.model_copy(
        update={
            "ideas": [
                idea
                for idea in population.ideas
                if idea.idea_form != "TENSION_SEED"
            ],
            "total_idea_count": 9,
            "idea_count_by_idea_form": {
                "RELATION_AXIS": 3,
                "HIGHER_ORDER_TOPOLOGY": 4,
                "COMPETING_EXPLANATION_SEED": 2,
            },
        }
    )
    with pytest.raises(ValueError, match="tension absent"):
        build_frontier_exploration_audit(
            population=population,
            higher_order_topology_cap=512,
            higher_order_topology_cap_source="TEST",
        )
