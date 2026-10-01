from types import SimpleNamespace as NS

from pipeline_core.discovery.idea_evolution import IdeaEvolutionIdea, EvolutionLineageRef
from pipeline_core.discovery.idea_evolution_audit import build_idea_evolution_audit


def _idea(eid, operator, form, relations, *, kinds=(), task="SUBORDINATE", mutation=None):
    refs = [
        EvolutionLineageRef(
            lineage_kind="FRONTIER_IDEA",
            source_object_id=f"{kind}:{index}",
            source_artifact="frontier.json",
            source_artifact_sha256="f" * 64,
            source_kind=kind,
        )
        for index, kind in enumerate(kinds)
    ]
    if not refs:
        refs = [
            EvolutionLineageRef(
                lineage_kind="SCIENTIFIC_REFRAME",
                source_object_id=eid + ":source",
                source_artifact="reframe.json",
                source_artifact_sha256="r" * 64,
                source_kind=operator,
            )
        ]
    return IdeaEvolutionIdea(
        evolution_id=eid,
        operator_id=operator,
        idea_form=form,
        source_context_id="ctx",
        source_context_sha256="c" * 64,
        parent_idea_ids=[ref.source_object_id for ref in refs if ref.lineage_kind == "FRONTIER_IDEA"],
        parent_source_kinds=sorted(set(kinds)),
        lineage_refs=refs,
        title=eid,
        scientific_intent=eid,
        conceptual_change_summary=eid,
        core_relations=relations,
        mutation_kind=mutation,
        differential_prediction="prediction",
        falsification_condition="falsifier",
        discriminating_observation="observation",
        task_relation_mode=task,
        conceptual_family_signature=eid + ":family",
        cross_source_composition=len(set(kinds)) > 1,
        open_world_parent_involved="OPEN_WORLD_AXIS" in kinds,
        candidate_parent_involved=operator == "CANDIDATE_INTERPRETATION",
    )


def test_audit_detects_four_large_transition_modes():
    baseline = NS(
        population_id="pop",
        audit_id="fa",
        source_context_id="ctx",
        source_context_sha256="c" * 64,
        raw_total_idea_count=528,
        subordinate_idea_fraction=0.9886,
        primitive_layer=NS(
            unique_primitive_family_count=8,
            unique_primitive_family_count_by_source_kind={"OPEN_WORLD_AXIS": 3},
            open_world_uncomposed_primitive_family_count=3,
        ),
        topology_layer=NS(
            unique_backbone_family_count=1,
            unique_modifier_family_count=267,
            backbone_families=[
                NS(backbone_relation_texts=["S --PROMOTES--> M", "M --PROMOTES--> T"])
            ],
        ),
    )
    ideas = [
        _idea(
            "bridge", "CROSS_SOURCE_BRIDGE", "CROSS_SOURCE_BRIDGE",
            ["O --MODULATES--> M", "M --CHANGES--> T"],
            kinds=("OPEN_WORLD_AXIS", "KG_AXIS"),
        ),
        _idea(
            "mut", "BACKBONE_MUTATION", "MUTATED_TOPOLOGY",
            ["S --ACTS_THROUGH--> X", "X --PROMOTES--> T"],
            kinds=("HIGHER_ORDER",), mutation="MECHANISM_INSERTION",
        ),
        _idea(
            "cand", "CANDIDATE_INTERPRETATION", "INTERPRETIVE_FORK",
            ["S --MAY_AFFECT--> T"], kinds=("HIGHER_ORDER",),
        ),
        _idea(
            "ref", "LATENT_VARIABLE", "SCIENTIFIC_REFRAME",
            ["latent construct --EXPLAINS--> response"], task="REFRAME",
        ),
    ]
    report = NS(
        report_id="evo",
        report_sha256="e" * 64,
        source_population_id="pop",
        source_exploration_audit_id="fa",
        source_context_id="ctx",
        source_context_sha256="c" * 64,
        research_question="q",
        ideas=ideas,
        native_llm_calls_attempted=3,
        native_llm_calls_succeeded=3,
    )
    audit = build_idea_evolution_audit(
        exploration_audit=baseline,
        evolution_report=report,
    )
    assert audit.cross_source_transition_observed is True
    assert audit.new_backbone_transition_observed is True
    assert audit.candidate_interpretation_observed is True
    assert audit.reframe_observed is True
    assert audit.conceptual_transition_observed is True
    assert audit.new_mutated_backbone_family_count == 1
    assert audit.reframe_count_by_operator == {"LATENT_VARIABLE": 1}
