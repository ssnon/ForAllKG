from types import SimpleNamespace as NS

import pytest

from pipeline_core.discovery.idea_evolution import (
    IdeaEvolutionDraft,
    IdeaEvolutionOperatorPlan,
    build_idea_evolution_plan,
    compile_native_evolution_draft,
)


def _lineage(*, external=False, candidate=False):
    return NS(
        external_work_ids=["EXT:1"] if external else [],
        external_literature_lineage=external,
        candidate_or_unverified_lineage=candidate,
    )


def _relation(idea_id, source_kind, relation, *, external=False):
    return NS(
        idea_id=idea_id,
        idea_form="RELATION_AXIS",
        source_kind=source_kind,
        task_relation_mode="SUBORDINATE",
        rendered_scientific_intent=relation,
        relation_signature=NS(
            subject=relation.split(" --")[0],
            relation="RELATED_TO",
            object=relation.split("-->")[-1],
            candidate_unit_id="",
        ),
        topology_signature=None,
        source_lineage=[_lineage(external=external)],
    )


def _topology(idea_id, *, candidate=False):
    return NS(
        idea_id=idea_id,
        idea_form="HIGHER_ORDER_TOPOLOGY",
        source_kind="HIGHER_ORDER",
        task_relation_mode="SUBORDINATE",
        rendered_scientific_intent="S -> M -> T with modifier",
        relation_signature=None,
        topology_signature=NS(
            task_source="S",
            task_target="T",
            modifier_text="modifier",
            modifier_anchor_role="mediator",
            modifier_anchor_text="M",
            modifier_component_id="m1",
            modifier_candidate_unit_id="candidate:1" if candidate else "",
            modifier_relation_text="M --VARIES_WITH--> modifier",
            backbone_component_ids=["b1", "b2"],
            backbone_relation_texts=["S --PROMOTES--> M", "M --PROMOTES--> T"],
        ),
        source_lineage=[_lineage(candidate=candidate)],
    )


def _population():
    ideas = [
        _relation("ow1", "OPEN_WORLD_AXIS", "O --MODULATES--> M", external=True),
        _relation("kg1", "KG_AXIS", "K --VARIES_WITH--> M"),
        _topology("ho1"),
        _topology("cand1", candidate=True),
    ]
    return NS(
        population_id="pop:1",
        population_sha256="p" * 64,
        source_context_id="ctx:1",
        source_context_sha256="c" * 64,
        research_question="How does S affect T?",
        task_source="S",
        task_target="T",
        ideas=ideas,
    )


def _audit():
    return NS(
        population_id="pop:1",
        population_sha256="p" * 64,
        audit_id="audit:1",
        audit_sha256="a" * 64,
        topology_layer=NS(
            backbone_families=[NS(topology_idea_ids=["ho1"])],
        ),
    )


def test_plan_exposes_three_native_transition_operators():
    plan = build_idea_evolution_plan(
        population=_population(),
        exploration_audit=_audit(),
    )
    by_id = {row.operator_id: row for row in plan.operator_plans}
    assert by_id["CROSS_SOURCE_BRIDGE"].enabled is True
    assert by_id["BACKBONE_MUTATION"].enabled is True
    assert by_id["CANDIDATE_INTERPRETATION"].enabled is True
    assert plan.open_world_primitive_idea_ids == ["ow1"]
    assert plan.backbone_representative_idea_ids == ["ho1"]
    assert plan.candidate_topology_idea_ids == ["cand1"]


def test_cross_source_requires_open_world_and_distinct_source_families():
    pop = _population()
    op = IdeaEvolutionOperatorPlan(
        operator_id="CROSS_SOURCE_BRIDGE",
        enabled=True,
        parent_pool_idea_ids=["kg1", "ho1"],
        max_output_count=2,
    )
    draft = IdeaEvolutionDraft(
        local_id="x",
        operator_id="CROSS_SOURCE_BRIDGE",
        parent_idea_ids=["kg1", "ho1"],
        title="bridge",
        scientific_intent="test bridge",
        conceptual_change_summary="cross-source bridge",
        core_relations=["K --MEDIATES--> T"],
        differential_prediction="different outcome",
        falsification_condition="no difference",
        discriminating_observation="measure T",
    )
    with pytest.raises(ValueError, match="OPEN_WORLD_AXIS"):
        compile_native_evolution_draft(
            draft=draft,
            operator_plan=op,
            population=pop,
            population_artifact="frontier.json",
            population_artifact_sha256="f" * 64,
        )


def test_backbone_mutation_rejects_unchanged_parent_backbone():
    pop = _population()
    op = IdeaEvolutionOperatorPlan(
        operator_id="BACKBONE_MUTATION",
        enabled=True,
        parent_pool_idea_ids=["ho1"],
        max_output_count=2,
    )
    draft = IdeaEvolutionDraft(
        local_id="m",
        operator_id="BACKBONE_MUTATION",
        parent_idea_ids=["ho1"],
        title="same backbone",
        scientific_intent="unchanged",
        conceptual_change_summary="none",
        core_relations=["S --PROMOTES--> M", "M --PROMOTES--> T"],
        mutation_kind="MECHANISM_INSERTION",
        differential_prediction="different outcome",
        falsification_condition="no difference",
        discriminating_observation="measure T",
    )
    with pytest.raises(ValueError, match="unchanged"):
        compile_native_evolution_draft(
            draft=draft,
            operator_plan=op,
            population=pop,
            population_artifact="frontier.json",
            population_artifact_sha256="f" * 64,
        )


def test_candidate_interpretation_requires_candidate_lineage():
    pop = _population()
    op = IdeaEvolutionOperatorPlan(
        operator_id="CANDIDATE_INTERPRETATION",
        enabled=True,
        parent_pool_idea_ids=["ho1"],
        max_output_count=2,
    )
    draft = IdeaEvolutionDraft(
        local_id="c",
        operator_id="CANDIDATE_INTERPRETATION",
        parent_idea_ids=["ho1"],
        title="fork",
        scientific_intent="fork interpretation",
        conceptual_change_summary="interpret candidate",
        core_relations=["S --MAY_AFFECT--> T"],
        challenged_assumption="one mechanism explains the signal",
        alternative_explanations=["mechanism A", "mechanism B"],
        differential_prediction="A and B diverge",
        falsification_condition="neither model predicts data",
        discriminating_observation="measure mediator and target",
    )
    with pytest.raises(ValueError, match="candidate/unverified"):
        compile_native_evolution_draft(
            draft=draft,
            operator_plan=op,
            population=pop,
            population_artifact="frontier.json",
            population_artifact_sha256="f" * 64,
        )


def test_successful_native_bridge_preserves_external_lineage_without_task_classification_authority():
    pop = _population()
    op = IdeaEvolutionOperatorPlan(
        operator_id="CROSS_SOURCE_BRIDGE",
        enabled=True,
        parent_pool_idea_ids=["ow1", "kg1"],
        max_output_count=2,
    )
    draft = IdeaEvolutionDraft(
        local_id="ok",
        operator_id="CROSS_SOURCE_BRIDGE",
        parent_idea_ids=["ow1", "kg1"],
        title="cross-source bridge",
        scientific_intent="Surface state may condition the task response through adsorption.",
        conceptual_change_summary="combine external adsorption context with KG task material",
        core_relations=["surface state --MODULATES--> adsorption", "adsorption --CONDITIONS--> T"],
        differential_prediction="matched surface state changes separate the response",
        falsification_condition="the response is invariant to the proposed bridge",
        discriminating_observation="measure adsorption state and T under matched conditions",
    )
    idea = compile_native_evolution_draft(
        draft=draft,
        operator_plan=op,
        population=pop,
        population_artifact="frontier.json",
        population_artifact_sha256="f" * 64,
    )
    assert idea.open_world_parent_involved is True
    assert idea.inherited_external_literature_lineage is True
    assert idea.inherited_external_work_ids == ["EXT:1"]
    assert idea.task_relation_mode == "UNKNOWN"
    assert idea.task_relation_classification_authority is False


def test_candidate_parent_pool_deduplicates_candidate_unit_lineage():
    pop = _population()
    duplicate = _topology("cand2", candidate=True)
    pop.ideas.append(duplicate)
    plan = build_idea_evolution_plan(
        population=pop,
        exploration_audit=_audit(),
        max_candidate_parent_pool=8,
    )
    assert plan.candidate_topology_idea_ids == ["cand1"]
