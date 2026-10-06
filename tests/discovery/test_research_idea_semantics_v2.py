from __future__ import annotations

from types import SimpleNamespace as NS

from pipeline_core.discovery.frontier_idea_population import FrontierIdea
from pipeline_core.discovery.idea_evolution import IdeaEvolutionIdea
from pipeline_core.discovery.research_idea_contracts import (
    ResearchIdeaKernel,
    ResearchIdeaNode,
    ResearchIdeaSearchState,
)
from pipeline_core.discovery.research_idea_projection import (
    project_evolution_idea,
    project_frontier_idea,
)
from pipeline_core.discovery.research_idea_semantics import (
    assess_idea_transition,
)


CONTEXT_ID = "ctx:test"
CONTEXT_SHA = "c" * 64


def _lineage():
    return NS(source_artifact="frontier.json")


def _frontier_relation() -> FrontierIdea:
    return FrontierIdea.model_construct(
        idea_id="frontier:relation",
        idea_form="RELATION_AXIS",
        source_kind="KG_AXIS",
        source_context_id=CONTEXT_ID,
        source_context_sha256=CONTEXT_SHA,
        source_lineage=[_lineage()],
        rendered_scientific_intent="Gap confinement changes hotspot distribution.",
        task_relation_mode="DIRECT",
        relation_signature=NS(
            subject="gap confinement",
            relation="MODULATES",
            object="hotspot distribution",
            candidate_unit_id="",
        ),
        topology_signature=None,
        tension_signature=None,
        competing_explanation_signature=None,
        exact_scientific_signature="legacy-signature",
    )


def _frontier_topology() -> FrontierIdea:
    return FrontierIdea.model_construct(
        idea_id="frontier:topology",
        idea_form="HIGHER_ORDER_TOPOLOGY",
        source_kind="HIGHER_ORDER",
        source_context_id=CONTEXT_ID,
        source_context_sha256=CONTEXT_SHA,
        source_lineage=[_lineage()],
        rendered_scientific_intent="Gap affects field through an interfacial state.",
        task_relation_mode="SUBORDINATE",
        relation_signature=None,
        topology_signature=NS(
            backbone_relation_texts=[
                "gap confinement --MODULATES--> interfacial state",
                "interfacial state --CONTROLS--> hotspot distribution",
            ],
            modifier_text="molecular orientation",
            modifier_anchor_role="mediator",
            modifier_anchor_text="interfacial state",
            modifier_relation_text=(
                "molecular orientation --MODULATES--> interfacial state"
            ),
        ),
        tension_signature=None,
        competing_explanation_signature=None,
        exact_scientific_signature="legacy-topology-signature",
    )


def _evolution(
    evolution_id: str,
    *,
    operator_id: str = "BACKBONE_MUTATION",
    core_relations: list[str] | None = None,
    parent_idea_ids: list[str] | None = None,
) -> IdeaEvolutionIdea:
    return IdeaEvolutionIdea.model_construct(
        evolution_id=evolution_id,
        operator_id=operator_id,
        idea_form="MUTATED_TOPOLOGY",
        source_context_id=CONTEXT_ID,
        source_context_sha256=CONTEXT_SHA,
        parent_idea_ids=(
            ["frontier:relation"]
            if parent_idea_ids is None
            else parent_idea_ids
        ),
        parent_source_kinds=["KG_AXIS"],
        lineage_refs=[NS(source_artifact="evolution.json")],
        title="mutated idea",
        scientific_intent="Gap confinement changes hotspot distribution.",
        conceptual_change_summary="test mutation",
        core_relations=(
            ["gap confinement --MODULATES--> hotspot distribution"]
            if core_relations is None
            else core_relations
        ),
        transformed_question=None,
        mutation_kind="MECHANISM_INSERTION",
        challenged_assumption=None,
        alternative_explanations=[],
        differential_prediction="measure a discriminating response",
        falsification_condition="no response difference",
        discriminating_observation="measure hotspot distribution",
        task_relation_mode="UNKNOWN",
        conceptual_family_signature="legacy-family",
    )


def _node(
    idea_id: str,
    *,
    core: list[str],
    scope: list[str] | None = None,
    contrast: list[str] | None = None,
    question: str | None = None,
    intent: str = "scientific intent",
    generation: int = 0,
) -> ResearchIdeaNode:
    kernel = ResearchIdeaKernel(
        canonical_intent=intent,
        core_scientific_commitments=core,
        scope_commitments=scope or [],
        contrastive_commitments=contrast or [],
        question_commitment=question,
    )
    import hashlib
    import json

    raw = json.dumps(
        kernel.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return ResearchIdeaNode(
        idea_id=idea_id,
        generation_index=generation,
        parent_idea_ids=[],
        source_context_id=CONTEXT_ID,
        source_context_sha256=CONTEXT_SHA,
        origin_kind="FRONTIER",
        source_object_id=f"source:{idea_id}",
        kernel=kernel,
        kernel_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        task_relation_mode="DIRECT",
    )


def test_frontier_relation_projects_core_commitment_without_legacy_family_identity():
    node = project_frontier_idea(_frontier_relation())
    assert node.generation_index == 0
    assert node.kernel.core_scientific_commitments == [
        "gap confinement --MODULATES--> hotspot distribution"
    ]
    assert not hasattr(node, "conceptual_family_signature")
    assert node.semantic_identity_authority is False
    assert node.production_selection_authority is False


def test_topology_projection_separates_backbone_from_modifier_scope():
    node = project_frontier_idea(_frontier_topology())
    assert node.kernel.core_scientific_commitments == [
        "gap confinement --MODULATES--> interfacial state",
        "interfacial state --CONTROLS--> hotspot distribution",
    ]
    assert "modifier: molecular orientation" in node.kernel.scope_commitments
    assert all(
        "molecular orientation" not in row
        for row in node.kernel.core_scientific_commitments
    )


def test_operator_id_does_not_change_kernel_identity_snapshot():
    mapping = {
        "frontier:relation": "research-parent",
    }
    first = project_evolution_idea(
        _evolution("evo:one", operator_id="BACKBONE_MUTATION"),
        parent_research_id_by_source_object_id=mapping,
    )
    second = project_evolution_idea(
        _evolution("evo:two", operator_id="CROSS_SOURCE_BRIDGE"),
        parent_research_id_by_source_object_id=mapping,
    )
    assert first.operator_id != second.operator_id
    assert first.kernel == second.kernel
    assert first.kernel_sha256 == second.kernel_sha256


def test_existing_imported_reframe_does_not_invent_research_idea_parent():
    imported = _evolution(
        "evo:imported",
        operator_id="REGIME_BOUNDARY",
        parent_idea_ids=[],
    )
    node = project_evolution_idea(imported)
    assert node.origin_kind == "IMPORTED_REFRAME"
    assert node.parent_idea_ids == []
    assert node.source_parent_object_ids == []


def test_same_kernel_is_same_idea_even_when_realization_details_change():
    parent = _node(
        "idea:parent",
        core=["gap --MODULATES--> hotspot"],
    )
    proposed = parent.model_copy(
        update={
            "idea_id": "idea:proposal",
            "source_object_id": "source:proposal",
            "generation_index": 1,
            "differential_prediction": "a new measurable prediction",
        }
    )
    assessment = assess_idea_transition(
        parents=[parent],
        proposed=proposed,
        operator_id="EVIDENCE_REAXIS",
    )
    assert assessment.identity_relation == "SAME_IDEA"
    assert assessment.genealogy_relation == "REFINEMENT_OF"
    assert assessment.operator_expectation_consistent is True


def test_replacing_core_mechanism_creates_child_idea():
    parent = _node(
        "idea:parent",
        core=[
            "gap --MODULATES--> mediator M",
            "mediator M --CONTROLS--> hotspot",
        ],
    )
    proposed = _node(
        "idea:proposal",
        core=[
            "gap --MODULATES--> mediator N",
            "mediator N --CONTROLS--> hotspot",
        ],
        generation=1,
    )
    assessment = assess_idea_transition(
        parents=[parent],
        proposed=proposed,
        operator_id="BACKBONE_MUTATION",
    )
    assert assessment.identity_relation == "DIFFERENT_IDEA"
    assert assessment.genealogy_relation == "CHILD_OF"
    assert assessment.operator_expectation_consistent is True


def test_scope_only_change_remains_indeterminate_in_v1():
    parent = _node(
        "idea:parent",
        core=["gap --MODULATES--> hotspot"],
        scope=["wavelength: visible"],
    )
    proposed = _node(
        "idea:proposal",
        core=["gap --MODULATES--> hotspot"],
        scope=["wavelength: near infrared"],
        generation=1,
    )
    assessment = assess_idea_transition(
        parents=[parent],
        proposed=proposed,
    )
    assert assessment.identity_relation == "INDETERMINATE"
    assert assessment.genealogy_relation == "DERIVED_FROM"
    assert assessment.semantic_identity_authority is False


def test_local_axis_mutation_can_escalate_to_research_idea_child():
    parent = _node(
        "idea:parent",
        core=["X --CAUSES--> Y"],
    )
    proposed = _node(
        "idea:proposal",
        core=["Y --CAUSES--> X"],
        generation=1,
    )
    assessment = assess_idea_transition(
        parents=[parent],
        proposed=proposed,
        operator_id="AXIS_MUTATION",
    )
    assert assessment.identity_relation == "DIFFERENT_IDEA"
    assert "LOCAL_REPAIR_ESCALATED_TO_IDEA_MUTATION" in assessment.diagnostic_codes
    assert assessment.operator_expectation_consistent is None


def test_nominal_mutation_that_preserves_kernel_is_reported_as_semantic_noop():
    parent = _node(
        "idea:parent",
        core=["X --CAUSES--> Y"],
    )
    proposed = _node(
        "idea:proposal",
        core=["X --CAUSES--> Y"],
        generation=1,
    )
    assessment = assess_idea_transition(
        parents=[parent],
        proposed=proposed,
        operator_id="BACKBONE_MUTATION",
    )
    assert assessment.identity_relation == "SAME_IDEA"
    assert assessment.operator_expectation_consistent is False
    assert "EXPECTED_IDEA_MUTATION_BUT_SEMANTIC_NOOP" in assessment.diagnostic_codes


def test_scheduler_state_is_separate_from_research_idea_identity():
    state = ResearchIdeaSearchState(idea_id="idea:one", visit_count=2)
    assert state.visit_count == 2
    assert state.compute_spent.llm_calls == 0
    assert not hasattr(state, "kernel")
    assert not hasattr(state, "fitness")
