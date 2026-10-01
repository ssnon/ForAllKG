from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
    DiscoveryAxisPlan,
    DiscoveryAxisPlannerPolicy,
)
from pipeline_core.discovery.frontier_idea_population import (
    build_frontier_idea_population,
)
from pipeline_core.discovery.higher_order_competing_explanations import (
    CompetingExplanation,
    CompetingExplanationPair,
    CompetingExplanationSet,
)
from pipeline_core.discovery.higher_order_tension_extractor import (
    ScientificTensionCandidate,
    ScientificTensionCandidateSet,
)
from pipeline_core.discovery.open_world_discovery_axis import (
    ExternalAxisProvenance,
    OpenWorldExternalAxisBundle,
)


CONTEXT_ID = "context:test"
CONTEXT_SHA = "context-sha"
QUESTION = "How does S relate to T?"
TASK_SOURCE = "S"
TASK_TARGET = "T"


def _axis(
    *,
    axis_id: str,
    subject: str,
    relation: str,
    obj: str,
    source_mode: str,
) -> DiscoveryAxis:
    return DiscoveryAxis(
        axis_id=axis_id,
        axis_rank=1,
        inspiration_id=f"inspiration:{axis_id}",
        source_path_id=f"path:{axis_id}",
        candidate_unit_id=f"unit:{axis_id}",
        label=f"{subject} {relation} {obj}",
        proposed_subject=subject,
        proposed_relation=relation,
        proposed_object=obj,
        rendered_path=f"{subject} --{relation}--> {obj}",
        source_mode=source_mode,
        exploration_score=1.0,
        candidate_unit_score=1.0,
        planner_score=1.0,
        mechanistic_continuity_band="high",
        requires_verification=True,
    )


def _plan(
    *,
    plan_id: str,
    bundle_id: str,
    axes: list[DiscoveryAxis],
) -> DiscoveryAxisPlan:
    return DiscoveryAxisPlan(
        plan_id=plan_id,
        plan_sha256=f"sha:{plan_id}",
        source_dual_context_id="dual:test",
        source_dual_context_sha256="dual-sha",
        source_bundle_id=bundle_id,
        source_bundle_sha256=f"sha:{bundle_id}",
        corpus_id="test-corpus",
        axes=axes,
        excluded_inspiration_ids=[],
        policy=DiscoveryAxisPlannerPolicy(),
    )


def _open_bundle(
    plan: DiscoveryAxisPlan,
) -> OpenWorldExternalAxisBundle:
    return OpenWorldExternalAxisBundle(
        bundle_id=plan.source_bundle_id,
        bundle_sha256=plan.source_bundle_sha256,
        source_dual_context_id=plan.source_dual_context_id,
        source_dual_context_sha256=plan.source_dual_context_sha256,
        provenance=[
            ExternalAxisProvenance(
                axis_id=axis.axis_id,
                raw_local_id=f"raw:{axis.axis_id}",
                source_work_ids=[f"work:{axis.axis_id}"],
                source_evidence_spans=["reported relation"],
                compatible_grounded_statement_ids=["stmt:1"],
                bounded_synthesis_note="",
                max_control_similarity=0.2,
                inspiration_role="EXPLORATORY_AXIS",
                external_relation_source_mode="BOUNDED_SYNTHESIS",
                second_order_gap_required=False,
            )
            for axis in plan.axes
        ],
    )


def _higher_order_topology(
    *,
    topology_id: str,
    direct: bool = False,
) -> dict:
    binding = {
        "modifier_text": "orientation",
        "modifier_anchor_text": "surface state",
    }
    if direct:
        binding["anchor_role"] = "source"
    else:
        binding["modifier_anchor_role"] = "mediator"

    return {
        "topology_id": topology_id,
        "backbone_topology_id": f"backbone:{topology_id}",
        "backbone": {
            "topology_id": f"backbone:{topology_id}",
            "requested_source": TASK_SOURCE,
            "requested_target": TASK_TARGET,
            "components": [
                {
                    "component_id": f"component:{topology_id}",
                    "subject": TASK_SOURCE,
                    "relation": "AFFECTS",
                    "object": "surface state",
                }
            ],
        },
        "modifier_component": {
            "component_id": f"modifier:{topology_id}",
            "subject": "orientation",
            "relation": "ASSOCIATED_WITH",
            "object": "surface state",
            "authority": "candidate_inspiration",
        },
        "role_binding": binding,
    }


def _tensions() -> ScientificTensionCandidateSet:
    row = ScientificTensionCandidate(
        tension_id="tension:1",
        tension_type="proxy_decoupling",
        source_arm_indices=[1],
        source_context_ids=["ho-context:1"],
        basis_premise_ids=["premise:1"],
        basis_relation_texts=[
            "orientation --ASSOCIATED_WITH--> surface state"
        ],
        trigger_codes=["MODIFIER_ANCHOR_BRIDGE_OMITTED"],
        requested_source=TASK_SOURCE,
        requested_target=TASK_TARGET,
        tension_statement=(
            "It remains unresolved whether orientation changes T "
            "or only changes the surface-state proxy."
        ),
        rationale="proxy/target distinction",
        modifier_text="orientation",
        modifier_anchor_text="surface state",
        candidate_inspiration_involved=True,
    )
    return ScientificTensionCandidateSet(
        candidate_count=1,
        type_counts={"proxy_decoupling": 1},
        candidate_inspiration_candidate_count=1,
        candidates=[row],
    )


def _explanations() -> CompetingExplanationSet:
    a = CompetingExplanation(
        explanation_id="explanation:a",
        tension_id="tension:1",
        tension_type="proxy_decoupling",
        explanation_type="target_state_change",
        explanation_role="A",
        explanation_statement=(
            "Orientation changes the requested target state."
        ),
        discriminator_requirement="Measure target and proxy independently.",
        source_arm_indices=[1],
        source_context_ids=["ho-context:1"],
        basis_premise_ids=["premise:1"],
        basis_relation_texts=[
            "orientation --ASSOCIATED_WITH--> surface state"
        ],
        candidate_inspiration_involved=True,
    )
    b = CompetingExplanation(
        explanation_id="explanation:b",
        tension_id="tension:1",
        tension_type="proxy_decoupling",
        explanation_type="anchor_or_proxy_only_change",
        explanation_role="B",
        explanation_statement=(
            "Orientation changes only the surface-state proxy."
        ),
        discriminator_requirement="Measure target and proxy independently.",
        source_arm_indices=[1],
        source_context_ids=["ho-context:1"],
        basis_premise_ids=["premise:1"],
        basis_relation_texts=[
            "orientation --ASSOCIATED_WITH--> surface state"
        ],
        candidate_inspiration_involved=True,
    )
    pair = CompetingExplanationPair(
        pair_id="pair:1",
        tension_id="tension:1",
        tension_type="proxy_decoupling",
        requested_source=TASK_SOURCE,
        requested_target=TASK_TARGET,
        tension_statement="Target change versus proxy-only change.",
        explanations=[a, b],
        candidate_inspiration_involved=True,
    )
    return CompetingExplanationSet(
        tension_count=1,
        pair_count=1,
        explanation_count=2,
        type_counts={"proxy_decoupling": 1},
        candidate_inspiration_pair_count=1,
        pairs=[pair],
    )


def _build(
    *,
    kg_plan: DiscoveryAxisPlan | None = None,
    open_plan: DiscoveryAxisPlan | None = None,
    higher_order_topologies=(),
    direct_topologies=(),
    tensions=None,
    explanations=None,
):
    kg_plan = kg_plan or _plan(
        plan_id="plan:kg",
        bundle_id="bundle:kg",
        axes=[
            _axis(
                axis_id="axis:kg",
                subject="S",
                relation="AFFECTS",
                obj="T",
                source_mode="persistent_kg",
            )
        ],
    )
    return build_frontier_idea_population(
        source_context_id=CONTEXT_ID,
        source_context_sha256=CONTEXT_SHA,
        research_question=QUESTION,
        task_source=TASK_SOURCE,
        task_target=TASK_TARGET,
        kg_axis_plan=kg_plan,
        kg_axis_artifact="/tmp/kg.json",
        kg_axis_artifact_sha256="kg-file-sha",
        open_world_axis_plan=open_plan,
        open_world_axis_artifact=(
            "/tmp/open.json"
            if open_plan is not None
            else None
        ),
        open_world_axis_artifact_sha256=(
            "open-file-sha"
            if open_plan is not None
            else None
        ),
        open_world_axis_bundle=(
            _open_bundle(open_plan)
            if open_plan is not None
            else None
        ),
        higher_order_topologies=higher_order_topologies,
        higher_order_topology_artifact=(
            "/tmp/ho.json"
            if higher_order_topologies
            else None
        ),
        higher_order_topology_artifact_sha256=(
            "ho-file-sha"
            if higher_order_topologies
            else None
        ),
        direct_higher_order_topologies=direct_topologies,
        direct_higher_order_topology_artifact=(
            "/tmp/direct-ho.json"
            if direct_topologies
            else None
        ),
        direct_higher_order_topology_artifact_sha256=(
            "direct-ho-file-sha"
            if direct_topologies
            else None
        ),
        tension_candidates=tensions,
        tension_artifact=(
            "/tmp/tensions.json"
            if tensions is not None
            else None
        ),
        tension_artifact_sha256=(
            "tension-file-sha"
            if tensions is not None
            else None
        ),
        competing_explanations=explanations,
        competing_explanations_artifact=(
            "/tmp/explanations.json"
            if explanations is not None
            else None
        ),
        competing_explanations_artifact_sha256=(
            "explanation-file-sha"
            if explanations is not None
            else None
        ),
    )


def test_union_semantics_preserves_all_supplied_source_families():
    open_plan = _plan(
        plan_id="plan:open",
        bundle_id="bundle:open",
        axes=[
            _axis(
                axis_id="axis:open",
                subject="S",
                relation="COUPLES_TO",
                obj="T",
                source_mode="external_open_world",
            )
        ],
    )

    population = _build(
        open_plan=open_plan,
        higher_order_topologies=[
            _higher_order_topology(
                topology_id="ho:1",
            )
        ],
        direct_topologies=[
            _higher_order_topology(
                topology_id="direct-ho:1",
                direct=True,
            )
        ],
        tensions=_tensions(),
        explanations=_explanations(),
    )

    assert population.total_idea_count == 7
    assert population.idea_count_by_source_kind == {
        "DIRECT_HIGHER_ORDER": 1,
        "HIGHER_ORDER": 1,
        "KG_AXIS": 1,
        "OPEN_WORLD_AXIS": 1,
        "TENSION_DERIVED": 3,
    }
    assert population.idea_count_by_idea_form == {
        "COMPETING_EXPLANATION_SEED": 2,
        "HIGHER_ORDER_TOPOLOGY": 2,
        "RELATION_AXIS": 2,
        "TENSION_SEED": 1,
    }


def test_missing_optional_inputs_still_builds_kg_population():
    population = _build()

    assert population.total_idea_count == 1
    assert [
        row.source_kind
        for row in population.ideas
    ] == ["KG_AXIS"]


def test_lineage_binds_exact_upstream_plan_and_artifact_identity():
    population = _build()
    idea = population.ideas[0]
    lineage = idea.source_lineage[0]

    assert lineage.source_object_id == "axis:kg"
    assert lineage.source_plan_id == "plan:kg"
    assert lineage.source_plan_sha256 == "sha:plan:kg"
    assert lineage.source_bundle_id == "bundle:kg"
    assert lineage.source_artifact_sha256 == "kg-file-sha"
    assert idea.source_context_id == CONTEXT_ID
    assert idea.source_context_sha256 == CONTEXT_SHA


def test_open_world_lineage_is_not_promoted_to_positive_evidence():
    open_plan = _plan(
        plan_id="plan:open",
        bundle_id="bundle:open",
        axes=[
            _axis(
                axis_id="axis:open",
                subject="S",
                relation="COUPLES_TO",
                obj="T",
                source_mode="external_open_world",
            )
        ],
    )
    population = _build(
        open_plan=open_plan,
    )

    open_idea = next(
        row
        for row in population.ideas
        if row.source_kind
        == "OPEN_WORLD_AXIS"
    )
    lineage = open_idea.source_lineage[0]

    assert lineage.external_work_ids == ["work:axis:open"]
    assert lineage.compatible_grounded_statement_ids == ["stmt:1"]
    assert open_idea.positive_premise_authority is False
    assert population.positive_premise_authority_created is False
    assert "premise_statement_ids" not in open_idea.model_dump(mode="json")


def test_authority_invariants_hold_for_every_frontier_idea():
    population = _build(
        higher_order_topologies=[
            _higher_order_topology(
                topology_id="ho:1",
            )
        ],
        tensions=_tensions(),
    )

    assert population.shadow_only is True
    assert population.new_llm_calls is False
    assert population.new_retrieval_calls is False
    assert population.stage8_input_changed is False
    assert population.production_selection_authority is False

    for idea in population.ideas:
        assert idea.epistemic_status == "INSPIRATION_ONLY"
        assert idea.requires_verification is True
        assert idea.positive_premise_authority is False
        assert idea.novelty_authority is False
        assert idea.generation_authority is False
        assert idea.production_selection_authority is False


def test_exact_duplicate_group_retains_both_source_lineages_as_ideas():
    kg_plan = _plan(
        plan_id="plan:kg",
        bundle_id="bundle:kg",
        axes=[
            _axis(
                axis_id="axis:kg",
                subject="S",
                relation="AFFECTS",
                obj="T",
                source_mode="persistent_kg",
            )
        ],
    )
    open_plan = _plan(
        plan_id="plan:open",
        bundle_id="bundle:open",
        axes=[
            _axis(
                axis_id="axis:open",
                subject="S",
                relation="AFFECTS",
                obj="T",
                source_mode="external_open_world",
            )
        ],
    )

    population = _build(
        kg_plan=kg_plan,
        open_plan=open_plan,
    )

    assert population.total_idea_count == 2
    assert len(population.exact_duplicate_groups) == 1
    group = population.exact_duplicate_groups[0]
    assert group.cross_source is True
    assert set(group.source_kinds) == {
        "KG_AXIS",
        "OPEN_WORLD_AXIS",
    }
    assert set(group.idea_ids) == {
        row.idea_id
        for row in population.ideas
    }


def test_similarity_diagnostic_has_no_candidate_deletion_authority():
    open_plan = _plan(
        plan_id="plan:open",
        bundle_id="bundle:open",
        axes=[
            _axis(
                axis_id="axis:open",
                subject="S",
                relation="AFFECTS",
                obj="T under condition",
                source_mode="external_open_world",
            )
        ],
    )

    population = _build(
        open_plan=open_plan,
    )

    assert population.total_idea_count == 2
    assert population.overlap_diagnostics
    assert all(
        row.candidate_deletion_authority is False
        for row in population.overlap_diagnostics
    )


def test_population_is_deterministic_for_identical_inputs():
    first = _build(
        higher_order_topologies=[
            _higher_order_topology(
                topology_id="ho:1",
            )
        ],
        tensions=_tensions(),
    )
    second = _build(
        higher_order_topologies=[
            _higher_order_topology(
                topology_id="ho:1",
            )
        ],
        tensions=_tensions(),
    )

    assert (
        first.model_dump(mode="json")
        == second.model_dump(mode="json")
    )


def test_direct_higher_order_remains_structural_inspiration_only():
    population = _build(
        direct_topologies=[
            _higher_order_topology(
                topology_id="direct-ho:1",
                direct=True,
            )
        ]
    )
    idea = next(
        row
        for row in population.ideas
        if row.source_kind
        == "DIRECT_HIGHER_ORDER"
    )

    assert idea.idea_form == "HIGHER_ORDER_TOPOLOGY"
    assert idea.task_relation_mode == "SUBORDINATE"
    assert idea.generation_authority is False
    assert idea.novelty_authority is False
    assert (
        idea.source_lineage[0]
        .candidate_or_unverified_lineage
        is True
    )


def test_e2e_shadow_wiring_precedes_unchanged_stage8_input_selection():
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "scripts"
        / "discovery"
        / "run_dac_discovery_e2e.py"
    )
    text = path.read_text(
        encoding="utf-8"
    )

    stage77 = text.index(
        "[7.7/13] Frontier idea population shadow"
    )
    stage8_input = text.index(
        "stage8_axis_plan_input = _stage8_axis_plan_input("
    )

    assert stage77 < stage8_input
    assert "--frontier-idea-population-shadow" in text
    assert "stage8_input_changed" in text
    assert (
        "frontier_population_shadow"
        in text
    )


def test_open_world_shadow_population_observes_control_and_external_axes_together():
    open_plan = _plan(
        plan_id="plan:open",
        bundle_id="bundle:open",
        axes=[
            _axis(
                axis_id="axis:open",
                subject="external descriptor",
                relation="MODULATES",
                obj="T",
                source_mode="external_open_world",
            )
        ],
    )

    population = _build(
        open_plan=open_plan,
    )

    assert {
        row.source_kind
        for row in population.ideas
    } == {
        "KG_AXIS",
        "OPEN_WORLD_AXIS",
    }
    assert population.stage8_input_changed is False


def test_structural_overlap_detects_axis_reused_as_topology_modifier():
    kg_plan = _plan(
        plan_id="plan:kg",
        bundle_id="bundle:kg",
        axes=[
            _axis(
                axis_id="axis:modifier",
                subject="orientation",
                relation="ASSOCIATED_WITH",
                obj="surface state",
                source_mode="persistent_kg",
            )
        ],
    )
    # Make the topology modifier explicitly reuse the same candidate unit.
    topology = _higher_order_topology(
        topology_id="direct-ho:modifier",
        direct=True,
    )
    topology["modifier_component"]["candidate_unit_id"] = (
        "unit:axis:modifier"
    )

    population = _build(
        kg_plan=kg_plan,
        direct_topologies=[topology],
    )

    assert population.total_idea_count == 2
    assert (
        population
        .cross_source_structural_overlap_pair_count
        == 1
    )

    row = population.structural_overlap_diagnostics[0]
    assert set(
        [row.source_kind_a, row.source_kind_b]
    ) == {
        "KG_AXIS",
        "DIRECT_HIGHER_ORDER",
    }
    assert "SAME_CANDIDATE_UNIT_V1" in row.match_methods
    assert (
        "EXACT_NORMALIZED_RELATION_COMPONENT_V1"
        in row.match_methods
    )
    assert row.shared_candidate_unit_ids == [
        "unit:axis:modifier"
    ]
    assert row.shared_relation_texts == [
        "orientation --ASSOCIATED_WITH--> surface state"
    ]
    assert "PRIMARY_RELATION" in row.relation_roles_a
    assert "MODIFIER_RELATION" in row.relation_roles_b
    assert row.diagnostic_only is True
    assert row.whole_idea_duplicate_asserted is False
    assert row.candidate_deletion_authority is False
    assert row.novelty_authority is False
    assert row.generation_authority is False
    assert row.production_selection_authority is False


def test_structural_overlap_can_use_candidate_lineage_when_text_differs():
    kg_plan = _plan(
        plan_id="plan:kg",
        bundle_id="bundle:kg",
        axes=[
            _axis(
                axis_id="axis:modifier",
                subject="Raman/SERS signal intensity",
                relation="VARIES_WITH",
                obj="excitation wavelength",
                source_mode="persistent_kg",
            )
        ],
    )
    topology = _higher_order_topology(
        topology_id="direct-ho:modifier",
        direct=True,
    )
    topology["modifier_component"].update(
        {
            "subject": "SERS intensity",
            "relation": "VARIES_WITH",
            "object": "excitation wavelength",
            "candidate_unit_id": "unit:axis:modifier",
        }
    )

    population = _build(
        kg_plan=kg_plan,
        direct_topologies=[topology],
    )

    assert (
        population
        .cross_source_structural_overlap_pair_count
        == 1
    )
    row = population.structural_overlap_diagnostics[0]
    assert row.match_methods == [
        "SAME_CANDIDATE_UNIT_V1"
    ]
    assert row.shared_relation_texts == []
    assert row.whole_idea_duplicate_asserted is False


def test_structural_overlap_is_deterministic_and_non_mutating():
    first = _build(
        direct_topologies=[
            _higher_order_topology(
                topology_id="direct-ho:1",
                direct=True,
            )
        ],
    )
    second = _build(
        direct_topologies=[
            _higher_order_topology(
                topology_id="direct-ho:1",
                direct=True,
            )
        ],
    )

    assert (
        first.structural_overlap_diagnostics
        == second.structural_overlap_diagnostics
    )
    assert first.stage8_input_changed is False
    assert first.production_selection_authority is False

