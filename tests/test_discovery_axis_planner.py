from types import SimpleNamespace

from pipeline_core.discovery.discovery_axis_planner import DiscoveryAxisPlanner
from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxisPlannerPolicy


def inspiration(
    name: str,
    *,
    exploration: float,
    unit: float,
    generic: float = 0.3,
    grounding: float = 0.85,
    reaction: float = 0.0,
):
    return SimpleNamespace(
        inspiration_id=f"insp:{name}",
        source_path_id=f"path:{name}",
        candidate_unit_id=f"unit:{name}",
        candidate_unit_label=name,
        candidate_entry_anchor_id=f"entry:{name}",
        candidate_entry_anchor_label=f"entry {name}",
        candidate_exit_anchor_id=f"exit:{name}",
        candidate_exit_anchor_label=f"exit {name}",
        candidate_proposed_subject=name,
        candidate_proposed_relation="modulates",
        candidate_proposed_object="HER",
        rendered_path=f"entry -> {name} -> exit",
        source_mode="exploratory",
        exploration_score=exploration,
        candidate_unit_score=unit,
        mechanistic_continuity_band="high",
        generic_entity_fraction=generic,
        registry_hop_fraction=0.1,
        semantic_similarity_to_grounding=grounding,
        reaction_domain_switch_penalty=reaction,
        requires_verification=True,
        reason_codes=["candidate_unit_route"],
    )


def test_planner_prefers_quality_candidate_units_and_filters_reaction_detour():
    bundle = SimpleNamespace(
        corpus_id="c1",
        bundle_id="bundle:1",
        bundle_sha256="b" * 64,
        inspirations=[
            inspiration("charge donation", exploration=0.55, unit=0.52),
            inspiration("geometry stability", exploration=0.53, unit=0.52),
            inspiration("ORR detour", exploration=0.60, unit=0.60, reaction=1.0),
            inspiration("weak unit", exploration=0.20, unit=0.10),
        ],
    )
    dual = SimpleNamespace(
        dual_context_id="dual:1",
        dual_context_sha256="d" * 64,
        grounded_context=SimpleNamespace(corpus_id="c1"),
        discovery_bundle=bundle,
    )
    plan = DiscoveryAxisPlanner(
        DiscoveryAxisPlannerPolicy(max_axes=5)
    ).build(dual)

    assert [axis.label for axis in plan.axes] == [
        "charge donation",
        "geometry stability",
    ]
    assert "insp:ORR detour" in plan.excluded_inspiration_ids
    assert "insp:weak unit" in plan.excluded_inspiration_ids
    assert all(axis.candidate_unit_id for axis in plan.axes)
