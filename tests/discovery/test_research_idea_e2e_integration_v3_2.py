from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from pipeline_core.discovery.research_idea_contracts import (
    ResearchIdeaKernel,
    ResearchIdeaNode,
)
from pipeline_core.discovery.research_idea_e2e_integration import (
    ResearchIdeaE2ESeedReport,
    build_e2e_seed_execution,
    select_retained_research_ideas,
)


def _sha(value) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _node(*, idea_id: str, source_object_id: str, origin_kind: str, generation: int):
    kernel = ResearchIdeaKernel(
        canonical_intent=f"intent {idea_id}",
        core_scientific_commitments=[f"relation {idea_id}"],
    )
    return ResearchIdeaNode(
        idea_id=idea_id,
        generation_index=generation,
        parent_idea_ids=[],
        source_context_id="ctx",
        source_context_sha256="c" * 64,
        origin_kind=origin_kind,
        source_object_id=source_object_id,
        source_parent_object_ids=[],
        source_kind=("KG_AXIS" if origin_kind == "FRONTIER" else None),
        operator_id=("AXIS_MUTATION" if origin_kind == "EVOLUTION" else None),
        source_artifact_refs=[],
        kernel=kernel,
        kernel_sha256=_sha(kernel),
        task_relation_mode="DIRECT",
    )


def _seed(nodes: list[ResearchIdeaNode]) -> ResearchIdeaE2ESeedReport:
    links = [
        {
            "candidate_id": f"cand-{i}",
            "source_object_id": node.source_object_id,
            "candidate_origin": "FRONTIER" if node.origin_kind == "FRONTIER" else "EVOLUTION",
            "research_idea_id": node.idea_id,
            "research_idea_origin_kind": node.origin_kind,
            "original_generation_index": node.generation_index,
        }
        for i, node in enumerate(nodes, start=1)
    ]
    return ResearchIdeaE2ESeedReport(
        seed_id="seed",
        seed_sha256="s" * 64,
        source_population_id="population",
        source_population_sha256="p" * 64,
        source_evolution_report_id="evolution",
        source_evolution_report_sha256="e" * 64,
        source_pool_id="pool",
        source_pool_sha256="o" * 64,
        source_selection_id="selection",
        source_selection_sha256="l" * 64,
        source_context_id="ctx",
        source_context_sha256="c" * 64,
        selected_candidate_count=len(nodes),
        selected_candidate_ids=[row["candidate_id"] for row in links],
        selected_source_object_ids=[row.source_object_id for row in nodes],
        selected_research_idea_ids=[row.idea_id for row in nodes],
        selected_count_by_candidate_origin={"FRONTIER": 1, "EVOLUTION": len(nodes) - 1},
        links=links,
        research_ideas=nodes,
    )


def test_retained_selection_maps_to_existing_research_idea_identity():
    frontier = _node(
        idea_id="ri-frontier",
        source_object_id="frontier-source",
        origin_kind="FRONTIER",
        generation=0,
    )
    evolution = _node(
        idea_id="ri-evolution",
        source_object_id="evolution-source",
        origin_kind="EVOLUTION",
        generation=1,
    )
    pool = [
        SimpleNamespace(
            candidate_id="c-frontier",
            source_object_id="frontier-source",
            origin="FRONTIER",
        ),
        SimpleNamespace(
            candidate_id="c-evolution",
            source_object_id="evolution-source",
            origin="EVOLUTION",
        ),
    ]
    entries = [
        SimpleNamespace(
            candidate_id="c-evolution",
            source_object_id="evolution-source",
            origin="EVOLUTION",
        ),
        SimpleNamespace(
            candidate_id="c-frontier",
            source_object_id="frontier-source",
            origin="FRONTIER",
        ),
    ]

    selected, links = select_retained_research_ideas(
        projected_nodes=[frontier, evolution],
        pool_candidates=pool,
        selection_entries=entries,
    )

    assert [row.idea_id for row in selected] == ["ri-evolution", "ri-frontier"]
    assert [row.research_idea_id for row in links] == ["ri-evolution", "ri-frontier"]
    assert selected[0].generation_index == 1
    assert selected[1].generation_index == 0


def test_retained_selection_rejects_missing_projection():
    frontier = _node(
        idea_id="ri-frontier",
        source_object_id="frontier-source",
        origin_kind="FRONTIER",
        generation=0,
    )
    pool = [
        SimpleNamespace(
            candidate_id="c-missing",
            source_object_id="missing-source",
            origin="EVOLUTION",
        )
    ]
    entries = [
        SimpleNamespace(
            candidate_id="c-missing",
            source_object_id="missing-source",
            origin="EVOLUTION",
        )
    ]

    with pytest.raises(ValueError, match="No canonical ResearchIdea projection"):
        select_retained_research_ideas(
            projected_nodes=[frontier],
            pool_candidates=pool,
            selection_entries=entries,
        )


def test_seed_execution_preserves_p0_identity_without_claiming_generation_authority():
    nodes = [
        _node(
            idea_id="ri-frontier",
            source_object_id="frontier-source",
            origin_kind="FRONTIER",
            generation=0,
        ),
        _node(
            idea_id="ri-evolution",
            source_object_id="evolution-source",
            origin_kind="EVOLUTION",
            generation=1,
        ),
    ]
    seed = _seed(nodes)
    execution = build_e2e_seed_execution(seed)

    assert execution.generation_index == 2
    assert execution.offspring_generation_executed is False
    assert execution.raw_offspring_count == 0
    assert execution.g4_population_count == 2
    assert execution.carried_forward_idea_ids == ["ri-frontier", "ri-evolution"]
    assert execution.carried_forward_count == 2
    assert [row["generation_index"] for row in execution.g4_population_nodes] == [0, 1]
    assert execution.production_generation_authority is False
    assert execution.production_selection_authority is False
    assert execution.canonical_graph_mutated is False
