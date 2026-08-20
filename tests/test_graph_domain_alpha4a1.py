from __future__ import annotations

import json
from types import SimpleNamespace

import networkx as nx
import pytest

from dac_her.domains.extraction_registry import get_extraction_adapter
from domains.graph_registry import (
    available_graph_adapters,
    get_graph_adapter,
)
from domains.registry import get_domain_profile
from dac_her.paper_graph_postprocess import load_resolution_plan
from dac_her.resolution_candidates import generate_resolution_candidates
from dac_her.strict_recovery import _domain_gate


def _role_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node("material", type="Material", label="test material")
    graph.add_node("experiment", type="Experiment", label="test experiment")
    graph.add_node(
        "measurement",
        type="Measurement",
        metric_id="metric:test",
        subject_id="material",
    )
    graph.add_edge(
        "material",
        "experiment",
        key="e1",
        relation="EVALUATED_IN",
    )
    graph.add_edge(
        "experiment",
        "measurement",
        key="e2",
        relation="HAS_MEASUREMENT",
    )
    graph.add_edge(
        "measurement",
        "material",
        key="e3",
        relation="MEASURED_FOR",
    )
    return graph


def test_graph_adapters_are_registered_and_profile_matched():
    assert set(available_graph_adapters()) >= {"dac_her", "sers_au_ag"}
    assert get_domain_profile("dac_her").graph_adapter_id == "dac_her"
    assert get_domain_profile("sers_au_ag").graph_adapter_id == "sers_au_ag"
    assert get_graph_adapter("dac_her").domain_profile_id == "dac_her"
    assert get_graph_adapter("sers_au_ag").domain_profile_id == "sers_au_ag"


def test_sers_graph_adapter_does_not_infer_catalyst_role():
    normalized, adjustments = get_graph_adapter(
        "sers_au_ag"
    ).normalize_semantic_roles(
        _role_graph(),
        chunk_id="sers:chunk",
    )
    assert normalized.nodes["material"]["type"] == "Material"
    assert adjustments == []


def test_dac_graph_adapter_preserves_existing_catalyst_role_inference():
    normalized, adjustments = get_graph_adapter(
        "dac_her"
    ).normalize_semantic_roles(
        _role_graph(),
        chunk_id="her:chunk",
    )
    assert normalized.nodes["material"]["type"] == "Catalyst"
    assert len(adjustments) == 1


def test_sers_resolution_uses_sers_profile_normalization():
    graph = nx.MultiDiGraph()
    graph.add_node(
        "gap_plural",
        type="StructuralMotif",
        label="Interior nanogaps",
    )
    graph.add_node(
        "gap_single",
        type="StructuralMotif",
        label="Interior nanogap",
    )

    sers_candidates, _ = generate_resolution_candidates(
        graph,
        domain_profile=get_domain_profile("sers_au_ag"),
    )
    assert len(sers_candidates) == 1
    assert sers_candidates[0].signature_equal
    assert not sers_candidates[0].auto_approve

    her_candidates, _ = generate_resolution_candidates(
        graph,
        domain_profile=get_domain_profile("dac_her"),
    )
    assert her_candidates == []


def test_resolution_plan_accepts_domain_owned_resolvable_types(tmp_path):
    graph = nx.MultiDiGraph()
    graph.add_node("left", type="PlasmonicSubstrate", label="Au@Ag")
    graph.add_node("right", type="PlasmonicSubstrate", label="Au@Ag")

    path = tmp_path / "decisions.jsonl"
    path.write_text(
        json.dumps(
            {
                "candidate_id": "resolution:test",
                "decision": "same_entity",
                "approved": True,
                "left_id": "left",
                "right_id": "right",
                "canonical_id": None,
                "reviewer": "tester",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    profile = get_domain_profile("sers_au_ag")
    plan = load_resolution_plan(
        path,
        graph=graph,
        resolvable_node_types=profile.resolution.resolvable_node_types,
    )
    assert plan.applied_aliases == 1

    with pytest.raises(ValueError, match="Unsupported node type"):
        load_resolution_plan(path, graph=graph)


def test_post_recovery_domain_gate_rejects_cross_domain_relation():
    adapter = get_extraction_adapter("sers_au_ag")
    bad = SimpleNamespace(
        entities=[SimpleNamespace(type="PlasmonicSubstrate")],
        edges=[SimpleNamespace(relation="CATALYZES")],
    )
    with pytest.raises(ValueError, match="vocabulary violation"):
        _domain_gate(
            bad,
            extraction_adapter=adapter,
        )
