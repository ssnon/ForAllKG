from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx

from pipeline_core.corpus.extraction.chemistry_signatures import (
    composition_signature,
)

def node_composition_signature(
    graph: nx.Graph,
    node_id: str,
) -> tuple[tuple[str, int], ...]:
    data = graph.nodes[node_id]
    node_type = str(data.get("type", ""))

    # 1. Node 자체 label
    signature = composition_signature(
        data.get("label")
    )
    if signature:
        return signature

    # 2. 명시적인 HAS_METAL 관계
    metals: Counter[str] = Counter()

    for _, target, edge_data in graph.out_edges(
        node_id,
        data=True,
    ):
        if edge_data.get("relation") != "HAS_METAL":
            continue

        for symbol, count in composition_signature(
            graph.nodes[target].get("label", "")
        ):
            metals[symbol] += count

    if metals:
        return tuple(sorted(metals.items()))

    # Support/CoordinationMotif description에서
    # 주변 catalyst metal을 가져오지 않는다.
    if node_type in {
        "Support",
        "CoordinationMotif",
        "Experiment",
        "Calculation",
        "ObservationClaim",
        "MechanismClaim",
    }:
        return ()

    # 필요할 때만 제한적인 fallback
    if node_type in {
        "Catalyst",
        "CatalystModel",
    }:
        return composition_signature(
            data.get("description")
        )

    return ()

def repair_model_of_targets(graph: nx.MultiDiGraph) -> list[dict[str, Any]]:
    """Retarget composition-incompatible MODEL_OF edges when unique and safe."""
    catalysts = [
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if str(data.get("type", "")) == "Catalyst"
    ]
    catalyst_by_signature: dict[tuple[tuple[str, int], ...], list[str]] = {}
    for catalyst_id in catalysts:
        signature = node_composition_signature(graph, catalyst_id)
        if signature:
            catalyst_by_signature.setdefault(signature, []).append(catalyst_id)

    repairs: list[dict[str, Any]] = []
    for source, target, key, data in list(graph.edges(keys=True, data=True)):
        if str(data.get("relation", "")) != "MODEL_OF":
            continue
        model_signature = node_composition_signature(graph, str(source))
        target_signature = node_composition_signature(graph, str(target))
        if not model_signature or not target_signature or model_signature == target_signature:
            continue
        candidates = [
            candidate
            for candidate in catalyst_by_signature.get(model_signature, [])
            if candidate != str(target)
        ]
        if len(candidates) != 1:
            continue
        new_target = candidates[0]
        edge_payload = dict(data)
        graph.remove_edge(source, target, key=key)
        new_key = key
        if graph.has_edge(source, new_target, key=new_key):
            suffix = 1
            base = str(key)
            while graph.has_edge(source, new_target, key=f"{base}:semantic:{suffix}"):
                suffix += 1
            new_key = f"{base}:semantic:{suffix}"
        graph.add_edge(source, new_target, key=new_key, **edge_payload)
        repairs.append({
            "source": str(source),
            "relation": "MODEL_OF",
            "old_target": str(target),
            "new_target": new_target,
            "edge_key": str(new_key),
            "model_composition": dict(model_signature),
            "old_target_composition": dict(target_signature),
            "action": "retargeted_to_unique_composition_match",
        })
    return repairs
