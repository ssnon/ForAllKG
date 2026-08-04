from __future__ import annotations

import re
from collections import Counter
from typing import Any

import networkx as nx

from dac_her.chemistry_signatures import (
    composition_signature,
    metal_signature,
)

_METAL_SYMBOLS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs",
    "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi",
}
_TOKEN_RE = re.compile(r"(?<![A-Za-z])([A-Z][a-z]?)(\d*)")


def composition_signature(value: Any) -> tuple[tuple[str, int], ...]:
    text = str(value or "")
    counts: Counter[str] = Counter()
    for symbol, raw_count in _TOKEN_RE.findall(text):
        if symbol not in metal_signature:
            continue
        counts[symbol] += int(raw_count) if raw_count else 1
    return tuple(sorted(counts.items()))

def node_composition_signature(
    graph: nx.Graph,
    node_id: str,
) -> tuple[tuple[str, int], ...]:
    data = graph.nodes[node_id]

    # 1. label 우선
    signature = composition_signature(data.get("label"))
    if signature:
        return signature

    # 2. 명시적인 HAS_METAL edge
    metals: Counter[str] = Counter()

    for _, target, edge_data in graph.out_edges(
        node_id,
        data=True,
    ):
        if edge_data.get("relation") != "HAS_METAL":
            continue

        label = graph.nodes[target].get("label", "")
        for symbol, count in composition_signature(label):
            metals[symbol] += count

    if metals:
        return tuple(sorted(metals.items()))

    # 3. 마지막 fallback만 description
    return composition_signature(data.get("description"))

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
