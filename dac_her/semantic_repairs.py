from __future__ import annotations

import re
from collections import Counter
from typing import Any

import networkx as nx


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
        if symbol not in _METAL_SYMBOLS:
            continue
        counts[symbol] += int(raw_count) if raw_count else 1
    return tuple(sorted(counts.items()))


def node_composition_signature(graph: nx.Graph, node_id: str) -> tuple[tuple[str, int], ...]:
    data = graph.nodes[node_id]
    for value in (data.get("label"), node_id, data.get("description")):
        signature = composition_signature(value)
        if signature:
            return signature

    metals: Counter[str] = Counter()
    for _, target, edge_data in graph.out_edges(node_id, data=True):
        if str(edge_data.get("relation", "")) != "HAS_METAL":
            continue
        label = graph.nodes[target].get("label", target)
        signature = composition_signature(label)
        for symbol, count in signature:
            metals[symbol] += count
    return tuple(sorted(metals.items()))


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
