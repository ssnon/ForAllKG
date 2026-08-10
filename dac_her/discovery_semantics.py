from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from dac_her.domain_profile import DiscoverySemantics


_ALIGNMENT_NODE_TYPES = frozenset({
    "CORPUSALIGNMENT",
    "CORPUSPATTERN",
})

_ALIGNMENT_EDGE_CLASSES = frozenset({
    "registry_alignment",
    "pattern_alignment",
})


def _value(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalized_node_type(node: Any) -> str:
    raw = (
        _value(node, "node_type", None)
        or _value(node, "type", "")
    )
    return "".join(
        ch for ch in str(raw).upper()
        if ch.isalnum()
    )


def normalized_relation(edge: Any) -> str:
    return str(
        _value(edge, "relation", "")
    ).strip().upper()


def is_alignment_node(node: Any) -> bool:
    return (
        str(_value(node, "graph_layer", "")).strip().lower()
        == "corpus_alignment"
        or str(_value(node, "corpus_node_kind", "")).strip().lower()
        == "alignment_hub"
        or normalized_node_type(node) in _ALIGNMENT_NODE_TYPES
    )


def is_alignment_edge(edge: Any) -> bool:
    return (
        str(_value(edge, "graph_layer", "")).strip().lower()
        == "corpus_alignment"
        or str(_value(edge, "evidence_status", "")).strip().lower()
        == "derived_corpus_alignment"
        or str(_value(edge, "edge_class", "")).strip().lower()
        in _ALIGNMENT_EDGE_CLASSES
    )


def is_mechanism_node(
    node_id: str,
    node: Any,
    semantics: DiscoverySemantics,
) -> bool:
    """Conservative, domain-owned mechanism-node classification."""
    node_type = normalized_node_type(node)
    if any(
        marker.upper() in node_type
        for marker in semantics.mechanism_node_markers
    ):
        return True

    local_id = str(node_id).split("::")[-1].strip().lower()
    return any(
        local_id.startswith(prefix.lower())
        for prefix in semantics.legacy_mechanism_id_prefixes
        if str(prefix).strip()
    )


def is_mechanism_edge(
    edge: Any,
    semantics: DiscoverySemantics,
) -> bool:
    relation = normalized_relation(edge)
    return any(
        marker.upper() in relation
        for marker in semantics.mechanism_relation_markers
    )


def is_scaffold_edge(
    edge: Any,
    semantics: DiscoverySemantics,
) -> bool:
    return normalized_relation(edge) in {
        str(value).strip().upper()
        for value in semantics.scaffold_relations
    }


def contains_strong_causal_language(
    text: str,
    semantics: DiscoverySemantics,
) -> bool:
    return any(
        re.search(pattern, str(text), re.I)
        for pattern in semantics.strong_causal_text_patterns
    )


def edge_has_strong_causal_semantics(
    edge: Any,
    semantics: DiscoverySemantics,
) -> bool:
    relation = normalized_relation(edge)
    return any(
        marker.upper() in relation
        for marker in semantics.strong_causal_relation_markers
    )


def is_shared_entity_node(
    node_id: str,
    node: Any,
    semantics: DiscoverySemantics,
) -> bool:
    del node_id
    if not semantics.shared_entity_types:
        return False
    return normalized_node_type(node) in {
        "".join(
            ch for ch in str(value).upper()
            if ch.isalnum()
        )
        for value in semantics.shared_entity_types
    }
