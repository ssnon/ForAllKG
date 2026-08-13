from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

import networkx as nx

from dac_her.chemistry_signatures import metal_signature


def normalize_sers_bridge_text(value: Any) -> str:
    text = unicodedata.normalize('NFKC', str(value or ''))
    text = text.replace('−', '-').replace('–', '-').replace('—', '-')
    text = text.casefold().strip()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z0-9+.%/@\-\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def architecture_signature(value: Any) -> tuple[str, ...]:
    text = normalize_sers_bridge_text(value)
    found: set[str] = set()
    patterns = {
        'core_shell': r'\bcore[- ]?shell\b|\bau\s*@\s*ag\b|\bag\s*@\s*au\b',
        'alloy': r'\balloy(?:ed|ing)?\b|\bnanoalloy\b',
        'hollow': r'\bhollow\b|\bnanobox\b|\bnanocage\b',
        'core_satellite': r'\bcore[- ]?satellite\b|\bsatellite assembly\b',
        'dimer': r'\bdimer\b|\bpaired nanoparticle',
    }
    for label, pattern in patterns.items():
        if re.search(pattern, text, re.I):
            found.add(label)
    return tuple(sorted(found))


def morphology_signature(value: Any) -> tuple[str, ...]:
    text = normalize_sers_bridge_text(value)
    found: set[str] = set()
    patterns = {
        'nanocube': r'\bnanocube',
        'nanorod': r'\bnanorod',
        'nanostar': r'\bnanostar',
        'nanoplate': r'\bnanoplate|\bnanoprism',
        'nanobox': r'\bnanobox',
        'nanoparticle': r'\bnanoparticle',
        'nanosphere': r'\bnanosphere',
        'film': r'\bfilm\b',
        'monolayer': r'\bmonolayer\b',
        'aggregate': r'\baggregate|\bassembly\b',
    }
    for label, pattern in patterns.items():
        if re.search(pattern, text, re.I):
            found.add(label)
    return tuple(sorted(found))


def structural_motif_signature(value: Any) -> tuple[str, ...]:
    text = normalize_sers_bridge_text(value)
    found: set[str] = set()
    patterns = {
        'nanogap': r'\bnano[- ]?gap|\binterparticle gap|\binterior gap',
        'hotspot': r'\bhot\s*spot|\bhotspot',
        'junction': r'\bjunction\b',
        'interface': r'\binterface|\binterfacial',
        'shell': r'\bshell\b',
        'rim': r'\brim\b',
    }
    for label, pattern in patterns.items():
        if re.search(pattern, text, re.I):
            found.add(label)
    return tuple(sorted(found))


def _condition_context(raw: Any) -> list[dict[str, Any]]:
    if raw in (None, ''):
        return []
    if isinstance(raw, list):
        payload = raw
    else:
        try:
            payload = json.loads(str(raw))
        except (TypeError, json.JSONDecodeError):
            return []
    if not isinstance(payload, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in payload[:12]:
        if not isinstance(item, dict):
            continue
        name = str(item.get('name', '')).strip()
        if not name:
            continue
        value = item.get('value_text')
        if value is None:
            value = item.get('value_numeric')
        rows.append({
            'name': name,
            'value': value,
            'unit': item.get('unit'),
        })
    return rows


def node_sers_signature(
    node_type: str,
    attrs: dict[str, Any],
) -> dict[str, Any]:
    label = str(
        attrs.get('label')
        or attrs.get('statement')
        or attrs.get('metric')
        or ''
    )
    description = str(attrs.get('description') or '')
    combined = ' '.join((label, description))
    return {
        'metal_signature': list(metal_signature(combined)),
        'architecture_signature': list(architecture_signature(combined)),
        'morphology_signature': list(morphology_signature(combined)),
        'structural_motif_signature': list(
            structural_motif_signature(combined)
        ),
        'measurement_context': _condition_context(
            attrs.get('conditions_json', '')
        ),
    }


def strict_node_catalog(graph: nx.Graph) -> list[dict[str, Any]]:
    """Build a compact SERS-aware catalog for Bridge extraction/validation."""
    rows: list[dict[str, Any]] = []
    for node_id, attrs_value in sorted(
        graph.nodes(data=True), key=lambda item: str(item[0])
    ):
        attrs = dict(attrs_value)
        node_type = str(attrs.get('type', ''))
        label = str(
            attrs.get('label')
            or attrs.get('statement')
            or attrs.get('metric')
            or node_id
        )
        incident_relations: set[str] = set()
        if graph.is_directed():
            for _, _, data in graph.in_edges(node_id, data=True):
                incident_relations.add(str(data.get('relation', '')))
            for _, _, data in graph.out_edges(node_id, data=True):
                incident_relations.add(str(data.get('relation', '')))
        else:
            for _, _, data in graph.edges(node_id, data=True):
                incident_relations.add(str(data.get('relation', '')))

        rows.append({
            'id': str(node_id),
            'type': node_type,
            'label': label,
            'description': str(attrs.get('description', ''))[:220],
            'metric_id': str(attrs.get('metric_id', '')),
            'subject_id': str(attrs.get('subject_id', '')),
            'incident_relations': sorted(filter(None, incident_relations)),
            **node_sers_signature(node_type, attrs),
        })
    return rows


_METAL_BEARING_TYPES = frozenset({
    'PlasmonicSubstrate',
    'Nanostructure',
    'Metal',
    'Material',
})


def strong_anchor_context_issues(
    *,
    concept_text: str,
    anchor: dict[str, Any],
    pattern_relation: str | None = None,
    pattern_support_mode: str | None = None,
    pattern_subject: str | None = None,
    pattern_object: str | None = None,
) -> list[str]:
    """Return only high-confidence SERS anchor contradictions.

    Missing architecture/morphology detail is not a contradiction. Hard failure
    is intentionally limited to explicit disjoint metal identity on a
    composition-bearing anchor. This keeps Bridge validation fail-closed without
    turning underspecified labels into false mismatches.
    """
    del pattern_relation, pattern_support_mode, pattern_subject, pattern_object

    anchor_type = str(anchor.get('type', ''))
    if anchor_type not in _METAL_BEARING_TYPES:
        return []

    concept_metals = set(metal_signature(concept_text))
    anchor_metals = set(anchor.get('metal_signature') or [])
    if not anchor_metals:
        anchor_metals = set(metal_signature(anchor.get('label', '')))

    if concept_metals and anchor_metals and concept_metals.isdisjoint(anchor_metals):
        return [
            'explicit Au/Ag metal identity conflicts with the selected SERS anchor'
        ]
    return []
