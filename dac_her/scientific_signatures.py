from __future__ import annotations

import re
import unicodedata
from typing import Any

import networkx as nx
from dac_her.chemistry_signatures import (
    METAL_NAMES,
    metal_signature,
)

_ELEMENT_NAMES = {
    name: symbol.lower()
    for name, symbol in METAL_NAMES.items()
}


def normalize_scientific_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.lower().strip()
    for name, symbol in metal_signature.items():
        text = re.sub(rf"\b{re.escape(name)}\b", symbol, text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9+.%/\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def nuclearity_signature(value: Any) -> str:
    text = normalize_scientific_text(value)
    if re.search(r"\b(?:dual[- ]?atom|diatomic|dimer|paired atoms?)\b", text):
        return "dual_atom"
    if re.search(r"\b(?:single[- ]?atom|isolated atom)\b", text):
        return "single_atom"
    if re.search(r"\b(?:nanoparticle|cluster|nanocluster)\b", text):
        return "particle_or_cluster"
    return "unspecified"


def support_signature(value: Any) -> tuple[str, ...]:
    text = normalize_scientific_text(value)
    signatures: set[str] = set()
    patterns = {
        "graphene": r"\bgraphene\b|\bgr\b",
        "pure_graphene": r"\b(?:pure|pristine|undoped) graphene\b",
        "nitrogen_doped_carbon": r"\b(?:n[- ]?doped|nitrogen[- ]?doped) (?:carbon|graphene)\b",
        "nitrogen_vacancy": r"\b(?:n\d*[- ]?)?(?:single[- ]?)?vacanc(?:y|ies)\b|\bsvgn\d*\b",
        "carbon": r"\bcarbon\b|\bcarbonaceous\b",
        "oxide": r"\boxide\b|\bo\d*\b",
        "sulfide": r"\bsulfide\b|\bs\d*\b",
        "nitride": r"\bnitride\b",
        "phosphide": r"\bphosphide\b",
    }
    for label, pattern in patterns.items():
        if re.search(pattern, text):
            signatures.add(label)
    return tuple(sorted(signatures))


def model_physical_signature(node_type: str, value: Any) -> str:
    text = normalize_scientific_text(value)
    if node_type == "CatalystModel" or re.search(
        r"\b(?:model|dft|calculation|computed|slab|supercell)\b", text
    ):
        return "model"
    if node_type == "Catalyst" or re.search(
        r"\b(?:synthesized|prepared|experimental catalyst|electrode)\b", text
    ):
        return "physical"
    return "unspecified"


def node_scientific_signature(node_type: str, attrs: dict[str, Any]) -> dict[str, Any]:
    label = str(attrs.get("label") or attrs.get("statement") or attrs.get("name") or "")
    description = str(attrs.get("description") or "")
    combined = " ".join((label, description))
    return {
        "metal_signature": list(metal_signature(combined)),
        "nuclearity": nuclearity_signature(combined),
        "support_signature": list(support_signature(combined)),
        "model_or_physical": model_physical_signature(node_type, combined),
    }


def strict_node_catalog(graph: nx.Graph) -> list[dict[str, Any]]:
    """Build a compact, chemistry-aware node catalog for bridge extraction."""
    rows: list[dict[str, Any]] = []
    for node_id, attrs_value in sorted(graph.nodes(data=True), key=lambda item: str(item[0])):
        attrs = dict(attrs_value)
        node_type = str(attrs.get("type", ""))
        label = str(
            attrs.get("label")
            or attrs.get("statement")
            or attrs.get("metric")
            or node_id
        )
        incident_relations: set[str] = set()
        if graph.is_directed():
            for _, _, data in graph.in_edges(node_id, data=True):
                incident_relations.add(str(data.get("relation", "")))
            for _, _, data in graph.out_edges(node_id, data=True):
                incident_relations.add(str(data.get("relation", "")))
        else:
            for _, _, data in graph.edges(node_id, data=True):
                incident_relations.add(str(data.get("relation", "")))

        row = {
            "id": str(node_id),
            "type": node_type,
            "label": label,
            "description": str(attrs.get("description", ""))[:220],
            "metric_id": str(attrs.get("metric_id", "")),
            "subject_id": str(attrs.get("subject_id", "")),
            "incident_relations": sorted(filter(None, incident_relations)),
            **node_scientific_signature(node_type, attrs),
        }
        rows.append(row)
    return rows


_METAL_COMPOSITION_ANCHOR_TYPES = frozenset({
    "Catalyst",
    "CatalystModel",
    "Metal",
    "Material",
})
_NUCLEARITY_ANCHOR_TYPES = frozenset({
    "Catalyst",
    "CatalystModel",
    "Material",
})
_SUPPORT_CONTEXT_ANCHOR_TYPES = frozenset({
    "Catalyst",
    "CatalystModel",
    "Support",
    "Material",
})

_METAL_IDENTITY_TERMS = re.compile(
    r"\b(?:metal|element|elemental)\s+(?:identity|type|species|choice|composition)\b",
    re.I,
)


def _is_cross_metal_comparison(
    *,
    pattern_relation: str | None,
    pattern_support_mode: str | None,
    pattern_subject: str | None,
    pattern_object: str | None,
) -> bool:
    """Return true when explicit metals are comparison values, not anchor identity.

    In a derived table comparison such as "binding energy VARIES_WITH metal
    identity", the source spans legitimately mention several disjoint metals.
    Those symbols describe the comparison axis and must not be interpreted as
    the composition of an aggregate MeasurementGroup/Claim anchor.
    """
    if pattern_support_mode != "derived_multi_span":
        return False
    if pattern_relation not in {"VARIES_WITH", "CORRELATES_WITH", "CONTRASTS_WITH"}:
        return False
    pattern_text = " ".join(filter(None, (pattern_subject, pattern_object)))
    return bool(_METAL_IDENTITY_TERMS.search(pattern_text))


def strong_anchor_context_issues(
    *,
    concept_text: str,
    anchor: dict[str, Any],
    pattern_relation: str | None = None,
    pattern_support_mode: str | None = None,
    pattern_subject: str | None = None,
    pattern_object: str | None = None,
) -> list[str]:
    """Return only high-precision context conflicts suitable for hard validation.

    Composition, nuclearity, and support are properties of material/entity
    anchors. Evidence-container anchors such as MeasurementGroup, Calculation,
    ObservationClaim, and MechanismClaim may summarize several materials and
    therefore must not be rejected merely because their label exposes only one
    or none of the compared metals.
    """
    issues: list[str] = []
    text = normalize_scientific_text(concept_text)
    anchor_label = str(anchor.get("label", ""))
    anchor_type = str(anchor.get("type", ""))
    metal_composition_bearing = anchor_type in _METAL_COMPOSITION_ANCHOR_TYPES
    nuclearity_bearing = anchor_type in _NUCLEARITY_ANCHOR_TYPES
    support_context_bearing = anchor_type in _SUPPORT_CONTEXT_ANCHOR_TYPES
    cross_metal_comparison = _is_cross_metal_comparison(
        pattern_relation=pattern_relation,
        pattern_support_mode=pattern_support_mode,
        pattern_subject=pattern_subject,
        pattern_object=pattern_object,
    )

    if metal_composition_bearing and not cross_metal_comparison:
        concept_metals = set(metal_signature(concept_text))
        anchor_metals = set(
            anchor.get("metal_signature") or metal_signature(anchor_label)
        )
        if concept_metals and anchor_metals and concept_metals.isdisjoint(anchor_metals):
            issues.append(
                "explicit metal composition conflicts with the selected anchor"
            )

    if nuclearity_bearing:
        concept_nuclearity = nuclearity_signature(text)
        anchor_nuclearity = str(anchor.get("nuclearity", "unspecified"))
        if (
            concept_nuclearity != "unspecified"
            and anchor_nuclearity != "unspecified"
            and concept_nuclearity != anchor_nuclearity
        ):
            issues.append("single/dual-atom context conflicts with the selected anchor")

    # This support conflict is meaningful only for a material/model-like anchor.
    if support_context_bearing:
        anchor_support = set(
            anchor.get("support_signature") or support_signature(anchor_label)
        )
        explicitly_n_coord = bool(re.search(
            r"\b(?:m\s*[-–—]\s*n|metal\s*[-–—]\s*n|n[- ]coordina|nitrogen[- ]coordina)\b",
            concept_text,
            re.I,
        ))
        if explicitly_n_coord and "pure_graphene" in anchor_support:
            issues.append(
                "nitrogen-coordination concept is anchored to an explicitly pure/pristine graphene context"
            )

    return issues
