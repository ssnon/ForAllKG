from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from pipeline_core.graph_domain import GraphDomainAdapter

SERS_GRAPH_DIAGNOSTICS_VERSION = "sers-alpha4a.5.2"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_json(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


MappingLike = dict[str, Any]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    if not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: _json(value) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def _safe_add_edge(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    key: str,
    attrs: dict[str, Any],
) -> None:
    final_key = key
    suffix = 1
    while graph.has_edge(source, target, final_key):
        final_key = f"{key}:domain_canonicalization:{suffix}"
        suffix += 1
    graph.add_edge(source, target, key=final_key, **attrs)


def _merge_text_attr(current: str | None, incoming: str | None) -> str:
    current = current or ""
    incoming = incoming or ""
    if not current:
        return incoming
    if not incoming:
        return current
    return current if len(current) >= len(incoming) else incoming


def _merge_alias_json(existing: Any, *extra_ids: str) -> str:
    aliases: set[str] = set()
    if existing:
        try:
            payload = json.loads(str(existing))
            if isinstance(payload, list):
                aliases.update(str(x) for x in payload)
        except Exception:
            aliases.add(str(existing))
    aliases.update(extra_ids)
    return json.dumps(sorted(aliases), ensure_ascii=False)


def _paper_node_score(
    node_id: str,
    attrs: MappingLike,
    *,
    paper_id: str,
) -> tuple[int, int, str]:
    label = str(attrs.get("label", ""))
    description = str(attrs.get("description", ""))
    normalized_label = label.strip().lower()
    normalized_paper = paper_id.strip().lower()
    informative = int(bool(label) and normalized_label != normalized_paper)
    length = len(label) + len(description)
    return (informative, length, str(node_id))


def merge_same_paper_nodes(
    graph: nx.MultiDiGraph,
    *,
    paper_id: str,
) -> tuple[nx.MultiDiGraph, list[dict[str, Any]]]:
    paper_nodes = [
        (str(node_id), dict(attrs))
        for node_id, attrs in graph.nodes(data=True)
        if str(attrs.get("type", "")) == "Paper"
    ]
    if len(paper_nodes) <= 1:
        return graph, []

    canonical_id, canonical_attrs = max(
        paper_nodes,
        key=lambda row: _paper_node_score(row[0], row[1], paper_id=paper_id),
    )
    rows: list[dict[str, Any]] = []

    for alias_id, alias_attrs in sorted(paper_nodes):
        if alias_id == canonical_id:
            continue

        graph.nodes[canonical_id]["label"] = _merge_text_attr(
            str(graph.nodes[canonical_id].get("label", "")),
            str(alias_attrs.get("label", "")),
        )
        graph.nodes[canonical_id]["description"] = _merge_text_attr(
            str(graph.nodes[canonical_id].get("description", "")),
            str(alias_attrs.get("description", "")),
        )
        graph.nodes[canonical_id]["aliases_json"] = _merge_alias_json(
            graph.nodes[canonical_id].get("aliases_json"),
            canonical_id,
            alias_id,
        )
        graph.nodes[canonical_id]["paper_identity_canonicalization"] = "same_paper_id"

        for source, _, key, edge_attrs in list(graph.in_edges(alias_id, keys=True, data=True)):
            source_id = canonical_id if source == alias_id else str(source)
            _safe_add_edge(
                graph,
                source_id,
                canonical_id,
                str(key),
                dict(edge_attrs),
            )

        for _, target, key, edge_attrs in list(graph.out_edges(alias_id, keys=True, data=True)):
            target_id = canonical_id if target == alias_id else str(target)
            _safe_add_edge(
                graph,
                canonical_id,
                target_id,
                str(key),
                dict(edge_attrs),
            )

        graph.remove_node(alias_id)
        rows.append({
            "action": "merge_same_paper_id",
            "paper_id": paper_id,
            "canonical_node_id": canonical_id,
            "alias_node_id": alias_id,
            "alias_label": alias_attrs.get("label", ""),
        })

    return graph, rows


def apply_graph_domain_canonicalization(
    graph: nx.MultiDiGraph,
    *,
    graph_adapter: GraphDomainAdapter,
    paper_id: str,
) -> tuple[nx.MultiDiGraph, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    graph, paper_rows = merge_same_paper_nodes(graph, paper_id=paper_id)
    rows.extend(paper_rows)

    summary = {
        "adapter_id": graph_adapter.adapter_id,
        "paper_identity_merges": len(paper_rows),
        "actions": rows,
    }
    return graph, summary


def _contains_computational_acronym(text: str) -> bool:
    """Match computational acronyms as tokens, not arbitrary substrings.

    This prevents false positives such as ``FEM`` matching ``femtomolar``.
    """
    import re

    acronym_pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:DDA|FDTD|FEM|BEM|DFT|TDDFT)(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    return bool(acronym_pattern.search(text))


def node_role_diagnostics(
    graph: nx.MultiDiGraph,
    *,
    domain_profile_id: str,
) -> list[dict[str, Any]]:
    """Detect collection-role contradictions without destructive retyping."""
    if domain_profile_id != "sers_au_ag":
        return []

    rows: list[dict[str, Any]] = []

    strong_calculation_phrases = (
        "discrete dipole",
        "finite-difference time-domain",
        "finite difference time domain",
        "finite element",
        "boundary element",
        "density functional",
        "time-dependent density functional",
        "simulation",
        "simulated",
        "computational model",
        "computational modeling",
        "computational modelling",
        "numerical model",
        "numerical simulation",
        "electromagnetic model",
        "electric-field distribution",
        "e-field simulation",
        "field simulation",
    )
    weak_calculation_markers = (
        "calculation",
        "calculated",
        "computational",
        "modeling",
        "modelling",
        "numerical",
    )
    experimental_identity_markers = (
        "sers",
        "serrs",
        "raman",
        "measurement",
        "measured",
        "spectrum",
        "spectra",
        "spectroscopy",
        "mapping",
        "single-molecule",
        "single molecule",
        "concentration",
        "experimental",
    )
    synthesis_markers = (
        "synthesis",
        "fabrication",
        "growth",
        "reduction",
        "deposition",
        "functionalization",
        "functionalisation",
        "reporter loading",
        "reporter-loading",
    )

    for node_id, attrs in graph.nodes(data=True):
        if str(attrs.get("type", "")) != "Experiment":
            continue

        identity_text = " ".join(
            str(attrs.get(field, ""))
            for field in (
                "label",
                "experiment_type",
                "experiment_family",
                "method_label",
                "raw_method_name",
            )
        ).lower()
        description_text = str(attrs.get("description", "")).lower()
        text = f"{identity_text} {description_text}".strip()

        simulated_by_target = any(
            str(edge_data.get("relation", "")) == "SIMULATED_BY"
            for _, _, _, edge_data in graph.in_edges(
                node_id,
                keys=True,
                data=True,
            )
        )

        explicit_method_signal = (
            _contains_computational_acronym(identity_text)
            or any(
                phrase in identity_text
                for phrase in strong_calculation_phrases
            )
        )

        explicit_description_signal = (
            _contains_computational_acronym(description_text)
            or any(
                phrase in description_text
                for phrase in (
                    "discrete dipole",
                    "finite-difference time-domain",
                    "finite difference time domain",
                    "finite element",
                    "boundary element",
                    "density functional",
                    "time-dependent density functional",
                    "simulation",
                    "simulated",
                    "numerical simulation",
                    "field simulation",
                    "e-field simulation",
                )
            )
        )

        weak_identity_signal = (
            any(
                marker in identity_text
                for marker in weak_calculation_markers
            )
            and not any(
                marker in identity_text
                for marker in experimental_identity_markers
            )
        )

        if (
            simulated_by_target
            or explicit_method_signal
            or explicit_description_signal
            or weak_identity_signal
        ):
            rows.append({
                "severity": "warning",
                "code": "calculation_encoded_as_experiment",
                "node_id": str(node_id),
                "actual_type": "Experiment",
                "expected_type": "Calculation",
                "label": attrs.get("label", ""),
                "message": (
                    f"Experiment {node_id!r} has explicit computational "
                    "role evidence (SIMULATED_BY target or a named "
                    "DDA/FDTD/FEM/BEM/DFT/TDDFT/simulation method); "
                    "review as Calculation."
                ),
            })
            continue

        experiment_family = str(
            attrs.get("experiment_family", "")
        ).lower()
        experiment_type = str(
            attrs.get("experiment_type", "")
        ).lower()
        if (
            experiment_family == "synthesis"
            or experiment_type.startswith("synthesis")
            or (
                any(marker in text for marker in synthesis_markers)
                and "spectroscop" not in text
                and "microscop" not in text
                and "measurement" not in text
                and "stability" not in text
            )
        ):
            rows.append({
                "severity": "warning",
                "code": "synthesis_encoded_as_experiment",
                "node_id": str(node_id),
                "actual_type": "Experiment",
                "expected_type": "SynthesisMethod",
                "label": attrs.get("label", ""),
                "message": (
                    f"Experiment {node_id!r} is synthesis/fabrication-like; "
                    "a protocol used to make or functionalize the substrate "
                    "should be SynthesisMethod."
                ),
            })

    return rows


def evidence_topology_diagnostics(
    graph: nx.MultiDiGraph,
    *,
    domain_profile_id: str,
) -> list[dict[str, Any]]:
    """Audit SERS measurement/evidence/claim topology without mutating graph."""
    if domain_profile_id != "sers_au_ag":
        return []
    rows: list[dict[str, Any]] = []
    def node_type(node_id: Any) -> str:
        return str(graph.nodes[node_id].get("type", ""))
    def incoming_with_relation(node_id: Any, relation: str):
        return [(s,t,k,d) for s,t,k,d in graph.in_edges(node_id, keys=True, data=True) if str(d.get("relation", "")) == relation]
    def outgoing_with_relation(node_id: Any, relation: str):
        return [(s,t,k,d) for s,t,k,d in graph.out_edges(node_id, keys=True, data=True) if str(d.get("relation", "")) == relation]
    evidence_types = {"Experiment", "Calculation", "Measurement"}
    for node_id, attrs in graph.nodes(data=True):
        current_type = str(attrs.get("type", ""))
        if current_type == "Measurement":
            producers = [s for s,_,_,_ in incoming_with_relation(node_id, "HAS_MEASUREMENT") if node_type(s) in {"Experiment", "Calculation"}]
            if not producers:
                rows.append({"severity":"warning","code":"measurement_without_producer","node_id":str(node_id),"node_type":current_type,"label":attrs.get("label", ""),"message":f"Measurement {node_id!r} has no incoming HAS_MEASUREMENT from Experiment/Calculation."})
        elif current_type == "ObservationClaim":
            supporters = [s for s,_,_,_ in incoming_with_relation(node_id, "SUPPORTS_CLAIM") if node_type(s) in evidence_types]
            if not supporters:
                rows.append({"severity":"warning","code":"observation_without_support","node_id":str(node_id),"node_type":current_type,"label":attrs.get("label", ""),"message":f"ObservationClaim {node_id!r} has no valid incoming SUPPORTS_CLAIM evidence."})
            if not outgoing_with_relation(node_id, "APPLIES_TO"):
                rows.append({"severity":"warning","code":"claim_without_application_target","node_id":str(node_id),"node_type":current_type,"label":attrs.get("label", ""),"message":f"ObservationClaim {node_id!r} has no APPLIES_TO target."})
        elif current_type == "MechanismClaim":
            direct_support = [s for s,_,_,_ in incoming_with_relation(node_id, "SUPPORTS_CLAIM") if node_type(s) in evidence_types]
            interpreted_from = [s for s,_,_,_ in incoming_with_relation(node_id, "INTERPRETED_AS") if node_type(s) == "ObservationClaim"]
            if not direct_support and not interpreted_from:
                rows.append({"severity":"warning","code":"mechanism_without_support","node_id":str(node_id),"node_type":current_type,"label":attrs.get("label", ""),"message":f"MechanismClaim {node_id!r} has neither valid direct SUPPORTS_CLAIM evidence nor ObservationClaim INTERPRETED_AS support."})
            if not outgoing_with_relation(node_id, "APPLIES_TO"):
                rows.append({"severity":"warning","code":"claim_without_application_target","node_id":str(node_id),"node_type":current_type,"label":attrs.get("label", ""),"message":f"MechanismClaim {node_id!r} has no APPLIES_TO target."})
    return rows


def _relation_constraint_valid(
    constraint: Any,
    *,
    source_type: str,
    target_type: str,
) -> bool:
    source_ok = (
        not constraint.source_types
        or source_type in constraint.source_types
    )
    target_ok = (
        not constraint.target_types
        or target_type in constraint.target_types
    )
    return source_ok and target_ok


def relation_contract_triage(
    graph: nx.MultiDiGraph,
    *,
    graph_adapter: GraphDomainAdapter,
) -> list[dict[str, Any]]:
    """Classify SERS relation-contract issues without mutating the graph."""
    if graph_adapter.domain_profile_id != "sers_au_ag":
        return []

    constraints = {
        item.relation: item
        for item in graph_adapter.relation_constraints
    }
    raw_issues = graph_adapter.diagnose_relation_contracts(graph)

    grouped: dict[
        tuple[str, str, str, str],
        list[Any],
    ] = defaultdict(list)
    for item in raw_issues:
        grouped[
            (
                item.source_id,
                item.target_id,
                item.edge_key,
                item.relation,
            )
        ].append(item)

    paper_ids = sorted(
        str(node_id)
        for node_id, attrs in graph.nodes(data=True)
        if str(attrs.get("type", "")) == "Paper"
    )
    biological_markers = (
        "cell",
        "cells",
        "tissue",
        "bacteria",
        "bacterial",
        "virus",
        "viral",
        "protein",
        "dna",
        "rna",
        "serum",
        "plasma",
    )

    rows: list[dict[str, Any]] = []
    for (
        source_id,
        target_id,
        edge_key,
        relation,
    ), issues in sorted(grouped.items()):
        source_type = str(graph.nodes[source_id].get("type", ""))
        target_type = str(graph.nodes[target_id].get("type", ""))
        source_label = str(graph.nodes[source_id].get("label", ""))
        target_label = str(graph.nodes[target_id].get("label", ""))

        edge_payload = graph.get_edge_data(
            source_id,
            target_id,
            edge_key,
            default={},
        ) or {}
        evidence_text = str(edge_payload.get("evidence_text", ""))

        category = "unclassified_contract_issue"
        confidence = "medium"
        suggested_action = (
            "Review the source evidence and relation endpoints; do not "
            "auto-repair this edge."
        )
        suggested_relation = ""
        suggested_source_id = ""
        suggested_target_id = ""

        constraint = constraints.get(relation)
        reverse_valid = bool(
            constraint is not None
            and _relation_constraint_valid(
                constraint,
                source_type=target_type,
                target_type=source_type,
            )
        )

        if reverse_valid:
            category = "likely_reversed_relation"
            confidence = "high"
            suggested_action = (
                f"Review reversing this edge to {target_id} "
                f"--{relation}--> {source_id}; source evidence remains "
                "authoritative."
            )
            suggested_relation = relation
            suggested_source_id = target_id
            suggested_target_id = source_id

        elif (
            relation == "TESTED_IN"
            and source_type == "RamanReporter"
            and target_type == "Experiment"
        ):
            category = "wrong_relation_for_role"
            confidence = "high"
            suggested_action = (
                "Prefer Experiment --USES_REPORTER--> RamanReporter when "
                "the reporter role is explicit."
            )
            suggested_relation = "USES_REPORTER"
            suggested_source_id = target_id
            suggested_target_id = source_id

        elif (
            relation == "TESTED_IN"
            and source_type == "Analyte"
            and target_type == "Experiment"
        ):
            category = "wrong_relation_for_role"
            confidence = "high"
            suggested_action = (
                "Prefer Experiment --USES_ANALYTE--> Analyte when the "
                "analyte role is explicit."
            )
            suggested_relation = "USES_ANALYTE"
            suggested_source_id = target_id
            suggested_target_id = source_id

        elif (
            relation == "CHARACTERIZED_IN"
            and source_type == "Experiment"
        ):
            category = "wrong_direction_or_scope"
            confidence = "high"
            suggested_action = (
                "CHARACTERIZED_IN must originate from the characterized "
                "scientific subject and target Experiment. Do not connect "
                "Experiment directly to Paper with this relation."
            )

        elif (
            relation == "HAS_COMPONENT"
            and source_type in {"StructuralMotif", "Morphology"}
        ):
            category = "owner_attachment_required"
            confidence = "high"
            suggested_action = (
                "Attach physical components to the source-grounded owning "
                "PlasmonicSubstrate/Nanostructure/Support, not to the "
                "StructuralMotif/Morphology descriptor."
            )

        elif relation == "PROPOSES_CLAIM" and source_type != "Paper":
            category = "paper_claim_scope_required"
            confidence = "high"
            suggested_action = (
                "PROPOSES_CLAIM is Paper-scoped. Review whether the unique "
                "Paper node explicitly proposes the claim; do not infer that "
                "edge from the current non-Paper source."
            )
            if len(paper_ids) == 1:
                suggested_source_id = paper_ids[0]
                suggested_target_id = target_id
                suggested_relation = "PROPOSES_CLAIM"

        elif relation == "PREPARED_BY" and source_type == "Metal":
            category = "scope_typing_mismatch"
            confidence = "high"
            suggested_action = (
                "Use an explicit produced Material/Nanostructure/Support as "
                "the PREPARED_BY source. An abstract Metal concept should not "
                "stand for a fabricated specimen."
            )

        elif (
            relation == "USES_MATERIAL"
            and (
                source_type == "Calculation"
                or target_type in {"Nanostructure", "Metal"}
            )
        ):
            category = "ontology_relation_gap"
            confidence = "medium"
            suggested_action = (
                "Do not widen USES_MATERIAL automatically. Preserve modeled "
                "medium or synthesis-feedstock semantics in conditions/"
                "description pending an ontology-v2 relation."
            )

        elif relation == "USES_ANALYTE" and target_type == "Material":
            label = target_label.lower()
            if any(marker in label for marker in biological_markers):
                category = "ontology_typing_gap"
                confidence = "medium"
                suggested_action = (
                    "Biological sample/target is not necessarily Analyte. "
                    "Preserve the Material node and review a future "
                    "BiologicalSample/Target ontology; do not retype merely "
                    "to satisfy USES_ANALYTE."
                )

        rows.append({
            "severity": "review",
            "category": category,
            "confidence": confidence,
            "relation": relation,
            "edge_key": edge_key,
            "source_id": source_id,
            "source_type": source_type,
            "source_label": source_label,
            "target_id": target_id,
            "target_type": target_type,
            "target_label": target_label,
            "issue_codes": sorted({
                str(item.code)
                for item in issues
            }),
            "suggested_relation": suggested_relation,
            "suggested_source_id": suggested_source_id,
            "suggested_target_id": suggested_target_id,
            "suggested_action": suggested_action,
            "auto_apply": False,
            "evidence_text": evidence_text[:500],
        })

    return rows


def relation_direction_diagnostics(
    graph: nx.MultiDiGraph,
    *,
    graph_adapter: GraphDomainAdapter,
) -> list[dict[str, Any]]:
    """Return only relation issues that look direction/role repairable."""
    directional_categories = {
        "likely_reversed_relation",
        "wrong_relation_for_role",
        "wrong_direction_or_scope",
    }
    return [
        row
        for row in relation_contract_triage(
            graph,
            graph_adapter=graph_adapter,
        )
        if row["category"] in directional_categories
    ]


_INTEGRATION_BRIDGE_PRIORITY = {
    "PlasmonicSubstrate": 0,
    "Nanostructure": 1,
}

_INTEGRATION_CORE_SUBJECT_TYPES = set(_INTEGRATION_BRIDGE_PRIORITY)

_INTEGRATION_CONTEXT_TYPES = {
    "Metal",
    "Precursor",
    "StructuralMotif",
    "Morphology",
    "Analyte",
    "RamanReporter",
    "OpticalCondition",
}

_INTEGRATION_EVIDENCE_TYPES = {
    "Experiment",
    "Calculation",
    "Measurement",
    "MeasurementGroup",
    "ObservationClaim",
    "MechanismClaim",
}

_REFERENCE_CONTROL_MARKERS = (
    "blank",
    "control",
    "reference",
    "baseline",
    "background",
    "normal raman",
    "normal-raman",
    "without substrate",
    "without sers substrate",
)


def _component_text(
    graph: nx.MultiDiGraph,
    component: set[Any],
) -> str:
    fields = (
        "label",
        "description",
        "experiment_type",
        "experiment_family",
        "method_label",
        "raw_method_name",
    )
    return " ".join(
        str(graph.nodes[node_id].get(field, ""))
        for node_id in component
        for field in fields
    ).lower()


def _integration_component_subtype(
    graph: nx.MultiDiGraph,
    component: set[Any],
) -> tuple[str, str]:
    node_types = {
        str(graph.nodes[node_id].get("type", ""))
        for node_id in component
    }
    component_text = _component_text(graph, component)

    has_reference_control_marker = any(
        marker in component_text
        for marker in _REFERENCE_CONTROL_MARKERS
    )
    has_evidence_chain = bool(
        node_types & _INTEGRATION_EVIDENCE_TYPES
    )
    has_core_subject = bool(
        node_types & _INTEGRATION_CORE_SUBJECT_TYPES
    )

    if has_reference_control_marker and has_evidence_chain:
        return (
            "reference_control_component",
            "Disconnected component is explicitly blank/control/reference-like; "
            "it may remain separate from the main study graph.",
        )

    if has_evidence_chain and not has_core_subject:
        return (
            "missing_subject_anchor",
            "Evidence-bearing component lacks a PlasmonicSubstrate/"
            "Nanostructure subject anchor. Review the modeled/measured subject "
            "rather than bridging a context entity to Paper.",
        )

    if (
        not has_evidence_chain
        and len(component) <= 3
        and node_types
        and node_types <= _INTEGRATION_CONTEXT_TYPES
    ):
        return (
            "isolated_context_entity",
            "Small disconnected component contains only contextual SERS "
            "entities and no evidence chain; do not create a Paper STUDIES "
            "bridge solely for connectivity.",
        )

    if has_core_subject:
        return (
            "scientific_subject_island",
            "Disconnected component contains a core SERS scientific subject. "
            "Review cross-chunk provenance or same-entity resolution before "
            "adding any Paper-level bridge.",
        )

    return (
        "other_disconnected_component",
        "Disconnected component does not match a safe bridge pattern; retain "
        "for review/information without automatic integration.",
    )


def integration_component_diagnostics(
    graph: nx.MultiDiGraph,
    *,
    domain_profile_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify disconnected SERS components and emit review-only bridges."""
    if domain_profile_id != "sers_au_ag":
        return [], []

    paper_ids = sorted(
        str(node_id)
        for node_id, attrs in graph.nodes(data=True)
        if str(attrs.get("type", "")) == "Paper"
    )

    components = sorted(
        (
            set(component)
            for component in nx.weakly_connected_components(graph)
        ),
        key=lambda component: (
            -len(component),
            min(str(node_id) for node_id in component),
        ),
    )

    component_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []

    for component_index, component in enumerate(components, start=1):
        contains_paper = any(
            str(node_id) in paper_ids
            for node_id in component
        )
        if contains_paper:
            continue

        node_ids = sorted(str(node_id) for node_id in component)
        type_counts: dict[str, int] = {}
        for node_id in component:
            node_type = str(graph.nodes[node_id].get("type", ""))
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        bridge_subject_ids = [
            str(node_id)
            for node_id in component
            if str(graph.nodes[node_id].get("type", ""))
            in _INTEGRATION_BRIDGE_PRIORITY
        ]
        bridge_subject_ids.sort(
            key=lambda node_id: (
                _INTEGRATION_BRIDGE_PRIORITY[
                    str(graph.nodes[node_id].get("type", ""))
                ],
                -int(graph.degree(node_id)),
                node_id,
            )
        )

        contains_primary_subject = bool(bridge_subject_ids)
        contains_evidence_chain = any(
            str(graph.nodes[node_id].get("type", ""))
            in _INTEGRATION_EVIDENCE_TYPES
            for node_id in component
        )

        component_subtype, review_reason = (
            _integration_component_subtype(
                graph,
                component,
            )
        )

        severity = (
            "review"
            if component_subtype in {
                "missing_subject_anchor",
                "scientific_subject_island",
            }
            else "info"
        )

        component_rows.append({
            "component_index": component_index,
            "severity": severity,
            "component_subtype": component_subtype,
            "node_count": len(node_ids),
            "edge_count": graph.subgraph(component).number_of_edges(),
            "node_type_counts": type_counts,
            "contains_primary_subject": contains_primary_subject,
            "contains_evidence_chain": contains_evidence_chain,
            "candidate_subject_ids": bridge_subject_ids[:8],
            "sample_node_ids": node_ids[:12],
            "review_reason": review_reason,
        })

        # alpha4a.5: Paper --STUDIES candidates are deliberately restricted to
        # core scientific subjects. Context entities (Support, reporter,
        # analyte, optical condition, etc.) never receive a candidate merely
        # because their component is disconnected.
        if (
            not paper_ids
            or component_subtype != "scientific_subject_island"
        ):
            continue

        for subject_id in bridge_subject_ids[:3]:
            attrs = graph.nodes[subject_id]
            bridge_rows.append({
                "component_index": component_index,
                "severity": "review",
                "component_subtype": component_subtype,
                "source_paper_id": paper_ids[0],
                "target_subject_id": subject_id,
                "target_subject_type": str(attrs.get("type", "")),
                "target_subject_label": str(attrs.get("label", "")),
                "suggested_relation": "STUDIES",
                "confidence": "review",
                "reason": (
                    "Candidate only: disconnected core SERS subject may "
                    "belong to the paper-level study graph. Verify source/"
                    "cross-chunk provenance or same-entity resolution before "
                    "adding any bridge."
                ),
                "auto_apply": False,
            })

    return component_rows, bridge_rows


def duplicate_label_groups(
    graph: nx.MultiDiGraph,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for node_id, attrs in graph.nodes(data=True):
        node_type = str(attrs.get("type", ""))
        label = str(attrs.get("label", "")).strip().lower()
        if not label:
            continue
        buckets[(node_type, label)].append(str(node_id))

    rows: list[dict[str, Any]] = []
    for (node_type, label), node_ids in sorted(buckets.items()):
        if len(node_ids) < 2:
            continue
        rows.append({
            "node_type": node_type,
            "normalized_label": label,
            "count": len(node_ids),
            "node_ids": node_ids,
            "severity": (
                "review"
                if node_type in {
                    "PlasmonicSubstrate",
                    "Nanostructure",
                    "StructuralMotif",
                    "Morphology",
                    "Analyte",
                    "RamanReporter",
                    "OpticalCondition",
                    "Paper",
                }
                else "info"
            ),
        })
    return rows


def component_diagnostics(
    graph: nx.MultiDiGraph,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, component in enumerate(nx.weakly_connected_components(graph), start=1):
        node_ids = sorted(str(node_id) for node_id in component)
        types = sorted({
            str(graph.nodes[node_id].get("type", ""))
            for node_id in component
        })
        has_paper = "Paper" in types
        has_primary = bool({
            "PlasmonicSubstrate",
            "Nanostructure",
            "Catalyst",
            "CatalystModel",
        } & set(types))
        rows.append({
            "component_index": index,
            "node_count": len(node_ids),
            "edge_count": graph.subgraph(component).number_of_edges(),
            "node_types": types,
            "contains_paper": has_paper,
            "contains_primary_subject": has_primary,
            "severity": "warning" if not has_paper and not has_primary else "info",
            "sample_node_ids": node_ids[:12],
        })
    return rows


def write_graph_semantics_report(
    run_dir: Path,
    graph: nx.MultiDiGraph,
    *,
    graph_adapter: GraphDomainAdapter,
) -> dict[str, Any]:
    output_dir = run_dir / "graph_semantics"
    output_dir.mkdir(parents=True, exist_ok=True)

    relation_issues = [
        issue.to_dict()
        for issue in graph_adapter.diagnose_relation_contracts(graph)
    ]
    role_issues = node_role_diagnostics(
        graph,
        domain_profile_id=graph_adapter.domain_profile_id,
    )
    evidence_topology_issues = evidence_topology_diagnostics(
        graph,
        domain_profile_id=graph_adapter.domain_profile_id,
    )
    relation_triage = relation_contract_triage(
        graph,
        graph_adapter=graph_adapter,
    )
    relation_direction_issues = relation_direction_diagnostics(
        graph,
        graph_adapter=graph_adapter,
    )
    integration_components, component_bridge_candidates = (
        integration_component_diagnostics(
            graph,
            domain_profile_id=graph_adapter.domain_profile_id,
        )
    )
    duplicates = duplicate_label_groups(graph)
    components = component_diagnostics(graph)

    _write_json(
        output_dir / "relation_contract_issues.json",
        {
            "adapter_id": graph_adapter.adapter_id,
            "issue_count": len(relation_issues),
            "issues": relation_issues,
        },
    )
    _write_csv(output_dir / "relation_contract_issues.csv", relation_issues)

    _write_json(
        output_dir / "node_role_issues.json",
        {
            "adapter_id": graph_adapter.adapter_id,
            "issue_count": len(role_issues),
            "issues": role_issues,
        },
    )
    _write_csv(output_dir / "node_role_issues.csv", role_issues)

    _write_json(
        output_dir / "evidence_topology_issues.json",
        {
            "adapter_id": graph_adapter.adapter_id,
            "issue_count": len(evidence_topology_issues),
            "issues": evidence_topology_issues,
        },
    )
    _write_csv(output_dir / "evidence_topology_issues.csv", evidence_topology_issues)

    _write_json(
        output_dir / "relation_contract_triage.json",
        {
            "adapter_id": graph_adapter.adapter_id,
            "count": len(relation_triage),
            "rows": relation_triage,
        },
    )
    _write_csv(
        output_dir / "relation_contract_triage.csv",
        relation_triage,
    )

    _write_json(
        output_dir / "relation_direction_issues.json",
        {
            "adapter_id": graph_adapter.adapter_id,
            "count": len(relation_direction_issues),
            "rows": relation_direction_issues,
        },
    )
    _write_csv(
        output_dir / "relation_direction_issues.csv",
        relation_direction_issues,
    )

    _write_json(
        output_dir / "integration_components.json",
        {
            "adapter_id": graph_adapter.adapter_id,
            "count": len(integration_components),
            "components": integration_components,
        },
    )
    _write_csv(
        output_dir / "integration_components.csv",
        integration_components,
    )

    _write_json(
        output_dir / "component_bridge_candidates.json",
        {
            "adapter_id": graph_adapter.adapter_id,
            "count": len(component_bridge_candidates),
            "candidates": component_bridge_candidates,
        },
    )
    _write_csv(
        output_dir / "component_bridge_candidates.csv",
        component_bridge_candidates,
    )

    _write_json(
        output_dir / "duplicate_label_groups.json",
        {
            "count": len(duplicates),
            "groups": duplicates,
        },
    )
    _write_csv(output_dir / "duplicate_label_groups.csv", duplicates)

    _write_json(
        output_dir / "components.json",
        {
            "count": len(components),
            "components": components,
        },
    )
    _write_csv(output_dir / "components.csv", components)

    warning_count = sum(
        1
        for row in relation_issues
        if str(row.get("severity", "")).lower() == "warning"
    )
    error_count = sum(
        1
        for row in relation_issues
        if str(row.get("severity", "")).lower() == "error"
    )
    summary = {
        "adapter_id": graph_adapter.adapter_id,
        "diagnostics_version": SERS_GRAPH_DIAGNOSTICS_VERSION,
        "relation_contract_issue_count": len(relation_issues),
        "relation_contract_warning_count": warning_count,
        "relation_contract_error_count": error_count,
        "node_role_issue_count": len(role_issues),
        "evidence_topology_issue_count": len(evidence_topology_issues),
        "relation_triage_count": len(relation_triage),
        "relation_direction_issue_count": len(relation_direction_issues),
        "relation_triage_category_counts": dict(sorted({
            category: sum(
                1
                for row in relation_triage
                if row["category"] == category
            )
            for category in {
                row["category"]
                for row in relation_triage
            }
        }.items())),
        "integration_review_component_count": sum(
            1
            for row in integration_components
            if row["severity"] == "review"
        ),
        "integration_component_subtype_counts": dict(sorted({
            subtype: sum(
                1
                for row in integration_components
                if row.get("component_subtype") == subtype
            )
            for subtype in {
                row.get("component_subtype", "")
                for row in integration_components
                if row.get("component_subtype")
            }
        }.items())),
        "component_bridge_candidate_count": len(component_bridge_candidates),
        "duplicate_label_group_count": len(duplicates),
        "component_count": len(components),
        "non_primary_component_count": sum(
            1
            for row in components
            if not row["contains_paper"] and not row["contains_primary_subject"]
        ),
        "report_dir": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary
