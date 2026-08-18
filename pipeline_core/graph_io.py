from __future__ import annotations

from pathlib import Path

import networkx as nx

from dac_her.schemas import KnowledgeGraph
import json

def knowledge_graph_to_networkx(
    result: KnowledgeGraph,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(
        paper_id=result.paper_id,
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        document_role=result.document_role,
        section=result.section,
        page_ids_json=json.dumps(result.page_ids, ensure_ascii=False),
        asset_ids_json=json.dumps(result.asset_ids, ensure_ascii=False),
    )

    for node in result.entities:
        graph.add_node(
            node.id,
            type=node.type,
            label=node.label,
            description=node.description or "",
        )

    for node in result.experiments:
        graph.add_node(
            node.id,
            type="Experiment",
            label=node.name,
            experiment_type=node.experiment_type,
            experiment_family=node.experiment_family,
            method_label=node.method_label,
            raw_method_name=node.raw_method_name or "",
            conditions_json=json.dumps(
                [
                    condition.model_dump()
                    for condition in node.conditions
                ],
                ensure_ascii=False,
            ),
            metric_parameters_json=json.dumps(
                {
                    condition.name: (
                        condition.value_text
                        if condition.value_text is not None
                        else condition.value_numeric
                    )
                    for condition in node.conditions
                    if condition.name.strip().lower()
                    in {"analyte", "orbital", "site", "component", "isotope", "phase"}
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            description=node.description or "",
        )

    for node in result.calculations:
        graph.add_node(
            node.id,
            type="Calculation",
            label=node.name,
            calculation_type=node.calculation_type,
            conditions_json=json.dumps(
                [
                    condition.model_dump()
                    for condition in node.conditions
                ],
                ensure_ascii=False,
            ),
            method_details=node.method_details or "",
        )

    for node in result.measurements:
        graph.add_node(
            node.id,
            type="Measurement",
            label=node.metric,
            metric_id=node.metric_id,
            metric=node.metric,
            subject_id=node.subject_id,
            source_expression=node.source_expression,
            group_id=node.group_id or "",
            value_numeric=(
                node.value_numeric
                if node.value_numeric is not None
                else ""
            ),
            value_text=node.value_text or "",
            unit=node.unit or "",
            uncertainty=node.uncertainty or "",
            qualifier=node.qualifier or "",
            basis=node.basis or "",
            conditions_json=json.dumps(
                [
                    condition.model_dump()
                    for condition in node.conditions
                ],
                ensure_ascii=False,
            ),
            metric_parameters_json=json.dumps(
                {
                    condition.name: (
                        condition.value_text
                        if condition.value_text is not None
                        else condition.value_numeric
                    )
                    for condition in node.conditions
                    if condition.name.strip().lower()
                    in {"analyte", "orbital", "site", "component", "isotope", "phase"}
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            description=node.description or "",
        )

    for node in result.measurement_groups:
        graph.add_node(
            node.id,
            type="MeasurementGroup",
            label=node.label,
            group_type=node.group_type,
            member_measurement_ids_json=json.dumps(
                node.member_measurement_ids, ensure_ascii=False
            ),
            description=node.description or "",
        )

    for node in result.observation_claims:
        graph.add_node(
            node.id,
            type="ObservationClaim",
            label=node.statement,
            claim_type=node.claim_type,
            statement=node.statement,
            basis=node.basis,
            description=node.description or "",
        )

    for node in result.mechanism_claims:
        graph.add_node(
            node.id,
            type="MechanismClaim",
            label=node.statement,
            claim_type=node.claim_type,
            statement=node.statement,
            basis=node.basis,
            description=node.description or "",
        )

    for index, edge in enumerate(result.edges):
        graph.add_edge(
            edge.source,
            edge.target,
            key=str(index),
            relation=edge.relation,
            title=edge.relation,

            paper_id=result.paper_id,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            document_role=result.document_role,
            section=result.section,
            page_ids_json=json.dumps(result.page_ids, ensure_ascii=False),
            asset_ids_json=json.dumps(result.asset_ids, ensure_ascii=False),
            evidence_pointers_json=json.dumps(
                [pointer.model_dump() for pointer in edge.evidence_pointers],
                ensure_ascii=False,
            ),
            evidence_asset_ids_json=json.dumps(
                sorted({
                    asset_id
                    for pointer in edge.evidence_pointers
                    for asset_id in pointer.asset_ids
                }),
                ensure_ascii=False,
            ),
            subsection=edge.subsection or "",

            evidence_type=edge.evidence_type,
            evidence_strength=edge.evidence_strength,
            evidence_text=edge.evidence_text,
            confidence=edge.confidence,
            human_verified=False,
        )

    return graph


def save_graphml(
    graph: nx.MultiDiGraph,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nx.write_graphml(
        graph,
        output_path,
    )

    return output_path
