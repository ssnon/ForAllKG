from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from dac_her.claim_overlap import write_claim_overlap_audit
from dac_her.measurement_scalarization import numeric_tokens


DEFAULT_GRAPHML = (
    "data_dac/extracted/Zhang2019_PtRu/"
    "Zhang2019_PtRu.graphml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and audit a merged DAC-HER GraphML graph."
        )
    )
    parser.add_argument(
        "--graphml",
        default=DEFAULT_GRAPHML,
        help="Path to the GraphML file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for CSV/audit outputs. "
            "Default: <graphml_parent>/audit"
        ),
    )
    parser.add_argument(
        "--asset-manifest",
        default=None,
        help=(
            "Optional asset_manifest.json. If omitted, the auditor looks "
            "beside the run GraphML and through latest_run.json."
        ),
    )
    parser.add_argument(
        "--show-limit",
        type=int,
        default=30,
        help="Maximum items printed per detailed section.",
    )
    return parser.parse_args()


def clean_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def edge_records(
    graph: nx.Graph,
) -> Iterable[tuple[str, str, str, dict[str, Any]]]:
    if graph.is_multigraph():
        for source, target, key, data in graph.edges(
            keys=True,
            data=True,
        ):
            yield str(source), str(target), str(key), dict(data)
    else:
        for index, (source, target, data) in enumerate(
            graph.edges(data=True)
        ):
            yield (
                str(source),
                str(target),
                str(index),
                dict(data),
            )


def normalized_label(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.lower().strip()
    value = re.sub(r"[\s_/–—-]+", " ", value)
    value = re.sub(r"[^\w\s]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def resolve_asset_manifest(graphml_path: Path, override: str | None) -> Path | None:
    if override:
        path = Path(override).resolve()
        return path if path.exists() else None
    direct = graphml_path.parent / "asset_manifest.json"
    if direct.exists():
        return direct
    latest_pointer = graphml_path.parent / "latest_run.json"
    if latest_pointer.exists():
        try:
            payload = json.loads(latest_pointer.read_text(encoding="utf-8"))
            candidate = Path(payload["run_directory"]) / "asset_manifest.json"
            if candidate.exists():
                return candidate
        except Exception:
            pass
    return None


def composite_measurement_rows(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    markers = (";", "respectively", " in acid", " in base", " before ", " after ")
    rows = []
    for row in measurements:
        text = str(row.get("value_text", "") or "")
        lowered = f" {text.lower()} "
        if len(numeric_tokens(text)) >= 2 and any(marker in lowered for marker in markers):
            rows.append({
                "id": row.get("id", ""),
                "metric_id": row.get("metric_id", ""),
                "subject_id": row.get("subject_id", ""),
                "value_text": text,
                "issue": "multiple scalar values/conditions in one Measurement",
            })
    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(
                            value,
                            ensure_ascii=False,
                        )
                        if isinstance(
                            value,
                            (dict, list, tuple),
                        )
                        else clean_value(value)
                    )
                    for key, value in row.items()
                }
            )


def component_sizes(
    graph: nx.Graph,
) -> list[int]:
    if graph.is_directed():
        components = nx.weakly_connected_components(graph)
    else:
        components = nx.connected_components(graph)

    return sorted(
        (len(component) for component in components),
        reverse=True,
    )


def relation_direction_errors(
    graph: nx.Graph,
) -> list[dict[str, Any]]:
    entity_types = {
        "Paper",
        "Catalyst",
        "CatalystModel",
        "Metal",
        "Support",
        "CoordinationMotif",
        "SynthesisMethod",
        "Precursor",
        "Reaction",
        "ReactionStep",
        "Intermediate",
        "Material",
    }

    def node_type(node_id: str) -> str:
        return str(
            graph.nodes[node_id].get("type", "")
        )

    errors: list[dict[str, Any]] = []

    for source, target, key, data in edge_records(graph):
        relation = str(data.get("relation", ""))
        source_type = node_type(source)
        target_type = node_type(target)

        valid = True
        expected = ""

        if relation == "EVALUATED_IN":
            valid = (
                source_type
                in {"Catalyst", "CatalystModel", "Material"}
                and target_type == "Experiment"
            )
            expected = (
                "Catalyst/CatalystModel/Material "
                "-> Experiment"
            )

        elif relation == "CHARACTERIZED_BY":
            valid = (
                source_type
                in {
                    "Catalyst",
                    "Support",
                    "Material",
                    "CoordinationMotif",
                }
                and target_type == "Experiment"
            )
            expected = (
                "physical entity -> Experiment"
            )

        elif relation == "MODELED_BY":
            valid = (
                source_type == "CatalystModel"
                and target_type == "Calculation"
            )
            expected = (
                "CatalystModel -> Calculation"
            )

        elif relation == "SYNTHESIZED_BY":
            valid = (
                source_type == "Catalyst"
                and target_type == "SynthesisMethod"
            )
            expected = (
                "Catalyst -> SynthesisMethod"
            )

        elif relation == "USES_PRECURSOR":
            valid = (
                source_type == "SynthesisMethod"
                and target_type == "Precursor"
            )
            expected = (
                "SynthesisMethod -> Precursor"
            )

        elif relation == "HAS_MEASUREMENT":
            valid = (
                source_type
                in {"Experiment", "Calculation"}
                and target_type == "Measurement"
            )
            expected = (
                "Experiment/Calculation -> Measurement"
            )

        elif relation == "MEASURED_FOR":
            valid = (
                source_type == "Measurement"
                and target_type in entity_types
            )
            expected = "Measurement -> scientific Entity"

        elif relation == "IN_MEASUREMENT_GROUP":
            valid = (
                source_type == "Measurement"
                and target_type == "MeasurementGroup"
            )
            expected = "Measurement -> MeasurementGroup"

        elif relation == "MODEL_OF":
            valid = (
                source_type == "CatalystModel"
                and target_type == "Catalyst"
            )
            expected = "CatalystModel -> Catalyst"

        elif relation == "SUPPORTS_CLAIM":
            valid = (
                source_type
                in {
                    "Measurement",
                    "Experiment",
                    "Calculation",
                }
                and target_type
                in {
                    "ObservationClaim",
                    "MechanismClaim",
                }
            )
            expected = (
                "evidence node -> claim"
            )

        elif relation == "INTERPRETED_AS":
            valid = (
                source_type == "ObservationClaim"
                and target_type == "MechanismClaim"
            )
            expected = (
                "ObservationClaim -> MechanismClaim"
            )

        elif relation == "APPLIES_TO":
            valid = (
                source_type
                in {
                    "ObservationClaim",
                    "MechanismClaim",
                }
                and target_type in entity_types
            )
            expected = (
                "claim -> scientific Entity"
            )

        elif relation == "HAS_METAL":
            valid = (
                source_type
                in {"Catalyst", "CatalystModel"}
                and target_type == "Metal"
            )
            expected = (
                "Catalyst/CatalystModel -> Metal"
            )

        elif relation == "SUPPORTED_ON":
            valid = (
                source_type
                in {"Catalyst", "CatalystModel"}
                and target_type == "Support"
            )
            expected = (
                "Catalyst/CatalystModel -> Support"
            )

        elif relation == "CATALYZES":
            valid = (
                source_type == "Catalyst"
                and target_type == "Reaction"
            )
            expected = (
                "Catalyst -> Reaction"
            )

        if not valid:
            errors.append(
                {
                    "source": source,
                    "source_type": source_type,
                    "relation": relation,
                    "target": target,
                    "target_type": target_type,
                    "edge_key": key,
                    "expected": expected,
                    "chunk_id": data.get("chunk_id", ""),
                }
            )

    return errors


def main() -> None:
    args = parse_args()

    graphml_path = Path(args.graphml).resolve()

    if not graphml_path.exists():
        raise FileNotFoundError(
            f"GraphML not found: {graphml_path}"
        )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else graphml_path.parent / "audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = nx.read_graphml(graphml_path)

    nodes = [
        {
            "id": str(node_id),
            **{
                key: clean_value(value)
                for key, value in data.items()
            },
            "in_degree": (
                graph.in_degree(node_id)
                if graph.is_directed()
                else graph.degree(node_id)
            ),
            "out_degree": (
                graph.out_degree(node_id)
                if graph.is_directed()
                else graph.degree(node_id)
            ),
        }
        for node_id, data in graph.nodes(data=True)
    ]

    edges = [
        {
            "source": source,
            "target": target,
            "edge_key": key,
            **{
                attr: clean_value(value)
                for attr, value in data.items()
            },
        }
        for source, target, key, data in edge_records(graph)
    ]

    node_type_counts = Counter(
        str(row.get("type", ""))
        for row in nodes
    )
    relation_counts = Counter(
        str(row.get("relation", ""))
        for row in edges
    )
    chunk_counts = Counter(
        str(row.get("chunk_id", ""))
        for row in edges
    )
    document_counts = Counter(
        (
            str(row.get("document_id", "")),
            str(row.get("document_role", "")),
        )
        for row in edges
    )

    isolated = [
        row
        for row in nodes
        if row["in_degree"] == 0
        and row["out_degree"] == 0
    ]

    missing_node_attrs = [
        row
        for row in nodes
        if not row.get("type")
        or not row.get("label")
    ]

    required_edge_attrs = {
        "relation",
        "title",
        "paper_id",
        "chunk_id",
        "document_id",
        "document_role",
        "evidence_text",
        "evidence_pointers_json",
    }

    missing_edge_attrs: list[dict[str, Any]] = []

    for row in edges:
        missing = [
            attr
            for attr in required_edge_attrs
            if not row.get(attr)
        ]

        if missing:
            missing_edge_attrs.append(
                {
                    "source": row["source"],
                    "relation": row.get("relation", ""),
                    "target": row["target"],
                    "edge_key": row["edge_key"],
                    "missing_attributes": ", ".join(missing),
                }
            )

    provenance_issues: list[dict[str, Any]] = []
    asset_evidence_edges: list[dict[str, Any]] = []

    asset_manifest_path = resolve_asset_manifest(graphml_path, args.asset_manifest)
    manifest_assets: dict[str, dict[str, Any]] = {}
    if asset_manifest_path is not None:
        try:
            manifest_payload = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
            manifest_assets = {
                str(item.get("asset_id")): item
                for item in manifest_payload.get("assets", [])
                if isinstance(item, dict) and item.get("asset_id")
            }
        except Exception as error:
            provenance_issues.append({
                "source": "", "relation": "", "target": "", "edge_key": "",
                "issue": f"invalid asset manifest: {error}",
            })

    for row in edges:
        try:
            pointers = json.loads(str(row.get("evidence_pointers_json", "")))
            chunk_assets = set(json.loads(str(row.get("asset_ids_json", "[]"))))
        except Exception as error:
            provenance_issues.append({
                "source": row["source"],
                "relation": row.get("relation", ""),
                "target": row["target"],
                "edge_key": row["edge_key"],
                "issue": f"invalid provenance JSON: {error}",
            })
            continue

        if not isinstance(pointers, list) or not pointers:
            provenance_issues.append({
                "source": row["source"],
                "relation": row.get("relation", ""),
                "target": row["target"],
                "edge_key": row["edge_key"],
                "issue": "no evidence pointers",
            })
            continue

        for pointer in pointers:
            if not isinstance(pointer, dict):
                provenance_issues.append({
                    "source": row["source"],
                    "relation": row.get("relation", ""),
                    "target": row["target"],
                    "edge_key": row["edge_key"],
                    "issue": "evidence pointer is not an object",
                })
                continue
            if pointer.get("document_id") != row.get("document_id"):
                provenance_issues.append({
                    "source": row["source"],
                    "relation": row.get("relation", ""),
                    "target": row["target"],
                    "edge_key": row["edge_key"],
                    "issue": "pointer document_id mismatch",
                })
            pointer_assets = set(pointer.get("asset_ids") or [])
            unknown = pointer_assets - chunk_assets
            if unknown:
                provenance_issues.append({
                    "source": row["source"],
                    "relation": row.get("relation", ""),
                    "target": row["target"],
                    "edge_key": row["edge_key"],
                    "issue": f"unknown pointer asset IDs: {sorted(unknown)}",
                })
            if pointer_assets and asset_manifest_path is None:
                provenance_issues.append({
                    "source": row["source"],
                    "relation": row.get("relation", ""),
                    "target": row["target"],
                    "edge_key": row["edge_key"],
                    "issue": "asset evidence exists but asset_manifest.json was not found",
                })
            for asset_id in sorted(pointer_assets):
                manifest_record = manifest_assets.get(asset_id)
                if asset_manifest_path is not None and manifest_record is None:
                    provenance_issues.append({
                        "source": row["source"], "relation": row.get("relation", ""),
                        "target": row["target"], "edge_key": row["edge_key"],
                        "issue": f"asset ID missing from manifest: {asset_id}",
                    })
                elif manifest_record is not None and not bool(manifest_record.get("exists", True)):
                    provenance_issues.append({
                        "source": row["source"], "relation": row.get("relation", ""),
                        "target": row["target"], "edge_key": row["edge_key"],
                        "issue": f"manifest asset file is missing: {asset_id}",
                    })
            if pointer_assets:
                asset_evidence_edges.append({
                    "source": row["source"],
                    "relation": row.get("relation", ""),
                    "target": row["target"],
                    "edge_key": row["edge_key"],
                    "document_id": row.get("document_id", ""),
                    "asset_ids": sorted(pointer_assets),
                    "page_id": pointer.get("page_id"),
                    "locator_text": pointer.get("locator_text"),
                })

    measurements = [
        row
        for row in nodes
        if row.get("type") == "Measurement"
    ]
    observation_claims = [
        row
        for row in nodes
        if row.get("type") == "ObservationClaim"
    ]
    mechanism_claims = [
        row
        for row in nodes
        if row.get("type") == "MechanismClaim"
    ]

    incoming_by_target: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)
    outgoing_by_source: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in edges:
        incoming_by_target[row["target"]].append(row)
        outgoing_by_source[row["source"]].append(row)

    unlinked_measurements = []

    for row in measurements:
        producers = [
            edge
            for edge in incoming_by_target[row["id"]]
            if edge.get("relation") == "HAS_MEASUREMENT"
        ]

        if not producers:
            unlinked_measurements.append(row)

    measurement_subject_issues: list[dict[str, Any]] = []
    for row in measurements:
        measured_for = [
            edge for edge in outgoing_by_source[row["id"]]
            if edge.get("relation") == "MEASURED_FOR"
        ]
        if len(measured_for) != 1 or str(measured_for[0].get("target", "")) != str(row.get("subject_id", "")):
            measurement_subject_issues.append({
                "id": row["id"],
                "subject_id": row.get("subject_id", ""),
                "measured_for_targets": [edge.get("target", "") for edge in measured_for],
                "issue": "expected exactly one MEASURED_FOR edge matching subject_id",
            })

    composite_measurements = composite_measurement_rows(measurements)
    unregistered_experiments = [
        row for row in nodes
        if row.get("type") == "Experiment"
        and str(row.get("experiment_type", "")).startswith("unregistered_")
    ]
    unregistered_metrics = [
        row for row in measurements
        if str(row.get("metric_id", "")).startswith("unregistered_")
    ]

    claim_rows: list[dict[str, Any]] = []
    claim_issues: list[dict[str, Any]] = []

    for claim in [
        *observation_claims,
        *mechanism_claims,
    ]:
        incoming = incoming_by_target[claim["id"]]
        outgoing = outgoing_by_source[claim["id"]]

        support_count = sum(
            edge.get("relation") == "SUPPORTS_CLAIM"
            for edge in incoming
        )
        interpreted_count = sum(
            edge.get("relation") == "INTERPRETED_AS"
            for edge in incoming
        )
        applies_count = sum(
            edge.get("relation") == "APPLIES_TO"
            for edge in outgoing
        )

        claim_row = {
            **claim,
            "support_count": support_count,
            "interpreted_as_count": interpreted_count,
            "applies_to_count": applies_count,
            "incoming_relations": [
                edge.get("relation", "")
                for edge in incoming
            ],
            "outgoing_relations": [
                edge.get("relation", "")
                for edge in outgoing
            ],
        }
        claim_rows.append(claim_row)

        issue_messages = []

        if claim.get("type") == "ObservationClaim":
            if support_count == 0:
                issue_messages.append(
                    "no SUPPORTS_CLAIM input"
                )
        else:
            if (
                support_count == 0
                and interpreted_count == 0
            ):
                issue_messages.append(
                    "no SUPPORTS_CLAIM or INTERPRETED_AS input"
                )

        if applies_count == 0:
            issue_messages.append(
                "no APPLIES_TO output"
            )

        if issue_messages:
            claim_issues.append(
                {
                    "id": claim["id"],
                    "type": claim.get("type", ""),
                    "statement": claim.get(
                        "statement",
                        claim.get("label", ""),
                    ),
                    "issues": "; ".join(issue_messages),
                }
            )

    labels: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in nodes:
        key = (
            str(row.get("type", "")),
            normalized_label(
                str(row.get("label", ""))
            ),
        )
        labels[key].append(row)

    duplicate_labels = []

    for (
        node_type,
        norm_label,
    ), group in labels.items():
        if norm_label and len(group) > 1:
            duplicate_labels.append(
                {
                    "type": node_type,
                    "normalized_label": norm_label,
                    "node_ids": [
                        row["id"]
                        for row in group
                    ],
                    "labels": [
                        row.get("label", "")
                        for row in group
                    ],
                }
            )

    direction_errors = relation_direction_errors(graph)

    sizes = component_sizes(graph)
    self_loops = list(nx.selfloop_edges(graph))

    report_lines = [
        f"GraphML: {graphml_path}",
        f"Graph class: {type(graph).__name__}",
        f"Directed: {graph.is_directed()}",
        f"Multigraph: {graph.is_multigraph()}",
        f"Nodes: {graph.number_of_nodes()}",
        f"Edges: {graph.number_of_edges()}",
        f"Self-loops: {len(self_loops)}",
        f"Components: {len(sizes)}",
        (
            "Largest component sizes: "
            + ", ".join(map(str, sizes[:10]))
        ),
        f"Isolated nodes: {len(isolated)}",
        (
            "Nodes missing type/label: "
            f"{len(missing_node_attrs)}"
        ),
        (
            "Edges missing required attributes: "
            f"{len(missing_edge_attrs)}"
        ),
        f"Provenance issues: {len(provenance_issues)}",
        f"Edges with asset evidence: {len(asset_evidence_edges)}",
        (
            "Unlinked measurements: "
            f"{len(unlinked_measurements)}"
        ),
        f"Measurement subject issues: {len(measurement_subject_issues)}",
        f"Composite measurements: {len(composite_measurements)}",
        f"Unregistered experiments: {len(unregistered_experiments)}",
        f"Unregistered metrics: {len(unregistered_metrics)}",
        f"Asset manifest: {asset_manifest_path or '<not found>'}",
        f"Claim issues: {len(claim_issues)}",
        (
            "Relation direction errors: "
            f"{len(direction_errors)}"
        ),
        (
            "Duplicate normalized labels: "
            f"{len(duplicate_labels)}"
        ),
        "",
        "Node type counts:",
    ]

    for key, value in node_type_counts.most_common():
        report_lines.append(
            f"  {key or '<missing>'}: {value}"
        )

    report_lines.extend(
        ["", "Relation counts:"]
    )

    for key, value in relation_counts.most_common():
        report_lines.append(
            f"  {key or '<missing>'}: {value}"
        )

    report_lines.extend(
        ["", "Edge counts by chunk:"]
    )

    for key, value in chunk_counts.most_common():
        report_lines.append(
            f"  {key or '<missing>'}: {value}"
        )

    report_lines.extend(["", "Edge counts by document:"])
    for (document_id, document_role), value in document_counts.most_common():
        report_lines.append(
            f"  {document_id or '<missing>'} "
            f"({document_role or '<missing>'}): {value}"
        )

    report_path = output_dir / "audit_report.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    write_csv(output_dir / "nodes.csv", nodes)
    write_csv(output_dir / "edges.csv", edges)
    write_csv(
        output_dir / "measurements.csv",
        measurements,
    )
    write_csv(
        output_dir / "claims.csv",
        claim_rows,
    )
    write_csv(
        output_dir / "isolated_nodes.csv",
        isolated,
    )
    write_csv(
        output_dir / "unlinked_measurements.csv",
        unlinked_measurements,
    )
    write_csv(output_dir / "measurement_subject_issues.csv", measurement_subject_issues)
    write_csv(output_dir / "composite_measurements.csv", composite_measurements)
    write_csv(output_dir / "unregistered_experiments.csv", unregistered_experiments)
    write_csv(output_dir / "unregistered_metrics.csv", unregistered_metrics)
    write_csv(
        output_dir / "claim_issues.csv",
        claim_issues,
    )
    write_csv(
        output_dir / "missing_node_attributes.csv",
        missing_node_attrs,
    )
    write_csv(
        output_dir / "missing_edge_attributes.csv",
        missing_edge_attrs,
    )
    write_csv(
        output_dir / "relation_direction_errors.csv",
        direction_errors,
    )
    write_csv(
        output_dir / "duplicate_labels.csv",
        duplicate_labels,
    )
    write_csv(
        output_dir / "provenance_issues.csv",
        provenance_issues,
    )
    write_csv(
        output_dir / "asset_evidence_edges.csv",
        asset_evidence_edges,
    )

    claim_overlap_summary = write_claim_overlap_audit(graph, output_dir / "claim_audit")
    experiment_total = max(1, node_type_counts.get("Experiment", 0))
    readiness = {
        "unlinked_measurements": len(unlinked_measurements),
        "composite_measurements": len(composite_measurements),
        "measurement_subject_issues": len(measurement_subject_issues),
        "provenance_issues": len(provenance_issues),
        "unregistered_experiment_ratio": len(unregistered_experiments) / experiment_total,
        "claim_overlap_review_candidates": claim_overlap_summary["review_required"],
        "claim_auto_merged": claim_overlap_summary["auto_merged"],
        "passes_structural_gate": (
            not unlinked_measurements
            and not composite_measurements
            and not measurement_subject_issues
            and not provenance_issues
            and not claim_issues
            and not direction_errors
        ),
    }
    (output_dir / "pilot_readiness.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print("Claim-overlap review candidates:", claim_overlap_summary["review_required"])
    print("Pilot structural gate:", readiness["passes_structural_gate"])

    print("\nObservation claims:")
    for row in observation_claims[: args.show_limit]:
        print(
            f"- {row['id']}: "
            f"{row.get('statement', row.get('label', ''))}"
        )

    print("\nMechanism claims:")
    for row in mechanism_claims[: args.show_limit]:
        print(
            f"- {row['id']}: "
            f"{row.get('statement', row.get('label', ''))}"
        )

    print("\nAudit files saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()
