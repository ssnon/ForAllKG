from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.relation_component_composition import (
    confirmed_known_component_from_mapping,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AcceptedRelationPatternExportAudit(StrictModel):
    schema_version: Literal[
        "accepted-relationpattern-canonical-export-v1"
    ] = "accepted-relationpattern-canonical-export-v1"

    canonical_graph: str
    canonical_graph_sha256: str
    output_csv: str
    output_csv_sha256: str

    accepted_relationpattern_node_count: int = Field(ge=0)
    validated_component_count: int = Field(ge=0)
    validation_rejection_count: int = Field(ge=0)
    component_ids: list[str] = Field(default_factory=list)

    source_authority: Literal[
        "CANONICAL_GRAPH_ACCEPTED_RELATIONPATTERN"
    ] = "CANONICAL_GRAPH_ACCEPTED_RELATIONPATTERN"
    deterministic_projection_only: Literal[True] = True
    new_scientific_claim_created: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    generation_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
    return value


def export_accepted_relationpatterns_from_canonical_graph(
    *,
    graph_path: str | Path,
    output_csv: str | Path,
) -> AcceptedRelationPatternExportAudit:
    graph_path = Path(graph_path).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()

    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)

    graph = nx.read_graphml(
        graph_path,
        force_multigraph=True,
    )

    rows: list[tuple[str, dict[str, Any], str]] = []
    rejections: list[dict[str, str]] = []
    accepted_count = 0

    for node_id, attrs in graph.nodes(data=True):
        if (
            str(attrs.get("retention_lane", "")).strip()
            != "accepted_pattern"
        ):
            continue
        if (
            str(attrs.get("concept_type", "")).strip()
            != "RelationPattern"
        ):
            continue

        accepted_count += 1
        row = dict(attrs)
        row["node_id"] = str(node_id)

        try:
            component = confirmed_known_component_from_mapping(
                row
            )
        except Exception as exc:
            rejections.append(
                {
                    "node_id": str(node_id),
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue

        rows.append(
            (
                str(node_id),
                row,
                component.component_id,
            )
        )

    if accepted_count == 0:
        raise RuntimeError(
            "canonical graph contains zero accepted RelationPattern nodes"
        )

    if rejections:
        preview = "; ".join(
            f"{row['node_id']}:{row['type']}:{row['message']}"
            for row in rejections[:5]
        )
        raise RuntimeError(
            "canonical accepted RelationPattern validation failed closed: "
            f"accepted={accepted_count}, rejected={len(rejections)}; "
            f"examples={preview}"
        )

    rows.sort(key=lambda row: row[0])

    fields: list[str] = []
    for _, row, _ in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with output_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        for _, row, _ in rows:
            writer.writerow(
                {
                    key: _csv_value(value)
                    for key, value in row.items()
                }
            )

    return AcceptedRelationPatternExportAudit(
        canonical_graph=str(graph_path),
        canonical_graph_sha256=_sha256_file(graph_path),
        output_csv=str(output_csv),
        output_csv_sha256=_sha256_file(output_csv),
        accepted_relationpattern_node_count=accepted_count,
        validated_component_count=len(rows),
        validation_rejection_count=0,
        component_ids=[
            component_id
            for _, _, component_id in rows
        ],
    )


__all__ = [
    "AcceptedRelationPatternExportAudit",
    "export_accepted_relationpatterns_from_canonical_graph",
]
