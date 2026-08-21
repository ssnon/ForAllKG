from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import networkx as nx

from pipeline_core.corpus.extraction.document_config import get_paper_config
from domains.extraction_registry import get_extraction_adapter
from domains.graph_registry import get_graph_adapter
from domains.registry import get_domain_profile
from pipeline_core.corpus.graph.graph_io import knowledge_graph_to_networkx, save_graphml
from pipeline_core.corpus.graph_semantics import (
    apply_graph_domain_canonicalization,
    write_graph_semantics_report,
)
from pipeline_core.corpus.extraction.extraction_policy import ExtractionPolicy
from pipeline_core.corpus.extraction_quality import (
    QUALITY_PARTIAL_CRITICAL,
    QUALITY_REJECTED,
    graph_quality_attributes,
    quality_from_active_payload,
)
from pipeline_core.corpus.extraction.locator_index import load_locator_index
from pipeline_core.corpus.measurement_merge_invariants import (
    MEASUREMENT_MERGE_INVARIANT_ID,
    assert_measurement_value_xor,
    measurement_mentions_conflict,
)
from pipeline_core.corpus.provenance_backfill import (
    backfill_edge_asset_provenance,
    refresh_run_asset_manifest,
)
from pipeline_core.corpus.semantic_repairs import repair_model_of_targets
from pipeline_core.corpus.graph_normalization import normalize_networkx_metric_vocabularies
from pipeline_core.corpus.extraction.node_references import remap_node_reference_attributes
from pipeline_core.corpus.paper_graph_postprocess import (
    canonicalize_paper_graph,
    load_resolution_plan,
    merge_node_attributes,
)
from pipeline_core.corpus.resolution_candidates import (
    build_raw_canonical_report,
    format_raw_canonical_report,
    generate_resolution_candidates,
    sync_decisions_jsonl,
    write_candidates_csv,
)
from domains.dac_her.claim_overlap import write_claim_overlap_audit
from domains.dac_her.semantic_roles import SemanticRoleAdjustment
from pipeline_core.runtime.run_lifecycle import (
    paper_output_root,
    resolve_run_directory,
)
from pipeline_core.runtime.serialization_primitives import (
    read_json,
    write_json,
)
from pipeline_core.corpus.schemas import KnowledgeGraph
from pipeline_core.runtime.validation import validate_graph_provenance
from pipeline_core.corpus.vocab_registry import load_default_registries


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "papers.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build raw and canonical paper-level GraphML graphs."
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--domain-profile", default="dac_her")
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Default: latest_run.json for the paper.",
    )
    parser.add_argument(
        "--attempt-id",
        default=None,
        help=(
            "Select one immutable extraction attempt within --run-id. "
            "Default: the run family's latest attempt; legacy flat runs "
            "remain supported."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Explicitly allow PARTIAL_CRITICAL extraction. "
            "PARTIAL_ACCEPTABLE is allowed automatically; REJECTED runs "
            "with unresolved failures/very low coverage remain blocked."
        ),
    )
    parser.add_argument(
        "--decisions",
        default=None,
        help=(
            "Override paper-level decisions file. Supported: reviewed .jsonl "
            "or legacy aliases .json."
        ),
    )
    parser.add_argument(
        "--no-resolution",
        action="store_true",
        help="Build canonical graph identical to raw graph.",
    )
    return parser.parse_args()


def _collision_safe_node_id(
    *,
    chunk_id: str,
    local_node_id: str,
    node_type: str,
) -> str:
    """Create a deterministic mention ID for a cross-chunk type collision.

    LLM node IDs are local identifiers, not globally trusted canonical IDs.
    If two chunks reuse the same local ID for different semantic types, both
    mentions must survive so that the paper-level resolver can review them.
    """
    digest = hashlib.sha256(
        f"{chunk_id}|{local_node_id}|{node_type}".encode("utf-8")
    ).hexdigest()[:12]
    safe_type = "".join(
        character.lower() if character.isalnum() else "_"
        for character in (node_type or "unknown")
    ).strip("_") or "unknown"
    return f"{local_node_id}__mention_{safe_type}_{digest}"


def merge_chunk_graph(
    merged: nx.MultiDiGraph,
    chunk_graph: nx.MultiDiGraph,
    *,
    chunk_id: str,
) -> list[dict[str, str]]:
    """Merge one chunk graph without stale embedded node references.

    LLM IDs are local mentions. Type collisions are namespaced, and every
    edge endpoint plus foreign-key-like node attribute is remapped through the
    same deterministic node_id_map.
    """
    node_id_map: dict[str, str] = {}
    collisions: list[dict[str, str]] = []

    # First pass: decide all target IDs before inserting any nodes. This is
    # required because a Measurement can refer to a subject that appears later
    # in the chunk node order.
    for raw_node_id, node_data in chunk_graph.nodes(data=True):
        local_node_id = str(raw_node_id)
        incoming_type = str(dict(node_data).get("type", ""))
        target_node_id = local_node_id

        if local_node_id in merged:
            existing_data = dict(merged.nodes[local_node_id])
            existing_type = str(existing_data.get("type", ""))
            collision_reason = ""
            collision_action = ""

            if (
                existing_type
                and incoming_type
                and existing_type != incoming_type
            ):
                collision_reason = "node_type_mismatch"
                collision_action = "preserved_as_chunk_scoped_mention"
            elif (
                existing_type == "Measurement"
                and incoming_type == "Measurement"
                and measurement_mentions_conflict(
                    existing_data,
                    dict(node_data),
                )
            ):
                collision_reason = "measurement_payload_conflict"
                collision_action = (
                    "preserved_as_chunk_scoped_measurement_mention"
                )

            if collision_reason:
                target_node_id = _collision_safe_node_id(
                    chunk_id=chunk_id,
                    local_node_id=local_node_id,
                    node_type=incoming_type,
                )
                suffix = 1
                base_target = target_node_id
                while (
                    target_node_id in merged
                    or target_node_id in node_id_map.values()
                ):
                    target_node_id = f"{base_target}_{suffix}"
                    suffix += 1
                collisions.append({
                    "chunk_id": chunk_id,
                    "local_node_id": local_node_id,
                    "existing_node_id": local_node_id,
                    "existing_type": existing_type,
                    "preserved_node_id": target_node_id,
                    "incoming_type": incoming_type,
                    "collision_reason": collision_reason,
                    "action": collision_action,
                })

        node_id_map[local_node_id] = target_node_id

    # Second pass: remap embedded references such as Measurement.subject_id
    # and MeasurementGroup member IDs before adding/merging the nodes.
    collision_by_local = {
        row["local_node_id"]: row
        for row in collisions
    }
    for raw_node_id, node_data in chunk_graph.nodes(data=True):
        local_node_id = str(raw_node_id)
        target_node_id = node_id_map[local_node_id]
        incoming = remap_node_reference_attributes(
            dict(node_data),
            node_id_map,
        )

        collision = collision_by_local.get(local_node_id)
        if collision is not None:
            incoming["source_local_id"] = local_node_id
            incoming["source_chunk_id"] = chunk_id
            incoming["id_collision_with"] = local_node_id
            incoming["id_collision_types_json"] = json.dumps(
                {
                    "existing_type": collision["existing_type"],
                    "incoming_type": collision["incoming_type"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            incoming["id_collision_reason"] = collision[
                "collision_reason"
            ]
            merged.add_node(target_node_id, **incoming)
        elif target_node_id in merged:
            merged.nodes[target_node_id].update(
                merge_node_attributes(
                    dict(merged.nodes[target_node_id]),
                    incoming,
                )
            )
        else:
            merged.add_node(target_node_id, **incoming)

    for source, target, local_key, edge_data in chunk_graph.edges(
        keys=True,
        data=True,
    ):
        source_id = node_id_map.get(str(source), str(source))
        target_id = node_id_map.get(str(target), str(target))
        global_key = f"{chunk_id}:{local_key}"
        merged.add_edge(
            source_id,
            target_id,
            key=global_key,
            **dict(edge_data),
        )

    return collisions

def write_id_collision_report(
    run_dir: Path,
    collisions: list[dict[str, str]],
) -> None:
    report = {
        "collision_count": len(collisions),
        "policy": (
            "Same local node ID with different types, or same-ID Measurement "
            "mentions with conflicting scientific payloads, are preserved as "
            "chunk-scoped mentions; no coercion or destructive value merge."
        ),
        "collisions": collisions,
    }
    write_json(run_dir / "id_collision_report.json", report)

    csv_path = run_dir / "id_collisions.csv"
    fieldnames = [
        "chunk_id",
        "local_node_id",
        "existing_node_id",
        "existing_type",
        "preserved_node_id",
        "incoming_type",
        "collision_reason",
        "action",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(collisions)



def write_semantic_role_report(
    run_dir: Path,
    adjustments: list[SemanticRoleAdjustment],
    *,
    policy: str,
) -> None:
    rows = [adjustment.to_dict() for adjustment in adjustments]
    write_json(
        run_dir / "semantic_role_report.json",
        {
            "adjustment_count": len(rows),
            "policy": policy,
            "adjustments": rows,
        },
    )
    path = run_dir / "semantic_role_adjustments.csv"
    fieldnames = [
        "chunk_id",
        "source_node_id",
        "original_type",
        "resolved_type",
        "action",
        "role_node_id",
        "measurement_ids",
        "experiment_ids",
        "reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "measurement_ids": json.dumps(
                    row.get("measurement_ids", []), ensure_ascii=False
                ),
                "experiment_ids": json.dumps(
                    row.get("experiment_ids", []), ensure_ascii=False
                ),
            })


def write_metric_normalization_report(
    run_dir: Path,
    issues: list[object],
) -> None:
    rows = [
        issue.to_dict()
        for issue in issues
        if hasattr(issue, "to_dict")
    ]
    write_json(
        run_dir / "metric_normalization_report.json",
        {
            "change_count": len(rows),
            "changes": rows,
        },
    )
    path = run_dir / "metric_normalization.csv"
    fieldnames = [
        "node_id",
        "vocabulary",
        "raw_id",
        "raw_label",
        "normalized_id",
        "status",
        "parameters",
        "matched_pattern",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "parameters": json.dumps(
                    row.get("parameters") or {}, ensure_ascii=False
                ),
            })

def write_generic_rows_report(
    *,
    run_dir: Path,
    stem: str,
    rows: list[dict[str, object]],
    summary_extra: dict[str, object] | None = None,
) -> None:
    write_json(
        run_dir / f"{stem}.json",
        {
            "count": len(rows),
            "rows": rows,
            **(summary_extra or {}),
        },
    )
    path = run_dir / f"{stem}.csv"
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
                key: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            })


def resolve_decisions_path(
    *,
    paper_resolution_file: Path | None,
    paper_root: Path,
    override: str | None,
    disabled: bool,
) -> Path | None:
    if disabled:
        return None
    if override:
        return Path(override).resolve()
    if paper_resolution_file is not None:
        return paper_resolution_file.resolve()

    default_jsonl = paper_root / "resolution" / "decisions.jsonl"
    if default_jsonl.exists():
        return default_jsonl.resolve()
    return None


def _candidate_summary(run_dir: Path) -> dict[str, object]:
    path = run_dir / "resolution" / "candidate_summary.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    domain_profile = get_domain_profile(args.domain_profile)
    extraction_adapter = get_extraction_adapter(domain_profile.profile_id)
    graph_adapter = get_graph_adapter(domain_profile.profile_id)
    data_root = args.data_root or extraction_adapter.default_data_root
    paper = get_paper_config(
        args.config,
        project_root=PROJECT_ROOT,
        paper_id=args.paper_id,
    )
    run_dir = resolve_run_directory(
        project_root=PROJECT_ROOT,
        paper_id=paper.paper_id,
        run_id=args.run_id,
        data_root=data_root,
        attempt_id=args.attempt_id,
    )

    run_metadata = read_json(run_dir / "run.json")
    active_payload = read_json(run_dir / "active_chunks.json")

    if active_payload.get("paper_id") != paper.paper_id:
        raise ValueError(
            "active_chunks.json paper_id does not match the requested paper."
        )
    if active_payload.get("run_id") != run_metadata.get("run_id"):
        raise ValueError(
            "active_chunks.json and run.json refer to different runs."
        )
    active_attempt_id = str(active_payload.get("attempt_id") or "").strip()
    metadata_attempt_id = str(run_metadata.get("attempt_id") or "").strip()
    if (
        active_attempt_id
        and metadata_attempt_id
        and active_attempt_id != metadata_attempt_id
    ):
        raise ValueError(
            "active_chunks.json and run.json refer to different attempts."
        )
    extraction_attempt_id = active_attempt_id or metadata_attempt_id
    extraction_quality = quality_from_active_payload(
        active_payload,
        policy=ExtractionPolicy(),
    )
    write_json(
        run_dir / "extraction_quality.json",
        extraction_quality,
    )
    materialization_status = str(
        extraction_quality["graph_materialization_status"]
    )

    if materialization_status == QUALITY_REJECTED:
        raise RuntimeError(
            "The extraction run is REJECTED: unresolved failed chunks, no "
            "usable strict-valid leaves, or source-token coverage below the "
            "minimum threshold. Quarantined/failed leaves are never merged."
        )

    if (
        materialization_status == QUALITY_PARTIAL_CRITICAL
        and not args.allow_incomplete
    ):
        raise RuntimeError(
            "The extraction run is PARTIAL_CRITICAL. Strict-valid leaves are "
            "available, but coverage is below the automatic graph gate. "
            "Review extraction_quality.json or pass --allow-incomplete "
            "explicitly."
        )

    chunk_records = active_payload.get("chunks")
    if not isinstance(chunk_records, list) or not chunk_records:
        raise RuntimeError("No active chunks are available for graph building.")

    refreshed_assets = refresh_run_asset_manifest(
        paper=paper,
        run_dir=run_dir,
    )

    quality_attrs = graph_quality_attributes(
        extraction_quality
    )
    merged = nx.MultiDiGraph(
        paper_id=paper.paper_id,
        run_id=str(run_metadata["run_id"]),
        run_fingerprint=str(run_metadata["run_fingerprint"]),
        source_extraction_attempt_id=extraction_attempt_id,
        graph_stage="raw_merged",
        domain_profile_id=domain_profile.profile_id,
        graph_adapter_id=graph_adapter.adapter_id,
        **quality_attrs,
    )

    loaded_chunks = 0
    loaded_chunk_ids: list[str] = []
    id_collisions: list[dict[str, str]] = []
    semantic_role_adjustments: list[SemanticRoleAdjustment] = []

    for record in chunk_records:
        if not isinstance(record, dict):
            raise ValueError("Invalid chunk record in active_chunks.json.")

        json_path = Path(str(record["output_path"]))
        if not json_path.exists():
            raise FileNotFoundError(f"Active chunk JSON not found: {json_path}")

        result = KnowledgeGraph.model_validate_json(
            json_path.read_text(encoding="utf-8")
        )
        validate_graph_provenance(
            result,
            paper_id=paper.paper_id,
            chunk_id=str(record["chunk_id"]),
            section=str(record["section"]),
            document_id=str(record.get("document_id", "main")),
            document_role=str(record.get("document_role", "main")),
            page_ids=list(record.get("page_ids", [])),
            asset_ids=list(record.get("asset_ids", [])),
        )

        chunk_graph = knowledge_graph_to_networkx(result)
        chunk_graph, chunk_role_adjustments = (
            graph_adapter.normalize_semantic_roles(
                chunk_graph,
                chunk_id=result.chunk_id,
            )
        )
        semantic_role_adjustments.extend(chunk_role_adjustments)
        id_collisions.extend(
            merge_chunk_graph(
                merged,
                chunk_graph,
                chunk_id=result.chunk_id,
            )
        )
        loaded_chunks += 1
        loaded_chunk_ids.append(result.chunk_id)

    assert_measurement_value_xor(
        merged,
        stage="raw_merged_after_chunk_merge",
    )
    write_id_collision_report(run_dir, id_collisions)
    write_semantic_role_report(
        run_dir,
        semantic_role_adjustments,
        policy=graph_adapter.semantic_role_policy,
    )
    if id_collisions:
        print(
            f"[NODE-ID COLLISIONS] {len(id_collisions)} collision(s) "
            f"preserved; see {run_dir / 'id_collisions.csv'}"
        )
    if semantic_role_adjustments:
        print(
            f"[SEMANTIC ROLES] {len(semantic_role_adjustments)} "
            f"adjustment(s); see {run_dir / 'semantic_role_adjustments.csv'}"
        )

    _, metric_registry = load_default_registries(PROJECT_ROOT)
    metric_normalization_issues = normalize_networkx_metric_vocabularies(
        merged,
        metric_registry=metric_registry,
    )
    write_metric_normalization_report(
        run_dir,
        metric_normalization_issues,
    )
    if metric_normalization_issues:
        print(
            f"[METRIC NORMALIZATION] {len(metric_normalization_issues)} "
            f"change(s); see {run_dir / 'metric_normalization.csv'}"
        )

    locator_index_records = load_locator_index(run_dir / "locator_index.json")
    provenance_backfills = backfill_edge_asset_provenance(
        merged,
        assets=refreshed_assets.values(),
        locator_index=locator_index_records,
    )
    write_generic_rows_report(
        run_dir=run_dir,
        stem="asset_provenance_backfill",
        rows=provenance_backfills,
        summary_extra={
            "policy": (
                "Backfill by Figure/Scheme/Table locator index, then caption/page "
                "fallback. Tables may use Markdown block provenance without pixels."
            )
        },
    )
    if provenance_backfills:
        print(
            f"[ASSET PROVENANCE BACKFILL] {len(provenance_backfills)} "
            f"pointer update(s); see {run_dir / 'asset_provenance_backfill.csv'}"
        )

    raw_graphml_path = run_dir / "raw_merged.graphml"
    save_graphml(merged, raw_graphml_path)

    paper_root = paper_output_root(
        PROJECT_ROOT, paper.paper_id, data_root=data_root
    )
    resolution_dir = run_dir / "resolution"
    resolution_dir.mkdir(parents=True, exist_ok=True)

    # Candidate generation is non-destructive. Only exact registry-safe
    # Metal/Reaction candidates are auto-approved; all others remain review.
    candidates, generated_candidate_summary = generate_resolution_candidates(
        merged,
        domain_profile=domain_profile,
    )
    write_candidates_csv(resolution_dir / "candidates.csv", candidates)
    candidate_summary_payload = generated_candidate_summary.to_dict()
    write_json(resolution_dir / "candidate_summary.json", candidate_summary_payload)

    stable_resolution_dir = paper_root / "resolution"
    stable_resolution_dir.mkdir(parents=True, exist_ok=True)
    default_decisions_jsonl = stable_resolution_dir / "decisions.jsonl"
    sync_decisions_jsonl(default_decisions_jsonl, candidates)

    decisions_path = resolve_decisions_path(
        paper_resolution_file=paper.resolution_file,
        paper_root=paper_root,
        override=args.decisions,
        disabled=args.no_resolution,
    )
    resolution_plan = load_resolution_plan(
        decisions_path,
        graph=merged,
        resolvable_node_types=(
            domain_profile.resolution.resolvable_node_types
        ),
    )

    canonical = canonicalize_paper_graph(
        merged,
        aliases=resolution_plan.aliases,
        drop_node_ids=resolution_plan.drop_node_ids,
    )
    canonical, graph_domain_canonicalization = (
        apply_graph_domain_canonicalization(
            canonical,
            graph_adapter=graph_adapter,
            paper_id=paper.paper_id,
        )
    )
    assert_measurement_value_xor(
        canonical,
        stage="canonical_after_resolution_and_domain_canonicalization",
    )
    canonical.graph["measurement_merge_invariant_id"] = (
        MEASUREMENT_MERGE_INVARIANT_ID
    )
    model_of_repairs = repair_model_of_targets(canonical)
    write_generic_rows_report(
        run_dir=run_dir,
        stem="semantic_edge_repairs",
        rows=model_of_repairs,
        summary_extra={
            "policy": (
                "MODEL_OF is retargeted only when the current target has a "
                "composition mismatch and exactly one Catalyst matches the "
                "CatalystModel composition signature."
            )
        },
    )
    if model_of_repairs:
        print(
            f"[SEMANTIC EDGE REPAIR] {len(model_of_repairs)} MODEL_OF "
            f"retarget(s); see {run_dir / 'semantic_edge_repairs.csv'}"
        )

    graph_semantics_summary = write_graph_semantics_report(
        run_dir,
        canonical,
        graph_adapter=graph_adapter,
    )
    if graph_semantics_summary["relation_contract_issue_count"]:
        print(
            "[GRAPH SEMANTICS] "
            f"{graph_semantics_summary['relation_contract_issue_count']} "
            "relation contract issue(s); see "
            f"{run_dir / 'graph_semantics' / 'relation_contract_issues.csv'}"
        )
    if graph_semantics_summary["node_role_issue_count"]:
        print(
            "[GRAPH SEMANTICS] "
            f"{graph_semantics_summary['node_role_issue_count']} "
            "node-role issue(s); see "
            f"{run_dir / 'graph_semantics' / 'node_role_issues.csv'}"
        )
    if graph_semantics_summary.get("relation_direction_issue_count", 0):
        print(
            "[RELATION DIRECTION] "
            f"{graph_semantics_summary['relation_direction_issue_count']} "
            "review candidate(s); see "
            f"{run_dir / 'graph_semantics' / 'relation_direction_issues.csv'}"
        )
    if graph_semantics_summary.get("integration_review_component_count", 0):
        print(
            "[GRAPH INTEGRATION] "
            f"{graph_semantics_summary['integration_review_component_count']} "
            "disconnected component(s) require review; see "
            f"{run_dir / 'graph_semantics' / 'integration_components.csv'}"
        )
    if graph_domain_canonicalization["paper_identity_merges"]:
        print(
            "[GRAPH DOMAIN CANONICALIZATION] "
            f"{graph_domain_canonicalization['paper_identity_merges']} "
            "same-paper node merge(s)."
        )

    canonical.graph["graph_stage"] = "canonical"
    canonical.graph.update(quality_attrs)
    canonical.graph["resolution_file"] = (
        str(decisions_path) if decisions_path is not None else ""
    )
    canonical.graph["resolution_source_format"] = (
        resolution_plan.source_format
    )
    canonical.graph["approved_same_entity_decisions"] = (
        resolution_plan.approved_same_entity
    )
    canonical.graph["applied_resolution_aliases"] = (
        resolution_plan.applied_aliases
    )

    canonical_graphml_path = run_dir / "canonical.graphml"
    save_graphml(canonical, canonical_graphml_path)

    reloaded = nx.read_graphml(canonical_graphml_path)
    if reloaded.number_of_nodes() != canonical.number_of_nodes():
        raise AssertionError("GraphML node count changed after serialization.")
    if reloaded.number_of_edges() != canonical.number_of_edges():
        raise AssertionError("GraphML edge count changed after serialization.")

    for _, _, edge_data in reloaded.edges(data=True):
        for required in (
            "relation",
            "title",
            "chunk_id",
            "paper_id",
            "evidence_text",
            "document_id",
            "document_role",
            "evidence_pointers_json",
        ):
            if not edge_data.get(required):
                raise AssertionError(
                    f"Serialized edge is missing {required!r}."
                )

    latest_graphml_path = paper_root / f"{paper.paper_id}.graphml"
    latest_raw_path = paper_root / f"{paper.paper_id}.raw.graphml"
    shutil.copyfile(canonical_graphml_path, latest_graphml_path)
    shutil.copyfile(raw_graphml_path, latest_raw_path)

    comparison_report = build_raw_canonical_report(
        raw_graph=merged,
        canonical_graph=canonical,
        candidate_summary=candidate_summary_payload,
        resolution_summary=resolution_plan.summary(),
        domain_profile=domain_profile,
    )
    write_json(
        resolution_dir / "resolution_report.json",
        comparison_report,
    )
    report_text = format_raw_canonical_report(comparison_report)
    (resolution_dir / "resolution_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    write_json(
        stable_resolution_dir / "latest_resolution_report.json",
        comparison_report,
    )
    (stable_resolution_dir / "latest_resolution_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    claim_audit_summary = write_claim_overlap_audit(
        canonical,
        run_dir / "claim_audit",
    )

    build_summary = {
        "paper_id": paper.paper_id,
        "run_id": run_metadata["run_id"],
        "domain_profile_id": domain_profile.profile_id,
        "graph_adapter_id": graph_adapter.adapter_id,
        "loaded_chunks": loaded_chunks,
        "loaded_chunk_ids": loaded_chunk_ids,
        "graph_materialization_status": materialization_status,
        "extraction_quality": extraction_quality,
        "loaded_documents": sorted({
            str(record.get("document_id", "main"))
            for record in chunk_records
        }),
        "linked_asset_ids": sorted({
            str(asset_id)
            for _, _, edge_data in canonical.edges(data=True)
            for asset_id in json.loads(str(edge_data.get("evidence_asset_ids_json", "[]")))
        }),
        "raw_nodes": merged.number_of_nodes(),
        "raw_edges": merged.number_of_edges(),
        "canonical_nodes": canonical.number_of_nodes(),
        "canonical_edges": canonical.number_of_edges(),
        "resolution": resolution_plan.summary(),
        "resolution_candidates": candidate_summary_payload,
        "semantic_role_adjustments": len(semantic_role_adjustments),
        "metric_normalization_changes": len(metric_normalization_issues),
        "asset_provenance_backfills": len(provenance_backfills),
        "locator_index_records": len(locator_index_records),
        "semantic_model_of_repairs": len(model_of_repairs),
        "graph_domain_canonicalization": graph_domain_canonicalization,
        "graph_semantics": graph_semantics_summary,
        "claim_overlap_audit": claim_audit_summary,
        "raw_graphml": str(raw_graphml_path),
        "canonical_graphml": str(canonical_graphml_path),
        "latest_graphml": str(latest_graphml_path),
        "resolution_report": str(
            resolution_dir / "resolution_report.json"
        ),
    }
    write_json(run_dir / "build_summary.json", build_summary)

    print("GraphML conversion successful")
    print("Paper:", paper.paper_id)
    print("Run ID:", run_metadata["run_id"])
    print("Chunks:", loaded_chunks)
    print(
        "Graph materialization status:",
        materialization_status,
    )
    print(
        "Source-token coverage:",
        extraction_quality.get("source_token_coverage"),
    )
    print("Raw nodes/edges:", merged.number_of_nodes(), merged.number_of_edges())
    print(
        "Canonical nodes/edges:",
        canonical.number_of_nodes(),
        canonical.number_of_edges(),
    )
    print(
        "Components:",
        comparison_report["raw"]["components"],
        "->",
        comparison_report["canonical"]["components"],
    )
    print(
        "Approved same_entity decisions:",
        resolution_plan.approved_same_entity,
    )
    print("Applied aliases:", resolution_plan.applied_aliases)
    print("Auto-approved safe candidates:", generated_candidate_summary.auto_approved_candidates)
    print("Claim-overlap review candidates:", claim_audit_summary["review_required"])
    print(
        "Graph semantics relation issues:",
        graph_semantics_summary["relation_contract_issue_count"],
    )
    print(
        "Graph semantics node-role issues:",
        graph_semantics_summary["node_role_issue_count"],
    )
    print(
        "Relation-direction review candidates:",
        graph_semantics_summary.get("relation_direction_issue_count", 0),
    )
    print(
        "Integration review components:",
        graph_semantics_summary.get("integration_review_component_count", 0),
    )
    print(
        "Component bridge candidates:",
        graph_semantics_summary.get("component_bridge_candidate_count", 0),
    )
    print("Resolution report:", resolution_dir / "resolution_report.txt")
    print("Saved:", latest_graphml_path)


if __name__ == "__main__":
    main()
