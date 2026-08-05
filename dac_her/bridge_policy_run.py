from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from dac_her.bridge_extraction import (
    bridge_raw_output_path,
)
from dac_her.bridge_filtering import (
    filter_bridge_raw_chunk,
)
from dac_her.bridge_graph import (
    build_bridge_graph,
    save_bridge_graph,
    write_bridge_tables,
)
from dac_her.bridge_policy import (
    BRIDGE_POLICY_VERSION,
)
from dac_her.bridge_prompts import (
    BRIDGE_PROMPT_VERSION,
)
from dac_her.bridge_run_state import (
    bridge_policy_run_directory,
    compute_bridge_policy_run_metadata,
)
from dac_her.bridge_schemas import (
    BridgeChunkGraph,
)
from dac_her.run_state import (
    paper_output_root,
    read_json,
    write_json,
)
from dac_her.schemas import KnowledgeGraph


def _now_utc() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:
    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def _load_rejections(
    records: Iterable[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for record in records:
        value = record.get(
            "rejections_path"
        )

        if not value:
            continue

        path = Path(str(value))

        if not path.exists():
            continue

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        if isinstance(payload, list):
            rows.extend(
                item
                for item in payload
                if isinstance(item, dict)
            )

    return rows

def _load_candidate_issues(
    records: Iterable[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    for record in records:
        value = record.get(
            "candidate_issues_path"
        )

        if not value:
            continue

        path = Path(str(value))

        if not path.exists():
            continue

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        if isinstance(payload, list):
            rows.extend(
                item
                for item in payload
                if isinstance(item, dict)
            )

    return rows

def _copy_latest_artifacts(
    *,
    source_dir: Path,
    legacy_dir: Path,
    policy_run_id: str,
    policy_fingerprint: str,
    extraction_id: str,
) -> None:
    """
    Maintain the historical strict-run/bridge path
    as a compatibility alias.
    """
    legacy_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    names = (
        "summary.json",
        "run.json",
        "bridge.raw.graphml",
        "bridge.graphml",
        "bridge_concepts.csv",
        "bridge_patterns.csv",
        "bridge_frontier.csv",
        "bridge_links.csv",
        "bridge_issues.csv",
        "bridge_rejected.csv",
        "bridge.candidates.graphml",
    )

    for name in names:
        source = source_dir / name

        if source.exists():
            shutil.copyfile(
                source,
                legacy_dir / name,
            )

    write_json(
        legacy_dir / "latest_run.json",
        {
            "bridge_run_id": (
                policy_run_id
            ),
            "bridge_run_directory": str(
                source_dir
            ),
            "bridge_run_fingerprint": (
                policy_fingerprint
            ),
            "bridge_extraction_id": (
                extraction_id
            ),
            "updated_at_utc": (
                _now_utc()
            ),
        },
    )


def materialize_bridge_policy_run(
    *,
    project_root: str | Path,
    paper_id: str,
    strict_run_dir: str | Path,
    active_payload: dict[str, Any],
    extraction_dir: str | Path,
    canonical_path: str | Path,
    strict_results: dict[
        str,
        KnowledgeGraph,
    ],
    source_payloads: dict[
        str,
        dict[str, Any],
    ],
    policy_implementation_paths: Iterable[
        str | Path
    ],
) -> dict[str, Any]:
    """
    Apply the current deterministic repair and Bridge
    policy to an existing frozen raw extraction.

    No LLM calls are performed here.
    """
    project_root = Path(
        project_root
    ).resolve()
    strict_run_dir = Path(
        strict_run_dir
    ).resolve()
    extraction_dir = Path(
        extraction_dir
    ).resolve()
    canonical_path = Path(
        canonical_path
    )

    extraction_run_path = (
        extraction_dir / "run.json"
    )

    if not extraction_run_path.exists():
        raise FileNotFoundError(
            "Bridge extraction metadata not found: "
            f"{extraction_run_path}"
        )

    extraction_metadata = read_json(
        extraction_run_path
    )

    raw_dir = (
        extraction_dir
        / "raw_chunks"
    )

    chunk_records = active_payload.get(
        "chunks",
        [],
    )

    if not isinstance(
        chunk_records,
        list,
    ):
        raise ValueError(
            "active_payload.chunks must be a list."
        )

    chunk_ids = [
        str(record["chunk_id"])
        for record in chunk_records
    ]

    raw_paths = [
        bridge_raw_output_path(
            chunk_id,
            raw_dir,
        )
        for chunk_id in chunk_ids
    ]

    missing_raw = [
        str(path)
        for path in raw_paths
        if not path.exists()
    ]

    if missing_raw:
        raise RuntimeError(
            "Raw Bridge extraction is incomplete. "
            f"Missing {len(missing_raw)} files; "
            f"examples: {missing_raw[:5]!r}"
        )

    missing_strict = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id not in strict_results
    ]
    missing_sources = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id not in source_payloads
    ]

    if missing_strict:
        raise RuntimeError(
            "Missing strict results for policy run: "
            f"{missing_strict[:5]!r}"
        )

    if missing_sources:
        raise RuntimeError(
            "Missing source payloads for policy run: "
            f"{missing_sources[:5]!r}"
        )

    policy_metadata = (
        compute_bridge_policy_run_metadata(
            strict_run_dir=strict_run_dir,
            extraction_metadata=(
                extraction_metadata
            ),
            raw_chunk_paths=raw_paths,
            canonical_graph_path=(
                canonical_path
                if canonical_path.exists()
                else None
            ),
            implementation_paths=(
                policy_implementation_paths
            ),
        )
    )

    policy_run_id = str(
        policy_metadata[
            "bridge_policy_run_id"
        ]
    )
    policy_fingerprint = str(
        policy_metadata[
            "bridge_policy_run_fingerprint"
        ]
    )

    policy_dir = (
        bridge_policy_run_directory(
            strict_run_dir,
            policy_run_id,
        )
    )
    filtered_dir = (
        policy_dir / "chunks"
    )
    manifest_path = (
        policy_dir / "manifest.jsonl"
    )

    for directory in (
        policy_dir,
        filtered_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest_path.write_text(
        "",
        encoding="utf-8",
    )
    write_json(
        policy_dir / "run.json",
        policy_metadata,
    )

    records: list[
        dict[str, Any]
    ] = []

    for chunk_id, raw_path in zip(
        chunk_ids,
        raw_paths,
        strict=True,
    ):
        try:
            raw_result = (
                BridgeChunkGraph
                .model_validate_json(
                    raw_path.read_text(
                        encoding="utf-8"
                    )
                )
            )

            record = (
                filter_bridge_raw_chunk(
                    raw_result=raw_result,
                    strict_result=(
                        strict_results[
                            chunk_id
                        ]
                    ),
                    source_payload=(
                        source_payloads[
                            chunk_id
                        ]
                    ),
                    output_dir=(
                        filtered_dir
                    ),
                )
            )

        except Exception as error:
            failed_record = {
                "status": "failed",
                "paper_id": paper_id,
                "chunk_id": chunk_id,
                "error_type": (
                    type(error).__name__
                ),
                "error_message": str(
                    error
                ),
                "bridge_policy_run_id": (
                    policy_run_id
                ),
                "recorded_at_utc": (
                    _now_utc()
                ),
            }

            records.append(
                failed_record
            )
            _append_jsonl(
                manifest_path,
                failed_record,
            )

            write_json(
                policy_dir / "summary.json",
                {
                    "paper_id": paper_id,
                    "complete": False,
                    "bridge_policy_run_id": (
                        policy_run_id
                    ),
                    "bridge_policy_run_fingerprint": (
                        policy_fingerprint
                    ),
                    "failed_chunks": [
                        failed_record
                    ],
                },
            )

            raise RuntimeError(
                "Bridge policy filtering failed "
                f"for {chunk_id!r}: {error}"
            ) from error

        record[
            "bridge_policy_run_id"
        ] = policy_run_id
        record[
            "recorded_at_utc"
        ] = _now_utc()

        records.append(record)
        _append_jsonl(
            manifest_path,
            record,
        )

        print(
            f"[POLICY] {chunk_id} "
            f"patterns="
            f"{record['pattern_count']} "
            f"frontier="
            f"{record['frontier_count']} "
            f"candidate="
            f"{record['candidate_count']} "
            f"fatal_rejected="
            f"{record['fatal_rejection_count']} "
            f"rejected="
            f"{record['rejection_count']} "
            f"repairs="
            f"{record['relation_repair_count']}",
            flush=True,
        )

    bridge_results = [
        BridgeChunkGraph
        .model_validate_json(
            Path(
                str(
                    record[
                        "output_path"
                    ]
                )
            ).read_text(
                encoding="utf-8"
            )
        )
        for record in records
    ]

    candidate_results = [
        BridgeChunkGraph
        .model_validate_json(
            Path(
                str(
                    record[
                        "candidates_path"
                    ]
                )
            ).read_text(
                encoding="utf-8"
            )
        )
        for record in records
    ]

    rejections = _load_rejections(
        records
    )

    candidate_records = (
        _load_candidate_issues(
            records
        )
    )

    canonical_graph = (
        nx.read_graphml(
            canonical_path,
            force_multigraph=True,
        )
        if canonical_path.exists()
        else None
    )

    raw_graph, raw_issues = (
        build_bridge_graph(
            bridge_results,
            strict_results=(
                strict_results
            ),
            canonical_graph=None,
        )
    )

    canonical_bridge_graph, (
        canonical_issues
    ) = build_bridge_graph(
        bridge_results,
        strict_results=strict_results,
        canonical_graph=canonical_graph,
    )

    candidate_bridge_graph, (
        candidate_issues
    ) = build_bridge_graph(
        candidate_results,
        strict_results=strict_results,
        canonical_graph=canonical_graph,
        graph_layer="bridge_candidate",
        evidence_status=(
            "semantic_candidate"
        ),
    )

    graph_metadata = {
        "bridge_prompt_version": str(
            extraction_metadata.get(
                "bridge_prompt_version",
                BRIDGE_PROMPT_VERSION,
            )
        ),
        "bridge_extraction_id": str(
            extraction_metadata[
                "bridge_extraction_id"
            ]
        ),
        "bridge_extraction_fingerprint": str(
            extraction_metadata[
                "bridge_extraction_fingerprint"
            ]
        ),
        "bridge_policy_version": (
            BRIDGE_POLICY_VERSION
        ),
        "bridge_policy_run_id": (
            policy_run_id
        ),
        "bridge_policy_run_fingerprint": (
            policy_fingerprint
        ),

        # Legacy metadata aliases.
        "bridge_run_id": (
            policy_run_id
        ),
        "bridge_run_fingerprint": (
            policy_fingerprint
        ),
    }

    raw_graph.graph.update(
        graph_metadata
    )
    canonical_bridge_graph.graph.update(
        graph_metadata
    )
    candidate_bridge_graph.graph.update(
        graph_metadata
    )
    candidate_bridge_graph.graph.update({
        "graph_layer": (
            "bridge_candidate"
        ),
        "evidence_status": (
            "semantic_candidate"
        ),
    })

    raw_graph_path = (
        save_bridge_graph(
            raw_graph,
            policy_dir
            / "bridge.raw.graphml",
        )
    )
    graph_path = save_bridge_graph(
        canonical_bridge_graph,
        policy_dir
        / "bridge.graphml",
    )
    candidate_graph_path = (
        save_bridge_graph(
            candidate_bridge_graph,
            policy_dir
            / "bridge.candidates.graphml",
        )
    )

    write_bridge_tables(
        canonical_bridge_graph,
        canonical_issues,
        rejections,
        policy_dir,
    )

    paper_root = paper_output_root(
        project_root,
        paper_id,
    )
    latest_bridge_path = (
        paper_root
        / f"{paper_id}.bridge.graphml"
    )
    latest_bridge_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copyfile(
        graph_path,
        latest_bridge_path,
    )
    latest_candidate_bridge_path = (
        paper_root
        / (
            f"{paper_id}"
            ".bridge.candidates.graphml"
        )
    )

    shutil.copyfile(
        candidate_graph_path,
        latest_candidate_bridge_path,
    )

    pattern_count = sum(
        concept.retention_lane
        == "accepted_pattern"
        for result in bridge_results
        for concept in result.concepts
    )
    frontier_count = sum(
        concept.retention_lane
        == "paper_local_frontier"
        for result in bridge_results
        for concept in result.concepts
    )
    concept_count = (
        pattern_count
        + frontier_count
    )
    link_count = sum(
        len(result.links)
        for result in bridge_results
    )
    repair_count = sum(
        int(
            record.get(
                "relation_repair_count",
                0,
            )
        )
        for record in records
    )
    candidate_concept_count = sum(
        len(result.concepts)
        for result in candidate_results
    )

    candidate_link_count = sum(
        len(result.links)
        for result in candidate_results
    )

    summary = {
        "paper_id": paper_id,
        "strict_run_id": (
            active_payload.get(
                "run_id"
            )
        ),
        "complete": True,
        "bridge_extraction_id": (
            extraction_metadata[
                "bridge_extraction_id"
            ]
        ),
        "bridge_extraction_fingerprint": (
            extraction_metadata[
                "bridge_extraction_fingerprint"
            ]
        ),
        "bridge_policy_run_id": (
            policy_run_id
        ),
        "bridge_policy_run_fingerprint": (
            policy_fingerprint
        ),

        # Legacy aliases.
        "bridge_run_id": (
            policy_run_id
        ),
        "bridge_run_fingerprint": (
            policy_fingerprint
        ),

        "prompt_version": str(
            extraction_metadata.get(
                "bridge_prompt_version",
                BRIDGE_PROMPT_VERSION,
            )
        ),
        "policy_version": (
            BRIDGE_POLICY_VERSION
        ),
        "chunks": len(
            bridge_results
        ),
        "concepts": concept_count,
        "patterns": pattern_count,
        "frontier_concepts": (
            frontier_count
        ),
        "links": link_count,
        "fatal_rejected_candidates": len(
            rejections
        ),
        "rejected_candidates": len(
            rejections
        ),
        "relation_repairs": (
            repair_count
        ),
        "failed_chunks": [],
        "raw_anchor_issues": len(
            raw_issues
        ),
        "canonical_anchor_issues": len(
            canonical_issues
        ),
        "canonical_graph_used": (
            str(canonical_path)
            if canonical_graph is not None
            else ""
        ),
        "raw_graphml": str(
            raw_graph_path
        ),
        "bridge_graphml": str(
            graph_path
        ),
        "latest_bridge_graphml": str(
            latest_bridge_path
        ),
        "semantic_candidates": (
            candidate_concept_count
        ),
        "semantic_candidate_links": (
            candidate_link_count
        ),
        "candidate_anchor_issues": len(
            candidate_issues
        ),
        "candidate_graphml": str(
            candidate_graph_path
        ),
        "latest_candidate_bridge_graphml": str(
            latest_candidate_bridge_path
        ),
    }

    write_json(
        policy_dir / "summary.json",
        summary,
    )

    _copy_latest_artifacts(
        source_dir=policy_dir,
        legacy_dir=(
            strict_run_dir
            / "bridge"
        ),
        policy_run_id=policy_run_id,
        policy_fingerprint=(
            policy_fingerprint
        ),
        extraction_id=str(
            extraction_metadata[
                "bridge_extraction_id"
            ]
        ),
    )

    latest_policy_payload = {
        "paper_id": paper_id,
        "strict_run_id": (
            active_payload.get(
                "run_id"
            )
        ),
        "bridge_extraction_id": (
            extraction_metadata[
                "bridge_extraction_id"
            ]
        ),
        "bridge_policy_run_id": (
            policy_run_id
        ),
        "bridge_policy_run_directory": str(
            policy_dir
        ),
        "bridge_policy_run_fingerprint": (
            policy_fingerprint
        ),
        "updated_at_utc": (
            _now_utc()
        ),
    }

    write_json(
        strict_run_dir
        / "latest_bridge_policy_run.json",
        latest_policy_payload,
    )

    # Existing tools may still read this pointer.
    write_json(
        strict_run_dir
        / "latest_bridge_run.json",
        {
            "paper_id": paper_id,
            "strict_run_id": (
                active_payload.get(
                    "run_id"
                )
            ),
            "bridge_run_id": (
                policy_run_id
            ),
            "bridge_run_directory": str(
                policy_dir
            ),
            "bridge_run_fingerprint": (
                policy_fingerprint
            ),
            "bridge_extraction_id": (
                extraction_metadata[
                    "bridge_extraction_id"
                ]
            ),
            "updated_at_utc": (
                _now_utc()
            ),
        },
    )

    return summary
