from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pipeline_core.bridge_filtering \
    as bridge_filtering_module
import dac_her.bridge_graph \
    as bridge_graph_module
import domains.dac_her.bridge_policy \
    as bridge_policy_module
import dac_her.bridge_relation_repairs \
    as bridge_relation_repairs_module
import dac_her.bridge_schemas \
    as bridge_schemas_module
import dac_her.schemas \
    as schemas_module
import domains.dac_her.scientific_signatures \
    as scientific_signatures_module
import pipeline_core.bridge_policy_run \
    as bridge_policy_run_module

from dac_her.bridge_policy_run import (
    materialize_bridge_policy_run,
)
from dac_her.run_state import (
    paper_output_root,
    read_json,
    resolve_run_directory,
)
from dac_her.schemas import KnowledgeGraph


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reapply the current Bridge policy "
            "to an existing frozen raw extraction."
        )
    )

    parser.add_argument(
        "--paper-id",
        required=True,
    )
    parser.add_argument(
        "--run-id",
        default=None,
    )
    parser.add_argument(
        "--extraction-id",
        default=None,
    )
    parser.add_argument(
        "--canonical-graphml",
        default=None,
    )

    return parser.parse_args()


def _source_path_for_record(
    run_dir: Path,
    record: dict[str, Any],
) -> Path | None:
    source_path_value = record.get(
        "source_path"
    )

    if source_path_value:
        path = Path(
            str(source_path_value)
        )

        if path.exists():
            return path

    safe_chunk_id = str(
        record.get(
            "chunk_id",
            "unknown",
        )
    ).replace(
        ":",
        "__",
    )

    deterministic = (
        run_dir
        / "source_chunks"
        / f"{safe_chunk_id}.json"
    )

    return (
        deterministic
        if deterministic.exists()
        else None
    )


def main() -> None:
    args = parse_args()

    strict_run_dir = (
        resolve_run_directory(
            project_root=PROJECT_ROOT,
            paper_id=args.paper_id,
            run_id=args.run_id,
        )
    )

    active_payload = read_json(
        strict_run_dir
        / "active_chunks.json"
    )
    strict_run_metadata = read_json(
        strict_run_dir
        / "run.json"
    )

    active_payload = {
        **active_payload,
        "run_fingerprint": (
            strict_run_metadata.get(
                "run_fingerprint",
                "",
            )
        ),
    }

    chunk_records = active_payload.get(
        "chunks"
    )

    if (
        not isinstance(
            chunk_records,
            list,
        )
        or not chunk_records
    ):
        raise RuntimeError(
            "No active strict chunks "
            "are available."
        )

    if args.extraction_id:
        extraction_dir = (
            strict_run_dir
            / "bridge_extractions"
            / args.extraction_id
        )
    else:
        pointer_path = (
            strict_run_dir
            / "latest_bridge_extraction.json"
        )

        if not pointer_path.exists():
            raise FileNotFoundError(
                "No latest Bridge extraction "
                f"pointer found: {pointer_path}"
            )

        pointer = read_json(
            pointer_path
        )
        extraction_dir = Path(
            pointer[
                "bridge_extraction_directory"
            ]
        )

    if not extraction_dir.exists():
        raise FileNotFoundError(
            extraction_dir
        )

    paper_root = paper_output_root(
        PROJECT_ROOT,
        args.paper_id,
    )

    canonical_path = (
        Path(
            args.canonical_graphml
        )
        if args.canonical_graphml
        else (
            paper_root
            / f"{args.paper_id}.graphml"
        )
    )

    strict_results: dict[
        str,
        KnowledgeGraph,
    ] = {}
    source_payloads: dict[
        str,
        dict[str, Any],
    ] = {}

    for record in chunk_records:
        strict_path = Path(
            str(record["output_path"])
        )

        strict_result = (
            KnowledgeGraph
            .model_validate_json(
                strict_path.read_text(
                    encoding="utf-8"
                )
            )
        )

        source_path = (
            _source_path_for_record(
                strict_run_dir,
                record,
            )
        )

        if source_path is None:
            raise RuntimeError(
                "Missing source payload for "
                f"{strict_result.chunk_id!r}"
            )

        source_payload = read_json(
            source_path
        )

        chunk_id = str(
            strict_result.chunk_id
        )
        strict_results[
            chunk_id
        ] = strict_result
        source_payloads[
            chunk_id
        ] = source_payload

    policy_implementation_paths = (
        bridge_filtering_module.__file__,
        bridge_policy_module.__file__,
        bridge_relation_repairs_module.__file__,
        bridge_graph_module.__file__,
        bridge_schemas_module.__file__,
        scientific_signatures_module.__file__,
        schemas_module.__file__,
        bridge_policy_run_module.__file__,
    )

    summary = (
        materialize_bridge_policy_run(
            project_root=PROJECT_ROOT,
            paper_id=args.paper_id,
            strict_run_dir=(
                strict_run_dir
            ),
            active_payload=(
                active_payload
            ),
            extraction_dir=(
                extraction_dir
            ),
            canonical_path=(
                canonical_path
            ),
            strict_results=(
                strict_results
            ),
            source_payloads=(
                source_payloads
            ),
            policy_implementation_paths=(
                policy_implementation_paths
            ),
        )
    )

    fatal_rejected = int(
        summary.get(
            "fatal_rejected_candidates",
            summary.get(
                "rejected_candidates",
                0,
            ),
        )
    )
    
    print(
        "Bridge refilter finished"
    )
    print(
        "Extraction ID:",
        summary[
            "bridge_extraction_id"
        ],
    )
    print(
        "Policy run ID:",
        summary[
            "bridge_policy_run_id"
        ],
    )
    print(
        "Patterns:",
        summary["patterns"],
    )
    print(
        "Frontier concepts:",
        summary[
            "frontier_concepts"
        ],
    )
    print(
        "Semantic candidates:",
        summary.get(
            "semantic_candidates",
            0,
        ),
    )
    print(
        "Fatal rejected candidates:",
        fatal_rejected,
    )
    print(
        "Relation repairs:",
        summary[
            "relation_repairs"
        ],
    )
    print(
        "Saved:",
        summary[
            "latest_bridge_graphml"
        ],
    )
    print(
        "Candidate graph:",
        summary.get(
            (
                "latest_candidate_"
                "bridge_graphml"
            ),
            "",
        ),
    )


if __name__ == "__main__":
    main()
