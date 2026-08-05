from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import dac_her.bridge_extraction \
    as bridge_extraction_module
import dac_her.bridge_filtering \
    as bridge_filtering_module
import dac_her.bridge_graph \
    as bridge_graph_module
import dac_her.bridge_policy \
    as bridge_policy_module
import dac_her.bridge_prompts \
    as bridge_prompts_module
import dac_her.bridge_relation_repairs \
    as bridge_relation_repairs_module
import dac_her.bridge_schemas \
    as bridge_schemas_module
import dac_her.bridge_validation \
    as bridge_validation_module
import dac_her.llm_openrouter \
    as llm_openrouter_module
import dac_her.schemas \
    as schemas_module
import dac_her.scientific_signatures \
    as scientific_signatures_module

from dac_her.bridge_extraction import (
    extract_bridge_raw_chunk,
)
from dac_her.bridge_policy_run import (
    materialize_bridge_policy_run,
)
from dac_her.bridge_run_state import (
    bridge_extraction_directory,
    compute_bridge_extraction_metadata,
)
from dac_her.run_state import (
    paper_output_root,
    read_json,
    resolve_run_directory,
    write_json,
)
from dac_her.schemas import KnowledgeGraph


load_dotenv()

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "papers.yaml"
)


def _now_utc() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen raw Bridge candidates "
            "and materialize the current policy run."
        )
    )

    parser.add_argument(
        "--paper-id",
        required=True,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
    )
    parser.add_argument(
        "--run-id",
        default=None,
    )
    parser.add_argument(
        "--model",
        default=(
            os.getenv(
                "OPENROUTER_BRIDGE_MODEL"
            )
            or os.getenv(
                "OPENROUTER_EXTRACT_MODEL"
            )
        ),
    )
    parser.add_argument(
        "--provider",
        default=(
            os.getenv(
                "OPENROUTER_PROVIDER"
            )
            or None
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
    )
    parser.add_argument(
        "--canonical-graphml",
        default=None,
    )
    parser.add_argument(
        "--seed-raw-dir",
        default=None,
        help=(
            "Copy existing *__raw.json files "
            "into the new extraction cache "
            "before extraction."
        ),
    )

    return parser.parse_args()


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


def _seed_raw_cache(
    *,
    seed_dir: Path,
    raw_dir: Path,
) -> int:
    if not seed_dir.exists():
        raise FileNotFoundError(
            seed_dir
        )

    copied = 0

    for source in sorted(
        seed_dir.glob(
            "*__raw.json"
        )
    ):
        destination = (
            raw_dir / source.name
        )

        if destination.exists():
            continue

        shutil.copyfile(
            source,
            destination,
        )
        copied += 1

    return copied


def main() -> None:
    args = parse_args()

    if not args.model:
        raise RuntimeError(
            "Set OPENROUTER_BRIDGE_MODEL "
            "or OPENROUTER_EXTRACT_MODEL, "
            "or pass --model."
        )

    if args.concurrency < 1:
        raise ValueError(
            "--concurrency must be "
            "at least 1."
        )

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

    if (
        not active_payload.get(
            "complete",
            False,
        )
        and not args.allow_incomplete
    ):
        raise RuntimeError(
            "Strict extraction is incomplete. "
            "Resolve failed chunks or pass "
            "--allow-incomplete explicitly."
        )

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

    jobs: list[
        tuple[
            dict[str, Any],
            KnowledgeGraph,
            dict[str, Any],
            Path,
            Path,
        ]
    ] = []

    strict_results: dict[
        str,
        KnowledgeGraph,
    ] = {}
    source_payloads: dict[
        str,
        dict[str, Any],
    ] = {}
    missing_sources: list[str] = []

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
            missing_sources.append(
                str(
                    record.get(
                        "chunk_id",
                        "unknown",
                    )
                )
            )
            continue

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

        jobs.append(
            (
                record,
                strict_result,
                source_payload,
                strict_path,
                source_path,
            )
        )

    if missing_sources:
        raise RuntimeError(
            "Active chunks do not contain "
            "source snapshots required for "
            "Bridge extraction. Missing: "
            f"{missing_sources[:8]!r}"
        )

    extraction_implementation_paths = (
        bridge_extraction_module.__file__,
        bridge_prompts_module.__file__,
        bridge_schemas_module.__file__,
        bridge_validation_module.__file__,
        scientific_signatures_module.__file__,
        schemas_module.__file__,
        llm_openrouter_module.__file__,
    )

    extraction_metadata = (
        compute_bridge_extraction_metadata(
            strict_run_dir=strict_run_dir,
            active_payload=active_payload,
            model=args.model,
            provider=args.provider,
            strict_chunk_paths=[
                item[3]
                for item in jobs
            ],
            source_chunk_paths=[
                item[4]
                for item in jobs
            ],
            implementation_paths=(
                extraction_implementation_paths
            ),
            runtime_options={
                "max_repairs": 2,
                "max_tokens": 3400,
                "reproducible": False,
                "zdr": True,
            },
        )
    )

    extraction_id = str(
        extraction_metadata[
            "bridge_extraction_id"
        ]
    )

    extraction_dir = (
        bridge_extraction_directory(
            strict_run_dir,
            extraction_id,
        )
    )
    raw_dir = (
        extraction_dir
        / "raw_chunks"
    )
    debug_dir = (
        extraction_dir
        / "debug"
    )
    manifest_path = (
        extraction_dir
        / "manifest.jsonl"
    )

    for directory in (
        extraction_dir,
        raw_dir,
        debug_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    seeded_count = 0

    if args.seed_raw_dir:
        seeded_count = (
            _seed_raw_cache(
                seed_dir=Path(
                    args.seed_raw_dir
                ),
                raw_dir=raw_dir,
            )
        )

    write_json(
        extraction_dir / "run.json",
        extraction_metadata,
    )
    manifest_path.write_text(
        "",
        encoding="utf-8",
    )

    records: list[
        dict[str, Any]
    ] = []

    with ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        future_map = {
            executor.submit(
                extract_bridge_raw_chunk,
                strict_result=(
                    strict_result
                ),
                source_payload=(
                    source_payload
                ),
                model=args.model,
                provider=args.provider,
                output_dir=raw_dir,
                debug_dir=debug_dir,
                force=args.force,
            ): strict_result.chunk_id
            for (
                _,
                strict_result,
                source_payload,
                _,
                _,
            ) in jobs
        }

        for future in as_completed(
            future_map
        ):
            chunk_id = future_map[
                future
            ]

            try:
                record = future.result()
            except Exception as error:
                record = {
                    "status": "failed",
                    "paper_id": (
                        args.paper_id
                    ),
                    "chunk_id": chunk_id,
                    "error_type": (
                        type(error).__name__
                    ),
                    "error_message": str(
                        error
                    ),
                }

            record[
                "bridge_extraction_id"
            ] = extraction_id
            record[
                "recorded_at_utc"
            ] = _now_utc()

            records.append(record)
            _append_jsonl(
                manifest_path,
                record,
            )

            print(
                f"[RAW-{record['status'].upper()}] "
                f"{chunk_id} "
                f"concepts="
                f"{record.get('raw_concept_count', 0)} "
                f"patterns="
                f"{record.get('raw_pattern_count', 0)} "
                f"frontier="
                f"{record.get('raw_frontier_count', 0)} "
                f"links="
                f"{record.get('raw_link_count', 0)}",
                flush=True,
            )

    failed = [
        record
        for record in records
        if record["status"]
        == "failed"
    ]

    extraction_summary = {
        "paper_id": args.paper_id,
        "strict_run_id": (
            active_payload.get(
                "run_id"
            )
        ),
        "bridge_extraction_id": (
            extraction_id
        ),
        "bridge_extraction_fingerprint": (
            extraction_metadata[
                "bridge_extraction_fingerprint"
            ]
        ),
        "complete": not failed,
        "model": args.model,
        "provider": (
            args.provider or ""
        ),
        "chunks": len(records),
        "failed_chunks": failed,
        "seed_raw_dir": (
            str(args.seed_raw_dir)
            if args.seed_raw_dir
            else ""
        ),
        "seeded_raw_files": (
            seeded_count
        ),
    }

    write_json(
        extraction_dir / "summary.json",
        extraction_summary,
    )

    if failed:
        raise SystemExit(2)

    write_json(
        strict_run_dir
        / "latest_bridge_extraction.json",
        {
            "paper_id": args.paper_id,
            "strict_run_id": (
                active_payload.get(
                    "run_id"
                )
            ),
            "bridge_extraction_id": (
                extraction_id
            ),
            "bridge_extraction_directory": str(
                extraction_dir
            ),
            "bridge_extraction_fingerprint": (
                extraction_metadata[
                    "bridge_extraction_fingerprint"
                ]
            ),
            "updated_at_utc": (
                _now_utc()
            ),
        },
    )

    policy_implementation_paths = (
        bridge_filtering_module.__file__,
        bridge_policy_module.__file__,
        bridge_relation_repairs_module.__file__,
        bridge_graph_module.__file__,
        bridge_schemas_module.__file__,
        scientific_signatures_module.__file__,
        schemas_module.__file__,
    )

    policy_summary = (
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

    print(
        "Bridge extraction and policy "
        "materialization finished"
    )
    print(
        "Extraction ID:",
        extraction_id,
    )
    print(
        "Policy run ID:",
        policy_summary[
            "bridge_policy_run_id"
        ],
    )
    print(
        "Patterns:",
        policy_summary["patterns"],
    )
    print(
        "Frontier concepts:",
        policy_summary[
            "frontier_concepts"
        ],
    )
    print(
        "Rejected candidates:",
        policy_summary[
            "rejected_candidates"
        ],
    )
    print(
        "Relation repairs:",
        policy_summary[
            "relation_repairs"
        ],
    )
    print(
        "Saved:",
        policy_summary[
            "latest_bridge_graphml"
        ],
    )


if __name__ == "__main__":
    main()
