from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dac_her.bridge_schemas import BridgeChunkGraph
from dac_her.run_state import read_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PAPERS = (
    "Kiwook_1",
    "Kiwook_2",
    "Kiwook_3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export frozen Bridge raw candidates and "
            "current policy decisions for semantic calibration."
        )
    )
    parser.add_argument(
        "--papers",
        nargs="+",
        default=list(DEFAULT_PAPERS),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_ROOT
            / "calibration"
            / "bridge_semantic"
        ),
    )
    return parser.parse_args()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_key(
    paper_id: str,
    chunk_id: str,
    concept_id: str,
) -> str:
    payload = (
        f"{paper_id}|{chunk_id}|{concept_id}"
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()[:20]


def _load_json_list(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, list):
        raise ValueError(
            f"Expected a JSON list: {path}"
        )

    return [
        item
        for item in payload
        if isinstance(item, dict)
    ]


def _json_text(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError(
            "No Bridge calibration rows were generated."
        )

    fields: list[str] = []

    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def _safe_chunk_id(
    chunk_id: str,
) -> str:
    return chunk_id.replace(
        ":",
        "__",
    )


def main() -> None:
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, Any]] = []
    manifest_papers: dict[
        str,
        dict[str, Any],
    ] = {}

    for paper_id in args.papers:
        paper_root = (
            PROJECT_ROOT
            / "data_dac"
            / "extracted"
            / paper_id
        )

        strict_pointer = read_json(
            paper_root
            / "latest_run.json"
        )
        strict_dir = Path(
            strict_pointer[
                "run_directory"
            ]
        )

        extraction_pointer = read_json(
            strict_dir
            / "latest_bridge_extraction.json"
        )
        policy_pointer = read_json(
            strict_dir
            / "latest_bridge_policy_run.json"
        )

        extraction_dir = Path(
            extraction_pointer[
                "bridge_extraction_directory"
            ]
        )
        policy_dir = Path(
            policy_pointer[
                "bridge_policy_run_directory"
            ]
        )

        extraction_metadata = read_json(
            extraction_dir / "run.json"
        )
        policy_metadata = read_json(
            policy_dir / "run.json"
        )

        extraction_id = str(
            extraction_metadata[
                "bridge_extraction_id"
            ]
        )
        policy_run_id = str(
            policy_metadata[
                "bridge_policy_run_id"
            ]
        )

        manifest_papers[paper_id] = {
            "strict_run_id": (
                strict_pointer["run_id"]
            ),
            "bridge_extraction_id": (
                extraction_id
            ),
            "bridge_extraction_fingerprint": (
                extraction_metadata[
                    "bridge_extraction_fingerprint"
                ]
            ),
            "current_bridge_policy_run_id": (
                policy_run_id
            ),
            "current_bridge_policy_version": (
                policy_metadata[
                    "bridge_policy_version"
                ]
            ),
        }

        raw_dir = (
            extraction_dir
            / "raw_chunks"
        )
        filtered_dir = (
            policy_dir
            / "chunks"
        )

        raw_paths = sorted(
            raw_dir.glob(
                "*__raw.json"
            )
        )

        for raw_path in raw_paths:
            raw_result = (
                BridgeChunkGraph
                .model_validate_json(
                    raw_path.read_text(
                        encoding="utf-8"
                    )
                )
            )

            safe_id = _safe_chunk_id(
                raw_result.chunk_id
            )

            filtered_path = (
                filtered_dir
                / f"{safe_id}.json"
            )
            rejection_path = (
                filtered_dir
                / (
                    f"{safe_id}"
                    "__rejections.json"
                )
            )
            repair_path = (
                filtered_dir
                / (
                    f"{safe_id}"
                    "__relation_repairs.json"
                )
            )
            candidate_path = (
                filtered_dir
                / (
                    f"{safe_id}"
                    "__candidates.json"
                )
            )

            candidate_issues_path = (
                filtered_dir
                / (
                    f"{safe_id}"
                    "__candidate_issues.json"
                )
            )
            
            filtered_result = (
                BridgeChunkGraph
                .model_validate_json(
                    filtered_path.read_text(
                        encoding="utf-8"
                    )
                )
            )
            candidate_result = (
                BridgeChunkGraph
                .model_validate_json(
                    candidate_path.read_text(
                        encoding="utf-8"
                    )
                )
            )

            candidate_by_id = {
                concept.id: concept
                for concept
                in candidate_result.concepts
            }
            filtered_by_id = {
                concept.id: concept
                for concept
                in filtered_result.concepts
            }

            rejection_by_id = {
                str(row["concept_id"]): row
                for row in _load_json_list(
                    rejection_path
                )
            }
            candidate_issue_by_id = {
                str(row["concept_id"]): row
                for row in _load_json_list(
                    candidate_issues_path
                )
            }
            repairs_by_id: dict[
                str,
                list[dict[str, Any]],
            ] = {}

            for repair in _load_json_list(
                repair_path
            ):
                repairs_by_id.setdefault(
                    str(
                        repair["concept_id"]
                    ),
                    [],
                ).append(repair)

            links_by_concept: dict[
                str,
                list[Any],
            ] = {}

            for link in raw_result.links:
                links_by_concept.setdefault(
                    link.concept_id,
                    [],
                ).append(link)

            for concept in raw_result.concepts:
                filtered_concept = (
                    filtered_by_id.get(
                        concept.id
                    )
                )
                candidate_concept = (
                    candidate_by_id.get(
                        concept.id
                    )
                )

                candidate_issue = (
                    candidate_issue_by_id.get(
                        concept.id
                    )
                )
                rejection = (
                    rejection_by_id.get(
                        concept.id
                    )
                )
                repairs = (
                    repairs_by_id.get(
                        concept.id,
                        [],
                    )
                )

                if filtered_concept is not None:
                    automatic_status = (
                        "ACCEPTED_PATTERN"
                        if (
                            filtered_concept
                            .retention_lane
                            == "accepted_pattern"
                        )
                        else "KEPT_FRONTIER"
                    )

                    effective_subject = (
                        filtered_concept
                        .pattern_subject
                        or ""
                    )
                    effective_relation = (
                        filtered_concept
                        .pattern_relation
                        or ""
                    )
                    effective_object = (
                        filtered_concept
                        .pattern_object
                        or ""
                    )
                elif candidate_concept is not None:
                    automatic_status = (
                        "SEMANTIC_CANDIDATE"
                    )

                    effective_subject = (
                        candidate_concept
                        .pattern_subject
                        or ""
                    )
                    effective_relation = (
                        candidate_concept
                        .pattern_relation
                        or ""
                    )
                    effective_object = (
                        candidate_concept
                        .pattern_object
                        or ""
                    )
                elif rejection is not None:
                    automatic_status = (
                        "REJECTED"
                    )
                    effective_subject = str(
                        rejection.get(
                            "pattern_subject",
                            concept.pattern_subject
                            or "",
                        )
                    )
                    effective_relation = str(
                        rejection.get(
                            "pattern_relation",
                            concept.pattern_relation
                            or "",
                        )
                    )
                    effective_object = str(
                        rejection.get(
                            "pattern_object",
                            concept.pattern_object
                            or "",
                        )
                    )

                else:
                    automatic_status = (
                        "UNACCOUNTED"
                    )
                    effective_subject = (
                        concept.pattern_subject
                        or ""
                    )
                    effective_relation = (
                        concept.pattern_relation
                        or ""
                    )
                    effective_object = (
                        concept.pattern_object
                        or ""
                    )

                links = links_by_concept.get(
                    concept.id,
                    [],
                )

                pointer_count = sum(
                    len(
                        link.evidence_pointers
                    )
                    for link in links
                )

                row = {
                    "candidate_key": (
                        _candidate_key(
                            paper_id,
                            raw_result.chunk_id,
                            concept.id,
                        )
                    ),
                    "paper_id": paper_id,
                    "bridge_extraction_id": (
                        extraction_id
                    ),
                    "bridge_policy_run_id": (
                        policy_run_id
                    ),
                    "chunk_id": (
                        raw_result.chunk_id
                    ),
                    "section": (
                        raw_result.section
                    ),
                    "document_id": (
                        raw_result.document_id
                    ),
                    "concept_id": concept.id,
                    "concept_type": (
                        concept.concept_type
                    ),
                    "in_relation_calibration": (
                        concept.retention_lane
                        == "accepted_pattern"
                    ),
                    "automatic_status": (
                        automatic_status
                    ),
                    "raw_retention_lane": (
                        concept.retention_lane
                    ),
                    "raw_pattern_subject": (
                        concept.pattern_subject
                        or ""
                    ),
                    "raw_pattern_relation": (
                        concept.pattern_relation
                        or ""
                    ),
                    "raw_pattern_object": (
                        concept.pattern_object
                        or ""
                    ),
                    "effective_pattern_subject": (
                        effective_subject
                    ),
                    "effective_pattern_relation": (
                        effective_relation
                    ),
                    "effective_pattern_object": (
                        effective_object
                    ),
                    "relation_strength": (
                        concept.relation_strength
                        or ""
                    ),
                    "pattern_support_mode": (
                        concept.pattern_support_mode
                        or ""
                    ),
                    "evidence_scope": (
                        concept.evidence_scope
                    ),
                    "label": concept.label,
                    "source_phrase": (
                        concept.source_phrase
                    ),
                    "supporting_phrases_json": (
                        _json_text(
                            concept.supporting_phrases
                        )
                    ),
                    "subject_evidence_phrase": (
                        concept.subject_evidence_phrase
                        or ""
                    ),
                    "relation_evidence_phrase": (
                        concept.relation_evidence_phrase
                        or ""
                    ),
                    "object_evidence_phrase": (
                        concept.object_evidence_phrase
                        or ""
                    ),
                    "comparison_items_json": (
                        _json_text(
                            [
                                item.model_dump()
                                for item
                                in concept.comparison_items
                            ]
                        )
                    ),
                    "qualifiers_json": (
                        _json_text(
                            [
                                item.model_dump()
                                for item
                                in concept.qualifiers
                            ]
                        )
                    ),
                    "anchor_ids_json": (
                        _json_text(
                            sorted(
                                {
                                    link.anchor_id
                                    for link in links
                                }
                            )
                        )
                    ),
                    "anchor_relations_json": (
                        _json_text(
                            sorted(
                                {
                                    link.relation
                                    for link in links
                                }
                            )
                        )
                    ),
                    "evidence_pointers_json": (
                        _json_text(
                            [
                                pointer.model_dump()
                                for link in links
                                for pointer
                                in link.evidence_pointers
                            ]
                        )
                    ),
                    "evidence_pointer_count": (
                        pointer_count
                    ),
                    "candidate_reason_codes_json": (
                        _json_text(
                            (
                                candidate_issue.get(
                                    "reason_codes",
                                    [],
                                )
                                if candidate_issue
                                else []
                            )
                        )
                    ),
                    "rejection_reason_codes_json": (
                        _json_text(
                            (
                                rejection.get(
                                    "reason_codes",
                                    [],
                                )
                                if rejection
                                else []
                            )
                        )
                    ),
                    "rejection_reason_details_json": (
                        _json_text(
                            (
                                rejection.get(
                                    "reason_details",
                                    [],
                                )
                                if rejection
                                else []
                            )
                        )
                    ),
                    "repair_rule_ids_json": (
                        _json_text(
                            [
                                repair.get(
                                    "rule_id",
                                    "",
                                )
                                for repair in repairs
                            ]
                        )
                    ),

                    # Human adjudication fields.
                    "manual_decision": "",
                    "manual_pattern_subject": "",
                    "manual_pattern_relation": "",
                    "manual_pattern_object": "",
                    "manual_anchor_correct": "",
                    "manual_reason": "",
                    "manual_confidence": "",
                    "manual_notes": "",
                    "reviewer": "",
                    "reviewed_at": "",
                }

                rows.append(row)

    unaccounted = [
        row["candidate_key"]
        for row in rows
        if row["automatic_status"]
        == "UNACCOUNTED"
    ]

    if unaccounted:
        raise RuntimeError(
            "Some raw candidates were neither "
            "accepted nor rejected: "
            f"{unaccounted[:10]!r}"
        )

    manifest = {
        "calibration_name": (
            "bridge-semantic-calibration-3-paper"
        ),
        "generated_at_utc": _now_utc(),
        "papers": manifest_papers,
    }

    manifest_path = (
        output_dir
        / "calibration_manifest.json"
    )

    if manifest_path.exists():
        previous = read_json(
            manifest_path
        )

        for paper_id in args.papers:
            previous_id = (
                previous["papers"][
                    paper_id
                ][
                    "bridge_extraction_id"
                ]
            )
            current_id = (
                manifest_papers[
                    paper_id
                ][
                    "bridge_extraction_id"
                ]
            )

            if previous_id != current_id:
                raise RuntimeError(
                    "Calibration raw extraction "
                    f"changed for {paper_id}: "
                    f"{previous_id} -> {current_id}. "
                    "Do not reuse the old gold file."
                )
    else:
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    predictions_path = (
        output_dir
        / "predictions.csv"
    )
    _write_csv(
        predictions_path,
        rows,
    )

    current_predictions = {
        "generated_at_utc": _now_utc(),
        "papers": manifest_papers,
        "candidate_count": len(rows),
        "relation_candidate_count": sum(
            bool(
                row[
                    "in_relation_calibration"
                ]
            )
            for row in rows
        ),
    }

    (
        output_dir
        / "current_predictions.json"
    ).write_text(
        json.dumps(
            current_predictions,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    gold_path = (
        output_dir
        / "gold.csv"
    )

    if not gold_path.exists():
        shutil.copyfile(
            predictions_path,
            gold_path,
        )
        print(
            "Created new gold template:",
            gold_path,
        )
    else:
        print(
            "Existing gold file preserved:",
            gold_path,
        )

    print(
        "Predictions:",
        predictions_path,
    )
    print(
        "All candidates:",
        len(rows),
    )
    print(
        "Relation candidates:",
        sum(
            bool(
                row[
                    "in_relation_calibration"
                ]
            )
            for row in rows
        ),
    )


if __name__ == "__main__":
    main()
