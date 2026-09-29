from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.llm.llm_telemetry import (
    run_instructor_structured_call,
)
from scripts.discovery.run_s226_validation_completion import (
    get_client,
)
from scripts.discovery.run_s233_blind_relationship_adjudication import (
    BlindRelationshipAdjudication,
    SYSTEM,
    build_packet_maps,
    evidence_spans_valid,
    packet,
    packet_paths_for,
    work_from_packets,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def adjudicate_once(
    *,
    client,
    model: str,
    claim_text: str,
    work_id: str,
    title: str,
    doi: str | None,
    abstract: str,
    temperature: float,
    max_retries: int,
    telemetry_context: dict[str, Any],
) -> tuple[BlindRelationshipAdjudication, Any]:
    user = "\n".join(
        [
            "ATOMIC CLAIM",
            "============",
            claim_text,
            "",
            "LITERATURE RECORD",
            "=================",
            f"work_id: {work_id}",
            f"title: {title}",
            f"doi: {doi}",
            "abstract:",
            abstract
            if abstract
            else "[NO ABSTRACT AVAILABLE]",
            "",
            (
                "Classify only the relationship "
                "between this record and this claim."
            ),
        ]
    )

    result, event = run_instructor_structured_call(
        client.chat.completions,
        model=model,
        response_model=BlindRelationshipAdjudication,
        messages=[
            {
                "role": "system",
                "content": SYSTEM,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
        temperature=temperature,
        max_retries=max_retries,
        telemetry_context=telemetry_context,
    )

    if not isinstance(
        result,
        BlindRelationshipAdjudication,
    ):
        result = (
            BlindRelationshipAdjudication
            .model_validate(result)
        )

    return result, event


def stability_class(
    *,
    original_relationship: str,
    replicate_relationships: list[str],
    all_spans_valid: bool,
) -> str:
    if not all_spans_valid:
        return "INVALID_EVIDENCE_SPAN_PRESENT"

    counts = Counter(
        replicate_relationships
    )
    top_relationship, top_count = (
        counts.most_common(1)[0]
    )

    if (
        len(counts) == 1
        and top_relationship
        == original_relationship
    ):
        return "UNANIMOUS_AND_MATCHES_S233"

    if len(counts) == 1:
        return "UNANIMOUS_BUT_DIFFERS_FROM_S233"

    if (
        top_count
        > len(replicate_relationships) / 2
    ):
        if (
            top_relationship
            == original_relationship
        ):
            return "MAJORITY_MATCHES_S233"
        return "MAJORITY_DIFFERS_FROM_S233"

    return "NO_MAJORITY"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "S234 repeated blind relationship adjudication stability "
            "check over the seven S233 claim-work pairs. "
            "No literature retrieval."
        )
    )
    p.add_argument(
        "--s233-summary",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--s227-2-summary",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--s229-summary",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--replicates",
        type=int,
        default=3,
    )
    p.add_argument(
        "--model",
        default=(
            os.getenv("OPENROUTER_AGENT_MODEL")
            or "openai/gpt-5.6-luna"
        ),
    )
    p.add_argument(
        "--base-url",
        default=(
            os.getenv("OPENAI_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ),
    )
    p.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
    )
    p.add_argument(
        "--instructor-mode",
        default="JSON",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=1,
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.replicates < 2:
        raise RuntimeError(
            "--replicates must be >= 2"
        )

    s233_path = (
        args.s233_summary.expanduser().resolve()
    )
    s227_path = (
        args.s227_2_summary.expanduser().resolve()
    )
    s229_path = (
        args.s229_summary.expanduser().resolve()
    )
    out_path = args.output.expanduser().resolve()

    for path in (
        s233_path,
        s227_path,
        s229_path,
    ):
        if not path.is_file():
            raise RuntimeError(
                "missing input: " + str(path)
            )

    if out_path.exists():
        raise RuntimeError(
            "output already exists; refusing overwrite: "
            + str(out_path)
        )

    api_key = os.getenv(
        args.api_key_env
    )
    if not api_key:
        raise RuntimeError(
            f"missing API key in {args.api_key_env}"
        )

    s233 = load_json(
        s233_path
    )
    s227 = load_json(
        s227_path
    )
    s229 = load_json(
        s229_path
    )

    rows227, rows229 = (
        build_packet_maps(
            s227,
            s229,
        )
    )

    client = get_client(
        api_key=api_key,
        base_url=args.base_url,
        instructor_mode=
            args.instructor_mode,
        timeout=args.timeout,
    )

    outputs: list[
        dict[str, Any]
    ] = []
    stability_counts: Counter[str] = (
        Counter()
    )
    consensus_counts: Counter[str] = (
        Counter()
    )

    print(
        "=== S234 REPEATED BLIND RELATIONSHIP STABILITY ==="
    )
    print(
        "tasks:",
        len(s233.get("rows", [])),
    )
    print(
        "new replicates per task:",
        args.replicates,
    )
    print(
        "literature retrieval: false"
    )
    print(
        "previous labels shown to adjudicator: false"
    )
    print(
        "production selection changed: false"
    )

    for index, source in enumerate(
        s233.get("rows", []),
        start=1,
    ):
        case_id = str(
            source["source_case_id"]
        )
        scope = str(
            source["scope"]
        )
        claim_id = str(
            source["claim_id"]
        )
        work_id = str(
            source["work_id"]
        )
        claim_text = str(
            source["claim_text"]
        )

        key = (
            case_id,
            scope,
        )

        before_packet_path, after_packet_path = (
            packet_paths_for(
                before_label=(
                    "S227_CONTROL_ORIGINAL_PACKET_REPLAY"
                    if (
                        "control"
                        in str(
                            source[
                                "before_report_path"
                            ]
                        ).lower()
                    )
                    else
                    "S227_POSTHOC_TREATMENT"
                ),
                after_label=(
                    "S227_TREATMENT_RESOLVED_PACKET_REPLAY"
                    if (
                        "control"
                        in str(
                            source[
                                "before_report_path"
                            ]
                        ).lower()
                    )
                    else
                    "S229_INTEGRATED_PRE_REVIEW"
                ),
                key=key,
                s227_rows=rows227,
                s229_rows=rows229,
            )
        )

        before_packet = packet(
            before_packet_path
        )
        after_packet = packet(
            after_packet_path
        )

        work = work_from_packets(
            work_id,
            before_packet,
            after_packet,
        )
        abstract = work.abstract or ""

        replicate_rows = []
        relationships = []
        span_validities = []

        print(
            "\n"
            f"[{index}/{len(s233.get('rows', []))}] "
            f"{case_id} | {claim_id} | {work_id}"
        )
        print(
            " S233 adjudication:",
            source["adjudication"][
                "relationship"
            ],
        )

        for replicate_index in range(
            1,
            args.replicates + 1,
        ):
            result, event = (
                adjudicate_once(
                    client=client,
                    model=args.model,
                    claim_text=claim_text,
                    work_id=work_id,
                    title=work.title,
                    doi=work.doi,
                    abstract=abstract,
                    temperature=
                        args.temperature,
                    max_retries=
                        args.max_retries,
                    telemetry_context={
                        "pipeline":
                            "external_novelty",
                        "stage":
                            "s234_repeat_blind_relationship_adjudication",
                        "claim_id":
                            claim_id,
                        "work_id":
                            work_id,
                        "replicate_index":
                            replicate_index,
                    },
                )
            )

            valid = evidence_spans_valid(
                abstract=abstract,
                spans=
                    result.evidence_spans,
                relationship=
                    result.relationship,
            )

            relationships.append(
                result.relationship
            )
            span_validities.append(
                valid
            )
            replicate_rows.append(
                {
                    "replicate_index":
                        replicate_index,
                    "adjudication":
                        result.model_dump(
                            mode="json"
                        ),
                    "evidence_spans_valid":
                        valid,
                    "telemetry_event_id":
                        getattr(
                            event,
                            "event_id",
                            None,
                        ),
                }
            )

            print(
                "  replicate",
                replicate_index,
                ":",
                result.relationship,
                "| span-valid=",
                valid,
            )

        counts = Counter(
            relationships
        )
        consensus_relationship, consensus_count = (
            counts.most_common(1)[0]
        )

        stability = stability_class(
            original_relationship=(
                source[
                    "adjudication"
                ]["relationship"]
            ),
            replicate_relationships=
                relationships,
            all_spans_valid=all(
                span_validities
            ),
        )

        stability_counts[
            stability
        ] += 1
        consensus_counts[
            consensus_relationship
        ] += 1

        row = {
            "source_case_id":
                case_id,
            "scope": scope,
            "hypothesis_id":
                source[
                    "hypothesis_id"
                ],
            "claim_id": claim_id,
            "work_id": work_id,
            "claim_text":
                claim_text,
            "title": work.title,
            "doi": work.doi,
            "abstract_length":
                len(abstract),
            "historical_before_relationship":
                source[
                    "before_relationship"
                ],
            "historical_after_relationship":
                source[
                    "after_relationship"
                ],
            "s233_relationship":
                source[
                    "adjudication"
                ]["relationship"],
            "replicates":
                replicate_rows,
            "replicate_relationship_counts":
                dict(
                    sorted(
                        counts.items()
                    )
                ),
            "consensus_relationship":
                consensus_relationship,
            "consensus_count":
                consensus_count,
            "replicate_count":
                args.replicates,
            "all_evidence_spans_valid":
                all(
                    span_validities
                ),
            "stability_class":
                stability,
            "authority": {
                "validation_only":
                    True,
                "literature_retrieval_performed":
                    False,
                "previous_labels_exposed_to_adjudicator":
                    False,
                "novelty_authority_created":
                    False,
                "production_selection_authority":
                    False,
            },
        }
        outputs.append(
            row
        )

        print(
            " stability:",
            stability,
            "| consensus=",
            consensus_relationship,
            f"({consensus_count}/{args.replicates})",
        )

    unanimous_count = sum(
        row[
            "consensus_count"
        ]
        == row[
            "replicate_count"
        ]
        for row in outputs
    )

    s233_consensus_agreement_count = sum(
        row[
            "consensus_relationship"
        ]
        == row[
            "s233_relationship"
        ]
        for row in outputs
    )

    summary = {
        "schema_version":
            "repeated-blind-relationship-stability-s234-v1",
        "source_s233_summary":
            str(s233_path),
        "task_count":
            len(outputs),
        "new_replicates_per_task":
            args.replicates,
        "unanimous_replicate_task_count":
            unanimous_count,
        "unanimous_replicate_fraction": (
            unanimous_count
            / len(outputs)
            if outputs
            else 1.0
        ),
        "s233_consensus_agreement_count":
            s233_consensus_agreement_count,
        "s233_consensus_agreement_fraction":
            (
                s233_consensus_agreement_count
                / len(outputs)
                if outputs
                else 1.0
            ),
        "stability_class_counts":
            dict(
                sorted(
                    stability_counts.items()
                )
            ),
        "consensus_relationship_counts":
            dict(
                sorted(
                    consensus_counts.items()
                )
            ),
        "rows":
            outputs,
        "interpretation_policy": {
            "UNANIMOUS_AND_MATCHES_S233":
                (
                    "All new replicates agree with each other "
                    "and with the original S233 blind adjudication."
                ),
            "UNANIMOUS_BUT_DIFFERS_FROM_S233":
                (
                    "New replicates are internally stable but disagree "
                    "with the original S233 adjudication."
                ),
            "MAJORITY_MATCHES_S233":
                (
                    "The decision tree has a majority label matching S233 "
                    "but is not fully deterministic."
                ),
            "MAJORITY_DIFFERS_FROM_S233":
                (
                    "The decision tree has a majority label different from S233."
                ),
            "NO_MAJORITY":
                (
                    "The decision tree does not produce a stable label."
                ),
            "literature_retrieval_performed":
                False,
            "production_selection_changed":
                False,
        },
    }

    write_json(
        out_path,
        summary,
    )

    print(
        "\nS234 stability check complete"
    )
    print(
        "tasks:",
        len(outputs),
    )
    print(
        "unanimous replicate tasks:",
        f"{unanimous_count}/{len(outputs)}",
        (
            f"({summary['unanimous_replicate_fraction']:.3f})"
            if outputs
            else ""
        ),
    )
    print(
        "S233/replicate-consensus agreement:",
        f"{s233_consensus_agreement_count}/{len(outputs)}",
        (
            f"({summary['s233_consensus_agreement_fraction']:.3f})"
            if outputs
            else ""
        ),
    )
    print(
        "stability classes:"
    )
    for key, count in sorted(
        stability_counts.items()
    ):
        print(
            " ",
            count,
            key,
        )
    print(
        "consensus relationships:"
    )
    for key, count in sorted(
        consensus_counts.items()
    ):
        print(
            " ",
            count,
            key,
        )
    print(
        "literature retrieval: false"
    )
    print(
        "production selection changed: false"
    )
    print(
        "artifact:",
        out_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
