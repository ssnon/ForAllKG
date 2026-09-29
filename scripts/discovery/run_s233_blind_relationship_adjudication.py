from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    PriorArtPacket,
)
from pipeline_core.llm.llm_telemetry import (
    run_instructor_structured_call,
)
from scripts.discovery.run_s226_validation_completion import (
    get_client,
)


AdjudicatedRelationship = Literal[
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
    "LOWER_ORDER_RELATION_PRIOR_ART",
    "DIRECTIONAL_COUNTEREVIDENCE",
    "COMPONENT_ONLY",
    "CONTEXTUAL_CONFLICT",
    "CONFLICTING_PRIOR_ART",
    "UNRELATED",
    "INSUFFICIENT_METADATA",
]


class BlindRelationshipAdjudication(BaseModel):
    relationship: AdjudicatedRelationship
    confidence: float = Field(ge=0.0, le=1.0)

    relation_nucleus_explicitly_supported: bool
    full_claim_relation_explicitly_supported: bool
    lower_order_multivariable_relation_supported: bool
    ordered_direction_materially_challenged: bool

    evidence_spans: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


SYSTEM = """You are a blinded scientific prior-art relationship adjudicator.

You receive ONE atomic scientific claim and ONE literature record. Use ONLY the supplied title and abstract. Do not use outside knowledge. You are not told any previous relationship labels.

Your job is to classify the record's relationship to the claim using this decision procedure.

DECISION PROCEDURE

1. DIRECT_PRIOR_ART
Use only when the title/abstract explicitly states, tests, compares, or demonstrates essentially the same scientific relation as the claim, including defining moderator/conditional/directional structure when that structure is part of the claim.

2. PARTIAL_PRIOR_ART
Use only when the abstract explicitly preserves the claim's RELATION NUCLEUS and establishes a substantial subset of the SAME scientific relation, but not the full formulation.
Shared variables, materials, mechanisms, or thematic proximity are not enough.

3. LOWER_ORDER_RELATION_PRIOR_ART
For a higher-order, moderator, conditional, mediated, or interaction claim, use this when the abstract explicitly establishes a nontrivial multivariable LOWER-ORDER SUBRELATION from the claim while omitting the higher-order moderator/conditional/interaction structure.
This is distinct from PARTIAL_PRIOR_ART:
- PARTIAL preserves the higher-level relation nucleus in incomplete form.
- LOWER_ORDER establishes a scientifically meaningful underlying relation after the higher-order extension is removed.

4. DIRECTIONAL_COUNTEREVIDENCE
Use when the claim has an ordered/directional prediction and the abstract materially challenges that direction in a scientifically relevant neighboring scope, such as an opposite trend, weak/absent correlation, regime dependence, or another determinant dominating.
Do not use this merely because the record fails to confirm the proposed direction.

5. COMPONENT_ONLY
Use when the record establishes relevant ingredients, variables, mechanisms, contexts, materials, separate main effects, or one comparison arm, but does not establish the claim's relation nucleus or a meaningful lower-order multivariable relation.

6. CONFLICTING_PRIOR_ART
Use only for a materially opposing relation/result in sufficiently overlapping scientific scope.

7. CONTEXTUAL_CONFLICT
Use when the record challenges a broader assumption or relation but materially differs in scientific scope.

8. UNRELATED
Use when the record does not materially bear on the claim.

9. INSUFFICIENT_METADATA
Use when the supplied title/abstract is inadequate to decide.

HIGHER-ORDER CONSISTENCY RULE
For a claim of logical form "M changes how X affects Y":
- M affects Y plus X affects Y separately => COMPONENT_ONLY.
- X affects Y explicitly, without M changing that relation => LOWER_ORDER_RELATION_PRIOR_ART.
- M changes/conditions the X-to-Y relation, but some scope/detail is incomplete => PARTIAL_PRIOR_ART.
- M changes/conditions the X-to-Y relation in essentially the claimed scope => DIRECT_PRIOR_ART.

EVIDENCE-SPAN CONTRACT
For every relationship other than UNRELATED or INSUFFICIENT_METADATA, provide 1-3 exact contiguous spans copied verbatim from the supplied abstract that justify the classification.
Do not paraphrase inside evidence_spans.
If the abstract does not contain a span supporting the asserted relationship, choose a weaker classification.

Return only the structured adjudication requested by the caller."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
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


def report(path: str | Path) -> ExternalNoveltyReport:
    p = Path(path).expanduser().resolve()
    return ExternalNoveltyReport.model_validate_json(
        p.read_text(encoding="utf-8")
    )


def packet(path: str | Path) -> PriorArtPacket:
    p = Path(path).expanduser().resolve()
    return PriorArtPacket.model_validate_json(
        p.read_text(encoding="utf-8")
    )


def card_for(
    report_obj: ExternalNoveltyReport,
    hypothesis_id: str,
):
    rows = [
        x
        for x in report_obj.cards
        if x.hypothesis_id == hypothesis_id
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one card for {hypothesis_id}; "
            f"found {len(rows)}"
        )
    return rows[0]


def review_for(card, claim_id: str):
    rows = [
        x
        for x in card.claim_reviews
        if x.claim_id == claim_id
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one review for {claim_id}; "
            f"found {len(rows)}"
        )
    return rows[0]


def work_from_packets(
    work_id: str,
    *packets: PriorArtPacket,
):
    hits = []
    for pkt in packets:
        for work in pkt.works:
            if work.work_id == work_id:
                hits.append(work)

    if not hits:
        raise RuntimeError(
            "work not found in supplied packets: "
            + work_id
        )

    # Prefer the richest abstract-bearing representation.
    hits.sort(
        key=lambda w: (
            bool(w.abstract),
            len(w.abstract or ""),
            len(w.title or ""),
        ),
        reverse=True,
    )
    return hits[0]


def evidence_spans_valid(
    *,
    abstract: str,
    spans: list[str],
    relationship: str,
) -> bool:
    if relationship in {
        "UNRELATED",
        "INSUFFICIENT_METADATA",
    }:
        return True

    if not abstract or not spans:
        return False

    return all(
        bool(span)
        and span in abstract
        for span in spans
    )


def build_packet_maps(
    s227_2: dict[str, Any],
    s229: dict[str, Any],
):
    p227 = {
        (
            str(row["source_case_id"]),
            str(row["scope"]),
        ): row
        for row in s227_2.get("rows", [])
        if row.get("measurement_status")
        == "AVAILABLE"
    }

    p229 = {
        (
            str(row["source_case_id"]),
            str(row["scope"]),
        ): row
        for row in s229.get("rows", [])
        if row.get("measurement_status")
        == "AVAILABLE"
    }

    return p227, p229


def packet_paths_for(
    *,
    before_label: str,
    after_label: str,
    key: tuple[str, str],
    s227_rows: dict,
    s229_rows: dict,
) -> tuple[str, str]:
    if (
        before_label
        == "S227_CONTROL_ORIGINAL_PACKET_REPLAY"
        and after_label
        == "S227_TREATMENT_RESOLVED_PACKET_REPLAY"
    ):
        row = s227_rows[key]
        return (
            row["original_packet_path"],
            row["resolved_packet_path"],
        )

    if (
        before_label
        == "S227_POSTHOC_TREATMENT"
        and after_label
        == "S229_INTEGRATED_PRE_REVIEW"
    ):
        row227 = s227_rows[key]
        row229 = s229_rows[key]
        return (
            row227["resolved_packet_path"],
            row229[
                "integrated_resolved_packet_path"
            ],
        )

    raise RuntimeError(
        "unsupported comparison labels: "
        f"{before_label} -> {after_label}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "S233 blinded adjudication of same-work relationship "
            "relabels identified by S232. No literature retrieval."
        )
    )
    p.add_argument(
        "--s232-summary",
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

    s232_path = args.s232_summary.expanduser().resolve()
    s227_path = args.s227_2_summary.expanduser().resolve()
    s229_path = args.s229_summary.expanduser().resolve()
    out_path = args.output.expanduser().resolve()

    for path in (
        s232_path,
        s227_path,
        s229_path,
    ):
        if not path.is_file():
            raise RuntimeError(
                "missing input: " + str(path)
            )

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"missing API key in {args.api_key_env}"
        )

    s232 = load_json(s232_path)
    s227 = load_json(s227_path)
    s229 = load_json(s229_path)

    rows227, rows229 = build_packet_maps(
        s227,
        s229,
    )

    client = get_client(
        api_key=api_key,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        timeout=args.timeout,
    )

    tasks = []

    for parent in s232.get("rows", []):
        case_id = str(
            parent["source_case_id"]
        )
        scope = str(parent["scope"])
        hypothesis_id = str(
            parent["hypothesis_id"]
        )
        key = (case_id, scope)

        before_report = report(
            parent["before_report_path"]
        )
        after_report = report(
            parent["after_report_path"]
        )

        before_packet_path, after_packet_path = (
            packet_paths_for(
                before_label=parent[
                    "before_label"
                ],
                after_label=parent[
                    "after_label"
                ],
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

        before_card = card_for(
            before_report,
            hypothesis_id,
        )
        after_card = card_for(
            after_report,
            hypothesis_id,
        )

        for claim in parent.get(
            "claim_rows",
            [],
        ):
            if (
                claim.get(
                    "classification"
                )
                !=
                "RELATION_BACKED_LOST_SAME_WORK_RELABELED"
            ):
                continue

            claim_id = str(
                claim["claim_id"]
            )
            before_review = review_for(
                before_card,
                claim_id,
            )
            after_review = review_for(
                after_card,
                claim_id,
            )

            for relabel in claim.get(
                "lost_relation_backed_same_work_relabeled",
                [],
            ):
                work_id = str(
                    relabel["work_id"]
                )
                work = work_from_packets(
                    work_id,
                    before_packet,
                    after_packet,
                )

                tasks.append(
                    {
                        "source_case_id":
                            case_id,
                        "scope": scope,
                        "hypothesis_id":
                            hypothesis_id,
                        "claim_id": claim_id,
                        "claim_text":
                            before_review.claim_text,
                        "work_id": work_id,
                        "title": work.title,
                        "doi": work.doi,
                        "abstract":
                            work.abstract or "",
                        "before_relationship":
                            relabel[
                                "before_relationship"
                            ],
                        "after_relationship":
                            relabel[
                                "after_relationship"
                            ],
                        "before_report_path":
                            parent[
                                "before_report_path"
                            ],
                        "after_report_path":
                            parent[
                                "after_report_path"
                            ],
                    }
                )

    # Deduplicate exact case/scope/claim/work tasks.
    deduped = []
    seen = set()
    for task in tasks:
        key = (
            task["source_case_id"],
            task["scope"],
            task["claim_id"],
            task["work_id"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(task)

    print(
        "=== S233 BLINDED SAME-WORK RELATIONSHIP ADJUDICATION ==="
    )
    print("tasks:", len(deduped))
    print("literature retrieval: false")
    print("previous labels shown to adjudicator: false")
    print("production selection changed: false")

    outputs = []
    verdict_counts = Counter()
    agreement_counts = Counter()

    for index, task in enumerate(
        deduped,
        start=1,
    ):
        abstract = task["abstract"]

        user = "\n".join(
            [
                "ATOMIC CLAIM",
                "============",
                task["claim_text"],
                "",
                "LITERATURE RECORD",
                "=================",
                f"work_id: {task['work_id']}",
                f"title: {task['title']}",
                f"doi: {task['doi']}",
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

        result, event = (
            run_instructor_structured_call(
                client.chat.completions,
                model=args.model,
                response_model=
                    BlindRelationshipAdjudication,
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
                temperature=args.temperature,
                max_retries=args.max_retries,
                telemetry_context={
                    "pipeline":
                        "external_novelty",
                    "stage":
                        "s233_blind_relationship_adjudication",
                    "call_kind":
                        "structured",
                    "claim_id":
                        task["claim_id"],
                    "work_id":
                        task["work_id"],
                },
            )
        )

        if not isinstance(
            result,
            BlindRelationshipAdjudication,
        ):
            result = (
                BlindRelationshipAdjudication
                .model_validate(result)
            )

        valid_spans = evidence_spans_valid(
            abstract=abstract,
            spans=result.evidence_spans,
            relationship=
                result.relationship,
        )

        if not valid_spans:
            adjudication_disposition = (
                "INVALID_EVIDENCE_SPAN"
            )
        elif (
            result.relationship
            == task["before_relationship"]
            and result.relationship
            != task["after_relationship"]
        ):
            adjudication_disposition = (
                "SUPPORTS_BEFORE_LABEL"
            )
        elif (
            result.relationship
            == task["after_relationship"]
            and result.relationship
            != task["before_relationship"]
        ):
            adjudication_disposition = (
                "SUPPORTS_AFTER_LABEL"
            )
        elif (
            result.relationship
            == task["before_relationship"]
            == task["after_relationship"]
        ):
            adjudication_disposition = (
                "LABELS_ALREADY_EQUAL"
            )
        else:
            adjudication_disposition = (
                "SUPPORTS_THIRD_LABEL"
            )

        row = {
            **{
                key: value
                for key, value
                in task.items()
                if key != "abstract"
            },
            "abstract_length":
                len(abstract),
            "adjudication":
                result.model_dump(
                    mode="json"
                ),
            "evidence_spans_valid":
                valid_spans,
            "adjudication_disposition":
                adjudication_disposition,
            "telemetry_event_id":
                getattr(
                    event,
                    "event_id",
                    None,
                ),
            "authority": {
                "validation_only": True,
                "blind_to_previous_labels":
                    True,
                "literature_retrieval_performed":
                    False,
                "novelty_authority_created":
                    False,
                "production_selection_authority":
                    False,
            },
        }
        outputs.append(row)

        verdict_counts[
            result.relationship
        ] += 1
        agreement_counts[
            adjudication_disposition
        ] += 1

        print(
            "\n"
            f"[{index}/{len(deduped)}] "
            f"{task['source_case_id']} | "
            f"{task['claim_id']} | "
            f"{task['work_id']}"
        )
        print(
            " hidden historical labels:",
            task["before_relationship"],
            "/",
            task["after_relationship"],
        )
        print(
            " blind adjudication:",
            result.relationship,
            "|",
            adjudication_disposition,
            "| evidence spans valid=",
            valid_spans,
        )

    summary = {
        "schema_version":
            "blind-same-work-relationship-adjudication-s233-v1",
        "source_s232_summary":
            str(s232_path),
        "task_count": len(outputs),
        "adjudicated_relationship_counts":
            dict(
                sorted(
                    verdict_counts.items()
                )
            ),
        "agreement_counts":
            dict(
                sorted(
                    agreement_counts.items()
                )
            ),
        "rows": outputs,
        "interpretation_policy": {
            "SUPPORTS_BEFORE_LABEL":
                (
                    "Blind adjudication independently agrees "
                    "with the earlier relation-backed label."
                ),
            "SUPPORTS_AFTER_LABEL":
                (
                    "Blind adjudication independently agrees "
                    "with the later non-relation-backed label."
                ),
            "SUPPORTS_THIRD_LABEL":
                (
                    "Blind adjudication selects neither previous label; "
                    "relationship taxonomy/prompt boundary needs review."
                ),
            "INVALID_EVIDENCE_SPAN":
                (
                    "Adjudication cannot be trusted because required "
                    "verbatim evidence was not grounded in the supplied abstract."
                ),
            "previous_labels_exposed_to_adjudicator":
                False,
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

    print("\nS233 adjudication complete")
    print("tasks:", len(outputs))
    print("adjudicated relationships:")
    for key, count in sorted(
        verdict_counts.items()
    ):
        print(" ", count, key)
    print("agreement:")
    for key, count in sorted(
        agreement_counts.items()
    ):
        print(" ", count, key)
    print("literature retrieval: false")
    print(
        "production selection changed: false"
    )
    print("artifact:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
