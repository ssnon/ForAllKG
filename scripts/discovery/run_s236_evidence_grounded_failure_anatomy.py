from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    PriorArtPacket,
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


def load_report(path: str | Path) -> ExternalNoveltyReport:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise RuntimeError("missing report: " + str(p))
    return ExternalNoveltyReport.model_validate_json(
        p.read_text(encoding="utf-8")
    )


def load_packet(path: str | Path) -> PriorArtPacket:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise RuntimeError("missing packet: " + str(p))
    return PriorArtPacket.model_validate_json(
        p.read_text(encoding="utf-8")
    )


def norm_ws(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", str(value or "")).split()
    )


def norm_loose(value: str) -> str:
    text = unicodedata.normalize(
        "NFKC",
        str(value or ""),
    ).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    return text.strip()


def classify_span(
    *,
    span: str,
    abstract: str,
    title: str,
) -> str:
    span = str(span or "")
    abstract = str(abstract or "")
    title = str(title or "")

    if not span:
        return "EMPTY_SPAN"

    if span in abstract:
        return "EXACT_ABSTRACT_MATCH"

    if span in title and span not in abstract:
        return "TITLE_ONLY_MATCH"

    if norm_ws(span) and norm_ws(span) in norm_ws(abstract):
        return "WHITESPACE_OR_UNICODE_NORMALIZATION_ONLY"

    if norm_loose(span) and norm_loose(span) in norm_loose(abstract):
        return "LOOSE_NORMALIZATION_ONLY"

    if span.endswith("…"):
        prefix = span[:-1].rstrip()
        if prefix and prefix in abstract:
            return "TRUNCATION_ELLIPSIS_ONLY"

    if span.endswith("..."):
        prefix = span[:-3].rstrip()
        if prefix and prefix in abstract:
            return "TRUNCATION_ELLIPSIS_ONLY"

    return "UNGROUNDED_OR_PARAPHRASED"


def work_map(packet: PriorArtPacket) -> dict[str, Any]:
    return {
        work.work_id: work
        for work in packet.works
    }


def claim_map(report: ExternalNoveltyReport) -> dict[tuple[str, str], Any]:
    result = {}
    for card in report.cards:
        for review in card.claim_reviews:
            result[
                (
                    card.hypothesis_id,
                    review.claim_id,
                )
            ] = review
    return result


def find_match(
    report: ExternalNoveltyReport,
    *,
    hypothesis_id: str,
    claim_id: str,
    work_id: str,
):
    for card in report.cards:
        if card.hypothesis_id != hypothesis_id:
            continue
        for review in card.claim_reviews:
            if review.claim_id != claim_id:
                continue
            for match in review.matches:
                if match.work_id == work_id:
                    return match
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "S236 artifact-only failure anatomy for S235 evidence-grounded "
            "review: grounding failures, S234 mismatches, and hypothesis "
            "status changes. No LLM or literature calls."
        )
    )
    p.add_argument(
        "--s235-summary",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--s234-summary",
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
    return p.parse_args()


def main() -> int:
    args = parse_args()

    s235_path = args.s235_summary.expanduser().resolve()
    s234_path = args.s234_summary.expanduser().resolve()
    s229_path = args.s229_summary.expanduser().resolve()
    out_path = args.output.expanduser().resolve()

    for path in (
        s235_path,
        s234_path,
        s229_path,
    ):
        if not path.is_file():
            raise RuntimeError("missing input: " + str(path))

    if out_path.exists():
        raise RuntimeError(
            "output already exists; refusing overwrite: "
            + str(out_path)
        )

    s235 = load_json(s235_path)
    s234 = load_json(s234_path)
    s229 = load_json(s229_path)

    p229 = {
        (
            str(row["source_case_id"]),
            str(row["scope"]),
        ): row
        for row in s229.get("rows", [])
        if row.get("measurement_status") == "AVAILABLE"
    }

    grounding_rows: list[dict[str, Any]] = []
    span_class_counts: Counter[str] = Counter()
    failure_claim_counts: Counter[str] = Counter()
    changed_hypotheses: list[dict[str, Any]] = []

    print("=== S236 EVIDENCE-GROUNDED REVIEW FAILURE ANATOMY ===")
    print("artifact only: true")
    print("LLM calls: false")
    print("literature retrieval: false")
    print("production selection changed: false")

    for row in s235.get("rows", []):
        if row.get("measurement_status") != "AVAILABLE":
            continue

        case_id = str(row["source_case_id"])
        scope = str(row["scope"])
        key = (case_id, scope)

        grounded_report = load_report(
            row["grounded_report_path"]
        )
        baseline_report = load_report(
            row["baseline_report_path"]
        )

        packet_path = p229[key][
            "integrated_resolved_packet_path"
        ]
        packet = load_packet(packet_path)
        works = work_map(packet)

        for status_change in row.get(
            "status_changes",
            [],
        ):
            if status_change.get("changed"):
                changed_hypotheses.append(
                    {
                        "source_case_id": case_id,
                        "scope": scope,
                        **status_change,
                    }
                )

        baseline_claims = claim_map(
            baseline_report
        )

        for card in grounded_report.cards:
            for review in card.claim_reviews:
                if not any(
                    code
                    in {
                        "invalid_or_missing_evidence_span",
                        "invalid_or_missing_evidence_span_prevents_relation_inference",
                    }
                    for code in review.reason_codes
                ):
                    continue

                failure_claim_counts[
                    review.status
                ] += 1

                baseline_review = baseline_claims.get(
                    (
                        card.hypothesis_id,
                        review.claim_id,
                    )
                )

                bad_matches = []
                for match in review.matches:
                    if (
                        match.relationship
                        != "INSUFFICIENT_METADATA"
                    ):
                        continue

                    work = works.get(
                        match.work_id
                    )
                    if work is None:
                        bad_matches.append(
                            {
                                "work_id":
                                    match.work_id,
                                "span_classes":
                                    ["WORK_NOT_IN_PACKET"],
                                "evidence_spans":
                                    list(
                                        match.evidence_spans
                                    ),
                            }
                        )
                        span_class_counts[
                            "WORK_NOT_IN_PACKET"
                        ] += 1
                        continue

                    classes = [
                        classify_span(
                            span=span,
                            abstract=work.abstract or "",
                            title=work.title or "",
                        )
                        for span in (
                            match.evidence_spans
                            or [""]
                        )
                    ]

                    for cls in classes:
                        span_class_counts[
                            cls
                        ] += 1

                    bad_matches.append(
                        {
                            "work_id":
                                match.work_id,
                            "title":
                                work.title,
                            "doi":
                                work.doi,
                            "abstract_available":
                                bool(work.abstract),
                            "evidence_spans":
                                list(
                                    match.evidence_spans
                                ),
                            "span_classes":
                                classes,
                        }
                    )

                grounding_rows.append(
                    {
                        "source_case_id":
                            case_id,
                        "scope":
                            scope,
                        "hypothesis_id":
                            card.hypothesis_id,
                        "claim_id":
                            review.claim_id,
                        "claim_text":
                            review.claim_text,
                        "grounded_claim_status":
                            review.status,
                        "baseline_claim_status":
                            (
                                baseline_review.status
                                if baseline_review
                                is not None
                                else None
                            ),
                        "reason_codes":
                            list(
                                review.reason_codes
                            ),
                        "bad_matches":
                            bad_matches,
                    }
                )

    consensus_checks = list(
        s235.get(
            "consensus_checks",
            [],
        )
    )
    mismatches = [
        row
        for row in consensus_checks
        if not row.get(
            "matches_consensus"
        )
    ]

    mismatch_rows = []
    for row in mismatches:
        case_id = str(
            row["source_case_id"]
        )
        scope = str(
            row["scope"]
        )
        hypothesis_id = str(
            row["hypothesis_id"]
        )
        claim_id = str(
            row["claim_id"]
        )
        work_id = str(
            row["work_id"]
        )

        s235_row = next(
            x
            for x in s235["rows"]
            if (
                str(x.get("source_case_id"))
                == case_id
                and str(x.get("scope"))
                == scope
                and x.get(
                    "measurement_status"
                )
                == "AVAILABLE"
            )
        )

        grounded = load_report(
            s235_row[
                "grounded_report_path"
            ]
        )
        baseline = load_report(
            s235_row[
                "baseline_report_path"
            ]
        )

        gmatch = find_match(
            grounded,
            hypothesis_id=
                hypothesis_id,
            claim_id=claim_id,
            work_id=work_id,
        )
        bmatch = find_match(
            baseline,
            hypothesis_id=
                hypothesis_id,
            claim_id=claim_id,
            work_id=work_id,
        )

        packet = load_packet(
            p229[
                (case_id, scope)
            ][
                "integrated_resolved_packet_path"
            ]
        )
        work = work_map(packet).get(
            work_id
        )

        span_classes = []
        if (
            gmatch is not None
            and work is not None
        ):
            span_classes = [
                classify_span(
                    span=span,
                    abstract=
                        work.abstract
                        or "",
                    title=
                        work.title
                        or "",
                )
                for span in (
                    gmatch.evidence_spans
                    or [""]
                )
            ]

        if gmatch is None:
            cause = (
                "MAIN_REVIEWER_OMITTED_MATCH"
            )
        elif (
            gmatch.relationship
            == "INSUFFICIENT_METADATA"
        ):
            cause = (
                "GROUNDING_VALIDATION_DOWNGRADE"
            )
        elif (
            str(
                gmatch.relationship
            )
            != str(
                row[
                    "expected_s234_consensus"
                ]
            )
        ):
            cause = (
                "MULTI_CANDIDATE_RELATIONSHIP_DRIFT"
            )
        else:
            cause = "UNKNOWN"

        mismatch_rows.append(
            {
                **row,
                "baseline_relationship":
                    (
                        str(
                            bmatch.relationship
                        )
                        if bmatch is not None
                        else None
                    ),
                "s235_match_present":
                    gmatch is not None,
                "s235_evidence_spans":
                    (
                        list(
                            gmatch.evidence_spans
                        )
                        if gmatch is not None
                        else []
                    ),
                "s235_span_classes":
                    span_classes,
                "cause":
                    cause,
            }
        )

    mismatch_cause_counts = Counter(
        row["cause"]
        for row in mismatch_rows
    )

    changed_case_counts = Counter(
        (
            row["source_case_id"],
            row["scope"],
        )
        for row in changed_hypotheses
    )

    summary = {
        "schema_version":
            "evidence-grounded-review-failure-anatomy-s236-v1",
        "source_s235_summary":
            str(s235_path),
        "source_s234_summary":
            str(s234_path),
        "source_s229_summary":
            str(s229_path),
        "grounding_failure_claim_count":
            len(grounding_rows),
        "grounding_failure_claim_status_counts":
            dict(
                sorted(
                    failure_claim_counts.items()
                )
            ),
        "span_failure_class_counts":
            dict(
                sorted(
                    span_class_counts.items()
                )
            ),
        "s234_consensus_mismatch_count":
            len(mismatch_rows),
        "s234_consensus_mismatch_cause_counts":
            dict(
                sorted(
                    mismatch_cause_counts.items()
                )
            ),
        "changed_hypothesis_count":
            len(changed_hypotheses),
        "changed_hypothesis_case_counts":
            {
                f"{case}|{scope}":
                    count
                for (
                    case,
                    scope,
                ), count
                in sorted(
                    changed_case_counts.items()
                )
            },
        "grounding_failures":
            grounding_rows,
        "consensus_mismatches":
            mismatch_rows,
        "changed_hypotheses":
            changed_hypotheses,
        "interpretation_policy": {
            "EMPTY_SPAN":
                "Reviewer returned a substantive relationship without any evidence span.",
            "TITLE_ONLY_MATCH":
                "Reviewer copied title text rather than abstract evidence.",
            "WHITESPACE_OR_UNICODE_NORMALIZATION_ONLY":
                "Evidence is contiguous after conservative Unicode/whitespace normalization.",
            "LOOSE_NORMALIZATION_ONLY":
                "Evidence is recoverable only after case/dash/whitespace normalization.",
            "TRUNCATION_ELLIPSIS_ONLY":
                "Evidence appears copied from the truncated prompt representation.",
            "UNGROUNDED_OR_PARAPHRASED":
                "Evidence span is not recoverable from the supplied abstract by conservative normalization.",
            "MULTI_CANDIDATE_RELATIONSHIP_DRIFT":
                "The match is grounded but the main multi-candidate reviewer assigns a different relationship from the stable S234 single-work consensus.",
            "GROUNDING_VALIDATION_DOWNGRADE":
                "The S235 deterministic span validator downgraded the relationship.",
            "MAIN_REVIEWER_OMITTED_MATCH":
                "The main reviewer did not return the S234 work at all.",
            "llm_calls":
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

    print("\nS236 audit complete")
    print(
        "grounding-failure claims:",
        len(grounding_rows),
    )
    print("span failure classes:")
    for key, count in sorted(
        span_class_counts.items()
    ):
        print(" ", count, key)

    print(
        "S234 consensus mismatches:",
        len(mismatch_rows),
    )
    print("mismatch causes:")
    for key, count in sorted(
        mismatch_cause_counts.items()
    ):
        print(" ", count, key)

    for row in mismatch_rows:
        print(
            " MISMATCH",
            row[
                "source_case_id"
            ],
            "|",
            row[
                "claim_id"
            ],
            "| expected=",
            row[
                "expected_s234_consensus"
            ],
            "| observed=",
            row[
                "s235_compiled_relationship"
            ],
            "| cause=",
            row["cause"],
        )

    print(
        "changed hypotheses:",
        len(changed_hypotheses),
    )
    print("LLM calls: false")
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
