from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    value: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
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


def slug(value: str) -> str:
    return "".join(
        ch
        if ch.isalnum()
        else "_"
        for ch
        in value.lower()
    ).strip("_")


def status_map(
    report: ExternalNoveltyReport,
) -> dict[str, str]:
    return {
        card.hypothesis_id:
            str(card.status)
        for card in report.cards
    }


def relationship_for(
    report: ExternalNoveltyReport,
    *,
    hypothesis_id: str,
    claim_id: str,
    work_id: str,
) -> str:
    for card in report.cards:
        if (
            card.hypothesis_id
            != hypothesis_id
        ):
            continue
        for review in (
            card.claim_reviews
        ):
            if (
                review.claim_id
                != claim_id
            ):
                continue
            for match in (
                review.matches
            ):
                if (
                    match.work_id
                    == work_id
                ):
                    return str(
                        match.relationship
                    )
    return "MISSING_MATCH"


def grounding_failure_count(
    report: ExternalNoveltyReport,
) -> int:
    return sum(
        any(
            code
            == "invalid_or_missing_evidence_span_prevents_relation_inference"
            for code
            in review.reason_codes
        )
        for card in report.cards
        for review in card.claim_reviews
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--s229-summary",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--s234-summary",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--model",
        default=(
            os.getenv(
                "OPENROUTER_AGENT_MODEL"
            )
            or "openai/gpt-5.6-luna"
        ),
    )
    p.add_argument(
        "--base-url",
        default=(
            os.getenv(
                "OPENAI_BASE_URL"
            )
            or "https://openrouter.ai/api/v1"
        ),
    )
    p.add_argument(
        "--api-key-env",
        default=
            "OPENROUTER_API_KEY",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    s229_path = (
        args.s229_summary
        .expanduser()
        .resolve()
    )
    s234_path = (
        args.s234_summary
        .expanduser()
        .resolve()
    )
    out_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    if (
        out_root.exists()
        and any(
            out_root.iterdir()
        )
    ):
        raise RuntimeError(
            "S237 output root must be fresh: "
            + str(out_root)
        )
    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    s229 = load_json(
        s229_path
    )
    s234 = load_json(
        s234_path
    )

    consensus = {
        (
            str(row[
                "source_case_id"
            ]),
            str(row["scope"]),
            str(row["claim_id"]),
            str(row["work_id"]),
        ): row
        for row
        in s234.get(
            "rows",
            [],
        )
    }

    rows_in = [
        row
        for row
        in s229.get(
            "rows",
            [],
        )
        if row.get(
            "measurement_status"
        )
        == "AVAILABLE"
    ]

    rows = []
    checks = []
    failures = 0

    print(
        "=== S237 INDEPENDENT EVIDENCE-GROUNDED REVIEW REPLAY ==="
    )
    print(
        "packets:",
        len(rows_in),
    )
    print(
        "literature retrieval: false"
    )
    print(
        "independent claim-work review: true"
    )
    print(
        "normalized span grounding: true"
    )
    print(
        "production selection changed: false"
    )

    for index, row in enumerate(
        rows_in,
        start=1,
    ):
        case_id = str(
            row[
                "source_case_id"
            ]
        )
        scope = str(
            row["scope"]
        )

        print(
            "\n"
            + "=" * 96
        )
        print(
            f"[{index}/{len(rows_in)}] "
            f"{case_id} | {scope}"
        )
        print(
            "=" * 96
        )

        result = {
            "source_case_id":
                case_id,
            "scope":
                scope,
            "measurement_status":
                "ERROR",
        }

        try:
            portfolio = Path(
                row[
                    "portfolio_path"
                ]
            ).expanduser().resolve()
            plan = Path(
                row[
                    "query_plan_path"
                ]
            ).expanduser().resolve()
            packet = Path(
                row[
                    "integrated_resolved_packet_path"
                ]
            ).expanduser().resolve()
            baseline_path = Path(
                row[
                    "integrated_report_path"
                ]
            ).expanduser().resolve()

            out_dir = (
                out_root
                / slug(case_id)
                / slug(scope)
            )
            out_dir.mkdir(
                parents=True,
                exist_ok=False,
            )
            prefix = (
                out_dir
                / "external_novelty_independent_grounded"
            )

            command = [
                sys.executable,
                "-m",
                "scripts.discovery.run_external_novelty",
                "--portfolio",
                str(portfolio),
                "--domain-profile",
                str(
                    row[
                        "domain_profile_id"
                    ]
                ),
                "--model",
                args.model,
                "--api-key-env",
                args.api_key_env,
                "--reuse-query-plan",
                str(plan),
                "--reuse-prior-art",
                str(packet),
                "--independent-evidence-review",
                "--output-prefix",
                str(prefix),
                "--pre-review-coverage-shadow",
            ]
            if args.base_url:
                command.extend(
                    [
                        "--base-url",
                        args.base_url,
                    ]
                )

            completed = (
                subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    check=False,
                )
            )

            log_path = (
                out_dir
                / "subprocess.log.txt"
            )
            log_path.write_text(
                "COMMAND\n=======\n"
                + " ".join(command)
                + "\n\nSTDOUT\n======\n"
                + completed.stdout
                + "\n\nSTDERR\n======\n"
                + completed.stderr,
                encoding="utf-8",
            )

            report_path = (
                prefix.with_name(
                    prefix.name
                    + ".report.json"
                )
            )

            if (
                completed.returncode
                != 0
                or not report_path.is_file()
            ):
                raise RuntimeError(
                    "S237 replay failed; "
                    f"log={log_path}"
                )

            baseline = (
                ExternalNoveltyReport
                .model_validate_json(
                    baseline_path
                    .read_text(
                        encoding="utf-8"
                    )
                )
            )
            report = (
                ExternalNoveltyReport
                .model_validate_json(
                    report_path
                    .read_text(
                        encoding="utf-8"
                    )
                )
            )

            before = status_map(
                baseline
            )
            after = status_map(
                report
            )
            ids = sorted(
                set(before)
                | set(after)
            )

            changes = [
                {
                    "hypothesis_id":
                        hid,
                    "before":
                        before.get(hid),
                    "after":
                        after.get(hid),
                    "changed":
                        before.get(hid)
                        != after.get(hid),
                }
                for hid in ids
            ]

            local_checks = []

            for key, expected_row in (
                consensus.items()
            ):
                (
                    c_case,
                    c_scope,
                    claim_id,
                    work_id,
                ) = key

                if (
                    c_case
                    != case_id
                    or c_scope
                    != scope
                ):
                    continue

                observed = (
                    relationship_for(
                        report,
                        hypothesis_id=
                            str(
                                expected_row[
                                    "hypothesis_id"
                                ]
                            ),
                        claim_id=
                            claim_id,
                        work_id=
                            work_id,
                    )
                )
                expected = str(
                    expected_row[
                        "consensus_relationship"
                    ]
                )

                check = {
                    "source_case_id":
                        case_id,
                    "scope":
                        scope,
                    "hypothesis_id":
                        str(
                            expected_row[
                                "hypothesis_id"
                            ]
                        ),
                    "claim_id":
                        claim_id,
                    "work_id":
                        work_id,
                    "expected":
                        expected,
                    "observed":
                        observed,
                    "matches":
                        observed
                        == expected,
                }

                local_checks.append(
                    check
                )
                checks.append(
                    check
                )

            result = {
                "source_case_id":
                    case_id,
                "scope":
                    scope,
                "measurement_status":
                    "AVAILABLE",
                "report_path":
                    str(report_path),
                "log_path":
                    str(log_path),
                "hypothesis_count":
                    len(ids),
                "changed_hypothesis_count":
                    sum(
                        row[
                            "changed"
                        ]
                        for row
                        in changes
                    ),
                "status_changes":
                    changes,
                "grounding_failure_claim_count":
                    grounding_failure_count(
                        report
                    ),
                "s234_checks":
                    local_checks,
            }
            rows.append(
                result
            )

            print(
                "status changes=",
                result[
                    "changed_hypothesis_count"
                ],
                "| grounding failures=",
                result[
                    "grounding_failure_claim_count"
                ],
                "| S234=",
                sum(
                    x["matches"]
                    for x
                    in local_checks
                ),
                "/",
                len(
                    local_checks
                ),
            )

        except Exception as exc:
            failures += 1
            result[
                "error"
            ] = repr(exc)
            rows.append(
                result
            )
            print(
                "ERROR:",
                repr(exc),
            )
            if not (
                args
                .continue_on_error
            ):
                raise

    available = [
        row
        for row in rows
        if row.get(
            "measurement_status"
        )
        == "AVAILABLE"
    ]

    match_count = sum(
        x["matches"]
        for x in checks
    )

    summary = {
        "schema_version":
            "independent-evidence-grounded-review-s237-v1",
        "source_s229_summary":
            str(s229_path),
        "source_s234_summary":
            str(s234_path),
        "completed_packet_count":
            len(available),
        "failed_packet_count":
            failures,
        "hypothesis_count":
            sum(
                int(
                    row.get(
                        "hypothesis_count"
                    )
                    or 0
                )
                for row
                in available
            ),
        "changed_hypothesis_count":
            sum(
                int(
                    row.get(
                        "changed_hypothesis_count"
                    )
                    or 0
                )
                for row
                in available
            ),
        "grounding_failure_claim_count":
            sum(
                int(
                    row.get(
                        "grounding_failure_claim_count"
                    )
                    or 0
                )
                for row
                in available
            ),
        "s234_check_count":
            len(checks),
        "s234_match_count":
            match_count,
        "s234_match_fraction":
            (
                match_count
                / len(checks)
                if checks
                else 1.0
            ),
        "checks":
            checks,
        "rows":
            rows,
        "authority": {
            "validation_only":
                True,
            "literature_retrieval_performed":
                False,
            "query_plan_changed":
                False,
            "prior_art_packet_changed":
                False,
            "production_selection_authority":
                False,
        },
    }

    summary_path = (
        out_root
        / "replay.summary.json"
    )
    write_json(
        summary_path,
        summary,
    )

    print(
        "\nS237 replay complete"
    )
    print(
        "completed packets:",
        len(available),
    )
    print(
        "failed packets:",
        failures,
    )
    print(
        "hypothesis status changes vs S229:",
        summary[
            "changed_hypothesis_count"
        ],
    )
    print(
        "grounding-failure claims:",
        summary[
            "grounding_failure_claim_count"
        ],
    )
    print(
        "S234 consensus agreement:",
        f"{match_count}/{len(checks)}",
        (
            f"({summary['s234_match_fraction']:.3f})"
            if checks
            else ""
        ),
    )
    print(
        "literature retrieval: false"
    )
    print(
        "production selection changed: false"
    )
    print(
        "artifact:",
        summary_path,
    )

    return (
        0
        if failures == 0
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
