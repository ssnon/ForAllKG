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
    LiteratureQueryPlan,
    PriorArtPacket,
)
from scripts.discovery.run_s227_2_external_novelty_resolution_ab import (
    find_portfolio,
    nearest_run_root,
    status_map,
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


def safe_slug(value: str) -> str:
    return "".join(
        ch if ch.isalnum() else "_"
        for ch in value.lower()
    ).strip("_")


def material_title_only_count(
    report: ExternalNoveltyReport,
) -> int:
    threshold = float(
        report.policy.min_match_confidence
    )
    count = 0
    for card in report.cards:
        for review in card.claim_reviews:
            if str(review.importance) != "core":
                continue
            for match in review.matches:
                if (
                    str(match.relationship)
                    == "TITLE_ONLY_NEIGHBOR"
                    and float(match.confidence) >= threshold
                ):
                    count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S229 production-shaped confirmation of integrated "
            "pre-review metadata resolution over the frozen cohort."
        )
    )
    parser.add_argument(
        "--s227-2-summary",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--model",
        default=(
            os.getenv("OPENROUTER_AGENT_MODEL")
            or "openai/gpt-5.6-luna"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=(
            os.getenv("OPENAI_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
    )
    parser.add_argument(
        "--lookup-limit",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = (
        args.s227_2_summary.expanduser().resolve()
    )
    out_root = args.output_root.expanduser().resolve()

    if not source_path.is_file():
        raise RuntimeError(
            "missing S227.2 summary: " + str(source_path)
        )
    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(
            "S229 output root must be fresh/nonempty=false: "
            + str(out_root)
        )
    out_root.mkdir(parents=True, exist_ok=True)

    source = load_json(source_path)
    rows_in = [
        row
        for row in source.get("rows", [])
        if row.get("measurement_status") == "AVAILABLE"
    ]

    rows: list[dict[str, Any]] = []
    failures = 0

    print("=== S229 INTEGRATED PRE-REVIEW RESOLUTION CONFIRMATION ===")
    print("packets:", len(rows_in))
    print("same frozen query plans: true")
    print("same original packets as input: true")
    print("pre-review selector integrated: true")
    print("new scientific literature queries: false")
    print("production selection authority: false")

    for index, row in enumerate(rows_in, start=1):
        case_id = str(row["source_case_id"])
        scope = str(row["scope"])
        domain = str(row["domain_profile_id"])

        print("\n" + "=" * 96)
        print(
            f"[{index}/{len(rows_in)}] "
            f"{case_id} | {scope}"
        )
        print("=" * 96)

        result: dict[str, Any] = {
            "source_case_id": case_id,
            "scope": scope,
            "measurement_status": "ERROR",
        }

        try:
            query_path = Path(
                row["query_plan_path"]
            ).expanduser().resolve()
            packet_path = Path(
                row["original_packet_path"]
            ).expanduser().resolve()
            historical_report_path = Path(
                row["original_report_path"]
            ).expanduser().resolve()
            s227_treatment_report_path = Path(
                row["rerun_report_path"]
            ).expanduser().resolve()

            for path in (
                query_path,
                packet_path,
                historical_report_path,
                s227_treatment_report_path,
            ):
                if not path.is_file():
                    raise RuntimeError(
                        "missing source artifact: "
                        + str(path)
                    )

            plan = LiteratureQueryPlan.model_validate_json(
                query_path.read_text(encoding="utf-8")
            )
            packet = PriorArtPacket.model_validate_json(
                packet_path.read_text(encoding="utf-8")
            )
            if packet.source_query_plan_id != plan.plan_id:
                raise RuntimeError(
                    "packet/query plan mismatch"
                )

            run_root = nearest_run_root(packet_path)
            portfolio_path = find_portfolio(
                run_root=run_root,
                packet_path=packet_path,
                expected_portfolio_id=
                    plan.source_portfolio_id,
            )

            out_dir = (
                out_root
                / safe_slug(case_id)
                / safe_slug(scope)
            )
            out_dir.mkdir(parents=True, exist_ok=False)
            prefix = out_dir / "external_novelty_integrated"

            command = [
                sys.executable,
                "-m",
                "scripts.discovery.run_external_novelty",
                "--portfolio",
                str(portfolio_path),
                "--domain-profile",
                domain,
                "--model",
                args.model,
                "--api-key-env",
                args.api_key_env,
                "--reuse-query-plan",
                str(query_path),
                "--reuse-prior-art",
                str(packet_path),
                "--pre-review-metadata-resolution",
                "--metadata-resolution-lookup-limit",
                str(args.lookup_limit),
                "--output-prefix",
                str(prefix),
                "--pre-review-coverage-shadow",
            ]
            if args.base_url:
                command.extend(
                    ["--base-url", str(args.base_url)]
                )

            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            log_path = out_dir / "subprocess.log.txt"
            log_path.write_text(
                "COMMAND\n=======\n"
                + " ".join(command)
                + "\n\nSTDOUT\n======\n"
                + completed.stdout
                + "\n\nSTDERR\n======\n"
                + completed.stderr,
                encoding="utf-8",
            )

            integrated_report_path = prefix.with_name(
                prefix.name + ".report.json"
            )
            resolution_audit_path = prefix.with_name(
                prefix.name
                + ".metadata_resolution.json"
            )
            resolved_packet_path = prefix.with_name(
                prefix.name
                + ".prior_art.resolved.json"
            )

            for path in (
                integrated_report_path,
                resolution_audit_path,
                resolved_packet_path,
            ):
                if not path.is_file():
                    raise RuntimeError(
                        "integrated output missing: "
                        + str(path)
                        + f"; returncode={completed.returncode}; "
                        + f"log={log_path}"
                    )

            historical = ExternalNoveltyReport.model_validate_json(
                historical_report_path.read_text(
                    encoding="utf-8"
                )
            )
            s227_treatment = (
                ExternalNoveltyReport.model_validate_json(
                    s227_treatment_report_path.read_text(
                        encoding="utf-8"
                    )
                )
            )
            integrated = ExternalNoveltyReport.model_validate_json(
                integrated_report_path.read_text(
                    encoding="utf-8"
                )
            )
            resolution = load_json(
                resolution_audit_path
            )

            hmap = status_map(historical)
            tmap = status_map(s227_treatment)
            imap = status_map(integrated)
            hypothesis_ids = sorted(
                set(hmap) | set(tmap) | set(imap)
            )

            transitions = [
                {
                    "hypothesis_id": hid,
                    "historical": hmap.get(hid),
                    "s227_posthoc_treatment":
                        tmap.get(hid),
                    "integrated_pre_review":
                        imap.get(hid),
                    "integrated_matches_s227_treatment":
                        imap.get(hid) == tmap.get(hid),
                }
                for hid in hypothesis_ids
            ]

            result = {
                "source_case_id": case_id,
                "scope": scope,
                "measurement_status": "AVAILABLE",
                "domain_profile_id": domain,
                "portfolio_path": str(portfolio_path),
                "query_plan_path": str(query_path),
                "original_packet_path": str(packet_path),
                "integrated_resolved_packet_path":
                    str(resolved_packet_path),
                "resolution_audit_path":
                    str(resolution_audit_path),
                "historical_report_path":
                    str(historical_report_path),
                "s227_treatment_report_path":
                    str(s227_treatment_report_path),
                "integrated_report_path":
                    str(integrated_report_path),
                "log_path": str(log_path),
                "selector_target_count":
                    resolution.get(
                        "selector",
                        {},
                    ).get("selected_work_count", 0),
                "abstract_recovered_count":
                    resolution.get(
                        "target_abstract_recovered_count",
                        0,
                    ),
                "historical_title_only_count":
                    material_title_only_count(historical),
                "s227_treatment_title_only_count":
                    material_title_only_count(
                        s227_treatment
                    ),
                "integrated_title_only_count":
                    material_title_only_count(integrated),
                "transitions": transitions,
                "integrated_matches_s227_treatment_count":
                    sum(
                        item[
                            "integrated_matches_s227_treatment"
                        ]
                        for item in transitions
                    ),
                "hypothesis_count":
                    len(transitions),
                "authority": {
                    "production_shaped_validation_only": True,
                    "query_plan_changed": False,
                    "scientific_query_coverage_changed": False,
                    "review_outcome_used_for_resolution_targeting":
                        False,
                    "novelty_selection_authority": False,
                    "production_selection_authority": False,
                },
            }
            rows.append(result)

            print(
                "targets=",
                result["selector_target_count"],
                "| abstracts recovered=",
                result["abstract_recovered_count"],
                "| title-only hist/posthoc/integrated=",
                result["historical_title_only_count"],
                "/",
                result["s227_treatment_title_only_count"],
                "/",
                result["integrated_title_only_count"],
                "| status agreement with posthoc=",
                f'{result["integrated_matches_s227_treatment_count"]}'
                f'/{result["hypothesis_count"]}',
            )

        except Exception as exc:
            failures += 1
            result["error"] = repr(exc)
            rows.append(result)
            print("ERROR:", repr(exc))
            if not args.continue_on_error:
                raise

    available = [
        row
        for row in rows
        if row.get("measurement_status") == "AVAILABLE"
    ]

    status_agree = sum(
        int(
            row.get(
                "integrated_matches_s227_treatment_count"
            )
            or 0
        )
        for row in available
    )
    hypothesis_total = sum(
        int(row.get("hypothesis_count") or 0)
        for row in available
    )

    summary = {
        "schema_version":
            "integrated-pre-review-resolution-confirmation-s229-v1",
        "source_s227_2_summary": str(source_path),
        "packet_count": len(rows_in),
        "completed_packet_count": len(available),
        "failed_packet_count": failures,
        "selector_target_count": sum(
            int(row.get("selector_target_count") or 0)
            for row in available
        ),
        "abstract_recovered_count": sum(
            int(
                row.get("abstract_recovered_count")
                or 0
            )
            for row in available
        ),
        "historical_title_only_count": sum(
            int(
                row.get("historical_title_only_count")
                or 0
            )
            for row in available
        ),
        "s227_treatment_title_only_count": sum(
            int(
                row.get(
                    "s227_treatment_title_only_count"
                )
                or 0
            )
            for row in available
        ),
        "integrated_title_only_count": sum(
            int(
                row.get("integrated_title_only_count")
                or 0
            )
            for row in available
        ),
        "status_agreement_with_s227_treatment_count":
            status_agree,
        "hypothesis_count": hypothesis_total,
        "status_agreement_fraction": (
            status_agree / hypothesis_total
            if hypothesis_total
            else 1.0
        ),
        "rows": rows,
        "interpretation_policy": {
            "same_original_packet_input": True,
            "same_query_plan": True,
            "pre_review_targeting_only": True,
            "review_outcome_used_for_targeting": False,
            "new_scientific_query_plan_created": False,
            "novelty_selection_authority": False,
            "production_selection_changed": False,
        },
    }

    summary_path = out_root / "confirmation.summary.json"
    write_json(summary_path, summary)

    print("\nS229 confirmation complete")
    print("completed packets:", len(available))
    print("failed packets:", failures)
    print(
        "selector targets:",
        summary["selector_target_count"],
    )
    print(
        "abstracts recovered:",
        summary["abstract_recovered_count"],
    )
    print(
        "material core TITLE_ONLY historical/posthoc/integrated:",
        summary["historical_title_only_count"],
        "/",
        summary["s227_treatment_title_only_count"],
        "/",
        summary["integrated_title_only_count"],
    )
    print(
        "status agreement with S227 treatment:",
        f'{summary["status_agreement_with_s227_treatment_count"]}'
        f'/{summary["hypothesis_count"]}',
        f'({summary["status_agreement_fraction"]:.3f})',
    )
    print("production selection changed: false")
    print("artifact:", summary_path)

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
