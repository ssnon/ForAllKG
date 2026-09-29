from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
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
    source_prefix_from_prior_art,
    status_map,
)


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


def safe_slug(value: str) -> str:
    return "".join(
        ch if ch.isalnum() else "_"
        for ch in value.lower()
    ).strip("_")


def material_title_only_count(report: ExternalNoveltyReport) -> int:
    count = 0
    min_conf = float(
        report.policy.min_match_confidence
    )
    for card in report.cards:
        for review in card.claim_reviews:
            if str(review.importance) != "core":
                continue
            for match in review.matches:
                if (
                    str(match.relationship) == "TITLE_ONLY_NEIGHBOR"
                    and float(match.confidence) >= min_conf
                ):
                    count += 1
    return count


def classify_triplet(
    historical: str | None,
    control: str | None,
    treatment: str | None,
) -> str:
    if historical == control == treatment:
        return "FULLY_STABLE"
    if historical == control and treatment != control:
        return "CONTROL_STABLE_TREATMENT_CHANGED"
    if historical != control and treatment == control:
        return "CONTROL_DRIFT_SHARED_BY_TREATMENT"
    if historical != control and treatment == historical:
        return "CONTROL_DRIFT_TREATMENT_RETURNS_HISTORICAL"
    return "CONTROL_AND_TREATMENT_DIVERGE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S227.3 control replay for external-novelty reviewer stability. "
            "Re-runs the original packet with the exact original query plan, "
            "then compares historical/original, control replay, and S227.2 "
            "resolved-packet treatment."
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
        "--continue-on-error",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.s227_2_summary.expanduser().resolve()
    out_root = args.output_root.expanduser().resolve()

    if not summary_path.is_file():
        raise RuntimeError(
            "missing S227.2 summary: " + str(summary_path)
        )
    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(
            "S227.3 output root must be fresh/nonempty=false: "
            + str(out_root)
        )
    out_root.mkdir(parents=True, exist_ok=True)

    source = load_json(summary_path)
    input_rows = [
        row
        for row in source.get("rows", [])
        if row.get("measurement_status") == "AVAILABLE"
    ]

    rows: list[dict[str, Any]] = []
    failures = 0

    print("=== S227.3 EXTERNAL NOVELTY CONTROL REPLAY ===")
    print("packets:", len(input_rows))
    print("historical packet: original")
    print("control packet: original, replayed")
    print("treatment packet: S227 resolved")
    print("query plans reused exactly: true")
    print("new literature retrieval performed: false")
    print("production selection changed: false")

    for index, row in enumerate(input_rows, start=1):
        case_id = str(row["source_case_id"])
        scope = str(row["scope"])
        domain_profile = str(row["domain_profile_id"])

        print("\n" + "=" * 96)
        print(f"[{index}/{len(input_rows)}] {case_id} | {scope}")
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
            original_packet_path = Path(
                row["original_packet_path"]
            ).expanduser().resolve()
            historical_report_path = Path(
                row["original_report_path"]
            ).expanduser().resolve()
            treatment_report_path = Path(
                row["rerun_report_path"]
            ).expanduser().resolve()

            for path in (
                query_path,
                original_packet_path,
                historical_report_path,
                treatment_report_path,
            ):
                if not path.is_file():
                    raise RuntimeError(
                        "missing required artifact: " + str(path)
                    )

            plan = LiteratureQueryPlan.model_validate_json(
                query_path.read_text(encoding="utf-8")
            )
            original_packet = PriorArtPacket.model_validate_json(
                original_packet_path.read_text(encoding="utf-8")
            )

            if original_packet.source_query_plan_id != plan.plan_id:
                raise RuntimeError(
                    "original packet/query-plan mismatch"
                )

            run_root = nearest_run_root(original_packet_path)
            portfolio_path = find_portfolio(
                run_root=run_root,
                packet_path=original_packet_path,
                expected_portfolio_id=plan.source_portfolio_id,
            )

            out_dir = (
                out_root
                / safe_slug(case_id)
                / safe_slug(scope)
            )
            out_dir.mkdir(parents=True, exist_ok=False)
            out_prefix = out_dir / "external_novelty_control"

            command = [
                sys.executable,
                "-m",
                "scripts.discovery.run_external_novelty",
                "--portfolio",
                str(portfolio_path),
                "--domain-profile",
                domain_profile,
                "--model",
                args.model,
                "--api-key-env",
                args.api_key_env,
                "--reuse-query-plan",
                str(query_path),
                "--reuse-prior-art",
                str(original_packet_path),
                "--output-prefix",
                str(out_prefix),
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

            control_report_path = out_prefix.with_name(
                out_prefix.name + ".report.json"
            )
            if (
                completed.returncode != 0
                or not control_report_path.is_file()
            ):
                raise RuntimeError(
                    "control replay failed; "
                    f"returncode={completed.returncode}; "
                    f"log={log_path}"
                )

            historical = ExternalNoveltyReport.model_validate_json(
                historical_report_path.read_text(encoding="utf-8")
            )
            control = ExternalNoveltyReport.model_validate_json(
                control_report_path.read_text(encoding="utf-8")
            )
            treatment = ExternalNoveltyReport.model_validate_json(
                treatment_report_path.read_text(encoding="utf-8")
            )

            historical_map = status_map(historical)
            control_map = status_map(control)
            treatment_map = status_map(treatment)

            hypothesis_ids = sorted(
                set(historical_map)
                | set(control_map)
                | set(treatment_map)
            )

            triplets = []
            for hypothesis_id in hypothesis_ids:
                h = historical_map.get(hypothesis_id)
                c = control_map.get(hypothesis_id)
                t = treatment_map.get(hypothesis_id)
                triplets.append(
                    {
                        "hypothesis_id": hypothesis_id,
                        "historical": h,
                        "control": c,
                        "treatment": t,
                        "classification":
                            classify_triplet(h, c, t),
                    }
                )

            counts = Counter(
                item["classification"]
                for item in triplets
            )

            result = {
                "source_case_id": case_id,
                "scope": scope,
                "measurement_status": "AVAILABLE",
                "domain_profile_id": domain_profile,
                "portfolio_path": str(portfolio_path),
                "query_plan_path": str(query_path),
                "original_packet_path":
                    str(original_packet_path),
                "historical_report_path":
                    str(historical_report_path),
                "control_report_path":
                    str(control_report_path),
                "treatment_report_path":
                    str(treatment_report_path),
                "log_path": str(log_path),
                "triplets": triplets,
                "classification_counts":
                    dict(sorted(counts.items())),
                "historical_title_only_count":
                    material_title_only_count(historical),
                "control_title_only_count":
                    material_title_only_count(control),
                "treatment_title_only_count":
                    material_title_only_count(treatment),
                "authority": {
                    "posthoc_validation_only": True,
                    "query_plan_changed": False,
                    "literature_retrieval_performed": False,
                    "scientific_query_coverage_changed": False,
                    "novelty_authority_created": False,
                    "production_selection_authority": False,
                },
            }

            print(
                "triplets=",
                len(triplets),
                "| stable=",
                counts["FULLY_STABLE"],
                "| control-stable/treatment-changed=",
                counts[
                    "CONTROL_STABLE_TREATMENT_CHANGED"
                ],
                "| control-drift-shared=",
                counts[
                    "CONTROL_DRIFT_SHARED_BY_TREATMENT"
                ],
                "| divergent=",
                counts[
                    "CONTROL_AND_TREATMENT_DIVERGE"
                ],
            )
            print(
                "title-only historical/control/treatment=",
                result["historical_title_only_count"],
                "/",
                result["control_title_only_count"],
                "/",
                result["treatment_title_only_count"],
            )

        except Exception as exc:
            failures += 1
            result["error"] = repr(exc)
            print("ERROR:", repr(exc))
            if not args.continue_on_error:
                rows.append(result)
                raise

        rows.append(result)

    available = [
        row
        for row in rows
        if row.get("measurement_status") == "AVAILABLE"
    ]

    total_counts: Counter[str] = Counter()
    for row in available:
        total_counts.update(
            row.get("classification_counts", {})
        )

    summary = {
        "schema_version":
            "external-novelty-reviewer-control-s227-3-v1",
        "source_s227_2_summary": str(summary_path),
        "packet_count": len(input_rows),
        "completed_packet_count": len(available),
        "failed_packet_count": failures,
        "classification_counts":
            dict(sorted(total_counts.items())),
        "historical_title_only_count": sum(
            int(row["historical_title_only_count"])
            for row in available
        ),
        "control_title_only_count": sum(
            int(row["control_title_only_count"])
            for row in available
        ),
        "treatment_title_only_count": sum(
            int(row["treatment_title_only_count"])
            for row in available
        ),
        "rows": rows,
        "interpretation_policy": {
            "CONTROL_STABLE_TREATMENT_CHANGED":
                "candidate resolution-attributable status change",
            "CONTROL_DRIFT_SHARED_BY_TREATMENT":
                "reviewer replay drift, not attributable to resolution",
            "CONTROL_DRIFT_TREATMENT_RETURNS_HISTORICAL":
                "indeterminate reviewer/treatment interaction",
            "CONTROL_AND_TREATMENT_DIVERGE":
                "indeterminate reviewer/treatment interaction",
            "FULLY_STABLE":
                "status stable across historical, control, treatment",
            "new_literature_retrieval_performed": False,
            "production_selection_changed": False,
        },
    }

    summary_out = out_root / "control.summary.json"
    write_json(summary_out, summary)

    print("\nS227.3 control replay complete")
    print("completed packets:", len(available))
    print("failed packets:", failures)
    print("classification counts:")
    for key, count in summary["classification_counts"].items():
        print(" ", count, key)
    print(
        "material core TITLE_ONLY historical/control/treatment:",
        summary["historical_title_only_count"],
        "/",
        summary["control_title_only_count"],
        "/",
        summary["treatment_title_only_count"],
    )
    print("new literature retrieval performed: false")
    print("production selection changed: false")
    print("artifact:", summary_out)

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
