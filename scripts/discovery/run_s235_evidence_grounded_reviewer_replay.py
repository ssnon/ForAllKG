from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import ExternalNoveltyReport


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def safe_slug(value: str) -> str:
    return "".join(
        ch if ch.isalnum() else "_"
        for ch in value.lower()
    ).strip("_")


def status_map(report: ExternalNoveltyReport) -> dict[str, str]:
    return {card.hypothesis_id: str(card.status) for card in report.cards}


def find_compiled_relationship(
    *,
    report: ExternalNoveltyReport,
    hypothesis_id: str,
    claim_id: str,
    work_id: str,
) -> str:
    cards = [c for c in report.cards if c.hypothesis_id == hypothesis_id]
    if len(cards) != 1:
        return "MISSING_HYPOTHESIS"

    reviews = [r for r in cards[0].claim_reviews if r.claim_id == claim_id]
    if len(reviews) != 1:
        return "MISSING_CLAIM"

    matches = [m for m in reviews[0].matches if m.work_id == work_id]
    if len(matches) != 1:
        return "MISSING_MATCH"

    return str(matches[0].relationship)


def grounding_failure_count(report: ExternalNoveltyReport) -> int:
    count = 0
    for card in report.cards:
        for review in card.claim_reviews:
            if any(
                code in {
                    "invalid_or_missing_evidence_span",
                    "invalid_or_missing_evidence_span_prevents_relation_inference",
                }
                for code in review.reason_codes
            ):
                count += 1
    return count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "S235 production-shaped evidence-grounded reviewer replay "
            "over S229 resolved packets. No literature retrieval."
        )
    )
    p.add_argument("--s229-summary", required=True, type=Path)
    p.add_argument("--s234-summary", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_AGENT_MODEL") or "openai/gpt-5.6-luna",
    )
    p.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1",
    )
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    s229_path = args.s229_summary.expanduser().resolve()
    s234_path = args.s234_summary.expanduser().resolve()
    out_root = args.output_root.expanduser().resolve()

    for path in (s229_path, s234_path):
        if not path.is_file():
            raise RuntimeError("missing input: " + str(path))

    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(
            "S235 output root must be fresh/nonempty=false: " + str(out_root)
        )
    out_root.mkdir(parents=True, exist_ok=True)

    s229 = load_json(s229_path)
    s234 = load_json(s234_path)

    consensus = {
        (
            str(row["source_case_id"]),
            str(row["scope"]),
            str(row["claim_id"]),
            str(row["work_id"]),
        ): row
        for row in s234.get("rows", [])
    }

    rows_in = [
        row
        for row in s229.get("rows", [])
        if row.get("measurement_status") == "AVAILABLE"
    ]

    rows: list[dict[str, Any]] = []
    failures = 0
    consensus_checks: list[dict[str, Any]] = []

    print("=== S235 EVIDENCE-GROUNDED MAIN REVIEWER REPLAY ===")
    print("packets:", len(rows_in))
    print("S229 resolved packets reused: true")
    print("literature retrieval performed: false")
    print("evidence-grounded reviewer enabled: true")
    print("production selection changed: false")

    for index, row in enumerate(rows_in, start=1):
        case_id = str(row["source_case_id"])
        scope = str(row["scope"])

        print("\n" + "=" * 96)
        print(f"[{index}/{len(rows_in)}] {case_id} | {scope}")
        print("=" * 96)

        result: dict[str, Any] = {
            "source_case_id": case_id,
            "scope": scope,
            "measurement_status": "ERROR",
        }

        try:
            portfolio_path = Path(row["portfolio_path"]).expanduser().resolve()
            query_path = Path(row["query_plan_path"]).expanduser().resolve()
            packet_path = Path(
                row["integrated_resolved_packet_path"]
            ).expanduser().resolve()
            baseline_report_path = Path(
                row["integrated_report_path"]
            ).expanduser().resolve()

            for path in (
                portfolio_path,
                query_path,
                packet_path,
                baseline_report_path,
            ):
                if not path.is_file():
                    raise RuntimeError("missing source artifact: " + str(path))

            out_dir = out_root / safe_slug(case_id) / safe_slug(scope)
            out_dir.mkdir(parents=True, exist_ok=False)
            prefix = out_dir / "external_novelty_evidence_grounded"

            command = [
                sys.executable,
                "-m",
                "scripts.discovery.run_external_novelty",
                "--portfolio",
                str(portfolio_path),
                "--domain-profile",
                str(row["domain_profile_id"]),
                "--model",
                args.model,
                "--api-key-env",
                args.api_key_env,
                "--reuse-query-plan",
                str(query_path),
                "--reuse-prior-art",
                str(packet_path),
                "--evidence-grounded-review",
                "--output-prefix",
                str(prefix),
                "--pre-review-coverage-shadow",
            ]
            if args.base_url:
                command.extend(["--base-url", str(args.base_url)])

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

            report_path = prefix.with_name(prefix.name + ".report.json")
            if completed.returncode != 0 or not report_path.is_file():
                raise RuntimeError(
                    "evidence-grounded replay failed; "
                    f"returncode={completed.returncode}; log={log_path}"
                )

            baseline = ExternalNoveltyReport.model_validate_json(
                baseline_report_path.read_text(encoding="utf-8")
            )
            grounded = ExternalNoveltyReport.model_validate_json(
                report_path.read_text(encoding="utf-8")
            )

            before = status_map(baseline)
            after = status_map(grounded)
            hypothesis_ids = sorted(set(before) | set(after))

            status_changes = [
                {
                    "hypothesis_id": hid,
                    "before": before.get(hid),
                    "after": after.get(hid),
                    "changed": before.get(hid) != after.get(hid),
                }
                for hid in hypothesis_ids
            ]

            local_consensus = []
            for key, consensus_row in consensus.items():
                c_case, c_scope, claim_id, work_id = key
                if c_case != case_id or c_scope != scope:
                    continue

                observed = find_compiled_relationship(
                    report=grounded,
                    hypothesis_id=str(consensus_row["hypothesis_id"]),
                    claim_id=claim_id,
                    work_id=work_id,
                )
                expected = str(consensus_row["consensus_relationship"])

                item = {
                    "source_case_id": case_id,
                    "scope": scope,
                    "hypothesis_id": str(consensus_row["hypothesis_id"]),
                    "claim_id": claim_id,
                    "work_id": work_id,
                    "expected_s234_consensus": expected,
                    "s235_compiled_relationship": observed,
                    "matches_consensus": observed == expected,
                }
                local_consensus.append(item)
                consensus_checks.append(item)

            result = {
                "source_case_id": case_id,
                "scope": scope,
                "measurement_status": "AVAILABLE",
                "baseline_report_path": str(baseline_report_path),
                "grounded_report_path": str(report_path),
                "log_path": str(log_path),
                "hypothesis_count": len(hypothesis_ids),
                "changed_hypothesis_count": sum(
                    item["changed"] for item in status_changes
                ),
                "status_changes": status_changes,
                "grounding_failure_claim_count": grounding_failure_count(grounded),
                "s234_consensus_checks": local_consensus,
                "authority": {
                    "validation_only": True,
                    "literature_retrieval_performed": False,
                    "query_plan_changed": False,
                    "prior_art_packet_changed": False,
                    "production_selection_authority": False,
                },
            }
            rows.append(result)

            print(
                "status changes=",
                result["changed_hypothesis_count"],
                "| grounding-failure claims=",
                result["grounding_failure_claim_count"],
                "| S234 checks=",
                sum(x["matches_consensus"] for x in local_consensus),
                "/",
                len(local_consensus),
            )

        except Exception as exc:
            failures += 1
            result["error"] = repr(exc)
            rows.append(result)
            print("ERROR:", repr(exc))
            if not args.continue_on_error:
                raise

    available = [
        row for row in rows if row.get("measurement_status") == "AVAILABLE"
    ]

    consensus_match_count = sum(
        bool(row["matches_consensus"]) for row in consensus_checks
    )

    summary = {
        "schema_version": "evidence-grounded-main-reviewer-replay-s235-v1",
        "source_s229_summary": str(s229_path),
        "source_s234_summary": str(s234_path),
        "packet_count": len(rows_in),
        "completed_packet_count": len(available),
        "failed_packet_count": failures,
        "hypothesis_count": sum(
            int(row.get("hypothesis_count") or 0) for row in available
        ),
        "changed_hypothesis_count": sum(
            int(row.get("changed_hypothesis_count") or 0) for row in available
        ),
        "grounding_failure_claim_count": sum(
            int(row.get("grounding_failure_claim_count") or 0)
            for row in available
        ),
        "s234_consensus_check_count": len(consensus_checks),
        "s234_consensus_match_count": consensus_match_count,
        "s234_consensus_match_fraction": (
            consensus_match_count / len(consensus_checks)
            if consensus_checks
            else 1.0
        ),
        "consensus_checks": consensus_checks,
        "rows": rows,
        "interpretation_policy": {
            "same_resolved_prior_art_packet": True,
            "same_query_plan": True,
            "literature_retrieval_performed": False,
            "evidence_grounded_review_enabled": True,
            "production_selection_changed": False,
        },
    }

    summary_path = out_root / "replay.summary.json"
    write_json(summary_path, summary)

    print("\nS235 replay complete")
    print("completed packets:", len(available))
    print("failed packets:", failures)
    print("hypotheses:", summary["hypothesis_count"])
    print(
        "hypothesis status changes vs S229:",
        summary["changed_hypothesis_count"],
    )
    print(
        "grounding-failure claims:",
        summary["grounding_failure_claim_count"],
    )
    print(
        "S234 consensus agreement:",
        f'{summary["s234_consensus_match_count"]}'
        f'/{summary["s234_consensus_check_count"]}',
        f'({summary["s234_consensus_match_fraction"]:.3f})',
    )
    print("literature retrieval: false")
    print("production selection changed: false")
    print("artifact:", summary_path)

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
