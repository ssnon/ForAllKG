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
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.prospective_novelty_validation_completion_s226 import (
    audit_external_report_payload,
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


def source_prefix_from_prior_art(path: Path) -> Path:
    name = path.name
    suffix = ".prior_art.json"
    if not name.endswith(suffix):
        raise RuntimeError(
            "expected .prior_art.json packet: " + str(path)
        )
    return path.with_name(name[: -len(suffix)])


def nearest_run_root(path: Path) -> Path:
    current = path.parent
    for candidate in [current, *current.parents]:
        if (candidate / "e2e_runner.manifest.json").is_file():
            return candidate
    raise RuntimeError(
        "could not find e2e_runner.manifest.json above "
        + str(path)
    )


def portfolio_candidates(run_root: Path, packet_path: Path) -> list[Path]:
    candidates: list[Path] = []

    local = packet_path.parent / "portfolio.json"
    if local.is_file():
        candidates.append(local)

    for pattern in (
        "*portfolio*.json",
        "direct_higher_order_shadow/downstream/arm_*/portfolio.json",
    ):
        for path in run_root.glob(pattern):
            if path.is_file() and path not in candidates:
                candidates.append(path)

    if not candidates:
        for path in run_root.rglob("*.json"):
            if "prior_art" in path.name or "report" in path.name:
                continue
            candidates.append(path)

    return candidates


def find_portfolio(
    *,
    run_root: Path,
    packet_path: Path,
    expected_portfolio_id: str,
) -> Path:
    matches: list[Path] = []

    for path in portfolio_candidates(run_root, packet_path):
        try:
            portfolio = HypothesisPortfolio.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        if portfolio.portfolio_id == expected_portfolio_id:
            matches.append(path.resolve())

    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(
            "expected exactly one portfolio matching "
            f"{expected_portfolio_id!r}; found {len(unique)}: "
            + ", ".join(str(x) for x in unique)
        )
    return unique[0]


def status_counts(report: ExternalNoveltyReport) -> dict[str, int]:
    return dict(
        Counter(str(card.status) for card in report.cards)
    )


def status_map(report: ExternalNoveltyReport) -> dict[str, str]:
    return {
        card.hypothesis_id: str(card.status)
        for card in report.cards
    }


def compare_reports(
    *,
    original: ExternalNoveltyReport,
    rerun: ExternalNoveltyReport,
    original_packet: PriorArtPacket,
    resolved_packet: PriorArtPacket,
) -> dict[str, Any]:
    before_map = status_map(original)
    after_map = status_map(rerun)

    all_ids = sorted(set(before_map) | set(after_map))
    transitions = [
        {
            "hypothesis_id": hypothesis_id,
            "before": before_map.get(hypothesis_id),
            "after": after_map.get(hypothesis_id),
            "changed": (
                before_map.get(hypothesis_id)
                != after_map.get(hypothesis_id)
            ),
        }
        for hypothesis_id in all_ids
    ]

    before_audit = audit_external_report_payload(
        original.model_dump(mode="json"),
        packet=original_packet.model_dump(mode="json"),
    )
    after_audit = audit_external_report_payload(
        rerun.model_dump(mode="json"),
        packet=resolved_packet.model_dump(mode="json"),
    )

    return {
        "original_status_counts": status_counts(original),
        "rerun_status_counts": status_counts(rerun),
        "status_transitions": transitions,
        "changed_hypothesis_count": sum(
            row["changed"] for row in transitions
        ),
        "original_resolution_disposition":
            before_audit["disposition"],
        "rerun_resolution_disposition":
            after_audit["disposition"],
        "original_material_core_title_only_count":
            before_audit["material_core_title_only_count"],
        "rerun_material_core_title_only_count":
            after_audit["material_core_title_only_count"],
        "material_core_title_only_reduction": (
            before_audit["material_core_title_only_count"]
            - after_audit["material_core_title_only_count"]
        ),
        "original_packet_abstract_count":
            before_audit["packet_abstract_count"],
        "resolved_packet_abstract_count":
            after_audit["packet_abstract_count"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S227.2 external-novelty metadata-resolution A/B review. "
            "Reuses the exact frozen query plan and the S227-resolved packet; "
            "no literature retrieval is performed."
        )
    )
    parser.add_argument(
        "--s226-matrix",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--s227-summary",
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

    s226_path = args.s226_matrix.expanduser().resolve()
    s227_path = args.s227_summary.expanduser().resolve()
    out_root = args.output_root.expanduser().resolve()

    for path in (s226_path, s227_path):
        if not path.is_file():
            raise RuntimeError("missing input: " + str(path))

    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(
            "S227.2 output root must be fresh/nonempty=false: "
            + str(out_root)
        )
    out_root.mkdir(parents=True, exist_ok=True)

    s226 = load_json(s226_path)
    s227 = load_json(s227_path)

    case_domains = {
        str(row.get("source_case_id")):
            str(row.get("domain_profile_id"))
        for row in s226.get("cases", [])
    }

    rows: list[dict[str, Any]] = []
    failures = 0

    available = [
        row
        for row in s227.get("rows", [])
        if row.get("measurement_status") == "AVAILABLE"
    ]

    print("=== S227.2 EXTERNAL NOVELTY RESOLUTION A/B ===")
    print("packets:", len(available))
    print("query plans reused exactly: true")
    print("literature retrieval performed: false")
    print("resolved metadata packets reused: true")
    print("production selection changed: false")

    for index, row in enumerate(available, start=1):
        case_id = str(row["source_case_id"])
        scope = str(row["scope"])
        domain_profile = case_domains.get(case_id)

        print("\n" + "=" * 96)
        print(f"[{index}/{len(available)}] {case_id} | {scope}")
        print("=" * 96)

        result: dict[str, Any] = {
            "source_case_id": case_id,
            "scope": scope,
            "measurement_status": "ERROR",
        }

        try:
            if not domain_profile:
                raise RuntimeError(
                    "missing domain profile for case " + case_id
                )

            source_packet_path = Path(
                row["source_packet_path"]
            ).expanduser().resolve()
            resolved_packet_path = Path(
                row["resolved_packet_path"]
            ).expanduser().resolve()

            prefix = source_prefix_from_prior_art(
                source_packet_path
            )
            query_path = prefix.with_name(
                prefix.name + ".claims_queries.json"
            )
            report_path = prefix.with_name(
                prefix.name + ".report.json"
            )

            for path in (
                source_packet_path,
                resolved_packet_path,
                query_path,
                report_path,
            ):
                if not path.is_file():
                    raise RuntimeError(
                        "missing required A/B source artifact: "
                        + str(path)
                    )

            plan = LiteratureQueryPlan.model_validate_json(
                query_path.read_text(encoding="utf-8")
            )
            original_packet = PriorArtPacket.model_validate_json(
                source_packet_path.read_text(encoding="utf-8")
            )
            resolved_packet = PriorArtPacket.model_validate_json(
                resolved_packet_path.read_text(encoding="utf-8")
            )
            original_report = (
                ExternalNoveltyReport.model_validate_json(
                    report_path.read_text(encoding="utf-8")
                )
            )

            if (
                original_packet.source_query_plan_id
                != plan.plan_id
            ):
                raise RuntimeError(
                    "source packet/query plan mismatch"
                )
            if (
                resolved_packet.source_query_plan_id
                != plan.plan_id
            ):
                raise RuntimeError(
                    "resolved packet/query plan mismatch"
                )

            run_root = nearest_run_root(source_packet_path)
            portfolio_path = find_portfolio(
                run_root=run_root,
                packet_path=source_packet_path,
                expected_portfolio_id=plan.source_portfolio_id,
            )

            safe_case = "".join(
                ch if ch.isalnum() else "_"
                for ch in case_id.lower()
            )
            safe_scope = "".join(
                ch if ch.isalnum() else "_"
                for ch in scope.lower()
            )
            out_dir = out_root / safe_case / safe_scope
            out_dir.mkdir(parents=True, exist_ok=False)
            out_prefix = out_dir / "external_novelty_resolved"

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
                str(resolved_packet_path),
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

            rerun_report_path = out_prefix.with_name(
                out_prefix.name + ".report.json"
            )

            if (
                completed.returncode != 0
                or not rerun_report_path.is_file()
            ):
                raise RuntimeError(
                    "resolved external-novelty rerun failed; "
                    f"returncode={completed.returncode}; "
                    f"log={log_path}"
                )

            rerun_report = (
                ExternalNoveltyReport.model_validate_json(
                    rerun_report_path.read_text(
                        encoding="utf-8"
                    )
                )
            )

            comparison = compare_reports(
                original=original_report,
                rerun=rerun_report,
                original_packet=original_packet,
                resolved_packet=resolved_packet,
            )

            result = {
                "source_case_id": case_id,
                "scope": scope,
                "measurement_status": "AVAILABLE",
                "domain_profile_id": domain_profile,
                "portfolio_path": str(portfolio_path),
                "query_plan_path": str(query_path),
                "original_packet_path":
                    str(source_packet_path),
                "resolved_packet_path":
                    str(resolved_packet_path),
                "original_report_path":
                    str(report_path),
                "rerun_report_path":
                    str(rerun_report_path),
                "log_path": str(log_path),
                **comparison,
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
                "status changes=",
                result["changed_hypothesis_count"],
                "| title-only=",
                f'{result["original_material_core_title_only_count"]}'
                "->"
                f'{result["rerun_material_core_title_only_count"]}',
                "| resolution=",
                f'{result["original_resolution_disposition"]}'
                "->"
                f'{result["rerun_resolution_disposition"]}',
            )
            for transition in result["status_transitions"]:
                if transition["changed"]:
                    print(
                        " ",
                        transition["hypothesis_id"],
                        transition["before"],
                        "->",
                        transition["after"],
                    )

        except Exception as exc:
            failures += 1
            result["error"] = repr(exc)
            print("ERROR:", repr(exc))
            if not args.continue_on_error:
                rows.append(result)
                raise

        rows.append(result)

    available_results = [
        row
        for row in rows
        if row.get("measurement_status") == "AVAILABLE"
    ]

    transition_counter: Counter[str] = Counter()
    for row in available_results:
        for transition in row.get("status_transitions", []):
            key = (
                str(transition.get("before"))
                + " -> "
                + str(transition.get("after"))
            )
            transition_counter[key] += 1

    summary = {
        "schema_version":
            "external-novelty-resolution-ab-s227-2-v1",
        "source_s226_matrix": str(s226_path),
        "source_s227_summary": str(s227_path),
        "packet_count": len(available),
        "completed_packet_count": len(available_results),
        "failed_packet_count": failures,
        "changed_hypothesis_count": sum(
            int(row.get("changed_hypothesis_count") or 0)
            for row in available_results
        ),
        "material_core_title_only_before": sum(
            int(
                row.get(
                    "original_material_core_title_only_count"
                )
                or 0
            )
            for row in available_results
        ),
        "material_core_title_only_after": sum(
            int(
                row.get(
                    "rerun_material_core_title_only_count"
                )
                or 0
            )
            for row in available_results
        ),
        "status_transition_counts":
            dict(sorted(transition_counter.items())),
        "rows": rows,
        "interpretation_policy": {
            "same_portfolio_required": True,
            "same_query_plan_required": True,
            "resolved_packet_only_change": True,
            "new_literature_retrieval_performed": False,
            "production_selection_changed": False,
            "overall_winner_computed": False,
        },
    }

    summary_path = out_root / "ab.summary.json"
    write_json(summary_path, summary)

    print("\nS227.2 A/B complete")
    print("completed packets:", summary["completed_packet_count"])
    print("failed packets:", failures)
    print(
        "changed hypotheses:",
        summary["changed_hypothesis_count"],
    )
    print(
        "material core TITLE_ONLY:",
        summary["material_core_title_only_before"],
        "->",
        summary["material_core_title_only_after"],
    )
    print("status transitions:")
    for key, count in summary["status_transition_counts"].items():
        print(" ", count, key)
    print("new literature retrieval performed: false")
    print("production selection changed: false")
    print("artifact:", summary_path)

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
