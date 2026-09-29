from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    PriorArtPacket,
)
from pipeline_core.discovery.prior_art_metadata_resolution_s227 import (
    PriorArtMetadataResolver,
)
from pipeline_core.discovery.prior_art_provider_plan import (
    build_literature_providers,
    require_standard_or_full_auto_plan,
    resolve_literature_provider_plan,
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
    out = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S227 post-hoc prior-art metadata resolution validation. "
            "Resolves only material core TITLE_ONLY work IDs already frozen "
            "by S226; original packets and reports are never modified."
        )
    )
    parser.add_argument(
        "--s226-matrix",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--providers",
        default="auto",
    )
    parser.add_argument(
        "--lookup-limit",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--disable-title-fallback",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_path = args.s226_matrix.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not matrix_path.is_file():
        raise RuntimeError(
            "missing S226 matrix: " + str(matrix_path)
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            "S227 output root must be fresh/nonempty=false: "
            + str(output_root)
        )
    output_root.mkdir(parents=True, exist_ok=True)

    matrix = load_json(matrix_path)

    requested = None
    if str(args.providers).strip().lower() != "auto":
        requested = [
            row.strip()
            for row in str(args.providers).split(",")
            if row.strip()
        ]

    provider_plan = resolve_literature_provider_plan(
        requested=requested,
    )
    if requested is None:
        require_standard_or_full_auto_plan(
            provider_plan
        )

    providers = build_literature_providers(
        provider_plan
    )
    resolver = PriorArtMetadataResolver(
        providers,
        lookup_limit=args.lookup_limit,
        enable_title_fallback=(
            not args.disable_title_fallback
        ),
    )

    rows: list[dict[str, Any]] = []
    seen_packet_keys: set[tuple[str, str]] = set()

    print("=== S227 PRIOR-ART METADATA RESOLUTION VALIDATION ===")
    print("source matrix:", matrix_path)
    print("provider mode:", provider_plan.mode)
    print(
        "scientific query coverage changed: false"
    )
    print(
        "production novelty authority changed: false"
    )

    for case in matrix.get("cases", []):
        case_id = str(
            case.get("source_case_id") or "UNKNOWN"
        )
        external = (
            case.get("external_retrieval_completion")
            or {}
        )

        print("\n" + "=" * 96)
        print(case_id)
        print("=" * 96)

        case_rows = 0

        for index, report in enumerate(
            external.get("reports", []),
            start=1,
        ):
            packet_raw = report.get("prior_art_path")
            if not packet_raw:
                continue

            packet_path = Path(
                packet_raw
            ).expanduser().resolve()

            if not packet_path.is_file():
                rows.append(
                    {
                        "source_case_id": case_id,
                        "scope": report.get("scope"),
                        "source_packet_path":
                            str(packet_path),
                        "measurement_status":
                            "SOURCE_PACKET_MISSING",
                    }
                )
                continue

            targets = {
                str(row.get("work_id") or "").strip()
                for row in report.get(
                    "material_core_title_only_matches",
                    [],
                )
                if str(row.get("work_id") or "").strip()
            }

            if not targets:
                rows.append(
                    {
                        "source_case_id": case_id,
                        "scope": report.get("scope"),
                        "source_packet_path":
                            str(packet_path),
                        "measurement_status":
                            "NOT_APPLICABLE_NO_MATERIAL_TITLE_ONLY",
                        "target_work_count": 0,
                    }
                )
                continue

            dedup_key = (
                str(packet_path),
                "|".join(sorted(targets)),
            )
            if dedup_key in seen_packet_keys:
                continue
            seen_packet_keys.add(dedup_key)

            packet = PriorArtPacket.model_validate_json(
                packet_path.read_text(encoding="utf-8")
            )

            resolved, audit = resolver.resolve(
                packet,
                target_work_ids=targets,
            )

            scope = str(
                report.get("scope")
                or f"report_{index}"
            )
            stem = (
                f"{safe_slug(case_id)}__"
                f"{safe_slug(scope)}"
            )
            packet_out = (
                output_root
                / "resolved_packets"
                / f"{stem}.prior_art.json"
            )
            audit_out = (
                output_root
                / "audits"
                / f"{stem}.resolution.json"
            )

            write_json(packet_out, resolved)
            write_json(audit_out, audit)

            row = {
                "source_case_id": case_id,
                "scope": scope,
                "source_packet_path":
                    str(packet_path),
                "resolved_packet_path":
                    str(packet_out),
                "audit_path":
                    str(audit_out),
                "measurement_status": "AVAILABLE",
                "target_work_count":
                    audit[
                        "requested_target_work_count"
                    ],
                "target_abstract_recovered_count":
                    audit[
                        "target_abstract_recovered_count"
                    ],
                "target_supplementary_doi_count":
                    audit[
                        "target_supplementary_doi_count"
                    ],
                "source_abstract_work_count":
                    audit[
                        "source_abstract_work_count"
                    ],
                "resolved_abstract_work_count":
                    audit[
                        "resolved_abstract_work_count"
                    ],
                "net_abstract_gain":
                    audit["net_abstract_gain"],
                "provider_failure_count":
                    audit["provider_failure_count"],
                "authority": audit["authority"],
            }
            rows.append(row)
            case_rows += 1

            print(
                scope,
                "| targets=",
                row["target_work_count"],
                "| recovered=",
                row[
                    "target_abstract_recovered_count"
                ],
                "| supplementary=",
                row[
                    "target_supplementary_doi_count"
                ],
                "| abstracts=",
                f'{row["source_abstract_work_count"]}'
                f'->{row["resolved_abstract_work_count"]}',
                "| provider_failures=",
                row["provider_failure_count"],
            )

        if case_rows == 0:
            print("no resolvable external packet in this case")

    available = [
        row
        for row in rows
        if row.get("measurement_status") == "AVAILABLE"
    ]

    summary = {
        "schema_version":
            "prior-art-metadata-resolution-validation-s227-v1",
        "source_s226_matrix": str(matrix_path),
        "provider_plan":
            provider_plan.model_dump(mode="json"),
        "packet_measurement_count":
            len(available),
        "target_work_count": sum(
            int(row.get("target_work_count") or 0)
            for row in available
        ),
        "target_abstract_recovered_count": sum(
            int(
                row.get(
                    "target_abstract_recovered_count"
                )
                or 0
            )
            for row in available
        ),
        "target_supplementary_doi_count": sum(
            int(
                row.get(
                    "target_supplementary_doi_count"
                )
                or 0
            )
            for row in available
        ),
        "net_abstract_gain": sum(
            int(row.get("net_abstract_gain") or 0)
            for row in available
        ),
        "provider_failure_count": sum(
            int(
                row.get("provider_failure_count")
                or 0
            )
            for row in available
        ),
        "rows": rows,
        "interpretation_policy": {
            "target_selection_basis":
                "S226 material core TITLE_ONLY_NEIGHBOR only",
            "scientific_query_coverage_changed": False,
            "original_packet_mutated": False,
            "original_external_report_mutated": False,
            "retrieval_recall_claimed_complete": False,
            "novelty_authority_created": False,
            "production_selection_changed": False,
        },
    }

    summary_path = (
        output_root / "resolution.summary.json"
    )
    write_json(summary_path, summary)

    print("\nS227 resolution validation complete")
    print(
        "packets:",
        summary["packet_measurement_count"],
    )
    print(
        "target works:",
        summary["target_work_count"],
    )
    print(
        "target abstracts recovered:",
        summary[
            "target_abstract_recovered_count"
        ],
    )
    print(
        "supplementary DOI targets:",
        summary[
            "target_supplementary_doi_count"
        ],
    )
    print(
        "net packet abstract gain:",
        summary["net_abstract_gain"],
    )
    print(
        "provider failures:",
        summary["provider_failure_count"],
    )
    print("original packets mutated: false")
    print("production selection changed: false")
    print("artifact:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
