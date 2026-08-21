from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_core.literature.acquisition.candidate_selection import assess_catalog, select_candidates
from pipeline_core.literature.acquisition.profile import load_acquisition_profile
from pipeline_core.literature.acquisition.progress import compact_text, progress_prefix
from pipeline_core.literature.catalog_contracts import (
    LiteratureCatalogPacket,
)


def _write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
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


def _write_jsonl(
    path: Path,
    rows: list[Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump(
                    mode="json"
                )
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M2: deterministically score and quota-select metadata "
            "candidates from an M1 literature catalog."
        )
    )
    parser.add_argument(
        "--profile",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--catalog",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = load_acquisition_profile(
        args.profile
    )
    packet = (
        LiteratureCatalogPacket
        .model_validate_json(
            args.catalog.read_text(
                encoding="utf-8"
            )
        )
    )

    def _assess_progress(event: dict[str, Any]) -> None:
        if event.get("stage") != "m2_assess":
            return
        print(
            progress_prefix(
                "M2 assess",
                int(event["current"]),
                int(event["total"]),
            ),
            f"{str(event['eligibility_status']):<13}",
            f"score={float(event['total_score']):>6.2f}",
            compact_text(str(event["title"]), max_length=72),
            flush=True,
        )

    def _select_progress(event: dict[str, Any]) -> None:
        if event.get("stage") != "m2_select":
            return
        axis = event.get("primary_quota_axis") or "-"
        print(
            progress_prefix(
                "M2 select",
                int(event["current"]),
                int(event["total"]),
            ),
            f"phase={event.get('phase')}",
            f"axis={axis}",
            f"work={event.get('work_id')}",
            flush=True,
        )

    assessments = assess_catalog(
        packet,
        profile,
        progress_callback=_assess_progress,
    )
    selected, report = select_candidates(
        packet=packet,
        profile=profile,
        assessments=assessments,
        progress_callback=_select_progress,
    )

    output = args.output_dir
    _write_jsonl(
        output / "assessments.jsonl",
        assessments,
    )
    _write_jsonl(
        output / "selected_works.jsonl",
        selected,
    )
    _write_json(
        output / "selection_report.json",
        report,
    )

    print("Generic corpus acquisition M2 complete")
    print("Profile:", profile.profile_id)
    print("Candidates:", report.candidate_count)
    print("Eligible:", report.eligible_count)
    print(
        "Manual review:",
        report.manual_review_count,
    )
    print("Excluded:", report.excluded_count)
    print(
        "Selected:",
        report.selected_count,
        "/ target",
        report.target_total,
    )
    print(
        "Unfilled axis quotas:",
        report.unfilled_axis_quotas,
    )
    print(
        "Positive-evidence promotion:",
        report.positive_evidence_promotion_performed,
    )
    print(
        "Selection report:",
        output / "selection_report.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
