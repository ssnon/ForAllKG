from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dac_her.corpus_acquisition.contracts import (
    CandidateAssessment,
    CorpusSelectionReport,
    SelectedCorpusWork,
)
from dac_her.corpus_acquisition.corpus_quality import (
    apply_quality_gate_and_reselect,
)
from dac_her.corpus_acquisition.profile import (
    load_acquisition_profile,
)
from dac_her.corpus_acquisition.progress import (
    compact_text,
    progress_prefix,
)
from dac_her.corpus_acquisition.quality_policy import (
    load_corpus_quality_policy,
)
from dac_her.literature_catalog_contracts import (
    LiteratureCatalogPacket,
)


def _read_jsonl(path: Path, model) -> list[Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(model.model_validate_json(line))
    return rows


def _write_json(path: Path, value: Any) -> None:
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


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump(mode="json")
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
            "M2.1: apply a deterministic corpus-quality gate to the entire "
            "M2 candidate pool, then re-run quota-aware selection so dropped "
            "papers can be backfilled without weakening quality semantics."
        )
    )
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--quality-policy", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--assessments", required=True, type=Path)
    parser.add_argument("--selected-works", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quality-gate-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = load_acquisition_profile(args.profile)
    policy = load_corpus_quality_policy(args.quality_policy)
    packet = LiteratureCatalogPacket.model_validate_json(
        args.catalog.read_text(encoding="utf-8")
    )
    upstream_assessments = _read_jsonl(
        args.assessments,
        CandidateAssessment,
    )
    upstream_selected = _read_jsonl(
        args.selected_works,
        SelectedCorpusWork,
    )
    upstream_report = CorpusSelectionReport.model_validate_json(
        args.selection_report.read_text(encoding="utf-8")
    )

    if packet.acquisition_profile_id != profile.profile_id:
        raise ValueError("Catalog/profile mismatch")
    if upstream_report.profile_id != profile.profile_id:
        raise ValueError("M2 report/profile mismatch")
    if upstream_report.source_catalog_id != packet.catalog_id:
        raise ValueError("M2 report/catalog mismatch")
    if upstream_report.selected_work_ids != [
        row.work_id for row in upstream_selected
    ]:
        raise ValueError(
            "M2 selected_works.jsonl does not match selection_report.json"
        )

    def progress(event: dict[str, Any]) -> None:
        stage = event.get("stage")
        if stage == "m2_1_quality":
            print(
                progress_prefix(
                    "M2.1 quality",
                    int(event["current"]),
                    int(event["total"]),
                ),
                f"{str(event['status']):<13}",
                (
                    "selected"
                    if event.get("originally_selected")
                    else "pool"
                ),
                compact_text(
                    str(event["title"]),
                    max_length=65,
                ),
                flush=True,
            )
        elif stage == "m2_1_select":
            axis = event.get("primary_quota_axis") or "-"
            print(
                progress_prefix(
                    "M2.1 select",
                    int(event["current"]),
                    int(event["total"]),
                ),
                f"phase={event.get('phase')}",
                f"axis={axis}",
                flush=True,
            )

    (
        quality_rows,
        selected,
        final_selection_report,
        quality_report,
    ) = apply_quality_gate_and_reselect(
        packet=packet,
        profile=profile,
        upstream_assessments=upstream_assessments,
        upstream_selected=upstream_selected,
        policy=policy,
        quality_gate_id=args.quality_gate_id,
        progress_callback=progress,
    )

    out = args.output_dir
    _write_jsonl(
        out / "quality_assessments.jsonl",
        quality_rows,
    )
    _write_jsonl(
        out / "selected_works.jsonl",
        selected,
    )
    _write_json(
        out / "selection_report.json",
        final_selection_report,
    )
    _write_json(
        out / "quality_gate_report.json",
        quality_report,
    )

    print()
    print("Generic corpus acquisition M2.1 complete")
    print("Policy:", policy.policy_id)
    print(
        "Quality:",
        f"pass={quality_report.quality_pass_count}",
        f"manual_review={quality_report.quality_manual_review_count}",
        f"exclude={quality_report.quality_exclude_count}",
    )
    print(
        "Original selection:",
        quality_report.original_selected_count,
    )
    print(
        "Retained/dropped/replacements:",
        quality_report.retained_original_selected_count,
        "/",
        quality_report.dropped_original_selected_count,
        "/",
        quality_report.replacement_selected_count,
    )
    print(
        "Final selected:",
        quality_report.final_selected_count,
        "/ target",
        quality_report.target_total,
    )
    print(
        "Unfilled axis quotas:",
        quality_report.final_unfilled_axis_quotas,
    )
    print(
        "Positive-evidence promotion:",
        quality_report.positive_evidence_promotion_performed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
