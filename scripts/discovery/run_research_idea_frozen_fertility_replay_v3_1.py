from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_core.discovery.research_idea_frozen_fertility_replay import (
    compare_frozen_fertility_audits,
    replay_frozen_fertility,
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=/path/to/cohort.json")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected NAME=/path/to/cohort.json")
    return name.strip(), Path(path).expanduser()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Replay SIS-v3.1 fertility decisions from frozen lifecycle/parallel artifacts. "
            "No LLM, web, evidence acquisition, or KG mutation is executed."
        )
    )
    p.add_argument(
        "--cohort",
        action="append",
        type=_named_path,
        required=True,
        help="Named v3.0/v3.1 cohort JSON as NAME=/path/file.json; repeatable.",
    )
    p.add_argument("--output", type=Path, required=True)
    return p


def _case_audit(snapshot_name: str, case_label: str, case: dict[str, Any]):
    lifecycle_path = Path(case["lifecycle"]).expanduser()
    parallel_path = Path(case["parallel_partial_search"]).expanduser()
    lifecycle = _read(lifecycle_path)
    parallel = _read(parallel_path)
    generation = int(case.get("current_generation_index") or lifecycle["generation_index"])

    recorded = case.get("fertility_summary")
    fertility_path = case.get("adaptive_fertility")
    if recorded is None and fertility_path:
        p = Path(fertility_path).expanduser()
        if p.exists():
            recorded = _read(p)

    return replay_frozen_fertility(
        snapshot_label=f"{snapshot_name}:{case_label}:G{generation}",
        parallel_report=parallel,
        lifecycle=lifecycle,
        cycle_generation_index=generation,
        recorded_fertility_report=recorded,
    )


def main() -> int:
    args = parser().parse_args()
    snapshots: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    ordered: list[tuple[str, str, Any]] = []

    for snapshot_name, path in args.cohort:
        cohort = _read(path)
        snapshots[snapshot_name] = {
            "path": str(path),
            "schema_version": cohort.get("schema_version"),
            "case_count": len(cohort.get("cases", {})),
        }
        for case_label, case in sorted(cohort.get("cases", {}).items()):
            audit = _case_audit(snapshot_name, case_label, case)
            key = f"{snapshot_name}:{case_label}:G{audit.cycle_generation_index}"
            audits[key] = audit.model_dump(mode="json")
            ordered.append((snapshot_name, case_label, audit))

            print("=" * 88)
            print(key)
            print("policy input sha256:", audit.policy_input_sha256)
            print("replay dispositions:", audit.replay_disposition_counts)
            print("fertile / persistent-nonfertile:", audit.replay_fertility_report.fertile_idea_count, "/", audit.replay_fertility_report.persistent_nonfertile_idea_count)
            print("repeat exact match:", audit.replay_repeat_exact_match)
            if audit.recorded_fertility_report_id is not None:
                print("recorded exact match:", audit.recorded_report_exact_match)
                print("recorded decision mismatches:", audit.recorded_decision_mismatch_count)
            else:
                print("recorded fertility report: NONE (policy replay only)")
            print("LLM_CALLS=0")
            print("EXTERNAL_SEARCH_EXECUTED=False")
            print("EVIDENCE_ACQUISITION_EXECUTED=False")
            print("CANONICAL_GRAPH_MUTATED=False")

    comparisons: list[dict[str, Any]] = []
    by_case_generation: dict[tuple[str, int], list[tuple[str, Any]]] = {}
    for snapshot_name, case_label, audit in ordered:
        by_case_generation.setdefault(
            (case_label, audit.cycle_generation_index), []
        ).append((snapshot_name, audit))
    for (case_label, generation), rows in sorted(by_case_generation.items()):
        if len(rows) < 2:
            continue
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                left_name, left = rows[i]
                right_name, right = rows[j]
                comparison = compare_frozen_fertility_audits(left, right)
                comparison.update({"case_label": case_label, "generation_index": generation})
                comparisons.append(comparison)

    exact_recorded = [
        row
        for _, _, row in ordered
        if row.recorded_fertility_report_id is not None
    ]
    payload = {
        "schema_version": "sis-v3-1-frozen-fertility-replay-cohort-v1",
        "snapshot_count": len(snapshots),
        "audit_count": len(audits),
        "snapshots": snapshots,
        "audits": audits,
        "comparisons": comparisons,
        "recorded_audit_count": len(exact_recorded),
        "recorded_exact_match_count": sum(
            row.recorded_report_exact_match is True for row in exact_recorded
        ),
        "recorded_mismatch_count": sum(
            row.recorded_report_exact_match is False for row in exact_recorded
        ),
        "all_recorded_reports_reproduced_exactly": bool(exact_recorded)
        and all(row.recorded_report_exact_match is True for row in exact_recorded),
        "llm_call_count": 0,
        "external_search_executed": False,
        "evidence_acquisition_executed": False,
        "canonical_graph_mutated": False,
        "policy_only_replay": True,
    }
    _write(args.output.expanduser(), payload)
    print()
    print("SIS-v3.1 Frozen Fertility Replay complete")
    print("audits:", len(audits))
    print("recorded exact / mismatch:", payload["recorded_exact_match_count"], "/", payload["recorded_mismatch_count"])
    print("LLM_CALLS=0")
    print("output:", args.output.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
