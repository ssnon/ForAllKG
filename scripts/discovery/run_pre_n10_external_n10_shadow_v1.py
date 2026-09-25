from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from pipeline_core.discovery.pre_n10_downstream_handoff_v1 import (
    PreN10DownstreamHandoffReportV1,
)
from pipeline_core.discovery.pre_n10_external_n10_shadow_v1 import (
    PreN10ExternalN10LineageResultV1,
    PreN10ExternalN10StagePlanV1,
    build_pre_n10_external_n10_lineage_result_v1,
    build_pre_n10_external_n10_shadow_report_v1,
    compile_pre_n10_external_n10_shadow_plan_v1,
)


def _canonical_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_exact_or_validate(path: Path, value: object) -> None:
    expected = _canonical_bytes(value)
    path = path.expanduser().resolve()
    if path.exists():
        if path.read_bytes() != expected:
            raise ValueError("existing write-once shadow artifact differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(expected)


def _load_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: " + str(path))
    return value


def _run(stage: PreN10ExternalN10StagePlanV1, *, ready_count: int | None = None) -> None:
    argv = list(stage.argv)
    if stage.dynamic_max_ready_claims_from_intake:
        if ready_count is None:
            raise ValueError("dynamic N9 full-closure stage lacks intake ready count")
        argv += ["--max-ready-claims", str(max(1, int(ready_count)))]
    subprocess.run([sys.executable, *argv], check=True)
    missing = [path for path in stage.expected_outputs if not Path(path).is_file()]
    if missing:
        raise ValueError(
            "external/N9/N10 stage completed without expected artifacts: "
            + repr(missing)
        )


def _intake_ready_count(path: str | Path) -> int:
    payload = _load_json(path)
    hypotheses = payload.get("hypotheses")
    if not isinstance(hypotheses, list):
        return 0
    return sum(
        len(row.get("ready_for_closure_claim_ids", []))
        for row in hypotheses
        if isinstance(row, dict)
        and isinstance(row.get("ready_for_closure_claim_ids", []), list)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Consume the frozen pre-N10 downstream authority handoff and run "
            "exactly one external-novelty -> N9 intake -> N9 full closure -> "
            "role-aware N10 shadow chain per eligible lineage."
        )
    )
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--hypothesis-context", required=True, type=Path)
    parser.add_argument("--provider-plan", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    handoff = PreN10DownstreamHandoffReportV1.model_validate_json(
        args.handoff.expanduser().resolve().read_text(encoding="utf-8")
    )
    plan = compile_pre_n10_external_n10_shadow_plan_v1(
        handoff=handoff,
        context_path=args.hypothesis_context,
        provider_plan_path=args.provider_plan,
        model=args.model,
        output_root=args.output_root,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        save_prompts=args.save_prompts,
    )

    print("Pre-N10 external/N9/N10 shadow")
    print("Handoff:", handoff.report_id)
    print("Eligible lineages:", plan.lineage_count)
    for row in plan.lineages:
        print("Lineage:", row.lineage_id, "origin=", row.origin)
        for stage in row.stages:
            suffix = (
                " --max-ready-claims <max(1,intake-ready-count)>"
                if stage.dynamic_max_ready_claims_from_intake
                else ""
            )
            print("  ", stage.stage, "$", sys.executable, *stage.argv, suffix)

    if args.dry_run:
        print("Dry-run only: no external/N9/N10 stage executed")
        return 0

    root = args.output_root.expanduser().resolve()
    plan_path = root / "external_n9_n10.execution_plan.json"
    report_path = root / "external_n9_n10.report.json"
    if report_path.exists():
        raise ValueError("external/N9/N10 shadow report is write-once")
    _write_exact_or_validate(plan_path, plan)

    results: list[PreN10ExternalN10LineageResultV1] = []
    for row in plan.lineages:
        outputs = [
            path
            for stage in row.stages
            for path in stage.expected_outputs
        ]
        existing = [path for path in outputs if Path(path).exists()]
        if existing:
            raise ValueError(
                "external/N9/N10 lineage outputs are write-once; existing="
                + repr(existing)
            )

        _run(row.stages[0])
        _run(row.stages[1])
        ready_count = _intake_ready_count(row.n9_intake)
        _run(row.stages[2], ready_count=ready_count)
        _run(row.stages[3])
        _run(row.stages[4])

        result = build_pre_n10_external_n10_lineage_result_v1(
            plan=row,
            production_gate=_load_json(row.n10_production_gate),
        )
        results.append(result)
        print(
            "  N10:",
            row.downstream_hypothesis_id,
            result.n10_selection_class,
            "->",
            result.certification_status,
        )

    report = build_pre_n10_external_n10_shadow_report_v1(
        execution_plan=plan,
        handoff=handoff,
        lineages=results,
    )
    _write_exact_or_validate(report_path, report)

    print("External/N9/N10 shadow complete")
    print("Report:", report.report_id)
    print(
        "Certified/unresolved/rejected:",
        report.certified_count,
        "/",
        report.unresolved_count,
        "/",
        report.rejected_count,
    )
    print("Endpoint binding/verifier performed: false")
    print("Second regeneration performed: false")
    print("Production selection changed: false")
    print("Output:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
