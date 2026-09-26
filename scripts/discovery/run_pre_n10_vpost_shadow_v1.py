from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pipeline_core.discovery.pre_n10_relational_binding_bridge_v1 import (
    PreN10RelationalBindingBridgeReportV1,
)
from pipeline_core.discovery.pre_n10_vpost_shadow_v1 import (
    build_pre_n10_vpost_lineage_result_v1,
    build_pre_n10_vpost_shadow_report_v1,
    compile_pre_n10_vpost_shadow_plan_v1,
    write_exact_or_validate,
)


def _run_stage(name: str, argv: list[str], expected_outputs: list[str]) -> None:
    existing = [
        value for value in expected_outputs if Path(value).expanduser().exists()
    ]
    if existing:
        raise ValueError(
            "V_post stage outputs are write-once; existing=" + repr(existing)
        )
    print("$", sys.executable, *argv)
    subprocess.run([sys.executable, *argv], check=True)
    missing = [
        value for value in expected_outputs if not Path(value).expanduser().is_file()
    ]
    if missing:
        raise ValueError(
            name + " completed without expected outputs: " + repr(missing)
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run literal endpoint binding and the frozen relational scientific "
            "verifier only for binding-ready lineages emitted by the unified "
            "pre-N10/N10 bridge."
        )
    )
    parser.add_argument("--binding-bridge", required=True, type=Path)
    parser.add_argument("--provider-plan", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--canonical-spec-bundle",
        type=Path,
        default=None,
    )
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument(
        "--allow-dirty-worktree",
        action="store_true",
        help="Calibration/debug only; prospective execution should be committed and clean.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bridge_path = args.binding_bridge.expanduser().resolve()
    bridge = PreN10RelationalBindingBridgeReportV1.model_validate_json(
        bridge_path.read_text(encoding="utf-8")
    )
    root = args.output_root.expanduser().resolve()

    execution_plan = compile_pre_n10_vpost_shadow_plan_v1(
        bridge=bridge,
        provider_plan_path=args.provider_plan,
        model=args.model,
        output_root=root,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        save_prompts=args.save_prompts,
        allow_dirty_worktree=args.allow_dirty_worktree,
        canonical_spec_bundle_path=args.canonical_spec_bundle,
    )
    plan_path = root / "vpost_execution.plan.json"
    write_exact_or_validate(plan_path, execution_plan)

    print("Unified pre-N10 V_post shadow")
    print("Plan:", execution_plan.plan_id)
    print("Lineages:", execution_plan.lineage_count)
    print(
        "Execution required/skipped:",
        execution_plan.execution_required_lineage_count,
        "/",
        execution_plan.skipped_not_binding_ready_count,
    )
    print("N10 status filters V_post reachability: false")
    print("Production authority: false")

    if args.dry_run:
        for lineage in execution_plan.lineages:
            print()
            print(lineage.lineage_id, lineage.binding_status)
            for stage in lineage.stages:
                print(" ", stage.stage)
                print("   $", sys.executable, *stage.argv)
                print("   outputs:", stage.expected_outputs)
        print()
        print("Execution performed: false")
        return 0

    results = []
    for lineage in execution_plan.lineages:
        for stage in lineage.stages:
            _run_stage(stage.stage, stage.argv, stage.expected_outputs)
        results.append(
            build_pre_n10_vpost_lineage_result_v1(plan=lineage)
        )

    report = build_pre_n10_vpost_shadow_report_v1(
        execution_plan=execution_plan,
        bridge=bridge,
        lineages=results,
    )
    report_path = root / "vpost_shadow.report.json"
    write_exact_or_validate(report_path, report)

    print()
    print("Unified V_post shadow complete")
    print("Report:", report.report_id)
    print("Completed/not binding-ready:", report.completed_count, "/", report.not_binding_ready_count)
    print("Scientific certification decisions:", report.certification_decision_counts)
    print("External novelty reassessed: false")
    print("Verifier result consumed by production: false")
    print("Production selection changed: false")
    print("Canonical graph mutated: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
