from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.pre_n10_downstream_handoff_v1 import (
    build_pre_n10_downstream_handoff_v1,
)
from pipeline_core.discovery.pre_n10_initial_semantic_gate_v1 import (
    PreN10InitialSemanticGateReportV1,
)
from pipeline_core.discovery.pre_n10_primary_router_v1 import (
    PreN10PrimaryRouterReportV1,
)
from pipeline_core.discovery.pre_n10_regeneration_reentry_v2 import (
    PreN10RegenerationReentryReportV2,
)
from pipeline_core.discovery.pre_n10_regeneration_v1 import (
    PreN10RegenerationExecutionReportV1,
)


def _load(path: Path, model):
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError("missing handoff input artifact: " + str(resolved))
    return model.model_validate_json(resolved.read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the pre-N10 downstream authority handoff. Only initial "
            "primary-ready or regenerated semantic-PASS + PRE_N10_READY "
            "lineages become eligible for external novelty/N9/N10."
        )
    )
    parser.add_argument("--initial-semantic-gate", required=True, type=Path)
    parser.add_argument("--initial-portfolio", required=True, type=Path)
    parser.add_argument("--post-primary-query-plan", required=True, type=Path)
    parser.add_argument("--primary-router-report", required=True, type=Path)
    parser.add_argument("--regeneration-report", type=Path, default=None)
    parser.add_argument("--regeneration-reentry-report", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    gate = _load(
        args.initial_semantic_gate,
        PreN10InitialSemanticGateReportV1,
    )
    primary = _load(
        args.primary_router_report,
        PreN10PrimaryRouterReportV1,
    )
    regeneration = (
        _load(args.regeneration_report, PreN10RegenerationExecutionReportV1)
        if args.regeneration_report is not None
        else None
    )
    reentry = (
        _load(
            args.regeneration_reentry_report,
            PreN10RegenerationReentryReportV2,
        )
        if args.regeneration_reentry_report is not None
        else None
    )
    if (regeneration is None) != (reentry is None):
        raise ValueError(
            "--regeneration-report and --regeneration-reentry-report must be "
            "supplied together"
        )

    report = build_pre_n10_downstream_handoff_v1(
        initial_semantic_gate=gate,
        initial_portfolio_path=args.initial_portfolio,
        post_primary_query_plan_path=args.post_primary_query_plan,
        primary_router_report=primary,
        regeneration_report=regeneration,
        regeneration_reentry_report=reentry,
        output_root=args.output_dir,
    )
    print("Pre-N10 downstream handoff complete")
    print("Report:", report.report_id)
    print("Source hypotheses:", report.source_hypothesis_count)
    print("Initial ready:", report.initial_ready_lineage_count)
    print("Regeneration fallbacks:", report.regeneration_fallback_source_count)
    print("Regenerated ready:", report.regenerated_ready_lineage_count)
    print("Blocked after regeneration:", report.blocked_after_regeneration_count)
    print("Eligible for external novelty:", report.eligible_for_external_novelty_count)
    print("External novelty/N9/N10 performed: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
