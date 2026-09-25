from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.pre_n10_prospective_campaign_v1 import (
    CAMPAIGN_STAGE_ORDER,
    build_pre_n10_prospective_campaign_plan_v1,
    execute_pre_n10_prospective_campaign_v1,
    write_exact_or_validate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete frozen prospective scientific-authority campaign "
            "from an existing initial semantic review through V_pre, the unified "
            "primary router, optional one-shot regeneration, external/N9/N10, "
            "literal relational binding, and relational V_post certification. "
            "Resume is allowed only between completed write-once stages."
        )
    )
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--semantic-run", required=True, type=Path)
    parser.add_argument("--semantic-review", type=Path, default=None)
    parser.add_argument("--hypothesis-context", required=True, type=Path)
    parser.add_argument("--regeneration-unit-freeze", required=True, type=Path)
    parser.add_argument("--provider-plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)

    parser.add_argument("--model", required=True)
    parser.add_argument("--decomposition-model", default=None)
    parser.add_argument("--primary-model", default=None)
    parser.add_argument("--specification-repair-model", default=None)
    parser.add_argument("--specification-audit-model", default=None)
    parser.add_argument("--source-alignment-model", default=None)
    parser.add_argument("--regeneration-model", default=None)
    parser.add_argument("--semantic-critic-model", default=None)
    parser.add_argument("--external-n10-model", default=None)
    parser.add_argument("--vpost-model", default=None)

    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-claims", type=int, default=4)
    parser.add_argument("--max-queries-per-claim", type=int, default=2)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    default_model = str(args.model)
    plan = build_pre_n10_prospective_campaign_plan_v1(
        portfolio_path=args.portfolio,
        semantic_run_path=args.semantic_run,
        semantic_review_path=args.semantic_review,
        hypothesis_context_path=args.hypothesis_context,
        regeneration_unit_freeze_path=args.regeneration_unit_freeze,
        provider_plan_path=args.provider_plan,
        output_root=args.output_root,
        decomposition_model=args.decomposition_model or default_model,
        primary_model=args.primary_model or default_model,
        specification_repair_model=(
            args.specification_repair_model or args.primary_model or default_model
        ),
        specification_audit_model=(
            args.specification_audit_model or args.primary_model or default_model
        ),
        source_alignment_model=(
            args.source_alignment_model or args.primary_model or default_model
        ),
        regeneration_model=args.regeneration_model or default_model,
        semantic_critic_model=args.semantic_critic_model or default_model,
        external_n10_model=args.external_n10_model or default_model,
        vpost_model=args.vpost_model or default_model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        parse_retries=args.parse_retries,
        timeout_seconds=args.timeout_seconds,
        max_claims=args.max_claims,
        max_queries_per_claim=args.max_queries_per_claim,
        save_prompts=args.save_prompts,
        allow_dirty_worktree_for_vpost=args.allow_dirty_worktree,
    )

    print("Pre-N10 prospective campaign")
    print("Plan:", plan.plan_id)
    print("Output root:", plan.output_root)
    print("Resume:", plan.resume_scope)
    print("Partial-stage resume allowed: false")
    print("Second regeneration allowed: false")
    print("Stages:")
    for index, name in enumerate(CAMPAIGN_STAGE_ORDER, start=1):
        print(" ", index, name)

    if args.dry_run:
        root = args.output_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        write_exact_or_validate(root / "campaign.plan.json", plan)
        print("Execution performed: false")
        print("Conditional route decisions deferred to frozen stage reports")
        return 0

    report = execute_pre_n10_prospective_campaign_v1(plan=plan)
    print()
    print("Campaign complete")
    print("Report:", report.report_id)
    print("Final status:", report.final_status)
    print("Initial semantic status:", report.initial_semantic_status)
    print(
        "Primary regeneration fallbacks:",
        report.primary_regeneration_fallback_count,
    )
    print(
        "External-eligible lineages:",
        report.handoff_external_eligible_count,
    )
    print(
        "N10 certified/unresolved/rejected:",
        report.n10_certified_count,
        "/",
        report.n10_unresolved_count,
        "/",
        report.n10_rejected_count,
    )
    print("Binding-ready lineages:", report.binding_ready_lineage_count)
    print("V_post completed:", report.vpost_completed_count)
    print(
        "Scientific certification:",
        report.scientific_certification_decision_counts,
    )
    print("Production selection changed: false")
    print("Canonical graph mutated: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
