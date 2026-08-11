from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dac_her.discovery_axis_contracts import (
    DiscoveryAxisPlannerPolicy,
)
from dac_her.discovery_axis_planner import DiscoveryAxisPlanner
from dac_her.discovery_axis_runtime import (
    DiscoveryAxisSynthesisRuntime,
)
from dac_her.dual_hypothesis_context import (
    DualHypothesisContext,
)
from dac_her.evidence_constituent_resolution import (
    ExistingConstituentResolutionReport,
)
from dac_her.hypothesis_llm import (
    InstructorOpenAICompatibleHypothesisBackend,
)
from dac_her.node_mapping import NodeMapper
from dac_her.synthesis_constituent_guidance import (
    SynthesisConstituentGuidedBackend,
    SynthesisConstituentHierarchy,
    SynthesisConstituentPromptAugmenter,
    audit_synthesis_constituent_selection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Header must be KEY=VALUE"
        )
    key, item = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(
            "Header key may not be empty"
        )
    return key, item


def _write_json(
    path: Path,
    value: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Alpha4 with EC2-D2 synthesis-constituent-aware "
            "premise-selection guidance. This is a standalone experimental "
            "runner; the canonical Alpha4 runner is not modified."
        )
    )
    parser.add_argument(
        "--dual-context",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--constituent-resolution-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
    )
    parser.add_argument(
        "--instructor-mode",
        default="JSON",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        type=_header,
        metavar="KEY=VALUE",
    )
    parser.add_argument("--device", default=None)

    parser.add_argument(
        "--max-axes",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--min-exploration-score",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--min-candidate-unit-score",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--max-reaction-switch-penalty",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--allow-non-candidate-axes",
        action="store_true",
    )

    parser.add_argument(
        "--max-compile-repairs",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument(
        "--max-fidelity-repairs",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument(
        "--max-novelty-repairs",
        type=int,
        choices=(0, 1),
        default=1,
    )

    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run-plan",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dual = DualHypothesisContext.model_validate_json(
        args.dual_context.read_text(
            encoding="utf-8"
        )
    )
    if not dual.discovery_bundle.inspirations:
        raise SystemExit(
            "DiscoveryBundle contains no inspirations. "
            "EC2-D2 refuses canonical fallback."
        )

    resolution = (
        ExistingConstituentResolutionReport.model_validate_json(
            args.constituent_resolution_report.read_text(
                encoding="utf-8"
            )
        )
    )
    hierarchy = SynthesisConstituentHierarchy.from_resolution_report(
        resolution,
        dual.grounded_context,
    )

    print(
        "EC2-D2 synthesis-constituent hierarchy:",
        hierarchy.hierarchy_id,
    )
    print("Hierarchy groups:", len(hierarchy.groups))
    for group in hierarchy.groups:
        print(" parent:", group.parent_statement_id)
        for member in group.members:
            print(
                "   ->",
                member.constituent_statement_id,
                "basis=",
                member.resolution_basis,
                "exact=",
                member.exact_equivalence,
            )

    planner = DiscoveryAxisPlanner(
        DiscoveryAxisPlannerPolicy(
            max_axes=args.max_axes,
            require_candidate_unit=(
                not args.allow_non_candidate_axes
            ),
            min_exploration_score=args.min_exploration_score,
            min_candidate_unit_score=(
                args.min_candidate_unit_score
            ),
            max_reaction_domain_switch_penalty=(
                args.max_reaction_switch_penalty
            ),
        )
    )
    plan = planner.build(dual)
    plan_path = Path(
        str(args.output_prefix)
        + ".axis_plan.json"
    )
    _write_json(plan_path, plan)

    print("DiscoveryAxisPlan built")
    print("Plan ID:", plan.plan_id)
    print("Axes:", len(plan.axes))
    for axis in plan.axes:
        print(
            f"[{axis.axis_rank}] "
            f"planner={axis.planner_score:.3f} "
            f"explore={axis.exploration_score:.3f} "
            f"unit={axis.candidate_unit_score:.3f} "
            f"reaction_penalty="
            f"{axis.reaction_domain_switch_penalty:.2f}"
        )
        print("     ", axis.label)

    if args.dry_run_plan:
        _write_json(
            Path(
                str(args.output_prefix)
                + ".constituent_hierarchy.json"
            ),
            hierarchy,
        )
        return 0

    if not plan.axes:
        raise SystemExit(
            "No discovery axis survived Alpha4 planner gates."
        )
    if not args.model:
        raise SystemExit(
            "--model is required unless "
            "GRAPHAGENTS_HYPOTHESIS_MODEL or "
            "OPENROUTER_AGENT_MODEL is set."
        )

    index_dir = args.index_dir or (
        PROJECT_ROOT
        / "data_dac"
        / "corpus"
        / dual.grounded_context.corpus_id
        / "mechanism"
        / "navigation"
        / "node_index"
    )
    mapper = NodeMapper.from_directory(
        index_dir,
        device=args.device,
    )

    base_backend = (
        InstructorOpenAICompatibleHypothesisBackend(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.temperature,
            parse_retries=args.parse_retries,
            timeout=args.timeout,
            extra_headers=dict(args.header),
        )
    )
    guided_backend = SynthesisConstituentGuidedBackend(
        base_backend,
        hierarchy,
    )

    # Do not pass EC2-C family_hierarchy even if that optional feature is
    # installed locally. EC2-D2 is intentionally isolated: only the backend
    # prompt adapter changes generation behavior.
    runtime = DiscoveryAxisSynthesisRuntime(
        guided_backend,
        mapper,
        max_compile_repairs=args.max_compile_repairs,
        max_fidelity_repairs=args.max_fidelity_repairs,
        max_novelty_repairs=args.max_novelty_repairs,
    )
    outcome = runtime.run(
        dual,
        plan,
    )

    _write_json(
        Path(str(args.output_prefix) + ".draft.json"),
        outcome.final_draft,
    )
    _write_json(
        Path(str(args.output_prefix) + ".portfolio.json"),
        outcome.portfolio,
    )
    _write_json(
        Path(str(args.output_prefix) + ".lineage.json"),
        outcome.report,
    )
    _write_json(
        Path(
            str(args.output_prefix)
            + ".internal_novelty.json"
        ),
        outcome.internal_novelty_report,
    )
    _write_json(
        Path(
            str(args.output_prefix)
            + ".constituent_hierarchy.json"
        ),
        hierarchy,
    )

    selection = audit_synthesis_constituent_selection(
        hierarchy,
        outcome.portfolio,
    )
    _write_json(
        Path(
            str(args.output_prefix)
            + ".constituent_selection.json"
        ),
        selection,
    )

    if args.save_prompts:
        augmenter = SynthesisConstituentPromptAugmenter(
            hierarchy
        )
        prompt_dir = Path(
            str(args.output_prefix)
            + ".prompts"
        )
        prompt_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        for row in outcome.axis_prompts:
            augmented = augmenter.augment(
                row.prompt
            )
            path = (
                prompt_dir
                / f"axis_{row.axis_rank:02d}.prompt.txt"
            )
            path.write_text(
                "SYSTEM\n======\n"
                + augmented.system_prompt
                + "\n\nUSER\n====\n"
                + augmented.user_prompt
                + "\n",
                encoding="utf-8",
            )

    print()
    print(
        "EC2-D2 synthesis-constituent selection:",
        f"constituent_use="
        f"{selection.hypotheses_using_any_constituent_count}/"
        f"{selection.hypothesis_count}",
        "parent_use=",
        f"{selection.hypotheses_using_any_parent_count}/"
        f"{selection.hypothesis_count}",
        "parent_incidence=",
        selection.parent_premise_incidence_count,
        "constituent_incidence=",
        selection.constituent_premise_incidence_count,
        "redundant_parent_all_constituents=",
        (
            selection
            .potential_parent_all_constituents_redundancy_count
        ),
    )

    print()
    print("Discovery-axis synthesis complete")
    print(
        "Attempted axes:",
        outcome.report.attempted_axis_count,
    )
    print(
        "Accepted hypotheses:",
        outcome.report.accepted_hypothesis_count,
    )
    print(
        "Portfolio:",
        outcome.portfolio.portfolio_id,
    )
    print(
        "External novelty:",
        outcome.report.external_novelty_status,
    )

    lineage_by_id = {
        row.hypothesis_id: row
        for row in outcome.report.lineages
    }
    novelty_by_id = {
        row.hypothesis_id: row
        for row in outcome.internal_novelty_report.cards
    }
    selection_by_id = {
        row.hypothesis_id: row
        for row in selection.cards
    }

    for index, card in enumerate(
        outcome.portfolio.hypotheses,
        start=1,
    ):
        lineage = lineage_by_id[card.hypothesis_id]
        novelty = novelty_by_id[card.hypothesis_id]
        sel = selection_by_id[card.hypothesis_id]
        print(
            f"[{index}] "
            f"fidelity={lineage.axis_fidelity_status} "
            f"internal_novelty={novelty.status} "
            f"constituent_usage={sel.usage_class}"
        )
        print("     ", card.title)
        print(
            "      premises:",
            card.premise_statement_ids,
        )
        print(
            "      parents:",
            sel.used_parent_statement_ids,
        )
        print(
            "      constituents:",
            sel.used_constituent_statement_ids,
        )

    print(
        "Saved portfolio:",
        Path(
            str(args.output_prefix)
            + ".portfolio.json"
        ),
    )
    print(
        "Saved selection:",
        Path(
            str(args.output_prefix)
            + ".constituent_selection.json"
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
