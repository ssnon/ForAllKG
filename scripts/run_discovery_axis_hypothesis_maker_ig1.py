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
from dac_her.hypothesis_llm import (
    InstructorOpenAICompatibleHypothesisBackend,
)
from dac_her.ig1_grounded_bridge import (
    IG1BlueprintGenerator,
    IG1HypothesisBackend,
    build_blueprint_report,
    build_conformance_report,
)
from dac_her.node_mapping import NodeMapper


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
            "Run experimental IG1 discovery-axis synthesis: "
            "two grounded endpoints + exactly one explicit novel bridge."
        )
    )
    parser.add_argument(
        "--dual-context",
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
        "--blueprint-model",
        default=None,
        help=(
            "Optional separate model for IG1 blueprint planning. "
            "Defaults to --model."
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
    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument("--max-axes", type=int, default=5)
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
        "--max-blueprint-grounding-repairs",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument(
        "--max-ig1-conformance-repairs",
        type=int,
        choices=(0, 1),
        default=1,
    )

    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
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
            "IG1 refuses canonical fallback."
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
    _write_json(
        Path(
            str(args.output_prefix)
            + ".axis_plan.json"
        ),
        plan,
    )

    print("IG1 DiscoveryAxisPlan built")
    print("Plan ID:", plan.plan_id)
    print("Axes:", len(plan.axes))
    for axis in plan.axes:
        print(
            f"[{axis.axis_rank}] "
            f"planner={axis.planner_score:.3f} "
            f"explore={axis.exploration_score:.3f} "
            f"unit={axis.candidate_unit_score:.3f}"
        )
        print("     ", axis.label)

    if args.dry_run_plan:
        return 0
    if not plan.axes:
        raise SystemExit(
            "No discovery axis survived planner gates."
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

    headers = dict(args.header)

    base_backend = (
        InstructorOpenAICompatibleHypothesisBackend(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.temperature,
            parse_retries=args.parse_retries,
            timeout=args.timeout,
            extra_headers=headers,
        )
    )

    blueprint_generator = IG1BlueprintGenerator(
        model=args.blueprint_model or args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=headers,
    )

    ig1_backend = IG1HypothesisBackend(
        base_backend,
        blueprint_generator,
        context=dual.grounded_context,
        max_blueprint_grounding_repairs=(
            args.max_blueprint_grounding_repairs
        ),
        max_ig1_conformance_repairs=(
            args.max_ig1_conformance_repairs
        ),
    )

    # Canonical Alpha4 orchestration/compiler/fidelity/novelty remain unchanged.
    runtime = DiscoveryAxisSynthesisRuntime(
        ig1_backend,
        mapper,
        max_compile_repairs=args.max_compile_repairs,
        max_fidelity_repairs=args.max_fidelity_repairs,
        max_novelty_repairs=args.max_novelty_repairs,
    )
    outcome = runtime.run(
        dual,
        plan,
    )

    blueprint_report = build_blueprint_report(
        ig1_backend,
        context=dual.grounded_context,
        axis_plan_id=plan.plan_id,
    )
    conformance_report = build_conformance_report(
        portfolio=outcome.portfolio,
        axis_report=outcome.report,
        blueprint_report=blueprint_report,
    )

    _write_json(
        Path(
            str(args.output_prefix)
            + ".draft.json"
        ),
        outcome.final_draft,
    )
    _write_json(
        Path(
            str(args.output_prefix)
            + ".portfolio.json"
        ),
        outcome.portfolio,
    )
    _write_json(
        Path(
            str(args.output_prefix)
            + ".lineage.json"
        ),
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
            + ".ig1_blueprints.json"
        ),
        blueprint_report,
    )
    _write_json(
        Path(
            str(args.output_prefix)
            + ".ig1_conformance.json"
        ),
        conformance_report,
    )

    if args.save_prompts:
        prompt_dir = Path(
            str(args.output_prefix)
            + ".prompts"
        )
        prompt_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        for row in outcome.axis_prompts:
            original_path = (
                prompt_dir
                / f"axis_{row.axis_rank:02d}.base.prompt.txt"
            )
            original_path.write_text(
                "SYSTEM\n======\n"
                + row.prompt.system_prompt
                + "\n\nUSER\n====\n"
                + row.prompt.user_prompt
                + "\n",
                encoding="utf-8",
            )
            augmented = ig1_backend.augmented_prompt(
                row.prompt.prompt_sha256
            )
            if augmented is not None:
                ig1_path = (
                    prompt_dir
                    / f"axis_{row.axis_rank:02d}.ig1.prompt.txt"
                )
                ig1_path.write_text(
                    "SYSTEM\n======\n"
                    + augmented.system_prompt
                    + "\n\nUSER\n====\n"
                    + augmented.user_prompt
                    + "\n",
                    encoding="utf-8",
                )

    print()
    print("IG1 blueprint summary")
    print(
        "blueprints / active / abstained / invalid:",
        f"{blueprint_report.blueprint_count}/"
        f"{blueprint_report.active_blueprint_count}/"
        f"{blueprint_report.abstained_blueprint_count}/"
        f"{blueprint_report.invalid_blueprint_count}",
    )
    print(
        "IG1 conformance repairs:",
        blueprint_report.ig1_conformance_repair_count,
    )
    for record in blueprint_report.records:
        bp = record.blueprint
        print()
        print("axis:", record.axis_id)
        print(
            " blueprint valid:",
            record.valid,
            "abstain:",
            bp.abstain,
        )
        if bp.abstain:
            print(
                " reason:",
                bp.abstention_reason,
            )
            continue
        assert bp.endpoint_a is not None
        assert bp.endpoint_b is not None
        assert bp.novel_bridge is not None
        assert bp.discriminative_test is not None
        print(
            " A:",
            bp.endpoint_a.anchor_statement_id,
            "=>",
            bp.endpoint_a.grounded_excerpt,
        )
        print(
            " B:",
            bp.endpoint_b.anchor_statement_id,
            "=>",
            bp.endpoint_b.grounded_excerpt,
        )
        print(
            " ONE BRIDGE:",
            bp.novel_bridge.relation,
        )
        print(
            " TEST:",
            bp.discriminative_test.observable,
            "/",
            bp.discriminative_test.expected_direction,
        )

    print()
    print("IG1 synthesis complete")
    print(
        "Attempted axes:",
        outcome.report.attempted_axis_count,
    )
    print(
        "Accepted hypotheses:",
        outcome.report.accepted_hypothesis_count,
    )
    print(
        "Final IG1 conformance:",
        f"{conformance_report.passing_count}/"
        f"{conformance_report.hypothesis_count}",
        "passing",
    )
    if conformance_report.issue_counts:
        print(
            "Conformance issues:",
            conformance_report.issue_counts,
        )

    lineage_by_id = {
        row.hypothesis_id: row
        for row in outcome.report.lineages
    }
    novelty_by_id = {
        row.hypothesis_id: row
        for row in outcome.internal_novelty_report.cards
    }
    conform_by_id = {
        row.hypothesis_id: row
        for row in conformance_report.cards
    }

    for index, card in enumerate(
        outcome.portfolio.hypotheses,
        start=1,
    ):
        lineage = lineage_by_id[card.hypothesis_id]
        novelty = novelty_by_id[card.hypothesis_id]
        conform = conform_by_id[card.hypothesis_id]
        print(
            f"[{index}] "
            f"fidelity={lineage.axis_fidelity_status} "
            f"internal_novelty={novelty.status} "
            f"ig1_conformance={conform.passes}"
        )
        print("     ", card.title)
        print(
            "      hypothesis:",
            card.hypothesis_statement,
        )
        print(
            "      premises:",
            card.premise_statement_ids,
        )

    print(
        "Saved blueprint report:",
        Path(
            str(args.output_prefix)
            + ".ig1_blueprints.json"
        ),
    )
    print(
        "Saved conformance report:",
        Path(
            str(args.output_prefix)
            + ".ig1_conformance.json"
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
