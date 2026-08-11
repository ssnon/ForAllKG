from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dac_her.discovery_axis_planner import DiscoveryAxisPlanner
from dac_her.discovery_axis_contracts import DiscoveryAxisPlannerPolicy
from dac_her.discovery_axis_runtime import DiscoveryAxisSynthesisRuntime
from dac_her.evidence_family_decomposition import (
    EvidenceFamilyDecompositionReport,
)
from dac_her.evidence_family_selection import (
    EvidenceFamilyHierarchy,
    audit_family_premise_selection,
)
from dac_her.dual_hypothesis_context import DualHypothesisContext
from dac_her.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
from dac_her.hypothesis_evidence_diversity import (
    HypothesisEvidenceDiversityAssessor,
)
from dac_her.node_mapping import NodeMapper


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Header must be KEY=VALUE")
    key, item = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Header key may not be empty")
    return key, item


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[attr-defined]
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run v2.8.0-alpha4 discovery-axis hypothesis synthesis: deterministic axis planning, "
            "one hypothesis per axis, axis-fidelity control, and bounded corpus-internal novelty repair."
        )
    )
    parser.add_argument("--dual-context", required=True, type=Path)
    parser.add_argument(
        "--evidence-family-decomposition-report",
        type=Path,
        default=None,
        help=(
            "Optional EC2-B report. When supplied, Alpha4 receives "
            "family-aware minimally-sufficient premise-selection guidance."
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help=(
            "Mechanism node index used for axis-fidelity and internal-novelty embeddings. "
            "Defaults to data_dac/corpus/<corpus>/mechanism/navigation/node_index."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL") or os.getenv("OPENROUTER_AGENT_MODEL"),
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--instructor-mode", default="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--header", action="append", default=[], type=_header, metavar="KEY=VALUE")
    parser.add_argument("--device", default=None)

    parser.add_argument("--max-axes", type=int, default=5)
    parser.add_argument("--min-exploration-score", type=float, default=0.05)
    parser.add_argument("--min-candidate-unit-score", type=float, default=0.30)
    parser.add_argument("--max-reaction-switch-penalty", type=float, default=0.50)
    parser.add_argument("--allow-non-candidate-axes", action="store_true")

    parser.add_argument("--max-compile-repairs", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-fidelity-repairs", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-novelty-repairs", type=int, choices=(0, 1), default=1)

    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dual = DualHypothesisContext.model_validate_json(
        args.dual_context.read_text(encoding="utf-8")
    )
    if not dual.discovery_bundle.inspirations:
        raise SystemExit(
            "DiscoveryBundle contains no inspirations. Alpha4 refuses to collapse back to canonical synthesis."
        )

    family_hierarchy = None
    if args.evidence_family_decomposition_report is not None:
        decomposition_report = (
            EvidenceFamilyDecompositionReport.model_validate_json(
                args.evidence_family_decomposition_report.read_text(
                    encoding="utf-8"
                )
            )
        )
        family_hierarchy = EvidenceFamilyHierarchy.from_decomposition_report(
            decomposition_report,
            dual.grounded_context,
        )
        print(
            "EC2-C family hierarchy:",
            family_hierarchy.hierarchy_id,
            "groups=",
            len(family_hierarchy.groups),
        )

    planner = DiscoveryAxisPlanner(
        DiscoveryAxisPlannerPolicy(
            max_axes=args.max_axes,
            require_candidate_unit=not args.allow_non_candidate_axes,
            min_exploration_score=args.min_exploration_score,
            min_candidate_unit_score=args.min_candidate_unit_score,
            max_reaction_domain_switch_penalty=args.max_reaction_switch_penalty,
        )
    )
    plan = planner.build(dual)
    plan_path = Path(str(args.output_prefix) + ".axis_plan.json")
    _write_json(plan_path, plan)

    print("DiscoveryAxisPlan built")
    print("Plan ID:", plan.plan_id)
    print("Axes:", len(plan.axes))
    for axis in plan.axes:
        print(
            f"[{axis.axis_rank}] planner={axis.planner_score:.3f} "
            f"explore={axis.exploration_score:.3f} unit={axis.candidate_unit_score:.3f} "
            f"reaction_penalty={axis.reaction_domain_switch_penalty:.2f}"
        )
        print("     ", axis.label)
        print("      entry:", axis.entry_anchor_label)
        print("      exit :", axis.exit_anchor_label)
    print("Saved axis plan:", plan_path)

    if args.dry_run_plan:
        return 0
    if not plan.axes:
        raise SystemExit(
            "No discovery axis survived alpha4 planner gates. Do not generate a canonical fallback."
        )
    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or OPENROUTER_AGENT_MODEL is set."
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
    mapper = NodeMapper.from_directory(index_dir, device=args.device)

    backend = InstructorOpenAICompatibleHypothesisBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=dict(args.header),
    )
    runtime = DiscoveryAxisSynthesisRuntime(
        backend,
        mapper,
        max_compile_repairs=args.max_compile_repairs,
        max_fidelity_repairs=args.max_fidelity_repairs,
        max_novelty_repairs=args.max_novelty_repairs,
        family_hierarchy=family_hierarchy,
    )
    outcome = runtime.run(dual, plan)
    evidence_diversity = (
        HypothesisEvidenceDiversityAssessor().assess(
            dual.grounded_context,
            outcome.portfolio,
        )
    )

    _write_json(Path(str(args.output_prefix) + ".draft.json"), outcome.final_draft)
    _write_json(Path(str(args.output_prefix) + ".portfolio.json"), outcome.portfolio)
    _write_json(Path(str(args.output_prefix) + ".lineage.json"), outcome.report)
    _write_json(
        Path(str(args.output_prefix) + ".internal_novelty.json"),
        outcome.internal_novelty_report,
    )
    _write_json(
        Path(str(args.output_prefix) + ".evidence_diversity.json"),
        evidence_diversity,
    )

    if family_hierarchy is not None:
        _write_json(
            Path(str(args.output_prefix) + ".family_hierarchy.json"),
            family_hierarchy,
        )
        family_selection = audit_family_premise_selection(
            family_hierarchy,
            outcome.portfolio,
        )
        _write_json(
            Path(str(args.output_prefix) + ".family_selection.json"),
            family_selection,
        )
        print(
            "EC2-C family selection:",
            f"child_use={family_selection.hypotheses_using_any_child_count}/"
            f"{family_selection.hypothesis_count}",
            "parent_incidence=",
            family_selection.parent_premise_incidence_count,
            "child_incidence=",
            family_selection.child_premise_incidence_count,
            "redundant_parent_all_children=",
            family_selection.potential_parent_all_children_redundancy_count,
        )

    if args.save_prompts:
        prompt_dir = Path(str(args.output_prefix) + ".prompts")
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for row in outcome.axis_prompts:
            path = prompt_dir / f"axis_{row.axis_rank:02d}.prompt.txt"
            path.write_text(
                "SYSTEM\n======\n"
                + row.prompt.system_prompt
                + "\n\nUSER\n====\n"
                + row.prompt.user_prompt
                + "\n",
                encoding="utf-8",
            )

    lineage_by_id = {row.hypothesis_id: row for row in outcome.report.lineages}
    novelty_by_id = {
        row.hypothesis_id: row for row in outcome.internal_novelty_report.cards
    }

    print()
    print("Discovery-axis synthesis complete")
    print("Attempted axes:", outcome.report.attempted_axis_count)
    print("Accepted hypotheses:", outcome.report.accepted_hypothesis_count)
    print("Portfolio:", outcome.portfolio.portfolio_id)
    print("External novelty:", outcome.report.external_novelty_status)
    for index, card in enumerate(outcome.portfolio.hypotheses, start=1):
        lineage = lineage_by_id[card.hypothesis_id]
        novelty = novelty_by_id[card.hypothesis_id]
        print(
            f"[{index}] fidelity={lineage.axis_fidelity_status} "
            f"internal_novelty={novelty.status} "
            f"fidelity_repaired={lineage.fidelity_repaired} "
            f"novelty_repaired={lineage.novelty_repaired}"
        )
        print("     ", card.title)
        print("      axis:", lineage.axis_id)
        print("      hypothesis:", card.hypothesis_statement)

    rejected = [row for row in outcome.report.attempts if row.decision != "accepted"]
    if rejected:
        print()
        print("Non-accepted axis attempts:")
        for row in rejected:
            print(
                f"- axis={row.axis_id} stage={row.stage} decision={row.decision} "
                f"fidelity={row.fidelity_status} novelty={row.internal_novelty_status}"
            )

    print("Saved portfolio:", Path(str(args.output_prefix) + ".portfolio.json"))
    print("Saved lineage:", Path(str(args.output_prefix) + ".lineage.json"))
    print(
        "Saved internal novelty:",
        Path(str(args.output_prefix) + ".internal_novelty.json"),
    )
    print(
        "Evidence statement coverage:",
        f"{evidence_diversity.used_statement_count}/"
        f"{evidence_diversity.eligible_statement_count}",
        f"({evidence_diversity.eligible_statement_coverage:.3f})",
    )
    print(
        "Shared-core premise statements:",
        evidence_diversity.shared_core_statement_count,
    )
    print(
        "Exact duplicate premise-set groups:",
        evidence_diversity.exact_premise_set_duplicate_group_count,
    )
    print(
        "Mean/max pairwise statement Jaccard:",
        f"{evidence_diversity.mean_pairwise_statement_jaccard:.3f}/"
        f"{evidence_diversity.max_pairwise_statement_jaccard:.3f}",
    )
    print(
        "Saved evidence diversity:",
        Path(str(args.output_prefix) + ".evidence_diversity.json"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
