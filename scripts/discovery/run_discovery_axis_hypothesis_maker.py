from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import hashlib
import networkx as nx

from domains.context_review_registry import (
    available_context_review_profiles,
    get_context_review_adapter,
)

from pipeline_core.discovery.discovery_axis_planner import DiscoveryAxisPlanner
from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxisPlannerPolicy
from pipeline_core.discovery.discovery_axis_runtime import DiscoveryAxisSynthesisRuntime
from pipeline_core.discovery.discovery_axis_inference_critic import (
    DiscoveryAxisInferenceCritic,
)
from pipeline_core.discovery.discovery_axis_inference_llm import (
    InstructorOpenAICompatibleAxisInferenceBackend,
)
from pipeline_core.discovery.evidence_family_decomposition import (
    EvidenceFamilyDecompositionReport,
)
from pipeline_core.discovery.evidence_family_selection import (
    EvidenceFamilyHierarchy,
    audit_family_premise_selection,
)
from pipeline_core.discovery.dual_hypothesis_context import DualHypothesisContext
from pipeline_core.discovery.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
from pipeline_core.discovery.hypothesis_evidence_diversity import (
    HypothesisEvidenceDiversityAssessor,
)
from pipeline_core.discovery.node_mapping import NodeMapper


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


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
    parser.add_argument("--max-inference-repairs", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-novelty-repairs", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--inference-critic-model",
        default=os.getenv("OPENROUTER_CRITIC_MODEL"),
        help=(
            "Model for discovery-axis inference-strength review. "
            "Defaults to OPENROUTER_CRITIC_MODEL, then --model."
        ),
    )
    parser.add_argument(
        "--context-critic-model",
        default=None,
        help=(
            "Optional SERS scientific-context model override. "
            "Defaults to --inference-critic-model."
        ),
    )
    parser.add_argument(
        "--context-graph",
        type=Path,
        default=None,
        help=(
            "Mechanism source graph.graphml used for "
            "claim-local scientific context compilation. "
            "Defaults from --index-dir."
        ),
    )

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
    inference_backend = InstructorOpenAICompatibleAxisInferenceBackend(
        model=args.inference_critic_model or args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=0.0,
        parse_retries=1,
        timeout=args.timeout,
        extra_headers=dict(args.header),
    )

    inference_critic = DiscoveryAxisInferenceCritic(
        inference_backend
    )

    context_adapter = None
    context_reviewer = None
    context_graph = None
    context_graph_path = None

    context_profile_id = (
        dual.grounded_context.domain_profile_id
    )

    if (
        context_profile_id
        in available_context_review_profiles()
    ):
        context_model = (
            args.context_critic_model
            or args.inference_critic_model
        )

        if not context_model:
            raise SystemExit(
                "Context-review capable domain requires "
                "--context-critic-model or "
                "--inference-critic-model."
            )

        context_graph_path = (
            args.context_graph
            or (
                index_dir.parents[1]
                / "graph.graphml"
            )
        )

        if not context_graph_path.is_file():
            raise FileNotFoundError(
                "Scientific context source graph missing: "
                f"{context_graph_path}"
            )

        context_graph = nx.read_graphml(
            context_graph_path,
            force_multigraph=True,
        )

        if not context_graph.is_directed():
            raise RuntimeError(
                "Scientific context source graph must be directed."
            )

        context_adapter = (
            get_context_review_adapter(
                context_profile_id
            )
        )

        context_reviewer = (
            context_adapter.build_openai_compatible(
                graph=context_graph,
                model=context_model,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                instructor_mode=args.instructor_mode,
                temperature=0.0,
                parse_retries=args.parse_retries,
                timeout=args.timeout,
                extra_headers=dict(args.header),
            )
        )

    runtime = DiscoveryAxisSynthesisRuntime(
        backend,
        mapper,
        inference_critic=inference_critic,
        context_reviewer=context_reviewer,
        max_compile_repairs=args.max_compile_repairs,
        max_fidelity_repairs=args.max_fidelity_repairs,
        max_inference_repairs=args.max_inference_repairs,
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

    inference_path = Path(
        str(args.output_prefix) + ".inference.json"
    )

    inference_records = []

    if outcome.inference_reviews:
        review_by_axis = {
            row.axis_id: row
            for row in outcome.inference_reviews
        }

        for lineage in outcome.report.lineages:
            review = review_by_axis.get(
                lineage.axis_id
            )

            if review is None:
                continue

            inference_records.append(
                {
                    "final_hypothesis_id":
                        lineage.hypothesis_id,
                    "axis_id":
                        lineage.axis_id,
                    "source_review_hypothesis_id":
                        review.hypothesis_id,
                    "status":
                        review.status,
                    "inference_repaired":
                        lineage.inference_repaired,
                    "review":
                        review.model_dump(
                            mode="json"
                        ),
                }
            )

    inference_review_history = [
        {
            "axis_id":
                row.axis_id,
            "generation_index":
                row.generation_index,
            "stage":
                row.stage,
            "review_id":
                row.review.review_id,
            "hypothesis_id":
                row.review.hypothesis_id,
            "status":
                row.review.status,
            "review":
                row.review.model_dump(
                    mode="json"
                ),
        }
        for row in outcome.inference_review_history
    ]

    _write_json(
        inference_path,
        {
            "schema_version":
                "discovery-axis-inference-artifact-v2",
            "portfolio_id":
                outcome.portfolio.portfolio_id,
            "policy_version":
                outcome.report.policy_version,

            # Backward-compatible final accepted records.
            "records":
                inference_records,

            # Complete critic history, including rejected and
            # pre-repair reviews.
            "review_history":
                inference_review_history,

            "final_record_count":
                len(inference_records),
            "review_history_count":
                len(inference_review_history),
        },
    )

    _write_json(
        Path(str(args.output_prefix) + ".internal_novelty.json"),
        outcome.internal_novelty_report,
    )
    _write_json(
        Path(str(args.output_prefix) + ".evidence_diversity.json"),
        evidence_diversity,
    )

    if context_reviewer is not None:
        if (
            context_adapter is None
            or context_graph is None
            or context_graph_path is None
        ):
            raise RuntimeError(
                "Context reviewer lacks source provenance."
            )

        context_payload = {
            "schema_version": (
                "discovery-axis-context-artifact-v1"
            ),
            "domain_profile_id": context_profile_id,
            "adapter_id": context_adapter.adapter_id,
            "model": context_model,
            "source_graph": str(context_graph_path),
            "source_graph_sha256": _sha256_file(
                context_graph_path
            ),
            "source_graph_node_count": (
                context_graph.number_of_nodes()
            ),
            "source_graph_edge_count": (
                context_graph.number_of_edges()
            ),
            "records": [
                (
                    row.model_dump(mode="json")
                    if hasattr(row, "model_dump")
                    else row
                )
                for row in outcome.context_reviews
            ],
            "review_history": [
                {
                    "axis_id": row.axis_id,
                    "generation_index": row.generation_index,
                    "stage": row.stage,
                    "review": (
                        row.review.model_dump(mode="json")
                        if hasattr(row.review, "model_dump")
                        else row.review
                    ),
                }
                for row in outcome.context_review_history
            ],
            "final_record_count": len(
                outcome.context_reviews
            ),
            "review_history_count": len(
                outcome.context_review_history
            ),
            "action_policy_applied": False,
            "g1_action_policy_deferred": True,
        }

        context_path = Path(
            str(args.output_prefix)
            + ".context.json"
        )

        _write_json(
            context_path,
            context_payload,
        )

        print(
            "Saved context review:",
            context_path,
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
            f"inference={lineage.inference_status} "
            f"inference_repaired={lineage.inference_repaired} "
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
                f"fidelity={row.fidelity_status} "
                f"inference={row.inference_status} "
                f"novelty={row.internal_novelty_status}"
            )

    print("Saved portfolio:", Path(str(args.output_prefix) + ".portfolio.json"))
    print("Saved lineage:", Path(str(args.output_prefix) + ".lineage.json"))
    print("Saved inference:", inference_path)
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
