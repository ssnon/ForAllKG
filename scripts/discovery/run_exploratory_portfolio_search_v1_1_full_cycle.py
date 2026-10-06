from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pipeline_core.discovery.adaptive_discovery_controller import (
    AdaptiveControllerDecision,
    external_boundaries,
    generate_axis_mutation,
    safe_unused_premise_ids,
    stable_id,
)
from pipeline_core.discovery.adaptive_graph_retraversal_generation import (
    generate_retraversal_hypothesis,
)
from pipeline_core.discovery.exploratory_portfolio_search_v1 import (
    allocated_graph_parent_ids,
    build_exploratory_portfolio_plan,
    candidate_record,
)
from pipeline_core.discovery.exploratory_portfolio_search_v1_1 import (
    reserve_diverse_search_portfolio,
)
from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.prospective_identification_materialization_shadow import (
    compact_shadow_record,
    run_prospective_identification_shadow,
)
from pipeline_core.discovery.scientific_portfolio_closed_loop import (
    compile_residual_epistemic_state,
)
from pipeline_core.discovery.sers_closed_loop_decision_consolidation_v2 import (
    build_effective_gen1_portfolio,
    consolidate_post_verification_decisions,
)
from pipeline_core.discovery.sers_novelty_feedback_closed_loop import (
    load_context_source,
    run_feedback_generation,
)
from scripts.discovery.run_adaptive_discovery_controller_shadow import (
    _generation_report,
    _resolve_stage775_seed_root,
    _run_fresh_verification,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--seed-closed-loop-dir", required=True, type=Path)
    p.add_argument("--grounding-traversal", required=True, type=Path)
    p.add_argument("--provider-plan", required=True, type=Path)
    p.add_argument("--domain-profile", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--critic-model", required=True)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--exploration-temperature", type=float, default=0.9)
    p.add_argument("--max-local-branches-per-parent", type=int, default=3)
    p.add_argument("--max-graph-retraversal-allocations", type=int, default=2)
    p.add_argument("--max-retained-candidates", type=int, default=8)
    p.add_argument("--results-per-query", type=int, default=12)
    p.add_argument("--max-results-per-query", type=int, default=30)
    p.add_argument("--max-ranked-works", type=int, default=12)
    p.add_argument("--parse-retries", type=int, default=3)
    p.add_argument("--max-fulltext-candidates", type=int, default=4)
    p.add_argument("--max-excerpt-chars", type=int, default=24000)
    p.add_argument("--retraversal-objective", default="explain_connection")
    p.add_argument("--retraversal-candidate-top-k", type=int, default=32)
    p.add_argument("--retraversal-selected-top-k", type=int, default=6)
    p.add_argument("--retraversal-node-map-k", type=int, default=30)
    p.add_argument("--retraversal-endpoint-pair-k", type=int, default=24)
    p.add_argument("--retraversal-max-depth-increment", type=int, default=2)
    p.add_argument("--retraversal-max-depth-cap", type=int, default=16)
    p.add_argument("--retraversal-min-new-edge-fraction", type=float, default=0.35)
    p.add_argument("--retraversal-min-new-paper-fraction", type=float, default=0.25)
    p.add_argument("--retraversal-max-prior-edge-jaccard", type=float, default=0.85)
    p.add_argument("--retraversal-max-selected-edge-jaccard", type=float, default=0.85)
    p.add_argument("--retraversal-paper-expansion-reserve", type=int, default=2)
    p.add_argument("--retraversal-allow-top-n-fallback", action="store_true")
    p.add_argument(
        "--override-card",
        type=Path,
        default=None,
        help=(
            "Optional historical sentinel card. When used, the source seed "
            "remains prior-art memory only; feedback branches that require a "
            "matching external card are skipped and the incumbent remains "
            "subject to Prospective Identification."
        ),
    )
    p.add_argument("--output-dir", required=True, type=Path)
    return p


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_cmd(label: str, cmd: list[str], output_dir: Path) -> None:
    print()
    print("=" * 96)
    print(label)
    print("=" * 96)
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    safe = "_".join(label.casefold().split())
    (output_dir / f"{safe}.stdout.txt").write_text(
        result.stdout or "",
        encoding="utf-8",
    )
    (output_dir / f"{safe}.stderr.txt").write_text(
        result.stderr or "",
        encoding="utf-8",
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with return code {result.returncode}"
        )


def one_card_portfolio(
    *,
    template: HypothesisPortfolio,
    card: HypothesisCard,
    context: HypothesisContext,
    label: str,
) -> HypothesisPortfolio:
    return template.model_copy(
        update={
            "portfolio_id": stable_id(
                "eps_v1_1_one_card_portfolio",
                label,
                card.hypothesis_id,
                context.context_id,
            ),
            "domain_profile_id": context.domain_profile_id,
            "source_context_id": context.context_id,
            "source_context_sha256": context.context_sha256,
            "source_report_id": context.source_report_id,
            "source_report_sha256": context.source_report_sha256,
            "hypotheses": [card],
            "abstention_reason": None,
        }
    )


def sentinel_portfolio(
    *,
    card: HypothesisCard,
    context: HypothesisContext,
) -> HypothesisPortfolio:
    return HypothesisPortfolio(
        portfolio_id=stable_id(
            "eps_v1_1_sentinel_portfolio",
            card.hypothesis_id,
            context.context_id,
        ),
        domain_profile_id=context.domain_profile_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_report_id=context.source_report_id,
        source_report_sha256=context.source_report_sha256,
        hypotheses=[card],
        abstention_reason=None,
    )


def feedback_plan(
    *,
    action: str,
    card: HypothesisCard,
    external: ExternalNoveltyReport,
    source_external_status: str,
    known: list[str],
    unresolved: list[str],
    target_claim_ids: list[str],
    safe_unused: list[str],
) -> dict[str, Any]:
    route = {
        "SAME_PREMISE_SHARPEN": "SAME_PREMISE_GAP_SHARPEN",
        "EVIDENCE_REAXIS": "FRESH_CONTEXT_REAXIS",
    }[action]
    body = {
        "schema_version": "eps-v1-1-single-feedback-plan-v1",
        "source_external_report_id": external.report_id,
        "targets": [{
            "source_hypothesis_id": str(card.hypothesis_id),
            "source_external_status": str(source_external_status),
            "route": route,
            "route_reason": "EPS v1.1 full-cycle parallel branch",
            "safe_unused_premise_statement_ids": list(safe_unused),
            "already_known_boundary": list(known),
            "unresolved_boundary": list(unresolved),
            "target_claim_ids": list(target_claim_ids),
            "external_prior_art_as_positive_premise": False,
        }],
    }
    body["plan_id"] = stable_id("eps_v1_1_feedback_plan", body)
    return body


def axis_decision(
    *,
    card: HypothesisCard,
    parent: Any,
    known: list[str],
    unresolved: list[str],
    target_claim_ids: list[str],
    safe_unused: list[str],
) -> AdaptiveControllerDecision:
    return AdaptiveControllerDecision(
        root_hypothesis_id=str(card.hypothesis_id),
        current_hypothesis_id=str(card.hypothesis_id),
        current_epistemic_state=parent.current_epistemic_state,
        current_external_status=parent.external_status,
        action="AXIS_MUTATION",
        allowed_actions=["AXIS_MUTATION"],
        action_scope_rank=3,
        rationale="EPS v1.1 full-cycle parallel axis mutation.",
        expected_information_gain=(
            "Explore a structurally different grounded relation while "
            "preserving the incumbent in the archive."
        ),
        lower_scope_exhausted_reason=None,
        safe_unused_premise_statement_ids=list(safe_unused),
        target_claim_ids=list(target_claim_ids),
        already_known_boundary=list(known),
        unresolved_boundary=list(unresolved),
        prior_action_counts={},
        prior_attempt_count=0,
        llm_advisory_used=False,
        llm_advisory_accepted=False,
        deterministic_fallback_used=False,
    )


def verification_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        provider_plan=args.provider_plan,
        domain_profile=args.domain_profile,
        critic_model=args.critic_model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        results_per_query=args.results_per_query,
        max_results_per_query=args.max_results_per_query,
        max_ranked_works=args.max_ranked_works,
        parse_retries=args.parse_retries,
        max_fulltext_candidates=args.max_fulltext_candidates,
        max_excerpt_chars=args.max_excerpt_chars,
        output_dir=args.output_dir,
    )


def main() -> int:
    args = parser().parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.provider_plan = args.provider_plan.expanduser().resolve()

    raw_seed = args.seed_closed_loop_dir.expanduser().resolve()
    seed, seed_mode = _resolve_stage775_seed_root(raw_seed)

    source_portfolio = HypothesisPortfolio.model_validate_json(
        (seed / "gen1.portfolio.json").read_text(encoding="utf-8")
    )
    source_external = ExternalNoveltyReport.model_validate_json(
        (seed / "gen1/external.report.json").read_text(encoding="utf-8")
    )
    source_query = seed / "gen1/external.claims_queries.json"
    source_prior = seed / "gen1/external.prior_art.json"
    source_binding = seed / "gen1/external.atomic_source_binding.json"
    source_completion = seed / "gen1/external.topology_completion_shadow.json"
    source_aggregation = seed / "gen1/residual_aggregation.report.json"

    context = load_context_source(args.context.expanduser().resolve())

    override_mode = args.override_card is not None
    if override_mode:
        override_card = HypothesisCard.model_validate_json(
            args.override_card.expanduser().resolve().read_text(encoding="utf-8")
        )
        search_portfolio = sentinel_portfolio(
            card=override_card,
            context=context,
        )
        initial_state = {
            "schema_version": "eps-v1-1-sentinel-epistemic-state-v1",
            "hypotheses": [{
                "hypothesis_id": str(override_card.hypothesis_id),
                "final_epistemic_state": "UNRESOLVED_EVIDENCE_GAP",
            }],
        }
    else:
        search_portfolio = source_portfolio
        initial_state = compile_residual_epistemic_state(
            portfolio=source_portfolio.model_dump(mode="json"),
            query_plan=load(source_query),
            external_report=source_external.model_dump(mode="json"),
            aggregation=load(source_aggregation),
        )

    cards = {
        str(card.hypothesis_id): card
        for card in search_portfolio.hypotheses
    }
    ext_by_h = {
        str(card.hypothesis_id): card
        for card in source_external.cards
    }

    # ------------------------------------------------------------------
    # Prospective Identification on incumbent parents.
    # ------------------------------------------------------------------
    prospective_by_h: dict[str, dict[str, Any]] = {}
    for index, card in enumerate(search_portfolio.hypotheses, start=1):
        artifact = run_prospective_identification_shadow(
            context=context,
            candidate=card,
            source_stage="exploratory_portfolio_search_v1_1_seed",
            model=args.critic_model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            parse_retries=3,
            output_prefix=(
                args.output_dir
                / "seed_prospective"
                / f"{index:02d}_{str(card.hypothesis_id).split(':')[-1]}"
            ),
        )
        prospective_by_h[str(card.hypothesis_id)] = compact_shadow_record(
            artifact
        )

    plan = build_exploratory_portfolio_plan(
        context=context,
        portfolio=search_portfolio,
        epistemic_state_report=initial_state,
        external_report=source_external.model_dump(mode="json"),
        prospective_by_hypothesis=prospective_by_h,
        max_local_branches_per_parent=args.max_local_branches_per_parent,
        max_graph_retraversal_allocations=args.max_graph_retraversal_allocations,
        exploration_temperature=args.exploration_temperature,
    )
    write(args.output_dir / "exploratory.plan.json", plan)

    parent_by_h = {
        row.hypothesis_id: row
        for row in plan.parent_states
    }

    candidate_records = []
    candidate_meta: dict[str, dict[str, Any]] = {}
    local_generation_records: list[dict[str, Any]] = []

    # Incumbents are always archived. NOT_OPERATIONALIZABLE incumbents are
    # marked sterile by candidate_record and cannot enter selected verification.
    for card in search_portfolio.hypotheses:
        hid = str(card.hypothesis_id)
        parent = parent_by_h[hid]
        candidate_records.append(
            candidate_record(
                card=card,
                source_hypothesis_id=hid,
                route="KEEP_ELITE",
                prospective_status=parent.prospective_identifiability,
                parent_state=parent,
                lineage_visit_count=1,
                total_candidate_count=max(1, len(search_portfolio.hypotheses)),
            )
        )
        candidate_meta[hid] = {
            "card": card,
            "context": context,
            "context_path": args.context.expanduser().resolve(),
            "template_portfolio": search_portfolio,
            "route": "KEEP_ELITE",
            "source_hypothesis_id": hid,
            "generation_record": {
                "source_hypothesis_id": hid,
                "route": "KEEP_ELITE",
                "decision": "CARRIED_FORWARD",
                "generated_hypothesis_id": hid,
                "reason_codes": ["incumbent_archived_for_eps_v1_1"],
            },
        }

    # ------------------------------------------------------------------
    # Parallel local offspring.
    # ------------------------------------------------------------------
    for p_index, parent in enumerate(plan.parent_states, start=1):
        original = cards[parent.hypothesis_id]
        ext_card = ext_by_h.get(parent.hypothesis_id)
        ext_payload = (
            ext_card.model_dump(mode="json")
            if ext_card is not None
            else {}
        )
        known, unresolved, target_claim_ids = external_boundaries(ext_payload)
        safe_unused = safe_unused_premise_ids(context, original)

        for b_index, action in enumerate(parent.branch_actions, start=1):
            branch_dir = (
                args.output_dir
                / "local_branches"
                / f"{p_index:02d}_{parent.hypothesis_id.split(':')[-1]}"
                / f"{b_index:02d}_{action.lower()}"
            )
            branch_dir.mkdir(parents=True, exist_ok=True)

            candidate = None
            record: dict[str, Any]

            if action in {"SAME_PREMISE_SHARPEN", "EVIDENCE_REAXIS"}:
                if ext_card is None:
                    record = {
                        "source_hypothesis_id": parent.hypothesis_id,
                        "route": action,
                        "decision": "SKIPPED_NO_MATCHING_EXTERNAL_CARD",
                        "generated_hypothesis_id": None,
                        "reason_codes": [
                            "historical_sentinel_external_card_not_in_frozen_seed",
                            "no_external_status_synthesized",
                        ],
                    }
                else:
                    report, generated = run_feedback_generation(
                        context=context,
                        portfolio=one_card_portfolio(
                            template=search_portfolio,
                            card=original,
                            context=context,
                            label=f"local_{action}",
                        ),
                        external=source_external,
                        plan=feedback_plan(
                            action=action,
                            card=original,
                            external=source_external,
                            source_external_status=str(ext_card.status),
                            known=known,
                            unresolved=unresolved,
                            target_claim_ids=target_claim_ids,
                            safe_unused=safe_unused,
                        ),
                        model=args.model,
                        critic_model=args.critic_model,
                        api_key_env=args.api_key_env,
                        base_url=args.base_url,
                        output_dir=branch_dir,
                        prospective_identification_stage=(
                            "exploratory_portfolio_search_v1_1_"
                            + action.lower()
                        ),
                        generation_temperature=args.exploration_temperature,
                    )
                    record = (
                        report["records"][0]
                        if report.get("records")
                        else {
                            "source_hypothesis_id": parent.hypothesis_id,
                            "route": action,
                            "decision": "NO_RECORD",
                        }
                    )
                    if generated.hypotheses:
                        candidate = generated.hypotheses[0]

            elif action == "AXIS_MUTATION":
                candidate, record = generate_axis_mutation(
                    context=context,
                    original=original,
                    decision=axis_decision(
                        card=original,
                        parent=parent,
                        known=known,
                        unresolved=unresolved,
                        target_claim_ids=target_claim_ids,
                        safe_unused=safe_unused,
                    ),
                    attempt_history=[],
                    model=args.model,
                    critic_model=args.critic_model,
                    api_key_env=args.api_key_env,
                    base_url=args.base_url,
                    output_prefix=str(branch_dir / "axis_mutation"),
                    generation_temperature=args.exploration_temperature,
                )
            else:
                continue

            local_generation_records.append(
                {"planned_action": action, **record}
            )

            if candidate is None:
                continue

            hid = str(candidate.hypothesis_id)
            shadow = record.get("prospective_identification_shadow") or {}
            status = str(
                shadow.get("prospective_identifiability")
                or "UNKNOWN"
            )
            candidate_records.append(
                candidate_record(
                    card=candidate,
                    source_hypothesis_id=parent.hypothesis_id,
                    route=action,
                    prospective_status=status,
                    parent_state=parent,
                    lineage_visit_count=0,
                    total_candidate_count=max(
                        1,
                        len(candidate_records) + 1,
                    ),
                )
            )
            candidate_meta[hid] = {
                "card": candidate,
                "context": context,
                "context_path": args.context.expanduser().resolve(),
                "template_portfolio": search_portfolio,
                "route": action,
                "source_hypothesis_id": parent.hypothesis_id,
                "generation_record": record,
            }

    write(
        args.output_dir / "local_generation.records.json",
        {
            "schema_version":
                "exploratory-portfolio-search-v1-1-local-generation-v1",
            "records": local_generation_records,
        },
    )

    # ------------------------------------------------------------------
    # Execute graph allocations with existing 7.77 executor.
    # ------------------------------------------------------------------
    graph_records: list[dict[str, Any]] = []
    graph_parent_ids = allocated_graph_parent_ids(plan)

    for graph_index, hid in enumerate(graph_parent_ids, start=1):
        original = cards[hid]
        parent = parent_by_h[hid]
        ext_card = ext_by_h.get(hid)
        ext_payload = (
            ext_card.model_dump(mode="json")
            if ext_card is not None
            else {}
        )
        known, unresolved, target_claim_ids = external_boundaries(ext_payload)

        request = {
            "schema_version": "adaptive-graph-retraversal-request-v2",
            "root_hypothesis_id": hid,
            "current_hypothesis_id": hid,
            "current_epistemic_state": parent.current_epistemic_state,
            "current_external_status": parent.external_status,
            "source_context_id": context.context_id,
            "source_context_sha256": context.context_sha256,
            "task_id": context.task_id,
            "question": context.question,
            "corpus_id": context.corpus_id,
            "domain_profile_id": context.domain_profile_id,
            "context_epoch": 0,
            "retraversal_index": graph_index,
            "request_reason": (
                "prospective_not_operationalizable_repair"
                if parent.prospective_identifiability
                == "NOT_OPERATIONALIZABLE"
                else "eps_v1_1_uncertainty_compute_allocation"
            ),
            "already_known_boundary": known,
            "unresolved_boundary": unresolved,
            "target_claim_ids": target_claim_ids,
            "attempt_history": [],
            "external_prior_art_as_positive_premise": False,
            "external_boundary_used_for_positive_path_selection": False,
            "canonical_graph_mutation_requested": False,
        }
        request["request_id"] = stable_id(
            "eps_v1_1_graph_request",
            request,
        )

        graph_dir = args.output_dir / "graph_branches" / f"{graph_index:02d}_{hid.split(':')[-1]}"
        request_path = graph_dir / "request.json"
        write(request_path, request)

        cmd = [
            sys.executable,
            "-m",
            "scripts.discovery.run_adaptive_graph_retraversal_shadow",
            "--source-context", str(args.context.expanduser().resolve()),
            "--prior-traversal", str(args.grounding_traversal.expanduser().resolve()),
            "--request", str(request_path),
            "--model", str(args.critic_model),
            "--api-key-env", str(args.api_key_env),
            "--objective", str(args.retraversal_objective),
            "--candidate-top-k", str(args.retraversal_candidate_top_k),
            "--selected-top-k", str(args.retraversal_selected_top_k),
            "--node-map-k", str(args.retraversal_node_map_k),
            "--endpoint-pair-k", str(args.retraversal_endpoint_pair_k),
            "--max-depth-increment", str(args.retraversal_max_depth_increment),
            "--max-depth-cap", str(args.retraversal_max_depth_cap),
            "--min-new-edge-fraction", str(args.retraversal_min_new_edge_fraction),
            "--min-new-paper-fraction", str(args.retraversal_min_new_paper_fraction),
            "--max-prior-edge-jaccard", str(args.retraversal_max_prior_edge_jaccard),
            "--max-selected-edge-jaccard", str(args.retraversal_max_selected_edge_jaccard),
            "--paper-expansion-reserve", str(args.retraversal_paper_expansion_reserve),
            "--output-dir", str(graph_dir),
        ]
        if args.retraversal_allow_top_n_fallback:
            cmd.append("--allow-top-n-fallback")
        if args.base_url:
            cmd += ["--base-url", str(args.base_url)]

        graph_row = {
            "source_hypothesis_id": hid,
            "request_id": request["request_id"],
            "execution_status": None,
            "generation_decision": None,
            "generated_hypothesis_id": None,
        }

        try:
            run_cmd(
                f"EPS v1.1 graph retraversal {graph_index}",
                cmd,
                args.output_dir,
            )
            summary = load(graph_dir / "retraversal.summary.json")
            graph_row["execution_status"] = summary.get("status")

            if summary.get("status") != "COMPLETE_NEW_GROUNDED_CONTEXT":
                graph_records.append(graph_row)
                continue

            new_context_path = graph_dir / "retraversal.context.json"
            new_context = load_context_source(new_context_path)
            lineage = load(graph_dir / "retraversal.lineage.json")

            generated_portfolio, gen_record = generate_retraversal_hypothesis(
                old_context=context,
                new_context=new_context,
                original=original,
                request=request,
                attempt_history=[],
                required_new_premise_ids=lineage.get(
                    "structurally_new_eligible_premise_ids",
                    [],
                ),
                model=args.model,
                critic_model=args.critic_model,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                output_prefix=str(graph_dir / "context_reset_generation"),
            )
            write(graph_dir / "generation.record.json", gen_record)
            graph_row["generation_decision"] = gen_record.get("decision")
            graph_row["generated_hypothesis_id"] = gen_record.get(
                "generated_hypothesis_id"
            )

            if generated_portfolio is None:
                graph_records.append(graph_row)
                continue

            graph_card = generated_portfolio.hypotheses[0]
            graph_hid = str(graph_card.hypothesis_id)
            shadow = gen_record.get("prospective_identification_shadow") or {}
            status = str(
                shadow.get("prospective_identifiability")
                or "UNKNOWN"
            )

            candidate_records.append(
                candidate_record(
                    card=graph_card,
                    source_hypothesis_id=hid,
                    route="REQUEST_GRAPH_RETRAVERSAL",
                    prospective_status=status,
                    parent_state=parent,
                    lineage_visit_count=0,
                    total_candidate_count=max(
                        1,
                        len(candidate_records) + 1,
                    ),
                )
            )
            candidate_meta[graph_hid] = {
                "card": graph_card,
                "context": new_context,
                "context_path": new_context_path,
                "template_portfolio": generated_portfolio,
                "route": "REQUEST_GRAPH_RETRAVERSAL",
                "source_hypothesis_id": hid,
                "generation_record": gen_record,
            }
            graph_records.append(graph_row)

        except Exception as exc:
            graph_row["execution_status"] = "EXECUTION_FAILED"
            graph_row["error"] = f"{type(exc).__name__}:{exc}"
            graph_records.append(graph_row)

    write(
        args.output_dir / "graph_execution.records.json",
        {
            "schema_version":
                "exploratory-portfolio-search-v1-1-graph-execution-v1",
            "allocated_parent_count": len(graph_parent_ids),
            "records": graph_records,
            "graph_retraversal_allocation_authority": True,
            "graph_retraversal_execution_authority": True,
            "canonical_graph_mutated": False,
        },
    )

    # ------------------------------------------------------------------
    # Reserved explore/exploit selection over local + graph offspring.
    # ------------------------------------------------------------------
    selection = reserve_diverse_search_portfolio(
        candidate_records,
        max_retained_candidates=args.max_retained_candidates,
    )
    write(args.output_dir / "exploratory.selection.v1_1.json", selection)

    selected_ids = list(selection["retained_candidate_ids"])

    # ------------------------------------------------------------------
    # Fresh verification: one candidate per cohort.
    # ------------------------------------------------------------------
    verification_rows = []
    effective_ids = []
    v_args = verification_args(args)

    for index, hid in enumerate(selected_ids, start=1):
        meta = candidate_meta[hid]
        card = meta["card"]
        candidate_context = meta["context"]
        template = meta["template_portfolio"]
        one = one_card_portfolio(
            template=template,
            card=card,
            context=candidate_context,
            label=f"verify_{index}",
        )
        verify_dir = args.output_dir / "fresh_verification" / f"{index:02d}_{hid.split(':')[-1]}"
        portfolio_path = verify_dir / "candidate.portfolio.json"
        write(portfolio_path, one)

        raw_record = dict(meta["generation_record"])
        raw_record.setdefault("source_hypothesis_id", meta["source_hypothesis_id"])
        raw_record["route"] = meta["route"]
        raw_record["decision"] = raw_record.get("decision") or "ACCEPTED_GENERATION_SHADOW"
        raw_record["generated_hypothesis_id"] = hid

        gen_report = _generation_report(
            source_portfolio_id=str(search_portfolio.portfolio_id),
            output_portfolio_id=str(one.portfolio_id),
            round_index=1,
            records=[raw_record],
        )
        write(verify_dir / "generation.report.json", gen_report)

        row = {
            "hypothesis_id": hid,
            "route": meta["route"],
            "source_hypothesis_id": meta["source_hypothesis_id"],
            "context_id": str(candidate_context.context_id),
            "verification_status": None,
            "post_verification_decision": None,
            "effective": False,
        }

        try:
            artifacts = _run_fresh_verification(
                args=v_args,
                round_dir=verify_dir,
                portfolio_path=portfolio_path,
                previous_query=source_query,
                previous_external=seed / "gen1/external.report.json",
                previous_prior=source_prior,
                previous_binding=source_binding,
                previous_completion=source_completion,
                retrieval_multiplier=1,
                reuse_query_plan=False,
            )

            decisions = consolidate_post_verification_decisions(
                gen1_portfolio=one.model_dump(mode="json"),
                generation_report=gen_report,
                query_plan=load(artifacts["query"]),
                external_report=load(artifacts["external"]),
                aggregation=load(artifacts["aggregation"]),
                cohort_audit=load(artifacts["cohort"]),
            )
            effective = build_effective_gen1_portfolio(
                gen1_portfolio=one.model_dump(mode="json"),
                decisions=decisions,
            )

            write(verify_dir / "post_verification_decisions.json", decisions)
            write(verify_dir / "effective.portfolio.json", effective)

            decision_row = (
                decisions.get("rows", [{}])[0]
                if decisions.get("rows")
                else {}
            )
            row["verification_status"] = "COMPLETE"
            row["post_verification_decision"] = decision_row.get(
                "post_verification_decision"
            )
            row["fresh_external_status"] = decision_row.get(
                "fresh_external_status"
            )
            row["effective"] = bool(effective.hypotheses)

            if effective.hypotheses:
                effective_ids.append(hid)

        except Exception as exc:
            row["verification_status"] = "FAILED"
            row["error"] = f"{type(exc).__name__}:{exc}"

        verification_rows.append(row)

    write(
        args.output_dir / "fresh_verification.summary.json",
        {
            "schema_version":
                "exploratory-portfolio-search-v1-1-fresh-verification-v1",
            "selected_count": len(selected_ids),
            "completed_count": sum(
                row["verification_status"] == "COMPLETE"
                for row in verification_rows
            ),
            "failed_count": sum(
                row["verification_status"] == "FAILED"
                for row in verification_rows
            ),
            "effective_count": len(effective_ids),
            "effective_hypothesis_ids": effective_ids,
            "rows": verification_rows,
        },
    )

    selected_records = {
        str(row["hypothesis_id"]): row
        for row in selection["records"]
        if row.get("selected_for_next_verification")
    }

    summary = {
        "schema_version":
            "exploratory-portfolio-search-v1-1-full-cycle-summary-v1",
        "seed_source_mode": seed_mode,
        "override_sentinel_mode": override_mode,
        "source_hypothesis_count": len(search_portfolio.hypotheses),
        "candidate_pool_count": len(candidate_records),
        "local_materialized_offspring_count": sum(
            str(row.get("decision")) == "ACCEPTED_GENERATION_SHADOW"
            for row in local_generation_records
        ),
        "graph_allocation_count": len(graph_parent_ids),
        "graph_execution_complete_context_count": sum(
            row.get("execution_status") == "COMPLETE_NEW_GROUNDED_CONTEXT"
            for row in graph_records
        ),
        "graph_materialized_offspring_count": sum(
            row.get("generation_decision") == "ACCEPTED_GENERATION_SHADOW"
            for row in graph_records
        ),
        "selected_count": len(selected_ids),
        "selected_ids": selected_ids,
        "selected_slot_counts": dict(sorted(Counter(
            str(row.get("selected_slot"))
            for row in selected_records.values()
        ).items())),
        "selected_route_counts": dict(sorted(Counter(
            str(candidate_meta[hid]["route"])
            for hid in selected_ids
        ).items())),
        "fresh_verification_completed_count": sum(
            row["verification_status"] == "COMPLETE"
            for row in verification_rows
        ),
        "fresh_verification_failed_count": sum(
            row["verification_status"] == "FAILED"
            for row in verification_rows
        ),
        "effective_count": len(effective_ids),
        "effective_hypothesis_ids": effective_ids,
        "effective_route_counts": dict(sorted(Counter(
            row["route"]
            for row in verification_rows
            if row["effective"]
        ).items())),
        "not_operationalizable_parent_count": sum(
            row.prospective_identifiability == "NOT_OPERATIONALIZABLE"
            for row in plan.parent_states
        ),
        "sterile_incumbent_selected_count": sum(
            (
                row.prospective_identifiability == "NOT_OPERATIONALIZABLE"
                and row.hypothesis_id in selected_ids
            )
            for row in candidate_records
        ),
        "authority": {
            "parallel_branching": True,
            "search_parent_selection": True,
            "compute_allocation": True,
            "graph_retraversal_allocation": True,
            "graph_retraversal_execution": True,
            "not_operationalizable_reproductive_block": True,
            "scientific_truth": False,
            "literature_wide_novelty": False,
            "production_selection": False,
            "stage8_input_changed": False,
            "canonical_graph_mutated": False,
        },
    }
    write(args.output_dir / "full_cycle.summary.json", summary)

    print()
    print("===== EPS v1.1 FULL-CYCLE SUMMARY =====")
    print("source hypotheses:", summary["source_hypothesis_count"])
    print("candidate pool:", summary["candidate_pool_count"])
    print("local offspring:", summary["local_materialized_offspring_count"])
    print("graph allocations:", summary["graph_allocation_count"])
    print("graph complete contexts:", summary["graph_execution_complete_context_count"])
    print("graph offspring:", summary["graph_materialized_offspring_count"])
    print("selected:", summary["selected_count"])
    print("selected slots:", summary["selected_slot_counts"])
    print("selected routes:", summary["selected_route_counts"])
    print("fresh verification completed:", summary["fresh_verification_completed_count"])
    print("fresh verification failed:", summary["fresh_verification_failed_count"])
    print("EFFECTIVE:", summary["effective_count"])
    print("effective routes:", summary["effective_route_counts"])
    print("NOT_OPERATIONALIZABLE parents:", summary["not_operationalizable_parent_count"])
    print("sterile incumbent selected:", summary["sterile_incumbent_selected_count"])
    print("SCIENTIFIC_TRUTH_AUTHORITY=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("STAGE8_INPUT_CHANGED=False")
    print("CANONICAL_GRAPH_MUTATED=False")
    print("output:", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
