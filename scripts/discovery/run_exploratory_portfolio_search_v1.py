from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.adaptive_discovery_controller import (
    AdaptiveControllerDecision,
    external_boundaries,
    generate_axis_mutation,
    safe_unused_premise_ids,
    stable_id,
)
from pipeline_core.discovery.adaptive_graph_retraversal import build_retraversal_request
from pipeline_core.discovery.exploratory_portfolio_search_v1 import (
    allocated_graph_parent_ids,
    build_exploratory_portfolio_plan,
    candidate_record,
    select_exploratory_portfolio,
)
from pipeline_core.discovery.external_novelty_contracts import ExternalNoveltyReport
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.prospective_identification_materialization_shadow import (
    compact_shadow_record,
    run_prospective_identification_shadow,
)
from pipeline_core.discovery.scientific_portfolio_closed_loop import compile_residual_epistemic_state
from pipeline_core.discovery.sers_novelty_feedback_closed_loop import load_context_source, run_feedback_generation


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--seed-closed-loop-dir", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--critic-model", required=True)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--exploration-temperature", type=float, default=0.9)
    p.add_argument("--max-local-branches-per-parent", type=int, default=3)
    p.add_argument("--max-graph-retraversal-allocations", type=int, default=2)
    p.add_argument("--max-retained-candidates", type=int, default=8)
    p.add_argument("--output-dir", required=True, type=Path)
    return p


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def single_feedback_plan(*, action: str, card: Any, external_report_id: str, known_boundary: list[str], unresolved_boundary: list[str], target_claim_ids: list[str], safe_unused: list[str], source_external_status: str) -> dict[str, Any]:
    if not str(source_external_status or "").strip():
        raise ValueError(
            "EPS feedback branch requires the parent ExternalNoveltyStatus"
        )
    route = {"SAME_PREMISE_SHARPEN": "SAME_PREMISE_GAP_SHARPEN", "EVIDENCE_REAXIS": "FRESH_CONTEXT_REAXIS"}[action]
    body = {
        "schema_version": "exploratory-portfolio-single-branch-plan-v1",
        "source_external_report_id": external_report_id,
        "targets": [{
            "source_hypothesis_id": str(card.hypothesis_id),
            "source_external_status": str(source_external_status),
            "route": route,
            "route_reason": "parallel exploratory branch",
            "safe_unused_premise_statement_ids": safe_unused,
            "already_known_boundary": known_boundary,
            "unresolved_boundary": unresolved_boundary,
            "target_claim_ids": target_claim_ids,
            "external_prior_art_as_positive_premise": False,
        }],
    }
    body["plan_id"] = stable_id("exploratory_single_branch", body)
    return body


def axis_decision(*, card: Any, parent: Any, known: list[str], unresolved: list[str], target_claim_ids: list[str], safe_unused: list[str]) -> AdaptiveControllerDecision:
    return AdaptiveControllerDecision(
        root_hypothesis_id=str(card.hypothesis_id),
        current_hypothesis_id=str(card.hypothesis_id),
        current_epistemic_state=parent.current_epistemic_state,
        current_external_status=parent.external_status,
        action="AXIS_MUTATION",
        allowed_actions=["AXIS_MUTATION"],
        action_scope_rank=3,
        rationale="Exploratory Portfolio Search parallel axis branch.",
        expected_information_gain="Test a structurally different grounded relation without replacing the incumbent.",
        lower_scope_exhausted_reason=None,
        safe_unused_premise_statement_ids=safe_unused,
        target_claim_ids=target_claim_ids,
        already_known_boundary=known,
        unresolved_boundary=unresolved,
        prior_action_counts={},
        prior_attempt_count=0,
        llm_advisory_used=False,
        llm_advisory_accepted=False,
        deterministic_fallback_used=False,
    )


def main() -> int:
    args = parser().parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed_closed_loop_dir.expanduser().resolve()
    context = load_context_source(args.context.expanduser().resolve())
    portfolio = HypothesisPortfolio.model_validate_json((seed / "gen1.portfolio.json").read_text(encoding="utf-8"))
    external = ExternalNoveltyReport.model_validate_json((seed / "gen1/external.report.json").read_text(encoding="utf-8"))
    state = compile_residual_epistemic_state(
        portfolio=portfolio.model_dump(mode="json"),
        query_plan=load(seed / "gen1/external.claims_queries.json"),
        external_report=external.model_dump(mode="json"),
        aggregation=load(seed / "gen1/residual_aggregation.report.json"),
    )
    write(out / "source.epistemic_state.json", state)

    prospective_by_h = {}
    cards = {str(x.hypothesis_id): x for x in portfolio.hypotheses}
    for i, card in enumerate(portfolio.hypotheses, start=1):
        artifact = run_prospective_identification_shadow(
            context=context,
            candidate=card,
            source_stage="exploratory_portfolio_search_v1_seed",
            model=args.critic_model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            parse_retries=3,
            output_prefix=out / "seed_prospective" / f"{i:02d}_{str(card.hypothesis_id).split(':')[-1]}",
        )
        prospective_by_h[str(card.hypothesis_id)] = compact_shadow_record(artifact)
    write(out / "seed.prospective.summary.json", prospective_by_h)

    plan = build_exploratory_portfolio_plan(
        context=context,
        portfolio=portfolio,
        epistemic_state_report=state,
        external_report=external.model_dump(mode="json"),
        prospective_by_hypothesis=prospective_by_h,
        max_local_branches_per_parent=args.max_local_branches_per_parent,
        max_graph_retraversal_allocations=args.max_graph_retraversal_allocations,
        exploration_temperature=args.exploration_temperature,
    )
    write(out / "exploratory.plan.json", plan)
    parent_by_h = {x.hypothesis_id: x for x in plan.parent_states}
    ext_by_h = {str(x.hypothesis_id): x for x in external.cards}

    all_cards = {}
    candidate_records = []
    generation_records = []
    for card in portfolio.hypotheses:
        hid = str(card.hypothesis_id)
        parent = parent_by_h[hid]
        all_cards[hid] = card
        candidate_records.append(candidate_record(
            card=card,
            source_hypothesis_id=hid,
            route="KEEP_ELITE",
            prospective_status=parent.prospective_identifiability,
            parent_state=parent,
            lineage_visit_count=1,
            total_candidate_count=max(1, len(portfolio.hypotheses)),
        ))

    for p_i, parent in enumerate(plan.parent_states, start=1):
        original = cards[parent.hypothesis_id]
        ext_card = ext_by_h.get(parent.hypothesis_id)
        ext_payload = ext_card.model_dump(mode="json") if ext_card else {}
        known, unresolved, target_claim_ids = external_boundaries(ext_payload)
        safe_unused = safe_unused_premise_ids(context, original)
        for b_i, action in enumerate(parent.branch_actions, start=1):
            branch = out / "branches" / f"{p_i:02d}_{parent.hypothesis_id.split(':')[-1]}" / f"{b_i:02d}_{action.lower()}"
            branch.mkdir(parents=True, exist_ok=True)
            candidate = None
            if action in {"SAME_PREMISE_SHARPEN", "EVIDENCE_REAXIS"}:
                report, generated = run_feedback_generation(
                    context=context,
                    portfolio=portfolio.model_copy(update={"hypotheses": [original], "abstention_reason": None}),
                    external=external,
                    plan=single_feedback_plan(
                        action=action,
                        card=original,
                        external_report_id=external.report_id,
                        known_boundary=known,
                        unresolved_boundary=unresolved,
                        target_claim_ids=target_claim_ids,
                        safe_unused=safe_unused,
                        source_external_status=str(parent.external_status or ""),
                    ),
                    model=args.model,
                    critic_model=args.critic_model,
                    api_key_env=args.api_key_env,
                    base_url=args.base_url,
                    output_dir=branch,
                    prospective_identification_stage="exploratory_portfolio_search_v1_" + action.lower(),
                    generation_temperature=args.exploration_temperature,
                )
                record = report["records"][0] if report.get("records") else {"source_hypothesis_id": parent.hypothesis_id, "decision": "NO_RECORD"}
                if generated.hypotheses:
                    candidate = generated.hypotheses[0]
            elif action == "AXIS_MUTATION":
                candidate, record = generate_axis_mutation(
                    context=context,
                    original=original,
                    decision=axis_decision(card=original, parent=parent, known=known, unresolved=unresolved, target_claim_ids=target_claim_ids, safe_unused=safe_unused),
                    attempt_history=[],
                    model=args.model,
                    critic_model=args.critic_model,
                    api_key_env=args.api_key_env,
                    base_url=args.base_url,
                    output_prefix=str(branch / "axis_mutation"),
                    generation_temperature=args.exploration_temperature,
                )
            else:
                raise RuntimeError(f"unsupported action {action}")
            generation_records.append({"planned_action": action, **record})
            if candidate is None:
                continue
            hid = str(candidate.hypothesis_id)
            all_cards[hid] = candidate
            shadow = record.get("prospective_identification_shadow") or {}
            candidate_records.append(candidate_record(
                card=candidate,
                source_hypothesis_id=parent.hypothesis_id,
                route=action,
                prospective_status=str(shadow.get("prospective_identifiability") or "UNKNOWN"),
                parent_state=parent,
                lineage_visit_count=0,
                total_candidate_count=max(1, len(candidate_records)+1),
            ))

    write(out / "generation.records.json", {
        "schema_version": "exploratory-portfolio-search-generation-records-v1",
        "records": generation_records,
        "parallel_branching_authority": True,
        "production_selection_authority": False,
    })

    graph_requests = []
    context_payload = context.model_dump(mode="json")
    for i, hid in enumerate(allocated_graph_parent_ids(plan), start=1):
        parent = parent_by_h[hid]
        ext_card = ext_by_h.get(hid)
        ext_payload = ext_card.model_dump(mode="json") if ext_card else {}
        known, unresolved, target_claim_ids = external_boundaries(ext_payload)
        graph_requests.append(build_retraversal_request(
            decision={
                "root_hypothesis_id": hid,
                "current_hypothesis_id": hid,
                "current_epistemic_state": parent.current_epistemic_state,
                "current_external_status": parent.external_status,
                "already_known_boundary": known,
                "unresolved_boundary": unresolved,
                "target_claim_ids": target_claim_ids,
            },
            source_context=context_payload,
            attempt_history=[],
            context_epoch=0,
            retraversal_index=i,
            reason="prospective_not_operationalizable_repair" if parent.prospective_identifiability == "NOT_OPERATIONALIZABLE" else "exploratory_uncertainty_compute_allocation",
        ))
    write(out / "graph_retraversal.allocations.json", {
        "schema_version": "exploratory-portfolio-graph-allocation-v1",
        "request_count": len(graph_requests),
        "requests": graph_requests,
        "allocation_authority": True,
        "execution_authority": False,
        "executor": "existing_adaptive_7_77_graph_retraversal",
        "canonical_graph_mutated": False,
        "external_prior_art_as_positive_premise": False,
    })

    selection = select_exploratory_portfolio(plan=plan, records=candidate_records, max_retained_candidates=args.max_retained_candidates)
    write(out / "exploratory.selection.json", selection)
    selected_cards = [all_cards[x] for x in selection.retained_candidate_ids if x in all_cards]
    selected_portfolio = portfolio.model_copy(update={
        "portfolio_id": stable_id("exploratory_portfolio_selected", portfolio.portfolio_id, selection.retained_candidate_ids),
        "hypotheses": selected_cards,
        "abstention_reason": None if selected_cards else "Exploratory search retained no verification candidate.",
    })
    all_materialized = portfolio.model_copy(update={
        "portfolio_id": stable_id("exploratory_portfolio_all_materialized", portfolio.portfolio_id, sorted(all_cards)),
        "hypotheses": list(all_cards.values()),
        "abstention_reason": None if all_cards else "No materialized candidates.",
    })
    write(out / "all_materialized.portfolio.json", all_materialized)
    write(out / "selected_for_verification.portfolio.json", selected_portfolio)

    summary = {
        "schema_version": "exploratory-portfolio-search-v1-summary",
        "source_hypothesis_count": len(portfolio.hypotheses),
        "materialized_candidate_count": len(all_cards),
        "selected_for_verification_count": len(selected_cards),
        "selected_for_verification_ids": selection.retained_candidate_ids,
        "graph_retraversal_allocation_count": len(graph_requests),
        "not_operationalizable_parent_count": sum(x.prospective_identifiability == "NOT_OPERATIONALIZABLE" for x in plan.parent_states),
        "branch_action_counts": dict(sorted(Counter(action for row in plan.parent_states for action in row.branch_actions).items())),
        "search_authority": {
            "parallel_branching": True,
            "parent_selection": True,
            "compute_allocation": True,
            "graph_retraversal_allocation": True,
            "not_operationalizable_reproductive_block": True,
        },
        "production_authority": {
            "scientific_truth": False,
            "literature_wide_novelty": False,
            "production_selection": False,
            "stage8_input_changed": False,
            "canonical_graph_mutated": False,
        },
    }
    write(out / "exploratory.summary.json", summary)
    print("===== EXPLORATORY PORTFOLIO SEARCH v1 =====")
    print("source hypotheses:", len(portfolio.hypotheses))
    print("materialized candidates:", len(all_cards))
    print("selected for next verification:", len(selected_cards))
    print("graph allocations:", len(graph_requests))
    print("branch actions:", summary["branch_action_counts"])
    print("SEARCH_PARENT_SELECTION_AUTHORITY=True")
    print("COMPUTE_ALLOCATION_AUTHORITY=True")
    print("GRAPH_RETRAVERSAL_ALLOCATION_AUTHORITY=True")
    print("SCIENTIFIC_TRUTH_AUTHORITY=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("STAGE8_INPUT_CHANGED=False")
    print("CANONICAL_GRAPH_MUTATED=False")
    print("output:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
