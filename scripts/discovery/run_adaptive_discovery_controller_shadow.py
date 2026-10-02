
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.adaptive_discovery_controller import (
    AdaptiveControllerDecision,
    InstructorOpenAICompatibleAdaptiveControllerBackend,
    build_controller_plan,
    canonical_json,
    generate_axis_mutation,
    normalize_seed_action,
    stable_id,
)
from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.scientific_portfolio_closed_loop import (
    compile_residual_epistemic_state,
)
from pipeline_core.discovery.sers_closed_loop_decision_consolidation_v2 import (
    consolidate_post_verification_decisions,
)
from pipeline_core.discovery.sers_novelty_feedback_closed_loop import (
    load_context_source,
    run_feedback_generation,
)
from scripts.discovery.run_scientific_portfolio_closed_loop_shadow import (
    load,
    lower_order,
    run,
    run_aggregation,
    run_fulltext,
    run_topology,
    write,
)


def _card_map(portfolio: HypothesisPortfolio) -> dict[str, Any]:
    return {
        str(card.hypothesis_id): card
        for card in portfolio.hypotheses
    }


def _row_map(report: dict[str, Any], key: str = "hypothesis_id"):
    return {
        str(row.get(key)): row
        for row in report.get("rows", [])
        if row.get(key)
    }


def _external_card_map(report: dict[str, Any]):
    return {
        str(row.get("hypothesis_id")): row
        for row in report.get("cards", [])
        if row.get("hypothesis_id")
    }


def _decision_by_current(plan: dict[str, Any]):
    return {
        str(row.get("current_hypothesis_id")): row
        for row in plan.get("decisions", [])
    }


def _single_feedback_plan(
    *,
    decision: dict[str, Any],
    external_report_id: str,
) -> dict[str, Any]:
    action = str(decision["action"])
    route = {
        "SAME_PREMISE_SHARPEN": "SAME_PREMISE_GAP_SHARPEN",
        "EVIDENCE_REAXIS": "FRESH_CONTEXT_REAXIS",
    }[action]
    body = {
        "schema_version": "adaptive-feedback-single-target-plan-v1",
        "source_external_report_id": external_report_id,
        "targets": [
            {
                "source_hypothesis_id": decision["current_hypothesis_id"],
                "source_external_status": decision.get(
                    "current_external_status"
                ),
                "route": route,
                "route_reason": decision.get("rationale"),
                "safe_unused_premise_statement_ids": decision.get(
                    "safe_unused_premise_statement_ids", []
                ),
                "already_known_boundary": decision.get(
                    "already_known_boundary", []
                ),
                "unresolved_boundary": decision.get(
                    "unresolved_boundary", []
                ),
                "target_claim_ids": decision.get(
                    "target_claim_ids", []
                ),
                "external_prior_art_as_positive_premise": False,
            }
        ],
    }
    body["plan_id"] = stable_id(
        "adaptive_feedback_single_target",
        body,
    )
    return body



def _retrieval_only_actions(
    actions: list[str],
) -> bool:
    normalized = [
        str(action or "")
        for action in actions
        if str(action or "")
    ]

    return bool(normalized) and all(
        action == "RETRIEVE_MORE"
        for action in normalized
    )


def _validate_retrieval_batch(
    actions: list[str],
) -> bool:
    normalized = [
        str(action or "")
        for action in actions
        if str(action or "")
    ]

    has_retrieval = (
        "RETRIEVE_MORE"
        in normalized
    )

    retrieval_only = (
        _retrieval_only_actions(
            normalized
        )
    )

    if (
        has_retrieval
        and not retrieval_only
    ):
        raise RuntimeError(
            "Adaptive controller v1d does not "
            "permit RETRIEVE_MORE in a mixed "
            "verification batch. Exact frozen "
            "query-plan reuse is required for "
            "retrieval-only evidence escalation."
        )

    return retrieval_only


def _assert_reused_query_plan_identity(
    *,
    previous: dict[str, Any],
    current: dict[str, Any],
) -> None:
    for key in (
        "source_portfolio_id",
        "plan_id",
        "plan_sha256",
        "claims",
        "queries",
    ):
        if previous.get(key) != current.get(key):
            raise RuntimeError(
                "RETRIEVE_MORE changed frozen "
                f"query-plan field: {key}"
            )



def _combined_portfolio(
    *,
    template: HypothesisPortfolio,
    cards: list[Any],
    round_index: int,
    preserve_portfolio_id: bool = False,
) -> HypothesisPortfolio:
    unique = {}
    for card in cards:
        unique[str(card.hypothesis_id)] = card
    ordered = list(unique.values())
    pid = (
        template.portfolio_id
        if preserve_portfolio_id
        else stable_id(
            "adaptive_discovery_round_portfolio",
            template.portfolio_id,
            round_index,
            [str(card.hypothesis_id) for card in ordered],
        )
    )
    return template.model_copy(
        update={
            "portfolio_id": pid,
            "hypotheses": ordered,
            "abstention_reason": (
                None
                if ordered
                else "Adaptive controller produced no candidate for this round."
            ),
        }
    )


def _generation_report(
    *,
    source_portfolio_id: str,
    output_portfolio_id: str,
    round_index: int,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    body = {
        "schema_version": "adaptive-discovery-generation-report-v1",
        "source_portfolio_id": source_portfolio_id,
        "output_portfolio_id": output_portfolio_id,
        "round_index": round_index,
        "records": records,
        "decision_counts": dict(
            sorted(
                Counter(
                    str(row.get("decision"))
                    for row in records
                ).items()
            )
        ),
        "route_counts": dict(
            sorted(
                Counter(
                    str(row.get("route"))
                    for row in records
                ).items()
            )
        ),
        "external_prior_art_as_positive_premise": False,
        "generation_authority_created": False,
    }
    body["report_id"] = stable_id(
        "adaptive_discovery_generation",
        body,
    )
    return body


def _seed_history(
    *,
    seed_generation: dict[str, Any],
    seed_decisions: dict[str, Any],
    seed_state: dict[str, Any],
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
    decision_by_h = _row_map(seed_decisions)
    state_by_h = {
        str(row.get("hypothesis_id")): row
        for row in seed_state.get("hypotheses", [])
    }

    root_by_h: dict[str, str] = {}
    history: dict[str, list[dict[str, Any]]] = {}

    for rec in seed_generation.get("records", []):
        source = str(rec.get("source_hypothesis_id") or "")
        generated = str(rec.get("generated_hypothesis_id") or "")
        if not source:
            continue
        current = generated or source
        root_by_h[current] = source
        action = normalize_seed_action(str(rec.get("route") or ""))
        row = {
            "round_index": 1,
            "source_hypothesis_id": source,
            "current_hypothesis_id": current,
            "action": action,
            "generation_route": rec.get("route"),
            "generation_decision": rec.get("decision"),
            "post_verification_decision": (
                decision_by_h.get(current, {})
                .get("post_verification_decision")
            ),
            "epistemic_state": (
                state_by_h.get(current, {})
                .get("final_epistemic_state")
            ),
        }
        history.setdefault(source, []).append(row)

    return root_by_h, history


def _append_history(
    *,
    history: dict[str, list[dict[str, Any]]],
    root_by_source: dict[str, str],
    plan: dict[str, Any],
    generation: dict[str, Any],
    decisions: dict[str, Any],
    state: dict[str, Any],
    round_index: int,
) -> dict[str, str]:
    controller_by_h = _decision_by_current(plan)
    post_by_h = _row_map(decisions)
    state_by_h = {
        str(row.get("hypothesis_id")): row
        for row in state.get("hypotheses", [])
    }
    next_root: dict[str, str] = {}

    recorded_sources: set[str] = set()

    for rec in generation.get("records", []):
        source = str(rec.get("source_hypothesis_id") or "")
        generated = str(rec.get("generated_hypothesis_id") or "")
        recorded_sources.add(source)
        controller = controller_by_h.get(source, {})
        root = str(
            root_by_source.get(
                source,
                controller.get("root_hypothesis_id") or source,
            )
        )
        current = generated or source
        if generated:
            next_root[generated] = root

        history.setdefault(root, []).append(
            {
                "round_index": round_index,
                "source_hypothesis_id": source,
                "current_hypothesis_id": current,
                "action": controller.get("action"),
                "generation_route": rec.get("route"),
                "generation_decision": rec.get("decision"),
                "post_verification_decision": (
                    post_by_h.get(current, {})
                    .get("post_verification_decision")
                ),
                "epistemic_state": (
                    state_by_h.get(current, {})
                    .get("final_epistemic_state")
                ),
            }
        )

    for row in plan.get("decisions", []):
        action = str(row.get("action") or "")
        if action not in {
            "REQUEST_GRAPH_RETRAVERSAL",
            "STOP",
            "KEEP",
        }:
            continue
        source = str(row.get("current_hypothesis_id") or "")
        if source in recorded_sources:
            continue
        root = str(
            root_by_source.get(
                source,
                row.get("root_hypothesis_id") or source,
            )
        )
        history.setdefault(root, []).append(
            {
                "round_index": round_index,
                "source_hypothesis_id": source,
                "current_hypothesis_id": source,
                "action": action,
                "generation_route": action,
                "generation_decision": "TERMINAL_CONTROLLER_ACTION",
                "post_verification_decision": None,
                "epistemic_state": row.get("current_epistemic_state"),
            }
        )

    return next_root



def _verification_results_per_query(
    *,
    base: int,
    maximum: int,
    multiplier: int,
) -> int:
    base = int(base)
    maximum = int(maximum)
    multiplier = int(multiplier)

    if base <= 0:
        raise ValueError(
            "base results-per-query must be positive"
        )
    if maximum < base:
        raise ValueError(
            "maximum results-per-query must be >= base"
        )
    if multiplier <= 0:
        raise ValueError(
            "retrieval multiplier must be positive"
        )

    return min(
        maximum,
        max(
            base,
            base * multiplier,
        ),
    )



def _run_fresh_verification(
    *,
    args,
    round_dir: Path,
    portfolio_path: Path,
    previous_query: Path,
    previous_external: Path,
    previous_prior: Path,
    previous_binding: Path,
    previous_completion: Path,
    retrieval_multiplier: int,
    reuse_query_plan: bool = False,
) -> dict[str, Path]:
    ext_prefix = round_dir / "external"
    results_per_query = (
        _verification_results_per_query(
            base=args.results_per_query,
            maximum=args.max_results_per_query,
            multiplier=retrieval_multiplier,
        )
    )

    cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_external_novelty",
        "--portfolio", str(portfolio_path),
        "--domain-profile", args.domain_profile,
        "--model", args.critic_model,
        "--api-key-env", args.api_key_env,
        "--provider-plan", str(args.provider_plan),
        "--results-per-query", str(results_per_query),
        "--max-ranked-works", str(args.max_ranked_works),
        "--parse-retries", str(args.parse_retries),
        "--pre-review-coverage-shadow",
        "--downstream-gate-shadow",
        "--source-bound-topology-shadow",
        "--prior-art-memory-query-plan", str(previous_query),
        "--prior-art-memory-report", str(previous_external),
        "--prior-art-memory-packet", str(previous_prior),
        "--output-prefix", str(ext_prefix),
        "--save-prompts",
    ]
    if reuse_query_plan:
        cmd += [
            "--reuse-query-plan",
            str(previous_query),
        ]

    if args.base_url:
        cmd += ["--base-url", args.base_url]

    run(
        f"adaptive round {round_dir.name} fresh external novelty",
        cmd,
        args.output_dir,
    )

    query = round_dir / "external.claims_queries.json"
    external = round_dir / "external.report.json"
    prior = round_dir / "external.prior_art.json"
    binding = round_dir / "external.atomic_source_binding.json"
    completion = round_dir / "external.topology_completion_shadow.json"

    if reuse_query_plan:
        _assert_reused_query_plan_identity(
            previous=load(previous_query),
            current=load(query),
        )

        if not previous_binding.is_file():
            raise RuntimeError(
                "RETRIEVE_MORE requires previous "
                "atomic source-binding artifact: "
                + str(previous_binding)
            )

        if not previous_completion.is_file():
            raise RuntimeError(
                "RETRIEVE_MORE requires previous "
                "topology-completion artifact: "
                + str(previous_completion)
            )

        shutil.copy2(
            previous_binding,
            binding,
        )
        shutil.copy2(
            previous_completion,
            completion,
        )

    lower_report, lower_reviews, lower_packet = lower_order(
        args=args,
        portfolio=portfolio_path,
        query_plan=query,
        external_report=external,
        prefix=round_dir / "lower",
        results_per_query=results_per_query,
    )
    topology_report = round_dir / "topology.report.json"
    run_topology(
        args=args,
        query_plan=query,
        saturation_report=lower_report,
        external_report=external,
        source_binding=binding,
        output=topology_report,
    )
    fulltext = round_dir / "fulltext.report.json"
    run_fulltext(
        args=args,
        query_plan=query,
        reviews=lower_reviews,
        packet=lower_packet,
        topology_report=topology_report,
        output=fulltext,
    )
    aggregation = round_dir / "residual_aggregation.report.json"
    run_aggregation(
        args=args,
        topology_report=topology_report,
        reviews=lower_reviews,
        fulltext_report=fulltext,
        output=aggregation,
    )

    cohort = round_dir / "cohort_audit.report.json"
    run(
        f"adaptive round {round_dir.name} residual cohort audit",
        [
            sys.executable,
            "-m",
            "scripts.discovery.run_residual_novelty_cohort_audit_shadow",
            "--aggregation", str(aggregation),
            "--query-plan", str(query),
            "--topology-completion", str(completion),
            "--source-binding", str(binding),
            "--topology-report", str(topology_report),
            "--output", str(cohort),
        ],
        args.output_dir,
        allow_code_2=True,
    )

    return {
        "query": query,
        "external": external,
        "prior": prior,
        "binding": binding,
        "completion": completion,
        "aggregation": aggregation,
        "cohort": cohort,
    }



def _audit_failed_hypothesis_ids(
    decisions: dict[str, Any],
) -> set[str]:
    return {
        str(row.get("hypothesis_id") or "")
        for row in decisions.get("rows", [])
        if (
            str(row.get("hypothesis_id") or "")
            and str(
                row.get("post_verification_decision") or ""
            )
            == "HOLD_COHORT_AUDIT_FAILED"
        )
    }


def _mark_audit_failure_states(
    state: dict[str, Any],
    failed_hypothesis_ids: set[str],
) -> dict[str, Any]:
    if not failed_hypothesis_ids:
        return state

    body = json.loads(canonical_json(state))

    for row in body.get("hypotheses", []):
        hid = str(row.get("hypothesis_id") or "")
        if hid not in failed_hypothesis_ids:
            continue

        row["pre_audit_epistemic_state"] = row.get(
            "final_epistemic_state"
        )
        row["pre_audit_state_reason"] = row.get(
            "state_reason"
        )
        row["final_epistemic_state"] = "AUDIT_FAILURE_HOLD"
        row["state_reason"] = (
            "cohort_audit_failed_fail_closed"
        )
        row["authority_ready_candidate_shadow"] = False
        row["authority_readiness_state"] = (
            "AUDIT_FAILURE_HOLD"
        )

    counts = Counter(
        str(row.get("final_epistemic_state") or "")
        for row in body.get("hypotheses", [])
        if str(row.get("final_epistemic_state") or "")
    )
    body["state_counts"] = dict(
        sorted(counts.items())
    )
    body["audit_fail_closed"] = True
    body["audit_failed_hypothesis_ids"] = sorted(
        failed_hypothesis_ids
    )

    return body


def _controller_summary_status(
    *,
    audit_failures: list[dict[str, Any]],
    graph_handoffs: list[dict[str, Any]],
    final_effective_count: int,
    terminal_stops: list[dict[str, Any]],
) -> str:
    if audit_failures:
        return "COMPLETE_FAIL_CLOSED_AUDIT"

    if graph_handoffs:
        return "COMPLETE_WITH_GRAPH_RETRAVERSAL_HANDOFF"

    if final_effective_count > 0:
        return "COMPLETE_WITH_EFFECTIVE_CANDIDATES"

    if terminal_stops:
        return "COMPLETE_SEARCH_EXHAUSTED"

    return "COMPLETE"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--seed-closed-loop-dir", required=True, type=Path)
    p.add_argument("--provider-plan", required=True, type=Path)
    p.add_argument("--domain-profile", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--critic-model", required=True)
    p.add_argument("--controller-model", default=None)
    p.add_argument("--disable-controller-llm", action="store_true")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--results-per-query", type=int, default=12)
    p.add_argument("--max-results-per-query", type=int, default=30)
    p.add_argument("--max-ranked-works", type=int, default=12)
    p.add_argument("--parse-retries", type=int, default=3)
    p.add_argument("--max-fulltext-candidates", type=int, default=4)
    p.add_argument("--max-excerpt-chars", type=int, default=24000)
    p.add_argument("--max-controller-rounds", type=int, default=4)
    p.add_argument(
        "--max-local-attempts-per-lineage",
        type=int,
        default=4,
    )
    p.add_argument("--output-dir", required=True, type=Path)
    return p


def main() -> int:
    args = parser().parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.seed_closed_loop_dir = (
        args.seed_closed_loop_dir.expanduser().resolve()
    )
    args.provider_plan = args.provider_plan.expanduser().resolve()

    context = load_context_source(args.context)

    seed_portfolio_path = args.seed_closed_loop_dir / "gen1.portfolio.json"
    seed_generation_path = args.seed_closed_loop_dir / "generation.report.json"
    seed_effective_path = (
        args.seed_closed_loop_dir / "effective_gen1.portfolio.json"
    )
    seed_decisions_path = (
        args.seed_closed_loop_dir / "post_verification_decisions.json"
    )
    seed_query = args.seed_closed_loop_dir / "gen1/external.claims_queries.json"
    seed_external = args.seed_closed_loop_dir / "gen1/external.report.json"
    seed_prior = args.seed_closed_loop_dir / "gen1/external.prior_art.json"
    seed_binding = (
        args.seed_closed_loop_dir
        / "gen1/external.atomic_source_binding.json"
    )
    seed_completion = (
        args.seed_closed_loop_dir
        / "gen1/external.topology_completion_shadow.json"
    )
    seed_aggregation = (
        args.seed_closed_loop_dir / "gen1/residual_aggregation.report.json"
    )

    for path in (
        seed_portfolio_path,
        seed_generation_path,
        seed_effective_path,
        seed_decisions_path,
        seed_query,
        seed_external,
        seed_prior,
        seed_binding,
        seed_completion,
        seed_aggregation,
    ):
        if not path.is_file():
            raise RuntimeError(
                "Adaptive controller requires complete Stage-7.75 seed artifacts; "
                f"missing {path}"
            )

    seed_portfolio = HypothesisPortfolio.model_validate_json(
        seed_portfolio_path.read_text(encoding="utf-8")
    )
    seed_effective = HypothesisPortfolio.model_validate_json(
        seed_effective_path.read_text(encoding="utf-8")
    )
    seed_generation = load(seed_generation_path)
    seed_decisions = load(seed_decisions_path)
    seed_external_payload = load(seed_external)
    seed_state = compile_residual_epistemic_state(
        portfolio=seed_portfolio.model_dump(mode="json"),
        query_plan=load(seed_query),
        external_report=seed_external_payload,
        aggregation=load(seed_aggregation),
    )

    root_by_h, history = _seed_history(
        seed_generation=seed_generation,
        seed_decisions=seed_decisions,
        seed_state=seed_state,
    )

    effective_cards = _card_map(seed_effective)
    active_cards = [
        card
        for card in seed_portfolio.hypotheses
        if str(card.hypothesis_id) not in effective_cards
    ]
    current_portfolio = seed_portfolio.model_copy(
        update={
            "hypotheses": active_cards,
            "abstention_reason": (
                None if active_cards else "No unresolved Stage-7.75 candidate."
            ),
        }
    )

    current_query = seed_query
    current_external = seed_external
    current_prior = seed_prior
    current_binding = seed_binding
    current_completion = seed_completion
    current_aggregation = seed_aggregation
    current_state = seed_state

    backend = None
    if not args.disable_controller_llm:
        backend = InstructorOpenAICompatibleAdaptiveControllerBackend(
            model=args.controller_model or args.critic_model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            parse_retries=args.parse_retries,
            telemetry_path=str(
                args.output_dir / "controller.telemetry.jsonl"
            ),
        )

    round_summaries: list[dict[str, Any]] = []
    graph_handoffs: list[dict[str, Any]] = []
    terminal_stops: list[dict[str, Any]] = []
    audit_failures: list[dict[str, Any]] = []

    for adaptive_round in range(1, args.max_controller_rounds + 1):
        if not current_portfolio.hypotheses:
            break

        round_index = adaptive_round + 1
        round_dir = args.output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        current_external_obj = ExternalNoveltyReport.model_validate_json(
            current_external.read_text(encoding="utf-8")
        )

        plan = build_controller_plan(
            context=context,
            portfolio=current_portfolio,
            epistemic_state_report=current_state,
            external_report=load(current_external),
            root_by_hypothesis=root_by_h,
            history_by_root=history,
            round_index=round_index,
            max_local_attempts_per_lineage=(
                args.max_local_attempts_per_lineage
            ),
            backend=backend,
        )
        plan_payload = plan.model_dump(mode="json")
        write(round_dir / "controller.plan.json", plan_payload)

        current_cards = _card_map(current_portfolio)
        generated_cards: list[Any] = []
        generation_records: list[dict[str, Any]] = []

        round_actions = [
            str(row.action)
            for row in plan.decisions
        ]
        retrieval_only_round = (
            _validate_retrieval_batch(
                round_actions
            )
        )
        retrieval_requested = False

        for decision_model in plan.decisions:
            decision = decision_model.model_dump(mode="json")
            hid = decision["current_hypothesis_id"]
            action = decision["action"]
            original = current_cards[hid]
            root = decision["root_hypothesis_id"]
            attempt_history = history.get(root, [])

            if action == "KEEP":
                effective_cards[hid] = original
                generation_records.append(
                    {
                        "source_hypothesis_id": hid,
                        "route": "KEEP",
                        "decision": "CARRIED_EFFECTIVE_RESIDUAL",
                        "generated_hypothesis_id": hid,
                        "reason_codes": [
                            "controller_preserved_surviving_residual"
                        ],
                    }
                )
                continue

            if action == "REQUEST_GRAPH_RETRAVERSAL":
                graph_handoffs.append(
                    {
                        **decision,
                        "round_index": round_index,
                        "attempt_history": attempt_history,
                        "handoff_state": "REQUESTED_NOT_EXECUTED_V1",
                    }
                )
                generation_records.append(
                    {
                        "source_hypothesis_id": hid,
                        "route": action,
                        "decision": "HELD_FOR_GRAPH_RETRAVERSAL",
                        "generated_hypothesis_id": None,
                        "reason_codes": [
                            "local_context_exhausted",
                            "graph_retraversal_not_executed_in_v1",
                        ],
                    }
                )
                continue

            if action == "STOP":
                terminal_stops.append(
                    {
                        **decision,
                        "round_index": round_index,
                        "attempt_history": attempt_history,
                    }
                )
                generation_records.append(
                    {
                        "source_hypothesis_id": hid,
                        "route": action,
                        "decision": "BOUNDED_SEARCH_STOPPED",
                        "generated_hypothesis_id": None,
                        "reason_codes": [
                            "bounded_adaptive_search_exhausted"
                        ],
                    }
                )
                continue

            if action == "RETRIEVE_MORE":
                retrieval_requested = True
                generated_cards.append(original)
                generation_records.append(
                    {
                        "source_hypothesis_id": hid,
                        "route": action,
                        "decision": "CARRIED_FORWARD_FOR_DEEPER_RETRIEVAL",
                        "generated_hypothesis_id": hid,
                        "reason_codes": [
                            "hypothesis_unchanged",
                            "evidence_budget_escalated",
                        ],
                    }
                )
                continue

            if action in {
                "SAME_PREMISE_SHARPEN",
                "EVIDENCE_REAXIS",
            }:
                subplan = _single_feedback_plan(
                    decision=decision,
                    external_report_id=current_external_obj.report_id,
                )
                one = current_portfolio.model_copy(
                    update={
                        "hypotheses": [original],
                        "abstention_reason": None,
                    }
                )
                subreport, subportfolio = run_feedback_generation(
                    context=context,
                    portfolio=one,
                    external=current_external_obj,
                    plan=subplan,
                    model=args.model,
                    critic_model=args.critic_model,
                    api_key_env=args.api_key_env,
                    base_url=args.base_url,
                    output_dir=round_dir / f"generation_{hid.split(':')[-1]}",
                )
                generated_cards.extend(subportfolio.hypotheses)
                generation_records.extend(subreport.get("records", []))
                continue

            if action == "AXIS_MUTATION":
                candidate, record = generate_axis_mutation(
                    context=context,
                    original=original,
                    decision=decision_model,
                    attempt_history=attempt_history,
                    model=args.model,
                    critic_model=args.critic_model,
                    api_key_env=args.api_key_env,
                    base_url=args.base_url,
                    output_prefix=str(
                        round_dir
                        / f"axis_mutation_{hid.split(':')[-1]}"
                    ),
                )
                if candidate is not None:
                    generated_cards.append(candidate)
                generation_records.append(record)
                continue

            raise RuntimeError(
                "Unhandled adaptive action: " + str(action)
            )

        next_portfolio = _combined_portfolio(
            template=current_portfolio,
            cards=generated_cards,
            round_index=round_index,
            preserve_portfolio_id=(
                retrieval_only_round
            ),
        )

        if retrieval_only_round:
            previous_ids = sorted(
                str(card.hypothesis_id)
                for card
                in current_portfolio.hypotheses
            )
            next_ids = sorted(
                str(card.hypothesis_id)
                for card
                in next_portfolio.hypotheses
            )

            if previous_ids != next_ids:
                raise RuntimeError(
                    "RETRIEVE_MORE changed "
                    "hypothesis population"
                )
        next_portfolio_path = round_dir / "candidate.portfolio.json"
        write(next_portfolio_path, next_portfolio)

        generation_report = _generation_report(
            source_portfolio_id=current_portfolio.portfolio_id,
            output_portfolio_id=next_portfolio.portfolio_id,
            round_index=round_index,
            records=generation_records,
        )
        generation_report_path = round_dir / "generation.report.json"
        write(generation_report_path, generation_report)

        if not next_portfolio.hypotheses:
            _append_history(
                history=history,
                root_by_source=root_by_h,
                plan=plan_payload,
                generation=generation_report,
                decisions={"rows": []},
                state=current_state,
                round_index=round_index,
            )
            round_summaries.append(
                {
                    "round_index": round_index,
                    "action_counts": plan.action_counts,
                    "generated_candidate_count": 0,
                    "status": "NO_CANDIDATE_TO_REVERIFY",
                }
            )
            current_portfolio = next_portfolio
            break

        artifacts = _run_fresh_verification(
            args=args,
            round_dir=round_dir,
            portfolio_path=next_portfolio_path,
            previous_query=current_query,
            previous_external=current_external,
            previous_prior=current_prior,
            previous_binding=current_binding,
            previous_completion=current_completion,
            retrieval_multiplier=(
                2 if retrieval_requested else 1
            ),
            reuse_query_plan=(
                retrieval_only_round
            ),
        )

        cohort_payload = load(artifacts["cohort"])

        next_state = compile_residual_epistemic_state(
            portfolio=next_portfolio.model_dump(mode="json"),
            query_plan=load(artifacts["query"]),
            external_report=load(artifacts["external"]),
            aggregation=load(artifacts["aggregation"]),
        )

        decisions = consolidate_post_verification_decisions(
            gen1_portfolio=next_portfolio.model_dump(mode="json"),
            generation_report=generation_report,
            query_plan=load(artifacts["query"]),
            external_report=load(artifacts["external"]),
            aggregation=load(artifacts["aggregation"]),
            cohort_audit=cohort_payload,
        )

        audit_failed_ids = _audit_failed_hypothesis_ids(
            decisions
        )
        if audit_failed_ids:
            next_state = _mark_audit_failure_states(
                next_state,
                audit_failed_ids,
            )
            audit_failures.append(
                {
                    "round_index": round_index,
                    "hypothesis_ids": sorted(
                        audit_failed_ids
                    ),
                    "cohort_failures": list(
                        cohort_payload.get(
                            "failures",
                            [],
                        )
                    ),
                    "cohort_warnings": list(
                        cohort_payload.get(
                            "warnings",
                            [],
                        )
                    ),
                }
            )

        write(
            round_dir / "epistemic_state.json",
            next_state,
        )
        write(
            round_dir / "post_verification_decisions.json",
            decisions,
        )

        decision_by_h = _row_map(decisions)
        for card in next_portfolio.hypotheses:
            hid = str(card.hypothesis_id)
            if decision_by_h.get(hid, {}).get(
                "advance_to_effective_gen1"
            ):
                effective_cards[hid] = card

        next_root = _append_history(
            history=history,
            root_by_source=root_by_h,
            plan=plan_payload,
            generation=generation_report,
            decisions=decisions,
            state=next_state,
            round_index=round_index,
        )

        active = [
            card
            for card in next_portfolio.hypotheses
            if (
                str(card.hypothesis_id)
                not in effective_cards
                and str(card.hypothesis_id)
                not in audit_failed_ids
            )
        ]
        active_ids = {str(card.hypothesis_id) for card in active}
        root_by_h = {
            hid: root
            for hid, root in next_root.items()
            if hid in active_ids
        }

        round_summaries.append(
            {
                "round_index": round_index,
                "action_counts": plan.action_counts,
                "generated_candidate_count": len(
                    next_portfolio.hypotheses
                ),
                "effective_added_count": sum(
                    1
                    for row in decisions.get("rows", [])
                    if row.get("advance_to_effective_gen1")
                ),
                "decision_counts": decisions.get(
                    "decision_counts", {}
                ),
                "epistemic_state_counts": next_state.get(
                    "state_counts", {}
                ),
                "retrieval_multiplier": (
                    2 if retrieval_requested else 1
                ),
                "query_plan_reused": (
                    retrieval_only_round
                ),
                "verification_results_per_query": (
                    _verification_results_per_query(
                        base=args.results_per_query,
                        maximum=args.max_results_per_query,
                        multiplier=(
                            2
                            if retrieval_requested
                            else 1
                        ),
                    )
                ),
                "audit_fail_closed_count": len(
                    audit_failed_ids
                ),
                "status": (
                    "COMPLETE_FAIL_CLOSED_AUDIT"
                    if audit_failed_ids
                    else "COMPLETE"
                ),
            }
        )

        current_portfolio = next_portfolio.model_copy(
            update={
                "hypotheses": active,
                "abstention_reason": (
                    None if active else "No unresolved adaptive candidate."
                ),
            }
        )
        current_query = artifacts["query"]
        current_external = artifacts["external"]
        current_prior = artifacts["prior"]
        current_binding = artifacts["binding"]
        current_completion = artifacts["completion"]
        current_aggregation = artifacts["aggregation"]
        current_state = next_state

    if current_portfolio.hypotheses:
        for card in current_portfolio.hypotheses:
            hid = str(card.hypothesis_id)
            root = root_by_h.get(hid, hid)
            history.setdefault(root, []).append(
                {
                    "round_index": args.max_controller_rounds + 2,
                    "source_hypothesis_id": hid,
                    "current_hypothesis_id": hid,
                    "action": "STOP",
                    "generation_route": "BUDGET_EXHAUSTED",
                    "generation_decision": "BOUNDED_SEARCH_STOPPED",
                    "post_verification_decision": None,
                    "epistemic_state": None,
                }
            )
            terminal_stops.append(
                {
                    "root_hypothesis_id": root,
                    "current_hypothesis_id": hid,
                    "reason": "adaptive_round_budget_exhausted",
                }
            )

    final_cards = list(effective_cards.values())
    final_portfolio = seed_portfolio.model_copy(
        update={
            "portfolio_id": stable_id(
                "adaptive_discovery_effective_portfolio",
                seed_portfolio.portfolio_id,
                sorted(effective_cards),
            ),
            "hypotheses": final_cards,
            "abstention_reason": (
                None
                if final_cards
                else "No candidate survived bounded adaptive search."
            ),
        }
    )
    write(
        args.output_dir / "adaptive_effective.portfolio.json",
        final_portfolio,
    )

    history_payload = {
        "schema_version": "adaptive-discovery-history-v1",
        "root_lineage_count": len(history),
        "history_by_root": history,
        "external_prior_art_as_positive_premise": False,
        "graph_retraversal_executes_in_v1": False,
    }
    write(
        args.output_dir / "adaptive_search.history.json",
        history_payload,
    )

    handoff = {
        "schema_version": "adaptive-graph-retraversal-handoff-v1",
        "request_count": len(graph_handoffs),
        "requests": graph_handoffs,
        "execution_performed": False,
        "reason": (
            "Graph/context reconstruction is an upstream search-space mutation "
            "and remains an explicit handoff in controller v1."
        ),
    }
    write(
        args.output_dir / "graph_retraversal.handoff.json",
        handoff,
    )

    all_actions = Counter()
    for row in round_summaries:
        all_actions.update(row.get("action_counts", {}))

    summary = {
        "schema_version": "adaptive-discovery-controller-summary-v1",
        "seed_portfolio_id": seed_portfolio.portfolio_id,
        "seed_effective_count": len(seed_effective.hypotheses),
        "adaptive_rounds_completed": len(round_summaries),
        "rounds": round_summaries,
        "action_counts": dict(sorted(all_actions.items())),
        "final_effective_count": len(final_cards),
        "final_effective_hypothesis_ids": sorted(effective_cards),
        "graph_retraversal_request_count": len(graph_handoffs),
        "terminal_stop_count": len(terminal_stops),
        "audit_fail_closed_count": len(audit_failures),
        "audit_fail_closed_events": audit_failures,
        "max_controller_rounds": args.max_controller_rounds,
        "max_local_attempts_per_lineage": (
            args.max_local_attempts_per_lineage
        ),
        "controller_llm_advisory_enabled": backend is not None,
        "external_prior_art_as_positive_premise": False,
        "controller_has_novelty_authority": False,
        "controller_has_generation_authority": False,
        "graph_retraversal_executes_in_v1": False,
        "n10_research_selection_authority": False,
        "stage8_input_changed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
        "status": _controller_summary_status(
            audit_failures=audit_failures,
            graph_handoffs=graph_handoffs,
            final_effective_count=len(final_cards),
            terminal_stops=terminal_stops,
        ),
    }
    write(
        args.output_dir / "adaptive_controller.summary.json",
        summary,
    )

    print()
    print("===== ADAPTIVE DISCOVERY CONTROLLER V1 =====")
    print("rounds:", len(round_summaries))
    print("actions:", summary["action_counts"])
    print("seed effective:", summary["seed_effective_count"])
    print("final effective:", summary["final_effective_count"])
    print(
        "graph retraversal requests:",
        summary["graph_retraversal_request_count"],
    )
    print("GRAPH_RETRAVERSAL_EXECUTED=False")
    print("STAGE8_INPUT_CHANGED=False")
    print("PRODUCTION_SELECTION_CHANGED=False")
    print("N10_RESEARCH_SELECTION_AUTHORITY=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
