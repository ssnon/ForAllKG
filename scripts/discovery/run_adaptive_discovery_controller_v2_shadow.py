from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from pipeline_core.discovery.adaptive_graph_retraversal import (
    build_retraversal_request,
)
from pipeline_core.discovery.adaptive_graph_retraversal_generation import (
    generate_retraversal_hypothesis,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.scientific_portfolio_closed_loop import (
    compile_residual_epistemic_state,
)
from pipeline_core.discovery.sers_closed_loop_decision_consolidation_v2 import (
    build_effective_gen1_portfolio,
    consolidate_post_verification_decisions,
)
from scripts.discovery.run_adaptive_discovery_controller_shadow import (
    _generation_report,
    _resolve_stage775_seed_root,
    _run_fresh_verification,
)
from scripts.discovery.run_scientific_portfolio_closed_loop_shadow import (
    load,
    run,
    write,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--grounding-traversal", required=True, type=Path)
    p.add_argument("--seed-closed-loop-dir", required=True, type=Path)
    p.add_argument(
        "--initial-local-controller-dir",
        type=Path,
        default=None,
        help=(
            "Optional completed Adaptive-v1 controller directory to reuse "
            "verbatim as context epoch 0. When supplied, v2 MUST NOT rerun "
            "the initial local search."
        ),
    )
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
    p.add_argument("--max-graph-retraversals", type=int, default=1)
    p.add_argument(
        "--retraversal-objective",
        default="explain_connection",
    )
    p.add_argument("--retraversal-candidate-top-k", type=int, default=32)
    p.add_argument("--retraversal-selected-top-k", type=int, default=6)
    p.add_argument("--retraversal-node-map-k", type=int, default=30)
    p.add_argument("--retraversal-endpoint-pair-k", type=int, default=24)
    p.add_argument("--retraversal-max-depth-increment", type=int, default=2)
    p.add_argument("--retraversal-max-depth-cap", type=int, default=16)
    p.add_argument(
        "--retraversal-min-new-edge-fraction",
        type=float,
        default=0.35,
    )
    p.add_argument(
        "--retraversal-min-new-paper-fraction",
        type=float,
        default=0.25,
    )
    p.add_argument(
        "--retraversal-max-prior-edge-jaccard",
        type=float,
        default=0.85,
    )
    p.add_argument(
        "--retraversal-max-selected-edge-jaccard",
        type=float,
        default=0.85,
    )
    p.add_argument(
        "--retraversal-paper-expansion-reserve",
        type=int,
        default=2,
    )
    p.add_argument(
        "--retraversal-allow-top-n-fallback",
        action="store_true",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    return p


def _context(path: Path) -> HypothesisContext:
    return HypothesisContext.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _portfolio(path: Path) -> HypothesisPortfolio:
    return HypothesisPortfolio.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _run_local_epoch(
    *,
    args: argparse.Namespace,
    context_path: Path,
    seed_dir: Path,
    output_dir: Path,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_adaptive_discovery_controller_shadow",
        "--context", str(context_path),
        "--seed-closed-loop-dir", str(seed_dir),
        "--provider-plan", str(args.provider_plan),
        "--domain-profile", str(args.domain_profile),
        "--model", str(args.model),
        "--critic-model", str(args.critic_model),
        "--controller-model", str(args.controller_model or args.critic_model),
        "--api-key-env", str(args.api_key_env),
        "--results-per-query", str(args.results_per_query),
        "--max-results-per-query", str(args.max_results_per_query),
        "--max-ranked-works", str(args.max_ranked_works),
        "--parse-retries", str(args.parse_retries),
        "--max-fulltext-candidates", str(args.max_fulltext_candidates),
        "--max-excerpt-chars", str(args.max_excerpt_chars),
        "--max-controller-rounds", str(args.max_controller_rounds),
        "--max-local-attempts-per-lineage",
        str(args.max_local_attempts_per_lineage),
        "--output-dir", str(output_dir),
    ]
    if args.disable_controller_llm:
        cmd.append("--disable-controller-llm")
    if args.base_url:
        cmd += ["--base-url", str(args.base_url)]
    run(
        f"adaptive v2 local epoch {output_dir.parent.name}",
        cmd,
        args.output_dir,
    )


def _validate_initial_local_controller_dir(
    path: Path,
) -> None:
    required = (
        "adaptive_controller.summary.json",
        "adaptive_effective.portfolio.json",
        "graph_retraversal.handoff.json",
        "adaptive_search.history.json",
    )
    missing = [
        str(path / name)
        for name in required
        if not (path / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Adaptive v2 initial-local-controller reuse requires a complete "
            "Stage-7.76 artifact set; missing "
            + repr(missing)
        )


def _find_card_with_portfolio_id(
    *,
    seed_dir: Path,
    local_dir: Path,
    hypothesis_id: str,
) -> tuple[Any, str] | None:
    candidates = sorted(
        local_dir.glob("round_*/candidate.portfolio.json"),
        reverse=True,
    )
    candidates.append(seed_dir / "gen1.portfolio.json")
    for path in candidates:
        if not path.is_file():
            continue
        portfolio = _portfolio(path)
        for card in portfolio.hypotheses:
            if str(card.hypothesis_id) == str(hypothesis_id):
                return card, str(portfolio.portfolio_id)
    return None



def _materialized_hypothesis_ids(
    *,
    seed_dir: Path,
    local_dir: Path,
) -> set[str]:
    ids: set[str] = set()

    candidates = sorted(
        local_dir.glob("round_*/candidate.portfolio.json"),
        reverse=True,
    )
    candidates.append(seed_dir / "gen1.portfolio.json")

    for path in candidates:
        if not path.is_file():
            continue
        portfolio = _portfolio(path)
        ids.update(
            str(card.hypothesis_id)
            for card in portfolio.hypotheses
        )

    return ids


def _latest_materialized_history_head(
    *,
    rows: list[dict[str, Any]],
    materialized_hypothesis_ids: set[str],
) -> tuple[str | None, int | None, list[str]]:
    # Generation records may retain generated_hypothesis_id for candidates
    # rejected by grounding/task-preservation checks. Those IDs are useful
    # diagnostics but are not valid lineage heads for graph retraversal.
    materialized = {
        str(value)
        for value in materialized_hypothesis_ids
        if str(value)
    }

    stale_tail: list[str] = []

    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        current = str(
            row.get("current_hypothesis_id") or ""
        )
        if not current:
            continue

        if current in materialized:
            return current, index, list(reversed(stale_tail))

        if current not in stale_tail:
            stale_tail.append(current)

    return None, None, list(reversed(stale_tail))


def _verification_artifacts(
    *,
    seed_dir: Path,
    local_dir: Path,
    hypothesis_id: str,
) -> dict[str, Path]:
    for round_dir in sorted(
        [
            path
            for path in local_dir.glob("round_*")
            if path.is_dir()
        ],
        reverse=True,
    ):
        report = round_dir / "external.report.json"
        if not report.is_file():
            continue
        payload = load(report)
        ids = {
            str(row.get("hypothesis_id") or "")
            for row in payload.get("cards", [])
        }
        if str(hypothesis_id) not in ids:
            continue
        return {
            "query": round_dir / "external.claims_queries.json",
            "external": report,
            "prior": round_dir / "external.prior_art.json",
            "binding": round_dir / "external.atomic_source_binding.json",
            "completion": round_dir / "external.topology_completion_shadow.json",
        }

    return {
        "query": seed_dir / "gen1/external.claims_queries.json",
        "external": seed_dir / "gen1/external.report.json",
        "prior": seed_dir / "gen1/external.prior_art.json",
        "binding": seed_dir / "gen1/external.atomic_source_binding.json",
        "completion": seed_dir / "gen1/external.topology_completion_shadow.json",
    }


def _latest_controller_decision(
    *,
    local_dir: Path,
    hypothesis_id: str,
) -> dict[str, Any]:
    for round_dir in sorted(
        [
            path
            for path in local_dir.glob("round_*")
            if path.is_dir()
        ],
        reverse=True,
    ):
        plan = round_dir / "controller.plan.json"
        if not plan.is_file():
            continue
        payload = load(plan)
        for row in payload.get("decisions", []):
            if (
                str(row.get("current_hypothesis_id") or "")
                == str(hypothesis_id)
            ):
                return dict(row)
    return {}


def _eligible_request(
    *,
    local_dir: Path,
    effective_ids: set[str],
    seed_dir: Path | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Resolve one v2 graph request from explicit v1 handoff or exhaustion.

    v2a deliberately executes at most one lineage per context reset. Multiple
    simultaneous graph requests are returned as deferred so that distinct
    contexts are never mixed into one HypothesisPortfolio.
    """

    handoff_path = local_dir / "graph_retraversal.handoff.json"
    handoff = (
        load(handoff_path)
        if handoff_path.is_file()
        else {"requests": []}
    )
    requests = [
        dict(row)
        for row in handoff.get("requests", [])
        if isinstance(row, dict)
    ]
    if len(requests) == 1:
        return requests[0], []
    if len(requests) > 1:
        return None, requests

    history_path = local_dir / "adaptive_search.history.json"
    if not history_path.is_file():
        return None, []
    history = load(history_path)

    synthesized: list[dict[str, Any]] = []

    # Backward-compatible helper contract:
    # production/v2 main always supplies seed_dir, so materialized-lineage
    # validation is enabled. Direct unit callers may omit it.
    materialization_guard_enabled = seed_dir is not None
    materialized_ids = (
        _materialized_hypothesis_ids(
            seed_dir=seed_dir,
            local_dir=local_dir,
        )
        if seed_dir is not None
        else set()
    )

    for root, rows in (history.get("history_by_root", {}) or {}).items():
        if not isinstance(rows, list):
            continue
        current_rows = [row for row in rows if isinstance(row, dict)]
        if not current_rows:
            continue

        if materialization_guard_enabled:
            (
                current_id,
                materialized_row_index,
                stale_tail_ids,
            ) = _latest_materialized_history_head(
                rows=current_rows,
                materialized_hypothesis_ids=materialized_ids,
            )

            # Fail closed in production: a lineage with only rejected /
            # non-materialized generated identities cannot seed graph
            # retraversal.
            if current_id is None:
                continue
        else:
            # Compatibility mode is only for direct helper/unit callers.
            # The production main path always passes seed_dir.
            current_id = str(
                current_rows[-1].get("current_hypothesis_id") or ""
            )
            materialized_row_index = len(current_rows) - 1
            stale_tail_ids = []

        if current_id in effective_ids:
            continue

        last_state = ""
        for row in reversed(current_rows):
            value = str(row.get("epistemic_state") or "")
            if value:
                last_state = value
                break

        actions = {
            str(row.get("action") or "")
            for row in current_rows
        }

        eligible = False
        reason = ""
        if (
            last_state == "PRIOR_ART_BACKED_OR_NO_RESIDUAL"
            and "AXIS_MUTATION" in actions
        ):
            eligible = True
            reason = "known_region_local_search_exhausted"
        elif last_state in {
            "UNRESOLVED_EVIDENCE_GAP",
            "UNRESOLVED_TOPOLOGY_GAP",
        }:
            eligible = True
            reason = "unresolved_local_search_exhausted"

        if not eligible:
            continue

        decision = _latest_controller_decision(
            local_dir=local_dir,
            hypothesis_id=current_id,
        )
        synthesized.append(
            {
                **decision,
                "root_hypothesis_id": (
                    decision.get("root_hypothesis_id") or str(root)
                ),
                "current_hypothesis_id": current_id,
                "current_epistemic_state": (
                    decision.get("current_epistemic_state") or last_state
                ),
                "attempt_history": current_rows,
                "handoff_state": "SYNTHESIZED_FROM_V1_EXHAUSTION",
                "v2_request_reason": reason,
                "materialized_lineage_head_recovered": True,
                "materialized_history_row_index": materialized_row_index,
                "stale_history_tail_count": len(stale_tail_ids),
                "stale_history_tail_hypothesis_ids": stale_tail_ids,
            }
        )

    if len(synthesized) == 1:
        return synthesized[0], []
    if len(synthesized) > 1:
        return None, synthesized
    return None, []


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _build_next_seed(
    *,
    seed_dir: Path,
    portfolio: HypothesisPortfolio,
    generation: dict[str, Any],
    effective: HypothesisPortfolio,
    decisions: dict[str, Any],
    artifacts: dict[str, Path],
) -> None:
    write(seed_dir / "gen1.portfolio.json", portfolio)
    write(seed_dir / "generation.report.json", generation)
    write(seed_dir / "effective_gen1.portfolio.json", effective)
    write(seed_dir / "post_verification_decisions.json", decisions)
    mapping = {
        "query": "external.claims_queries.json",
        "external": "external.report.json",
        "prior": "external.prior_art.json",
        "binding": "external.atomic_source_binding.json",
        "completion": "external.topology_completion_shadow.json",
        "aggregation": "residual_aggregation.report.json",
    }
    for key, name in mapping.items():
        _copy(
            artifacts[key],
            seed_dir / "gen1" / name,
        )


def main() -> int:
    args = parser().parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.provider_plan = args.provider_plan.expanduser().resolve()
    initial_local_controller_dir = (
        args.initial_local_controller_dir.expanduser().resolve()
        if args.initial_local_controller_dir is not None
        else None
    )
    if initial_local_controller_dir is not None:
        _validate_initial_local_controller_dir(
            initial_local_controller_dir
        )
    current_context_path = args.context.expanduser().resolve()
    raw_initial_seed_dir = args.seed_closed_loop_dir.expanduser().resolve()
    (
        resolved_initial_seed_dir,
        initial_seed_source_mode,
    ) = _resolve_stage775_seed_root(raw_initial_seed_dir)
    current_seed_dir = resolved_initial_seed_dir
    prior_traversals = [args.grounding_traversal.expanduser().resolve()]

    global_effective: dict[str, dict[str, Any]] = {}
    context_events: list[dict[str, Any]] = []
    remaining_handoffs: list[dict[str, Any]] = []
    graph_count = 0
    epoch_summaries: list[dict[str, Any]] = []

    for epoch_index in range(args.max_graph_retraversals + 1):
        epoch_dir = args.output_dir / f"epoch_{epoch_index:02d}"

        if (
            epoch_index == 0
            and initial_local_controller_dir is not None
        ):
            local_dir = initial_local_controller_dir
            local_execution_mode = "REUSED_STAGE_7_76"
        else:
            local_dir = epoch_dir / "local_controller"
            local_execution_mode = "EXECUTED_IN_V2"
            _run_local_epoch(
                args=args,
                context_path=current_context_path,
                seed_dir=current_seed_dir,
                output_dir=local_dir,
            )

        local_summary = load(
            local_dir / "adaptive_controller.summary.json"
        )
        local_effective = _portfolio(
            local_dir / "adaptive_effective.portfolio.json"
        )

        for card in local_effective.hypotheses:
            hid = str(card.hypothesis_id)
            global_effective[hid] = {
                "hypothesis_id": hid,
                "context_id": str(card.source_context_id),
                "context_sha256": str(card.source_context_sha256),
                "portfolio_path": str(
                    local_dir / "adaptive_effective.portfolio.json"
                ),
                "epoch_index": epoch_index,
                "title": str(card.title),
            }

        effective_ids = set(global_effective)
        request, deferred = _eligible_request(
            seed_dir=current_seed_dir,
            local_dir=local_dir,
            effective_ids=effective_ids,
        )

        epoch_summaries.append(
            {
                "epoch_index": epoch_index,
                "context": str(current_context_path),
                "local_summary": str(
                    local_dir / "adaptive_controller.summary.json"
                ),
                "local_status": local_summary.get("status"),
                "local_action_counts": local_summary.get("action_counts", {}),
                "local_effective_count": len(local_effective.hypotheses),
                "local_execution_mode": local_execution_mode,
                "local_controller_dir": str(local_dir),
                "retraversal_candidate": bool(request),
                "deferred_retraversal_count": len(deferred),
            }
        )

        if deferred:
            remaining_handoffs.extend(
                [
                    {
                        **row,
                        "handoff_state": "DEFERRED_MULTI_LINEAGE_V2B",
                        "reason": (
                            "v2b executes one context reset at a time; "
                            "cross-context hypotheses are not merged into one "
                            "HypothesisPortfolio."
                        ),
                    }
                    for row in deferred
                ]
            )
            break

        if request is None:
            break

        if graph_count >= args.max_graph_retraversals:
            remaining_handoffs.append(
                {
                    **request,
                    "handoff_state": "DEFERRED_GRAPH_RETRAVERSAL_BUDGET",
                    "reason": "max_graph_retraversals reached",
                }
            )
            break

        old_context = _context(current_context_path)
        current_hid = str(request.get("current_hypothesis_id") or "")
        recovered = _find_card_with_portfolio_id(
            seed_dir=current_seed_dir,
            local_dir=local_dir,
            hypothesis_id=current_hid,
        )
        if recovered is None:
            raise RuntimeError(
                "Could not recover current hypothesis for graph retraversal: "
                + current_hid
            )
        original, source_portfolio_id = recovered

        history = list(request.get("attempt_history", []) or [])
        if not history:
            history_payload = load(
                local_dir / "adaptive_search.history.json"
            )
            root = str(
                request.get("root_hypothesis_id") or current_hid
            )
            history = list(
                (history_payload.get("history_by_root", {}) or {}).get(
                    root,
                    [],
                )
                or []
            )

        request_reason = str(
            request.get("v2_request_reason")
            or request.get("handoff_state")
            or "v1_graph_retraversal_handoff"
        )
        request_payload = build_retraversal_request(
            decision=request,
            source_context=old_context.model_dump(mode="json"),
            attempt_history=history,
            context_epoch=epoch_index,
            retraversal_index=graph_count + 1,
            reason=request_reason,
        )

        retrav_dir = epoch_dir / "graph_retraversal"
        request_path = retrav_dir / "request.json"
        write(request_path, request_payload)

        cmd = [
            sys.executable,
            "-m",
            "scripts.discovery.run_adaptive_graph_retraversal_shadow",
            "--source-context", str(current_context_path),
            "--request", str(request_path),
            "--model", str(args.critic_model),
            "--api-key-env", str(args.api_key_env),
            "--objective", str(args.retraversal_objective),
            "--candidate-top-k", str(args.retraversal_candidate_top_k),
            "--selected-top-k", str(args.retraversal_selected_top_k),
            "--node-map-k", str(args.retraversal_node_map_k),
            "--endpoint-pair-k", str(args.retraversal_endpoint_pair_k),
            "--max-depth-increment", str(
                args.retraversal_max_depth_increment
            ),
            "--max-depth-cap", str(args.retraversal_max_depth_cap),
            "--min-new-edge-fraction", str(
                args.retraversal_min_new_edge_fraction
            ),
            "--min-new-paper-fraction", str(
                args.retraversal_min_new_paper_fraction
            ),
            "--max-prior-edge-jaccard", str(
                args.retraversal_max_prior_edge_jaccard
            ),
            "--max-selected-edge-jaccard", str(
                args.retraversal_max_selected_edge_jaccard
            ),
            "--paper-expansion-reserve", str(
                args.retraversal_paper_expansion_reserve
            ),
            "--output-dir", str(retrav_dir),
        ]
        for path in prior_traversals:
            cmd += ["--prior-traversal", str(path)]
        if args.retraversal_allow_top_n_fallback:
            cmd.append("--allow-top-n-fallback")
        if args.base_url:
            cmd += ["--base-url", str(args.base_url)]

        run(
            f"adaptive v2 grounded graph retraversal epoch {epoch_index}",
            cmd,
            args.output_dir,
        )

        retrav_summary = load(
            retrav_dir / "retraversal.summary.json"
        )
        if retrav_summary.get("status") != "COMPLETE_NEW_GROUNDED_CONTEXT":
            remaining_handoffs.append(
                {
                    **request_payload,
                    "handoff_state": "GRAPH_RETRAVERSAL_EXHAUSTED",
                    "retraversal_summary": retrav_summary,
                }
            )
            break

        new_context_path = retrav_dir / "retraversal.context.json"
        new_context = _context(new_context_path)
        lineage = load(
            retrav_dir / "retraversal.lineage.json"
        )

        generated_portfolio, gen_record = generate_retraversal_hypothesis(
            old_context=old_context,
            new_context=new_context,
            original=original,
            request=request_payload,
            attempt_history=history,
            required_new_premise_ids=lineage.get(
                "structurally_new_eligible_premise_ids",
                [],
            ),
            model=args.model,
            critic_model=args.critic_model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            output_prefix=str(
                retrav_dir / "context_reset_generation"
            ),
        )
        write(
            retrav_dir / "generation.record.json",
            gen_record,
        )

        graph_count += 1
        context_events.append(
            {
                "epoch_index": epoch_index,
                "retraversal_index": graph_count,
                "request_id": request_payload.get("request_id"),
                "root_hypothesis_id": request_payload.get(
                    "root_hypothesis_id"
                ),
                "source_hypothesis_id": current_hid,
                "source_context_id": old_context.context_id,
                "output_context_id": new_context.context_id,
                "selection_id": lineage.get("selection_id"),
                "selected_path_ids": lineage.get(
                    "selected_path_ids",
                    [],
                ),
                "new_eligible_premise_ids": lineage.get(
                    "new_eligible_premise_ids",
                    [],
                ),
                "structurally_new_eligible_premise_ids": lineage.get(
                    "structurally_new_eligible_premise_ids",
                    [],
                ),
                "context_delta_audit_id": lineage.get(
                    "context_delta_audit_id"
                ),
                "generation_decision": gen_record.get("decision"),
                "generated_hypothesis_id": gen_record.get(
                    "generated_hypothesis_id"
                ),
            }
        )

        if generated_portfolio is None:
            remaining_handoffs.append(
                {
                    **request_payload,
                    "handoff_state": "CONTEXT_RESET_GENERATION_STOPPED",
                    "generation_record": gen_record,
                }
            )
            break

        generated_path = retrav_dir / "generated.portfolio.json"
        write(generated_path, generated_portfolio)

        generation_report = _generation_report(
            source_portfolio_id=source_portfolio_id,
            output_portfolio_id=generated_portfolio.portfolio_id,
            round_index=1,
            records=[gen_record],
        )
        write(
            retrav_dir / "generation.report.json",
            generation_report,
        )

        previous = _verification_artifacts(
            seed_dir=current_seed_dir,
            local_dir=local_dir,
            hypothesis_id=current_hid,
        )

        verification_dir = retrav_dir / "verification"
        artifacts = _run_fresh_verification(
            args=args,
            round_dir=verification_dir,
            portfolio_path=generated_path,
            previous_query=previous["query"],
            previous_external=previous["external"],
            previous_prior=previous["prior"],
            previous_binding=previous["binding"],
            previous_completion=previous["completion"],
            retrieval_multiplier=1,
            reuse_query_plan=False,
        )

        cohort = load(artifacts["cohort"])
        state = compile_residual_epistemic_state(
            portfolio=generated_portfolio.model_dump(mode="json"),
            query_plan=load(artifacts["query"]),
            external_report=load(artifacts["external"]),
            aggregation=load(artifacts["aggregation"]),
        )
        write(
            verification_dir / "epistemic_state.json",
            state,
        )

        decisions = consolidate_post_verification_decisions(
            gen1_portfolio=generated_portfolio.model_dump(mode="json"),
            generation_report=generation_report,
            query_plan=load(artifacts["query"]),
            external_report=load(artifacts["external"]),
            aggregation=load(artifacts["aggregation"]),
            cohort_audit=cohort,
        )
        write(
            verification_dir / "post_verification_decisions.json",
            decisions,
        )

        if not cohort.get("pass"):
            remaining_handoffs.append(
                {
                    **request_payload,
                    "handoff_state": "FAIL_CLOSED_RETRAVERSAL_AUDIT",
                    "cohort_failures": cohort.get("failures", []),
                }
            )
            break

        effective = build_effective_gen1_portfolio(
            gen1_portfolio=generated_portfolio.model_dump(mode="json"),
            decisions=decisions,
        )

        if effective.hypotheses:
            effective_path = retrav_dir / "effective.portfolio.json"
            write(effective_path, effective)
            for card in effective.hypotheses:
                hid = str(card.hypothesis_id)
                global_effective[hid] = {
                    "hypothesis_id": hid,
                    "context_id": str(card.source_context_id),
                    "context_sha256": str(card.source_context_sha256),
                    "portfolio_path": str(effective_path),
                    "epoch_index": epoch_index + 1,
                    "title": str(card.title),
                }
            break

        next_seed = retrav_dir / "next_seed"
        _build_next_seed(
            seed_dir=next_seed,
            portfolio=generated_portfolio,
            generation=generation_report,
            effective=effective,
            decisions=decisions,
            artifacts=artifacts,
        )

        current_context_path = new_context_path
        current_seed_dir = next_seed
        prior_traversals.append(
            retrav_dir / "retraversal.selected_traversal.json"
        )

    effective_population = {
        "schema_version": "adaptive-v2-effective-population-v1",
        "effective_count": len(global_effective),
        "hypothesis_ids": sorted(global_effective),
        "entries": [
            global_effective[key]
            for key in sorted(global_effective)
        ],
        "cross_context_population": (
            len(
                {
                    row["context_id"]
                    for row in global_effective.values()
                }
            )
            > 1
        ),
        "single_hypothesis_portfolio_merge_allowed": False,
        "stage8_input_changed": False,
        "production_selection_changed": False,
    }
    write(
        args.output_dir / "adaptive_v2.effective_population.json",
        effective_population,
    )

    lineage_payload = {
        "schema_version": "adaptive-v2-context-lineage-v1",
        "retraversal_execution_count": graph_count,
        "events": context_events,
        "external_prior_art_as_positive_premise": False,
        "canonical_graph_mutated": False,
    }
    write(
        args.output_dir / "context_retraversal.lineage.json",
        lineage_payload,
    )

    handoff = {
        "schema_version": "adaptive-graph-retraversal-handoff-v2",
        "request_count": len(remaining_handoffs),
        "requests": remaining_handoffs,
        "execution_count": graph_count,
        "execution_performed": graph_count > 0,
    }
    write(
        args.output_dir / "graph_retraversal.handoff.json",
        handoff,
    )

    if remaining_handoffs:
        status = "COMPLETE_WITH_UNRESOLVED_GRAPH_HANDOFF"
    elif global_effective:
        status = "COMPLETE_WITH_EFFECTIVE_CANDIDATES"
    elif graph_count:
        status = "COMPLETE_GRAPH_RETRAVERSAL_SEARCH_EXHAUSTED"
    else:
        status = "COMPLETE_LOCAL_SEARCH_EXHAUSTED"

    summary = {
        "schema_version": "adaptive-discovery-controller-summary-v2",
        "implementation_version": "adaptive-discovery-controller-v2b",
        "status": status,
        "epoch_count": len(epoch_summaries),
        "epochs": epoch_summaries,
        "graph_retraversal_execution_count": graph_count,
        "graph_retraversal_handoff_count": len(remaining_handoffs),
        "context_lineage_event_count": len(context_events),
        "final_effective_count": len(global_effective),
        "final_effective_hypothesis_ids": sorted(global_effective),
        "max_graph_retraversals": args.max_graph_retraversals,
        "initial_local_controller_reused": (
            initial_local_controller_dir is not None
        ),
        "initial_local_controller_dir": (
            str(initial_local_controller_dir)
            if initial_local_controller_dir is not None
            else None
        ),
        "initial_seed_source_mode": initial_seed_source_mode,
        "initial_seed_root": str(resolved_initial_seed_dir),
        "external_prior_art_as_positive_premise": False,
        "controller_has_novelty_authority": False,
        "controller_has_generation_authority": False,
        "n10_research_selection_authority": False,
        "stage8_input_changed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    write(
        args.output_dir / "adaptive_v2.summary.json",
        summary,
    )

    print()
    print("===== ADAPTIVE DISCOVERY CONTROLLER V2B =====")
    print("epochs:", summary["epoch_count"])
    print("graph retraversals:", graph_count)
    print("effective candidates:", len(global_effective))
    print("remaining handoffs:", len(remaining_handoffs))
    print("EXTERNAL_PRIOR_ART_AS_POSITIVE_PREMISE=False")
    print("CANONICAL_GRAPH_MUTATED=False")
    print("STAGE8_INPUT_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
