
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
)
from pipeline_core.discovery.hypothesis_compiler import (
    HypothesisCompileError,
    HypothesisCompiler,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
)
from pipeline_core.discovery.hypothesis_llm import (
    InstructorOpenAICompatibleHypothesisBackend,
)
from pipeline_core.discovery.hypothesis_validation import (
    HypothesisValidator,
)
from pipeline_core.discovery.novelty_reaxis_prompt import (
    FreshNoveltyReaxisPromptAssembler,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyGap,
)
from pipeline_core.discovery.novelty_refinement_prompt import (
    NoveltyRefinementPromptAssembler,
)
from pipeline_core.discovery.question_axis_responsiveness_llm import (
    OpenRouterQuestionAxisResponsivenessBackend,
)
from pipeline_core.discovery.question_hypothesis_responsiveness import (
    evaluate_hypothesis_task_preservation,
)


OPERATORS = [
    "MODERATOR",
    "INTERACTION",
    "RESIDUAL",
    "BOUNDARY",
    "PROXY_DECOUPLING",
    "COMPENSATION_LIMIT",
]


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(canonical_json(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def load_context_source(path: Path) -> HypothesisContext:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(
        payload.get("grounded_context"), dict
    ):
        payload = payload["grounded_context"]
    return HypothesisContext.model_validate(payload)


def safe_unused_premise_ids(
    context: HypothesisContext,
    original: Any,
) -> list[str]:
    used = set(map(str, original.premise_statement_ids))
    return sorted(
        str(row.statement_id)
        for row in context.evidence_statements
        if (
            row.eligible_as_premise
            and not row.requires_verification
            and not row.premise_restrictions
            and row.epistemic_role in {"reported", "evidence_synthesis"}
            and str(row.statement_id) not in used
        )
    )


def _external_boundaries(card: Any) -> tuple[list[str], list[str], list[str]]:
    known: list[str] = []
    unresolved: list[str] = []
    target_claim_ids: list[str] = []

    for review in card.claim_reviews:
        base = f"{review.status}: {review.claim_text}"
        titles = [
            str(m.title).strip()
            for m in review.matches[:2]
            if str(m.title).strip()
        ]
        if titles:
            base += " | prior art: " + " ; ".join(titles)

        if review.status in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}:
            known.append(base)
            if review.importance == "core":
                target_claim_ids.append(review.claim_id)
        else:
            unresolved.append(base)

    if not target_claim_ids:
        target_claim_ids = [
            review.claim_id
            for review in card.claim_reviews
            if review.status in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}
        ]

    return known[:8], unresolved[:8], target_claim_ids[:4]


def _materialization_lineage(
    report: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not report:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in report.get("records", []):
        hid = str(row.get("hypothesis_id") or "")
        if not hid:
            continue
        result[hid] = {
            "candidate_id": row.get("candidate_id"),
            "source_object_id": row.get("source_object_id"),
            "materialization_status": row.get("status"),
        }
    return result


def build_feedback_plan(
    *,
    context: HypothesisContext,
    portfolio: HypothesisPortfolio,
    closeout: dict[str, Any],
    external: ExternalNoveltyReport,
    materialization_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    closeout_by_id = {
        str(row["hypothesis_id"]): row
        for row in closeout.get("hypotheses", [])
    }
    external_by_id = {
        str(card.hypothesis_id): card
        for card in external.cards
    }
    lineage = _materialization_lineage(materialization_report)

    portfolio_ids = {
        str(card.hypothesis_id)
        for card in portfolio.hypotheses
    }
    if set(closeout_by_id) != portfolio_ids:
        raise ValueError(
            "closeout/portfolio hypothesis sets do not match"
        )
    if set(external_by_id) != portfolio_ids:
        raise ValueError(
            "external-report/portfolio hypothesis sets do not match"
        )

    targets: list[dict[str, Any]] = []

    for card in portfolio.hypotheses:
        hid = str(card.hypothesis_id)
        frozen = closeout_by_id[hid]
        ext = external_by_id[hid]
        state = str(frozen.get("final_epistemic_state") or "")
        unused = safe_unused_premise_ids(context, card)
        known, unresolved, target_claim_ids = _external_boundaries(ext)

        if state == "RESIDUAL_AUTHORITY_CANDIDATE_SHADOW":
            route = "KEEP_RESIDUAL"
            reason = (
                "The current higher-order residual survived the SERS "
                "evidence accounting; do not novelty-optimize it further."
            )
        elif state == "UNRESOLVED_EVIDENCE_GAP":
            route = "HOLD_FOR_EVIDENCE"
            reason = (
                "A required lower-order component remains unresolved; "
                "generation is blocked rather than treating missing evidence "
                "as novelty."
            )
        elif state == "UNRESOLVED_TOPOLOGY_GAP":
            if unused:
                route = "FRESH_CONTEXT_REAXIS"
                reason = (
                    "The current decomposition does not justify residual "
                    "component topology. Re-axis from a different grounded "
                    "premise combination instead of inventing topology."
                )
            else:
                route = "HOLD_FOR_EVIDENCE"
                reason = (
                    "No justified topology and no safe unused grounded "
                    "premise are available for a fresh re-axis."
                )
        elif state == "PRIOR_ART_BACKED_OR_NO_RESIDUAL":
            if unused:
                route = "FRESH_CONTEXT_REAXIS"
                reason = (
                    "The current relation is already directly/partially "
                    "backed. Prefer a genuinely different grounded axis over "
                    "a cosmetic novelty rewrite."
                )
            elif target_claim_ids:
                route = "SAME_PREMISE_GAP_SHARPEN"
                reason = (
                    "The relation is prior-art-backed and no safe unused "
                    "premise is available; attempt one structural sharpen "
                    "over the same grounded premises."
                )
            else:
                route = "HOLD_FOR_EVIDENCE"
                reason = (
                    "The current relation is prior-art-backed but there is "
                    "no safe re-axis capacity and no claim-bound sharpen target."
                )
        else:
            route = "HOLD_FOR_EVIDENCE"
            reason = "Unrecognized closeout state; fail closed."

        lin = lineage.get(hid, {})
        targets.append(
            {
                "source_hypothesis_id": hid,
                "source_title": card.title,
                "source_state": state,
                "source_external_status": ext.status,
                "route": route,
                "route_reason": reason,
                "safe_unused_premise_statement_ids": unused,
                "target_claim_ids": target_claim_ids,
                "already_known_boundary": known,
                "unresolved_boundary": unresolved,
                "source_candidate_id": lin.get("candidate_id"),
                "source_object_id": lin.get("source_object_id"),
                "frontier_lineage_available": bool(
                    lin.get("source_object_id")
                ),
                "external_prior_art_as_positive_premise": False,
            }
        )

    counts = Counter(row["route"] for row in targets)
    body = {
        "schema_version": "closed-loop-novelty-feedback-plan-v1",
        "source_portfolio_id": portfolio.portfolio_id,
        "source_external_report_id": external.report_id,
        "source_closeout_schema": closeout.get("schema_version"),
        "target_count": len(targets),
        "route_counts": dict(sorted(counts.items())),
        "targets": targets,
        "single_generation_only": True,
        "external_prior_art_as_positive_premise": False,
        "memory_is_evidence": False,
        "novelty_authority_created": False,
        "generation_authority_created": False,
        "production_selection_changed": False,
    }
    body["plan_id"] = stable_id(
        "closed_loop_novelty_feedback",
        body["source_portfolio_id"],
        body["source_external_report_id"],
        [
            (r["source_hypothesis_id"], r["route"])
            for r in targets
        ],
    )
    return body


def _make_gap(
    *,
    target: dict[str, Any],
) -> NoveltyGap:
    route = target["route"]
    if route == "KEEP_RESIDUAL":
        action = "keep"
        operators: list[str] = []
    elif route == "SAME_PREMISE_GAP_SHARPEN":
        action = "gap_sharpen"
        operators = list(OPERATORS)
    elif route == "FRESH_CONTEXT_REAXIS":
        action = "targeted_search_then_refine"
        operators = []
    else:
        action = "targeted_search_only"
        operators = []

    target_ids = list(target.get("target_claim_ids", []))
    if action == "gap_sharpen" and not target_ids:
        raise ValueError(
            "SAME_PREMISE_GAP_SHARPEN requires target claim IDs"
        )

    return NoveltyGap(
        gap_id=stable_id(
            "feedback_gap",
            target["source_hypothesis_id"],
            route,
            target_ids,
        ),
        hypothesis_id=target["source_hypothesis_id"],
        source_external_status=target["source_external_status"],
        action=action,
        target_claim_ids=target_ids,
        differentiator=(
            "Generate a scientifically different, falsifiable SERS relation "
            "that preserves the original research task while moving beyond "
            "the prior-art-backed formulation. Do not add cosmetic qualifiers."
        ),
        already_known_boundary=list(
            target.get("already_known_boundary", [])
        ),
        unresolved_boundary=list(
            target.get("unresolved_boundary", [])
        ),
        targeted_queries=[],
        sharpening_operators=operators,
        reason_codes=[
            "closed_loop_novelty_feedback_v1",
            "external_prior_art_is_boundary_only",
            "single_generation_budget",
            "route:" + route.lower(),
        ],
    )


def _compile(
    *,
    context: HypothesisContext,
    draft: HypothesisPortfolioDraft,
) -> tuple[HypothesisPortfolio | None, list[str]]:
    try:
        portfolio = HypothesisCompiler().compile(context, draft)
    except HypothesisCompileError as exc:
        return None, [
            f"{x.code}:{x.location}:{x.message}"
            for x in exc.issues
        ]
    except Exception as exc:
        return None, [f"{type(exc).__name__}:{exc}"]

    validation = HypothesisValidator().validate(
        context,
        portfolio,
    )
    if not validation.passes:
        return None, [
            f"{x.code}:{x.location}:{x.message}"
            for x in validation.issues
            if x.severity == "error"
        ]
    return portfolio, []


def _lock_same_premise(
    draft: HypothesisPortfolioDraft,
    original: Any,
) -> HypothesisPortfolioDraft:
    if len(draft.hypotheses) != 1:
        return draft
    row = draft.hypotheses[0].model_copy(
        update={
            "premise_statement_ids": list(
                original.premise_statement_ids
            ),
            "gap_statement_ids": list(
                original.gap_statement_ids
            ),
            "hypothesis_type": original.hypothesis_type,
        }
    )
    return draft.model_copy(
        update={
            "hypotheses": [row],
            "abstention_reason": None,
        }
    )


def _reaxis_grounding_valid(
    *,
    context: HypothesisContext,
    original: Any,
    candidate: Any,
    safe_unused: list[str],
) -> bool:
    original_ids = set(map(str, original.premise_statement_ids))
    safe = set(map(str, safe_unused))
    candidate_ids = set(map(str, candidate.premise_statement_ids))

    if not candidate_ids:
        return False
    if not candidate_ids.issubset(original_ids | safe):
        return False
    if not candidate_ids.intersection(safe):
        return False

    allowed_gaps = {
        str(row.statement_id)
        for row in context.evidence_statements
        if row.eligible_as_gap
    }
    return set(
        map(str, candidate.gap_statement_ids)
    ).issubset(allowed_gaps)


def run_feedback_generation(
    *,
    context: HypothesisContext,
    portfolio: HypothesisPortfolio,
    external: ExternalNoveltyReport,
    plan: dict[str, Any],
    model: str,
    critic_model: str,
    api_key_env: str,
    base_url: str | None,
    output_dir: Path,
) -> tuple[dict[str, Any], HypothesisPortfolio]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    backend = InstructorOpenAICompatibleHypothesisBackend(
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        temperature=0.0,
        parse_retries=2,
        telemetry_path=output_dir / "generation.telemetry.jsonl",
        telemetry_context={
            "pipeline": "closed_loop_novelty_feedback_v1",
            "scope": "SERS_SHADOW",
        },
    )
    task_backend = OpenRouterQuestionAxisResponsivenessBackend(
        model=critic_model,
        temperature=0.0,
        reasoning_effort="medium",
        telemetry_path=output_dir / "task.telemetry.jsonl",
        telemetry_context={
            "pipeline": "closed_loop_novelty_feedback_v1",
            "scope": "SERS_SHADOW",
        },
    )

    by_h = {
        str(card.hypothesis_id): card
        for card in portfolio.hypotheses
    }
    ext_by_h = {
        str(card.hypothesis_id): card
        for card in external.cards
    }

    output_cards = []
    records: list[dict[str, Any]] = []

    for index, target in enumerate(plan["targets"], start=1):
        hid = target["source_hypothesis_id"]
        original = by_h[hid]
        route = target["route"]

        if route == "KEEP_RESIDUAL":
            output_cards.append(original)
            records.append(
                {
                    "source_hypothesis_id": hid,
                    "route": route,
                    "decision": "CARRIED_FORWARD",
                    "generated_hypothesis_id": original.hypothesis_id,
                    "task_preservation": "NOT_RERUN_CARRIED_ORIGINAL",
                    "reason_codes": [
                        "residual_candidate_preserved_without_novelty_optimization"
                    ],
                }
            )
            continue

        if route == "HOLD_FOR_EVIDENCE":
            records.append(
                {
                    "source_hypothesis_id": hid,
                    "route": route,
                    "decision": "HELD_FOR_EVIDENCE",
                    "generated_hypothesis_id": None,
                    "task_preservation": "NOT_RUN",
                    "reason_codes": [
                        "generation_blocked_by_feedback_policy"
                    ],
                }
            )
            continue

        gap = _make_gap(target=target)
        targeted_card = ext_by_h[hid]

        if route == "FRESH_CONTEXT_REAXIS":
            safe_unused = list(
                target["safe_unused_premise_statement_ids"]
            )
            allowed = sorted(
                set(map(str, original.premise_statement_ids))
                | set(safe_unused)
            )
            assembler = FreshNoveltyReaxisPromptAssembler(
                original=original,
                gap=gap,
                targeted_card=targeted_card,
                allowed_premise_ids=allowed,
                required_unused_premise_ids=safe_unused,
            )
        else:
            safe_unused = []
            assembler = NoveltyRefinementPromptAssembler(
                original=original,
                gap=gap,
                targeted_card=targeted_card,
            )

        prompt = assembler.build(context)
        (prompt_dir / f"{index:02d}_{hid.split(':')[-1]}.json").write_text(
            json.dumps(
                {
                    "source_hypothesis_id": hid,
                    "route": route,
                    "prompt_version": prompt.prompt_version,
                    "prompt_sha256": prompt.prompt_sha256,
                    "system_prompt": prompt.system_prompt,
                    "user_prompt": prompt.user_prompt,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        generation = backend.generate(prompt)
        draft = generation.draft

        if not draft.hypotheses:
            records.append(
                {
                    "source_hypothesis_id": hid,
                    "route": route,
                    "decision": "ABSTAINED",
                    "generated_hypothesis_id": None,
                    "task_preservation": "NOT_RUN",
                    "reason_codes": ["model_abstained"],
                }
            )
            continue

        if len(draft.hypotheses) != 1:
            records.append(
                {
                    "source_hypothesis_id": hid,
                    "route": route,
                    "decision": "REJECTED_CARDINALITY",
                    "generated_hypothesis_id": None,
                    "task_preservation": "NOT_RUN",
                    "reason_codes": [
                        "feedback_generation_requires_exactly_one_hypothesis"
                    ],
                }
            )
            continue

        if route == "SAME_PREMISE_GAP_SHARPEN":
            draft = _lock_same_premise(draft, original)

        compiled, issues = _compile(
            context=context,
            draft=draft,
        )

        if compiled is None:
            if route == "SAME_PREMISE_GAP_SHARPEN":
                feedback = assembler.repair_feedback(
                    previous_draft=draft,
                    issues=issues,
                )
            else:
                feedback = "\n".join(
                    [
                        "CLOSED-LOOP RE-AXIS REPAIR",
                        "Repair the single proposed hypothesis while preserving the fresh-context re-axis contract.",
                        "Use only allowed grounded premise IDs and at least one required unused premise ID.",
                        "Do not use external prior art as a positive premise.",
                        "Issues:",
                        *[f"- {x}" for x in issues],
                        "Return exactly one corrected hypothesis or abstain.",
                    ]
                )
            repaired = backend.repair(
                prompt,
                draft,
                feedback,
            ).draft
            if route == "SAME_PREMISE_GAP_SHARPEN":
                repaired = _lock_same_premise(
                    repaired,
                    original,
                )
            compiled, issues = _compile(
                context=context,
                draft=repaired,
            )

        if compiled is None:
            records.append(
                {
                    "source_hypothesis_id": hid,
                    "route": route,
                    "decision": "COMPILE_OR_VALIDATION_REJECTED",
                    "generated_hypothesis_id": None,
                    "task_preservation": "NOT_RUN",
                    "reason_codes": issues[:12],
                }
            )
            continue

        candidate = compiled.hypotheses[0]

        if (
            route == "FRESH_CONTEXT_REAXIS"
            and not _reaxis_grounding_valid(
                context=context,
                original=original,
                candidate=candidate,
                safe_unused=safe_unused,
            )
        ):
            records.append(
                {
                    "source_hypothesis_id": hid,
                    "route": route,
                    "decision": "GROUNDING_DRIFT_REJECTED",
                    "generated_hypothesis_id": candidate.hypothesis_id,
                    "task_preservation": "NOT_RUN",
                    "reason_codes": [
                        "fresh_reaxis_requires_new_safe_grounded_premise"
                    ],
                }
            )
            continue

        task, stability = evaluate_hypothesis_task_preservation(
            question=context.question,
            hypothesis=candidate,
            backend=task_backend,
            debug_path_prefix=str(
                output_dir
                / "task_reviews"
                / f"{index:02d}_{candidate.hypothesis_id.split(':')[-1]}"
            ),
        )

        task_ok = (
            task.decision_stable
            and task.task_class in {"DIRECT", "SUBORDINATE"}
        )

        if not task_ok:
            records.append(
                {
                    "source_hypothesis_id": hid,
                    "route": route,
                    "decision": "TASK_PRESERVATION_REJECTED",
                    "generated_hypothesis_id": candidate.hypothesis_id,
                    "task_preservation": task.task_class,
                    "task_decision_stable": task.decision_stable,
                    "task_source_decision_stable": (
                        task.source_decision_stable
                    ),
                    "task_stability": stability.model_dump(mode="json"),
                    "reason_codes": [
                        "generated_candidate_replaced_or_lost_original_task"
                    ],
                }
            )
            continue

        output_cards.append(candidate)
        records.append(
            {
                "source_hypothesis_id": hid,
                "route": route,
                "decision": "ACCEPTED_GENERATION_SHADOW",
                "generated_hypothesis_id": candidate.hypothesis_id,
                "generated_title": candidate.title,
                "task_preservation": task.task_class,
                "task_decision_stable": task.decision_stable,
                "task_source_decision_stable": (
                    task.source_decision_stable
                ),
                "task_stability": stability.model_dump(mode="json"),
                "introduced_premise_statement_ids": sorted(
                    set(map(str, candidate.premise_statement_ids))
                    - set(map(str, original.premise_statement_ids))
                ),
                "reason_codes": [
                    "compiled_and_validated",
                    "task_preservation_passed",
                    "external_prior_art_used_as_boundary_only",
                ],
            }
        )

    seen = set()
    unique_cards = []
    for card in output_cards:
        if card.hypothesis_id in seen:
            continue
        seen.add(card.hypothesis_id)
        unique_cards.append(card)

    portfolio_id = stable_id(
        "sers_feedback_gen1_portfolio",
        portfolio.portfolio_id,
        [
            card.model_dump(mode="json")
            for card in unique_cards
        ],
    )
    gen1 = portfolio.model_copy(
        update={
            "portfolio_id": portfolio_id,
            "hypotheses": unique_cards,
            "abstention_reason": (
                None
                if unique_cards
                else "No SERS closed-loop feedback candidate survived shadow generation."
            ),
        }
    )

    decisions = Counter(row["decision"] for row in records)
    route_counts = Counter(row["route"] for row in records)
    report = {
        "schema_version": "closed-loop-novelty-feedback-generation-v1",
        "source_portfolio_id": portfolio.portfolio_id,
        "source_external_report_id": external.report_id,
        "feedback_plan_id": plan["plan_id"],
        "output_portfolio_id": gen1.portfolio_id,
        "source_hypothesis_count": len(portfolio.hypotheses),
        "output_hypothesis_count": len(gen1.hypotheses),
        "route_counts": dict(sorted(route_counts.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "records": records,
        "single_generation_only": True,
        "standard_hypothesis_compiler_used": True,
        "standard_hypothesis_validator_used": True,
        "task_preservation_two_pass_used": True,
        "external_prior_art_as_positive_premise": False,
        "novelty_authority_created": False,
        "generation_authority_created": False,
        "production_selection_changed": False,
    }
    report["report_id"] = stable_id(
        "closed_loop_novelty_feedback_generation",
        report["source_portfolio_id"],
        report["output_portfolio_id"],
        records,
    )
    return report, gen1


def audit_gen1(
    *,
    generation_report: dict[str, Any],
    external_report: dict[str, Any],
    aggregation: dict[str, Any],
    cohort_audit: dict[str, Any],
) -> dict[str, Any]:
    external_by_h = {
        str(row.get("hypothesis_id")): row
        for row in external_report.get("cards", [])
        if row.get("hypothesis_id")
    }
    aggregate_by_h: dict[str, list[dict[str, Any]]] = {}
    for row in aggregation.get("composites", []):
        hid = str(row.get("hypothesis_id") or "")
        if hid:
            aggregate_by_h.setdefault(hid, []).append(row)

    rows = []
    for rec in generation_report.get("records", []):
        decision = rec.get("decision")
        source = rec.get("source_hypothesis_id")
        generated = rec.get("generated_hypothesis_id")
        route = rec.get("route")

        if decision == "CARRIED_FORWARD":
            outcome = "CARRIED_RESIDUAL_CANDIDATE"
            progress = True
            fresh_status = None
            dispositions = []
        elif decision == "HELD_FOR_EVIDENCE":
            outcome = "HELD_BEFORE_GENERATION"
            progress = True
            fresh_status = None
            dispositions = []
        elif decision != "ACCEPTED_GENERATION_SHADOW":
            outcome = "GENERATION_NOT_ACCEPTED"
            progress = False
            fresh_status = None
            dispositions = []
        else:
            ext = external_by_h.get(str(generated), {})
            fresh_status = ext.get("status")
            agg_rows = aggregate_by_h.get(str(generated), [])
            dispositions = sorted(
                {
                    str(x.get("aggregation_disposition"))
                    for x in agg_rows
                    if x.get("aggregation_disposition")
                }
            )

            if "NO_RESIDUAL" in dispositions:
                outcome = "KNOWN_AXIS_REPEAT"
                progress = False
            elif "RESIDUAL_CANDIDATE_SHADOW" in dispositions:
                outcome = "GEN1_RESIDUAL_CANDIDATE_SHADOW"
                progress = True
            elif any(x.startswith("HOLD_") for x in dispositions):
                outcome = "GEN1_HELD_FOR_EVIDENCE"
                progress = True
            elif fresh_status in {
                "WELL_ESTABLISHED",
                "LITERATURE_SUPPORTED_EXTENSION",
                "CONFLICTING_PRIOR_ART",
            }:
                outcome = "KNOWN_AXIS_REPEAT"
                progress = False
            elif fresh_status == "INSUFFICIENT_SEARCH_EVIDENCE":
                outcome = "GEN1_HELD_FOR_EVIDENCE"
                progress = True
            else:
                outcome = "GEN1_EXTERNALLY_DISTINCT_SHADOW"
                progress = True

        rows.append(
            {
                "source_hypothesis_id": source,
                "route": route,
                "generation_decision": decision,
                "gen1_hypothesis_id": generated,
                "fresh_external_status": fresh_status,
                "residual_dispositions": dispositions,
                "outcome": outcome,
                "feedback_progress_shadow": progress,
            }
        )

    counts = Counter(row["outcome"] for row in rows)
    accepted = [
        row
        for row in rows
        if row["generation_decision"] == "ACCEPTED_GENERATION_SHADOW"
    ]
    known_repeat_count = sum(
        row["outcome"] == "KNOWN_AXIS_REPEAT"
        for row in accepted
    )
    report = {
        "schema_version": "closed-loop-novelty-feedback-audit-v1",
        "generation_report_id": generation_report.get("report_id"),
        "cohort_audit_pass": bool(cohort_audit.get("pass")),
        "row_count": len(rows),
        "accepted_generation_count": len(accepted),
        "known_axis_repeat_count": known_repeat_count,
        "outcome_counts": dict(sorted(counts.items())),
        "rows": rows,
        "closed_loop_completed": bool(
            cohort_audit.get("pass")
            and accepted
        ),
        "no_known_axis_repeat_among_generated": (
            known_repeat_count == 0
        ),
        "shadow_only": True,
        "novelty_authority_created": False,
        "generation_authority_created": False,
        "production_selection_changed": False,
    }
    report["report_id"] = stable_id(
        "closed_loop_novelty_feedback_audit",
        report["generation_report_id"],
        rows,
    )
    return report
