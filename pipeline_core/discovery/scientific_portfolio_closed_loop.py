
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)


_BACKED_EXTERNAL = {
    "WELL_ESTABLISHED",
    "LITERATURE_SUPPORTED_EXTENSION",
    "CONFLICTING_PRIOR_ART",
}


def _canonical(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(_canonical(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _claim_index(query_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for group in query_plan.get("claims", []):
        for claim in group.get("claims", []):
            cid = str(claim.get("claim_id") or "")
            if cid:
                out[cid] = claim
    return out


def _role_class(claim: dict[str, Any] | None) -> str:
    if not claim:
        return "UNKNOWN"
    role = claim.get("novelty_selection_role")
    importance = str(claim.get("importance") or "core")
    if role == "NOVELTY_BEARING":
        return "NOVELTY_BEARING"
    if role == "REQUIRED_ENABLING_RELATION":
        return "REQUIRED_ENABLING"
    if role in {"TESTING_PREDICTION", "AUXILIARY"}:
        return "NONBLOCKING"
    if importance == "core":
        return "FALLBACK_CORE_NOVELTY_BEARING"
    return "NONBLOCKING"


def _external_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("hypothesis_id")): row
        for row in report.get("cards", [])
        if row.get("hypothesis_id")
    }


def _aggregation_index(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in report.get("composites", []):
        hid = str(row.get("hypothesis_id") or "")
        if hid:
            out.setdefault(hid, []).append(row)
    return out


def compile_residual_epistemic_state(
    *,
    portfolio: dict[str, Any],
    query_plan: dict[str, Any],
    external_report: dict[str, Any],
    aggregation: dict[str, Any],
) -> dict[str, Any]:
    """Compile domain-neutral residual state for one feedback generation."""
    portfolio_id = str(portfolio.get("portfolio_id") or "")
    if str(query_plan.get("source_portfolio_id") or "") != portfolio_id:
        raise ValueError("query-plan / portfolio lineage mismatch")
    if str(external_report.get("source_portfolio_id") or "") != portfolio_id:
        raise ValueError("external-report / portfolio lineage mismatch")

    claim_index = _claim_index(query_plan)
    claims_by_h: dict[str, list[dict[str, Any]]] = {
        str(group.get("hypothesis_id") or ""): list(group.get("claims", []))
        for group in query_plan.get("claims", [])
    }
    ext_by_h = _external_index(external_report)
    agg_by_h = _aggregation_index(aggregation)

    rows: list[dict[str, Any]] = []
    for card in portfolio.get("hypotheses", []):
        hid = str(card.get("hypothesis_id") or "")
        if not hid:
            continue
        ext = ext_by_h.get(hid, {})
        composites = agg_by_h.get(hid, [])
        query_claims = claims_by_h.get(hid, [])
        query_composite_claims = [
            claim
            for claim in query_claims
            if str(claim.get("kind") or "") == "composite"
        ]

        enriched = []
        for comp in composites:
            cid = str(comp.get("claim_id") or "")
            claim = claim_index.get(cid)
            enriched.append(
                {
                    **comp,
                    "_role_class": _role_class(claim),
                }
            )

        novelty_rows = [
            row
            for row in enriched
            if row["_role_class"]
            in {"NOVELTY_BEARING", "FALLBACK_CORE_NOVELTY_BEARING"}
        ]
        enabling_rows = [
            row
            for row in enriched
            if row["_role_class"] == "REQUIRED_ENABLING"
        ]
        nonblocking_rows = [
            row
            for row in enriched
            if row["_role_class"] in {"NONBLOCKING", "UNKNOWN"}
        ]

        dispositions = {
            str(row.get("aggregation_disposition") or "")
            for row in novelty_rows
        }
        enabling_dispositions = {
            str(row.get("aggregation_disposition") or "")
            for row in enabling_rows
        }

        if "HOLD_FOR_TOPOLOGY" in enabling_dispositions:
            state = "UNRESOLVED_TOPOLOGY_GAP"
            reason = "required_enabling_topology_unresolved"
        elif "HOLD_FOR_BASE_EVIDENCE" in enabling_dispositions:
            state = "UNRESOLVED_EVIDENCE_GAP"
            reason = "required_enabling_evidence_unresolved"
        elif "NO_RESIDUAL" in dispositions:
            state = "PRIOR_ART_BACKED_OR_NO_RESIDUAL"
            reason = "novelty_bearing_relation_prior_art_backed"
        elif "HOLD_FOR_TOPOLOGY" in dispositions:
            state = "UNRESOLVED_TOPOLOGY_GAP"
            reason = "novelty_bearing_topology_unresolved"
        elif "HOLD_FOR_BASE_EVIDENCE" in dispositions:
            state = "UNRESOLVED_EVIDENCE_GAP"
            reason = "novelty_bearing_base_evidence_unresolved"
        elif (
            novelty_rows
            and dispositions == {"RESIDUAL_CANDIDATE_SHADOW"}
            and str(ext.get("status") or "")
            == "INSUFFICIENT_SEARCH_EVIDENCE"
        ):
            state = "UNRESOLVED_EVIDENCE_GAP"
            reason = (
                "insufficient_external_search_evidence_"
                "for_residual_authority"
            )
        elif (
            novelty_rows
            and dispositions == {"RESIDUAL_CANDIDATE_SHADOW"}
        ):
            state = "RESIDUAL_AUTHORITY_CANDIDATE_SHADOW"
            reason = "novelty_bearing_residual_survives"
        elif not novelty_rows and str(ext.get("status") or "") in _BACKED_EXTERNAL:
            state = "PRIOR_ART_BACKED_OR_NO_RESIDUAL"
            reason = "known_external_axis_without_residual_composite"
        elif not novelty_rows and query_composite_claims:
            state = "UNRESOLVED_TOPOLOGY_GAP"
            reason = "composite_claim_present_but_no_novelty_bearing_topology"
        elif not novelty_rows:
            state = "UNRESOLVED_EVIDENCE_GAP"
            reason = "topology_not_applicable_no_composite_claim"
        else:
            state = "UNRESOLVED_EVIDENCE_GAP"
            reason = "mixed_or_incomplete_residual_state"

        rows.append(
            {
                "hypothesis_id": hid,
                "display_text": card.get("title"),
                "final_epistemic_state": state,
                "state_reason": reason,
                "fresh_external_status": ext.get("status"),
                "composite_count": len(enriched),
                "query_plan_composite_claim_count": len(query_composite_claims),
                "topology_applicable": bool(query_composite_claims),
                "novelty_bearing_composite_count": len(novelty_rows),
                "required_enabling_composite_count": len(enabling_rows),
                "nonblocking_composite_count": len(nonblocking_rows),
                "novelty_bearing_dispositions": sorted(dispositions),
                "required_enabling_dispositions": sorted(
                    enabling_dispositions
                ),
                "nonblocking_hold_count": sum(
                    str(row.get("aggregation_disposition") or "").startswith(
                        "HOLD_"
                    )
                    for row in nonblocking_rows
                ),
                "authority_ready_candidate_shadow": (
                    state == "RESIDUAL_AUTHORITY_CANDIDATE_SHADOW"
                ),
                "authority_readiness_state": state,
            }
        )

    counts = Counter(row["final_epistemic_state"] for row in rows)
    report = {
        "schema_version": "scientific-portfolio-residual-epistemic-state-v1",
        "source_portfolio_id": portfolio_id,
        "state_counts": dict(sorted(counts.items())),
        "hypotheses": rows,
        "claim_role_aware": True,
        "shadow_only": True,
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
    }
    report["report_id"] = _stable_id(
        "scientific_portfolio_residual_state",
        portfolio_id,
        [(r["hypothesis_id"], r["final_epistemic_state"]) for r in rows],
    )
    return report


def build_effective_portfolio(
    *,
    gen1_portfolio: HypothesisPortfolio,
    decisions: dict[str, Any],
) -> HypothesisPortfolio:
    decision_by_h = {
        str(row.get("hypothesis_id")): row
        for row in decisions.get("rows", [])
        if row.get("hypothesis_id")
    }
    retained = [
        card
        for card in gen1_portfolio.hypotheses
        if decision_by_h.get(str(card.hypothesis_id), {}).get(
            "advance_to_effective_gen1"
        )
    ]
    pid = _stable_id(
        "scientific_portfolio_effective_gen1",
        gen1_portfolio.portfolio_id,
        [str(x.hypothesis_id) for x in retained],
    )
    return gen1_portfolio.model_copy(
        update={
            "portfolio_id": pid,
            "hypotheses": retained,
            "abstention_reason": (
                None
                if retained
                else (
                    "No closed-loop Gen1 hypothesis survived fresh "
                    "post-verification residual consolidation."
                )
            ),
        }
    )


def summarize_closed_loop(
    *,
    gen0_state: dict[str, Any],
    feedback_plan: dict[str, Any],
    generation_report: dict[str, Any],
    gen1_aggregation: dict[str, Any],
    gen1_cohort_audit: dict[str, Any],
    decisions: dict[str, Any],
    effective_portfolio: HypothesisPortfolio,
) -> dict[str, Any]:
    return {
        "schema_version": "scientific-portfolio-closed-loop-shadow-summary-v1",
        "gen0_state_counts": gen0_state.get("state_counts", {}),
        "feedback_route_counts": feedback_plan.get("route_counts", {}),
        "generation_decision_counts": generation_report.get(
            "decision_counts", {}
        ),
        "gen1_residual_state_counts": gen1_aggregation.get(
            "residual_state_counts", {}
        ),
        "gen1_residual_disposition_counts": gen1_aggregation.get(
            "disposition_counts", {}
        ),
        "gen1_cohort_audit_pass": bool(gen1_cohort_audit.get("pass")),
        "post_verification_decision_counts": decisions.get(
            "decision_counts", {}
        ),
        "effective_gen1_portfolio_id": effective_portfolio.portfolio_id,
        "effective_gen1_hypothesis_count": len(
            effective_portfolio.hypotheses
        ),
        "effective_gen1_hypothesis_ids": [
            str(card.hypothesis_id)
            for card in effective_portfolio.hypotheses
        ],
        "single_feedback_generation_only": True,
        "external_prior_art_as_positive_premise": False,
        "n10_research_selection_authority": False,
        "shadow_only": True,
        "novelty_authority_created": False,
        "generation_authority_created": False,
        "production_selection_changed": False,
        "stage8_input_changed": False,
        "canonical_graph_mutated": False,
    }
