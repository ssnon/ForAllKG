
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio


BACKED_EXTERNAL_STATES = {
    "ALL_RELATION_BACKED",
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _claim_index(query_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for group in query_plan.get("claims", []):
        for claim in group.get("claims", []):
            cid = str(claim.get("claim_id") or "")
            if cid:
                out[cid] = claim
    return out


def _external_by_h(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("hypothesis_id")): row
        for row in report.get("cards", [])
        if row.get("hypothesis_id")
    }


def _aggregation_by_h(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in report.get("composites", []):
        hid = str(row.get("hypothesis_id") or "")
        if hid:
            out.setdefault(hid, []).append(row)
    return out


def _generation_by_gen1(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("generated_hypothesis_id")): row
        for row in report.get("records", [])
        if row.get("generated_hypothesis_id")
    }


def _role_class(claim: dict[str, Any] | None) -> str:
    if not claim:
        return "UNKNOWN"

    role = claim.get("novelty_selection_role")
    importance = str(claim.get("importance") or "core")

    if role == "NOVELTY_BEARING":
        return "NOVELTY_BEARING"

    if role == "REQUIRED_ENABLING_RELATION":
        return "REQUIRED_ENABLING"

    if role in {
        "TESTING_PREDICTION",
        "AUXILIARY",
    }:
        return "NONBLOCKING"

    # Older / partially specified decompositions may not assign a role.
    # Fail closed for core claims, but do not let an unlabelled supporting
    # claim suppress a valid novelty-bearing residual.
    if importance == "core":
        return "FALLBACK_CORE_NOVELTY_BEARING"

    return "NONBLOCKING"


def _depth_state(card: dict[str, Any]) -> str | None:
    profile = card.get("novelty_depth_profile")
    if not isinstance(profile, dict):
        return None
    value = profile.get("novelty_bearing_prior_art_state")
    return str(value) if value else None


def _row_summary(
    row: dict[str, Any],
    claim_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cid = str(row.get("claim_id") or "")
    claim = claim_index.get(cid)
    return {
        "claim_id": cid,
        "claim_text": row.get("claim_text"),
        "importance": (
            claim.get("importance")
            if claim
            else None
        ),
        "novelty_selection_role": (
            claim.get("novelty_selection_role")
            if claim
            else None
        ),
        "role_class": _role_class(claim),
        "full_relation_status": row.get("full_relation_status"),
        "topology_state": row.get("topology_state"),
        "aggregation_disposition": row.get("aggregation_disposition"),
        "aggregated_residual_state": row.get("aggregated_residual_state"),
        "component_count": len(
            row.get("aggregated_component_claim_ids", [])
        ),
        "component_backed_count": len(
            row.get(
                "aggregated_relation_backed_component_claim_ids",
                [],
            )
        ),
        "evidence_depth": row.get("evidence_depth"),
    }


def consolidate_post_verification_decisions(
    *,
    gen1_portfolio: dict[str, Any],
    generation_report: dict[str, Any],
    query_plan: dict[str, Any],
    external_report: dict[str, Any],
    aggregation: dict[str, Any],
    cohort_audit: dict[str, Any],
) -> dict[str, Any]:
    portfolio_id = str(gen1_portfolio.get("portfolio_id") or "")
    if str(query_plan.get("source_portfolio_id") or "") != portfolio_id:
        raise ValueError(
            "query-plan source_portfolio_id does not match Gen1 portfolio"
        )
    if str(external_report.get("source_portfolio_id") or "") != portfolio_id:
        raise ValueError(
            "external-report source_portfolio_id does not match Gen1 portfolio"
        )

    claim_index = _claim_index(query_plan)
    external = _external_by_h(external_report)
    aggregates = _aggregation_by_h(aggregation)
    gen_records = _generation_by_gen1(generation_report)

    portfolio_ids = {
        str(row.get("hypothesis_id"))
        for row in gen1_portfolio.get("hypotheses", [])
        if row.get("hypothesis_id")
    }

    if set(external) != portfolio_ids:
        raise ValueError(
            "fresh external report does not exactly cover Gen1 portfolio"
        )

    rows: list[dict[str, Any]] = []

    for hypothesis in gen1_portfolio.get("hypotheses", []):
        hid = str(hypothesis.get("hypothesis_id") or "")
        ext = external[hid]
        gen = gen_records.get(hid)
        composite_rows = [
            _row_summary(row, claim_index)
            for row in aggregates.get(hid, [])
        ]

        blocking_novelty = [
            row
            for row in composite_rows
            if row["role_class"] in {
                "NOVELTY_BEARING",
                "FALLBACK_CORE_NOVELTY_BEARING",
            }
        ]
        required_enabling = [
            row
            for row in composite_rows
            if row["role_class"] == "REQUIRED_ENABLING"
        ]
        nonblocking = [
            row
            for row in composite_rows
            if row["role_class"] in {
                "NONBLOCKING",
                "UNKNOWN",
            }
        ]

        warnings: list[str] = []
        reasons: list[str] = []

        if not cohort_audit.get("pass"):
            decision = "HOLD_COHORT_AUDIT_FAILED"
            reasons.append("gen1_residual_cohort_audit_not_passed")

        elif _depth_state(ext) in BACKED_EXTERNAL_STATES:
            decision = "REJECT_KNOWN_AXIS_REPEAT"
            reasons.append(
                "all_novelty_bearing_relations_are_prior_art_backed"
            )

        elif not composite_rows:
            decision = "HOLD_REVALIDATION_INCOMPLETE_NO_COMPOSITE"
            reasons.append(
                "fresh_residual_revalidation_emitted_no_composite_record"
            )

        elif any(
            row["aggregation_disposition"].startswith("HOLD_")
            for row in required_enabling
        ):
            decision = (
                "HOLD_REVALIDATION_INCOMPLETE_REQUIRED_ENABLING"
            )
            reasons.append(
                "required_enabling_relation_remains_unresolved"
            )

        elif not blocking_novelty:
            decision = (
                "HOLD_REVALIDATION_INCOMPLETE_NO_NOVELTY_BEARING_COMPOSITE"
            )
            reasons.append(
                "no_novelty_bearing_composite_available_for_residual_authority"
            )

        elif any(
            row["aggregation_disposition"] == "NO_RESIDUAL"
            for row in blocking_novelty
        ):
            decision = "REJECT_KNOWN_AXIS_REPEAT"
            reasons.append(
                "novelty_bearing_composite_is_prior_art_backed"
            )

        elif any(
            row["aggregation_disposition"].startswith("HOLD_")
            for row in blocking_novelty
        ):
            if any(
                row["aggregation_disposition"]
                == "HOLD_FOR_TOPOLOGY"
                for row in blocking_novelty
            ):
                decision = "HOLD_CORE_TOPOLOGY_UNRESOLVED"
                reasons.append(
                    "novelty_bearing_core_composite_has_unresolved_topology"
                )
            else:
                decision = "HOLD_CORE_EVIDENCE_UNRESOLVED"
                reasons.append(
                    "novelty_bearing_core_composite_has_incomplete_base_evidence"
                )

        elif (
            all(
                row["aggregation_disposition"]
                == "RESIDUAL_CANDIDATE_SHADOW"
                for row in blocking_novelty
            )
            and str(ext.get("status") or "")
            == "INSUFFICIENT_SEARCH_EVIDENCE"
        ):
            decision = "HOLD_CORE_EVIDENCE_UNRESOLVED"
            reasons.append(
                "insufficient_external_search_evidence_"
                "for_residual_authority"
            )

        elif all(
            row["aggregation_disposition"]
            == "RESIDUAL_CANDIDATE_SHADOW"
            for row in blocking_novelty
        ):
            decision = "ADVANCE_RESIDUAL_CANDIDATE_SHADOW"
            reasons.append(
                "all_novelty_bearing_composites_survive_fresh_residual_review"
            )
            if any(
                row["aggregation_disposition"].startswith("HOLD_")
                for row in nonblocking
            ):
                warnings.append(
                    "nonblocking_composite_remains_unresolved"
                )
        else:
            decision = "HOLD_REVALIDATION_INCOMPLETE"
            reasons.append(
                "unhandled_mixed_novelty_bearing_residual_state"
            )

        rows.append(
            {
                "hypothesis_id": hid,
                "title": hypothesis.get("title"),
                "source_hypothesis_id": (
                    gen.get("source_hypothesis_id")
                    if gen
                    else None
                ),
                "generation_route": (
                    gen.get("route")
                    if gen
                    else None
                ),
                "generation_decision": (
                    gen.get("decision")
                    if gen
                    else None
                ),
                "fresh_external_status": ext.get("status"),
                "fresh_novelty_bearing_prior_art_state": _depth_state(ext),
                "composite_count": len(composite_rows),
                "novelty_bearing_composite_count": len(blocking_novelty),
                "required_enabling_composite_count": len(required_enabling),
                "nonblocking_composite_count": len(nonblocking),
                "composites": composite_rows,
                "post_verification_decision": decision,
                "advance_to_effective_gen1": (
                    decision == "ADVANCE_RESIDUAL_CANDIDATE_SHADOW"
                ),
                "reason_codes": reasons,
                "warnings": warnings,
                "fresh_evidence_revalidated": True,
                "carried_forward_has_no_epistemic_immunity": True,
            }
        )

    counts = Counter(
        row["post_verification_decision"]
        for row in rows
    )
    report = {
        "schema_version":
            "closed-loop-post-verification-decision-consolidation-v1",
        "source_gen1_portfolio_id": portfolio_id,
        "generation_report_id": generation_report.get("report_id"),
        "cohort_audit_pass": bool(cohort_audit.get("pass")),
        "hypothesis_count": len(rows),
        "decision_counts": dict(sorted(counts.items())),
        "rows": rows,
        "carried_forward_has_no_epistemic_immunity": True,
        "claim_role_aware_aggregation": True,
        "novelty_authority_created": False,
        "generation_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
    }
    report["report_id"] = _stable_id(
        "closed_loop_post_verification_decisions",
        portfolio_id,
        [
            (
                row["hypothesis_id"],
                row["post_verification_decision"],
            )
            for row in rows
        ],
    )
    return report


def build_effective_gen1_portfolio(
    *,
    gen1_portfolio: dict[str, Any],
    decisions: dict[str, Any],
) -> HypothesisPortfolio:
    source = HypothesisPortfolio.model_validate(
        gen1_portfolio
    )
    decision_by_h = {
        str(row["hypothesis_id"]): row
        for row in decisions.get("rows", [])
    }

    retained = [
        card
        for card in source.hypotheses
        if decision_by_h.get(
            str(card.hypothesis_id),
            {},
        ).get("advance_to_effective_gen1")
    ]

    portfolio_id = _stable_id(
        "sers_closed_loop_effective_gen1",
        source.portfolio_id,
        [
            str(card.hypothesis_id)
            for card in retained
        ],
    )

    return source.model_copy(
        update={
            "portfolio_id": portfolio_id,
            "hypotheses": retained,
            "abstention_reason": (
                None
                if retained
                else (
                    "No Gen1 hypothesis survived fresh post-verification "
                    "residual consolidation."
                )
            ),
        }
    )


def build_replacement_map(
    *,
    generation_report: dict[str, Any],
    decisions: dict[str, Any],
) -> dict[str, Any]:
    post = {
        str(row["hypothesis_id"]): row
        for row in decisions.get("rows", [])
    }
    rows: list[dict[str, Any]] = []

    for gen in generation_report.get("records", []):
        source = str(gen.get("source_hypothesis_id") or "")
        gid = gen.get("generated_hypothesis_id")
        generation_decision = str(gen.get("decision") or "")
        route = str(gen.get("route") or "")

        if generation_decision == "HELD_FOR_EVIDENCE":
            final = "HOLD_BEFORE_GENERATION"
            effective = None
            post_decision = None

        elif gid and str(gid) in post:
            row = post[str(gid)]
            post_decision = row["post_verification_decision"]

            if row["advance_to_effective_gen1"]:
                if generation_decision == "CARRIED_FORWARD":
                    final = "CARRIED_AND_FRESH_REVALIDATED"
                else:
                    final = "REPLACED_BY_GEN1_RESIDUAL_CANDIDATE"
                effective = str(gid)

            elif post_decision == "REJECT_KNOWN_AXIS_REPEAT":
                final = "REJECT_AFTER_FRESH_PRIOR_ART"
                effective = None

            else:
                final = "HOLD_AFTER_FRESH_REVALIDATION"
                effective = None
        else:
            final = "GENERATION_NOT_AVAILABLE"
            effective = None
            post_decision = None

        rows.append(
            {
                "source_hypothesis_id": source,
                "route": route,
                "generation_decision": generation_decision,
                "generated_hypothesis_id": gid,
                "post_verification_decision": post_decision,
                "final_transition": final,
                "effective_hypothesis_id": effective,
            }
        )

    counts = Counter(
        row["final_transition"]
        for row in rows
    )
    return {
        "schema_version": "sers-closed-loop-replacement-map-v1",
        "row_count": len(rows),
        "transition_counts": dict(sorted(counts.items())),
        "rows": rows,
        "shadow_only": True,
        "production_selection_changed": False,
    }


def build_provenance_pin(
    *,
    artifacts: dict[str, Path],
    identifiers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = {}
    for name, path in artifacts.items():
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"cannot pin missing artifact {name}: {path}"
            )
        rows[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    body = {
        "schema_version": "sers-gen0-epistemic-provenance-pin-v1",
        "artifacts": rows,
        "identifiers": dict(identifiers or {}),
        "pin_is_semantic_authority": False,
        "pinned_artifacts_are_immutable_inputs": True,
        "production_selection_changed": False,
    }
    body["pin_id"] = _stable_id(
        "sers_gen0_epistemic_pin",
        rows,
        body["identifiers"],
    )
    return body


def build_final_freeze(
    *,
    provenance_pin: dict[str, Any],
    decisions: dict[str, Any],
    effective_portfolio: HypothesisPortfolio,
    replacement_map: dict[str, Any],
    old_audit: dict[str, Any],
) -> dict[str, Any]:
    effective_ids = [
        str(card.hypothesis_id)
        for card in effective_portfolio.hypotheses
    ]
    decision_by_h = {
        str(row["hypothesis_id"]): row
        for row in decisions.get("rows", [])
    }

    if set(effective_ids) != {
        hid
        for hid, row in decision_by_h.items()
        if row.get("advance_to_effective_gen1")
    }:
        raise ValueError(
            "effective portfolio does not equal consolidated advance set"
        )

    stale_audit_mismatches = []
    old_by_source = {
        str(row.get("source_hypothesis_id")): row
        for row in old_audit.get("rows", [])
        if row.get("source_hypothesis_id")
    }

    expected_old_to_new = {
        "HELD_BEFORE_GENERATION": {
            "HOLD_BEFORE_GENERATION",
        },
        "KNOWN_AXIS_REPEAT": {
            "REJECT_AFTER_FRESH_PRIOR_ART",
        },
        "GEN1_HELD_FOR_EVIDENCE": {
            "HOLD_AFTER_FRESH_REVALIDATION",
        },
        "GEN1_RESIDUAL_CANDIDATE_SHADOW": {
            "REPLACED_BY_GEN1_RESIDUAL_CANDIDATE",
        },
        "CARRIED_RESIDUAL_CANDIDATE": {
            "CARRIED_AND_FRESH_REVALIDATED",
        },
    }

    for row in replacement_map.get("rows", []):
        source = str(row.get("source_hypothesis_id") or "")
        old = old_by_source.get(source)
        if not old:
            continue

        old_outcome = str(old.get("outcome") or "")
        consolidated = str(row.get("final_transition") or "")
        expected = expected_old_to_new.get(old_outcome)

        if expected is not None and consolidated not in expected:
            stale_audit_mismatches.append(
                {
                    "source_hypothesis_id": source,
                    "old_outcome": old_outcome,
                    "consolidated_transition": consolidated,
                    "reason": (
                        "claim-role-aware fresh post-verification consolidation "
                        "changed the v1 source-level outcome"
                    ),
                }
            )

    body = {
        "schema_version": "sers-closed-loop-decision-freeze-v2",
        "provenance_pin_id": provenance_pin.get("pin_id"),
        "decision_report_id": decisions.get("report_id"),
        "source_gen1_portfolio_id": decisions.get(
            "source_gen1_portfolio_id"
        ),
        "effective_gen1_portfolio_id": effective_portfolio.portfolio_id,
        "effective_gen1_hypothesis_ids": effective_ids,
        "effective_gen1_count": len(effective_ids),
        "decision_counts": decisions.get("decision_counts", {}),
        "replacement_transition_counts": replacement_map.get(
            "transition_counts",
            {},
        ),
        "superseded_v1_audit_mismatches": stale_audit_mismatches,
        "supersedes_closed_loop_audit_v1_for_final_candidate_state": True,
        "single_generation_only": True,
        "claim_role_aware_aggregation": True,
        "carried_forward_has_no_epistemic_immunity": True,
        "shadow_only": True,
        "novelty_authority_created": False,
        "generation_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
    }
    body["freeze_id"] = _stable_id(
        "sers_closed_loop_decision_freeze",
        body["provenance_pin_id"],
        body["decision_report_id"],
        effective_ids,
        stale_audit_mismatches,
    )
    return body


def render_final_report(
    *,
    freeze: dict[str, Any],
    decisions: dict[str, Any],
    replacement_map: dict[str, Any],
) -> str:
    lines = [
        "# SERS Closed-loop Decision Consolidation",
        "",
        "This report consolidates the single-generation SERS closed loop after fresh external novelty and residual verification.",
        "",
        "## Final effective Gen1",
        "",
        f"- Effective candidate count: `{freeze['effective_gen1_count']}`",
        f"- Effective portfolio: `{freeze['effective_gen1_portfolio_id']}`",
        f"- Carried candidates have epistemic immunity: `False`",
        f"- Claim-role-aware aggregation: `True`",
        "",
        "## Post-verification decisions",
        "",
        "| Hypothesis | Source | Route | Fresh external | Composite roles | Decision |",
        "|---|---|---|---|---|---|",
    ]

    for row in decisions.get("rows", []):
        roles = Counter(
            comp.get("role_class")
            for comp in row.get("composites", [])
        )
        role_text = ", ".join(
            f"{key}:{value}"
            for key, value in sorted(roles.items())
        ) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("hypothesis_id") or "—"),
                    str(row.get("source_hypothesis_id") or "—"),
                    str(row.get("generation_route") or "—"),
                    str(row.get("fresh_external_status") or "—"),
                    role_text,
                    str(row.get("post_verification_decision") or "—"),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Gen0 → Gen1 transition map",
        "",
        "| Gen0 | Route | Generated / carried | Final transition | Effective |",
        "|---|---|---|---|---|",
    ]

    for row in replacement_map.get("rows", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("source_hypothesis_id") or "—"),
                    str(row.get("route") or "—"),
                    str(row.get("generated_hypothesis_id") or "—"),
                    str(row.get("final_transition") or "—"),
                    str(row.get("effective_hypothesis_id") or "—"),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Authority contract",
        "",
        "- This is a shadow consolidation only.",
        "- Prior art remains exclusion / boundary information, not positive premise evidence.",
        "- A carried candidate is re-evaluated under fresh evidence exactly like a newly generated candidate.",
        "- Novelty-bearing core composites block advancement if any is prior-art-backed or unresolved.",
        "- Supporting / auxiliary unresolved composites are warnings only.",
        "- No novelty, N9, N10, generation, or production authority is created.",
        "",
    ]

    if freeze.get("superseded_v1_audit_mismatches"):
        lines += [
            "## Superseded v1 audit findings",
            "",
        ]
        for row in freeze["superseded_v1_audit_mismatches"]:
            lines.append(
                "- "
                + str(row["source_hypothesis_id"])
                + ": `"
                + str(row["old_outcome"])
                + "` → `"
                + str(row["consolidated_transition"])
                + "` because fresh post-verification evidence changed the candidate state."
            )
        lines.append("")

    return "\n".join(lines)
