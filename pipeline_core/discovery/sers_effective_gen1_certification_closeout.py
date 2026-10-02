
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def build_effective_subset_query_plan(
    *,
    source_plan: LiteratureQueryPlan,
    effective_portfolio: HypothesisPortfolio,
) -> LiteratureQueryPlan:
    effective_ids = {
        str(card.hypothesis_id)
        for card in effective_portfolio.hypotheses
    }
    if not effective_ids:
        raise ValueError("effective Gen1 portfolio is empty")

    source_claim_ids = {
        str(group.hypothesis_id)
        for group in source_plan.claims
    }
    source_query_ids = {
        str(query.hypothesis_id)
        for query in source_plan.queries
    }
    source_ids = source_claim_ids | source_query_ids

    missing = sorted(effective_ids - source_ids)
    if missing:
        raise ValueError(
            "effective hypothesis IDs are absent from source query plan: "
            + ", ".join(missing)
        )

    claims = [
        group
        for group in source_plan.claims
        if str(group.hypothesis_id) in effective_ids
    ]
    queries = [
        query
        for query in source_plan.queries
        if str(query.hypothesis_id) in effective_ids
    ]

    covered_claim_ids = {
        str(group.hypothesis_id)
        for group in claims
    }
    if covered_claim_ids != effective_ids:
        raise ValueError(
            "subset query plan does not contain one claim group for every "
            "effective hypothesis"
        )

    if {
        str(query.hypothesis_id)
        for query in queries
    } != effective_ids:
        raise ValueError(
            "subset query plan does not contain queries for every "
            "effective hypothesis"
        )

    plan_id = _stable_id(
        "literature_query_plan",
        effective_portfolio.portfolio_id,
        *[query.query_id for query in queries],
    )
    body = {
        "schema_version": "literature-query-plan-v1",
        "plan_id": plan_id,
        "source_portfolio_id": effective_portfolio.portfolio_id,
        "queries": [
            query.model_dump(mode="json")
            for query in queries
        ],
        "claims": [
            group.model_dump(mode="json")
            for group in claims
        ],
        "policy_version": "external-novelty-query-policy-v1",
    }
    return LiteratureQueryPlan(
        **body,
        plan_sha256=_sha256_json(body),
    )


def _recursive_hypothesis_rows(
    value: Any,
    hypothesis_ids: set[str],
    *,
    path: str = "$",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if isinstance(value, dict):
        hid = value.get("hypothesis_id")
        if isinstance(hid, str) and hid in hypothesis_ids:
            interesting = {}
            for key in (
                "hypothesis_id",
                "decision",
                "state",
                "shadow_state",
                "full_shadow_state",
                "selection_class",
                "certification_status",
                "novelty_certified",
                "positive_nonobviousness_authority",
                "fallback_allowed",
                "action",
                "reason_codes",
                "ready_for_closure_claim_ids",
                "unresolved_dimensions",
            ):
                if key in value:
                    interesting[key] = value.get(key)
            interesting["_json_path"] = path
            rows.append(interesting)

        for key, child in value.items():
            rows.extend(
                _recursive_hypothesis_rows(
                    child,
                    hypothesis_ids,
                    path=f"{path}.{key}",
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(
                _recursive_hypothesis_rows(
                    child,
                    hypothesis_ids,
                    path=f"{path}[{index}]",
                )
            )

    return rows


def _n9_intake_summary(
    payload: dict[str, Any],
    ids: set[str],
) -> dict[str, Any]:
    rows = []
    states = Counter()

    for hypothesis in payload.get("hypotheses", []):
        hid = str(hypothesis.get("hypothesis_id") or "")
        if hid not in ids:
            continue
        claim_states = Counter(
            str(claim.get("shadow_state") or "UNKNOWN")
            for claim in hypothesis.get("claims", [])
        )
        for key, count in claim_states.items():
            states[key] += count

        rows.append(
            {
                "hypothesis_id": hid,
                "ready_for_closure_claim_ids": list(
                    hypothesis.get(
                        "ready_for_closure_claim_ids",
                        [],
                    )
                ),
                "claim_state_counts": dict(
                    sorted(claim_states.items())
                ),
            }
        )

    return {
        "hypothesis_count": len(rows),
        "claim_state_counts": dict(sorted(states.items())),
        "rows": rows,
    }


def _n9_full_summary(
    payload: dict[str, Any],
    ids: set[str],
) -> dict[str, Any]:
    states = Counter()
    rows_by_h: dict[str, list[dict[str, Any]]] = {}

    for claim in payload.get("claims", []):
        hid = str(claim.get("hypothesis_id") or "")
        if hid not in ids:
            continue
        state = str(
            claim.get("full_shadow_state")
            or claim.get("state")
            or "UNKNOWN"
        )
        states[state] += 1
        rows_by_h.setdefault(hid, []).append(
            {
                "claim_id": claim.get("claim_id"),
                "intake_state": claim.get("intake_state"),
                "full_shadow_state": state,
                "adjudication": claim.get(
                    "adjudication_decision"
                ),
            }
        )

    return {
        "claim_state_counts": dict(sorted(states.items())),
        "rows": [
            {
                "hypothesis_id": hid,
                "claims": rows,
            }
            for hid, rows in sorted(rows_by_h.items())
        ],
    }


def _gate_rows(
    payload: dict[str, Any],
    ids: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("gates", []):
        hid = str(row.get("hypothesis_id") or "")
        if hid not in ids:
            continue
        rows.append(
            {
                "hypothesis_id": hid,
                "selection_class": row.get("selection_class"),
                "positive_nonobviousness_authority": row.get(
                    "positive_nonobviousness_authority"
                ),
                "fallback_allowed": row.get("fallback_allowed"),
                "action": row.get("action"),
                "reason_codes": list(
                    row.get("reason_codes") or []
                ),
            }
        )
    return rows


def _certification_summary(
    payload: dict[str, Any],
    ids: set[str],
) -> dict[str, Any]:
    rows = []
    counts = Counter()

    for row in payload.get("decisions", []):
        hid = str(row.get("hypothesis_id") or "")
        if hid not in ids:
            continue
        status = str(
            row.get("certification_status")
            or "UNKNOWN"
        )
        counts[status] += 1
        rows.append(
            {
                "hypothesis_id": hid,
                "selection_class": row.get("selection_class"),
                "positive_nonobviousness_authority": row.get(
                    "positive_nonobviousness_authority"
                ),
                "fallback_allowed": row.get("fallback_allowed"),
                "certification_status": status,
                "novelty_certified": bool(
                    row.get("novelty_certified")
                ),
                "action": row.get("action"),
                "unresolved_dimensions": list(
                    row.get("unresolved_dimensions") or []
                ),
                "reason_codes": list(
                    row.get("reason_codes") or []
                ),
            }
        )

    return {
        "counts": dict(sorted(counts.items())),
        "rows": rows,
    }


def _load_if_file(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        value = json.loads(
            candidate.read_text(encoding="utf-8")
        )
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def summarize_gen0_baseline(
    *,
    projection: dict[str, Any],
    baseline_comparison: dict[str, Any],
) -> dict[str, Any]:
    ids = {
        str(row.get("hypothesis_id"))
        for row in projection.get("projected", [])
        if row.get("hypothesis_id")
    }

    artifact_rows = []
    for row in baseline_comparison.get("comparisons", []):
        path = row.get("path")
        payload = _load_if_file(path)
        if payload is None:
            artifact_rows.append(
                {
                    "name": row.get("name"),
                    "path": path,
                    "load_status": "UNAVAILABLE",
                    "matching_rows": [],
                }
            )
            continue

        artifact_rows.append(
            {
                "name": row.get("name"),
                "path": path,
                "load_status": "LOADED",
                "matching_rows": _recursive_hypothesis_rows(
                    payload,
                    ids,
                ),
            }
        )

    return {
        "gen0_projected_hypothesis_ids": sorted(ids),
        "artifact_count": len(artifact_rows),
        "artifacts": artifact_rows,
        "comparison_semantics": (
            "historical persisted N9/N10 records for the Gen0 projected "
            "candidate set; Gen0 and Gen1 are different hypothesis sets and "
            "must not be treated as paired identical-candidate measurements"
        ),
    }


def build_certification_closeout(
    *,
    effective_portfolio: HypothesisPortfolio,
    n9_intake: dict[str, Any],
    n9_full: dict[str, Any],
    candidate_gate: dict[str, Any],
    production_gate: dict[str, Any],
    certification: dict[str, Any],
    certified_portfolio: HypothesisPortfolio,
    manifest: dict[str, Any],
    gen0_projection: dict[str, Any],
    gen0_baseline_comparison: dict[str, Any],
) -> dict[str, Any]:
    ids = {
        str(card.hypothesis_id)
        for card in effective_portfolio.hypotheses
    }
    certified_ids = {
        str(card.hypothesis_id)
        for card in certified_portfolio.hypotheses
    }
    if not certified_ids <= ids:
        raise ValueError(
            "certified portfolio is not a subset of effective Gen1"
        )

    cert = _certification_summary(certification, ids)
    cert_by_h = {
        row["hypothesis_id"]: row
        for row in cert["rows"]
    }

    if set(cert_by_h) != ids:
        raise ValueError(
            "N10 certification report does not exactly cover effective Gen1"
        )

    final_rows = []
    for card in effective_portfolio.hypotheses:
        hid = str(card.hypothesis_id)
        row = cert_by_h[hid]
        status = row["certification_status"]

        if status == "NOVELTY_CERTIFIED":
            n10_diagnostic = "N10_REFERENCE_POSITIVE"
        elif status == "NOVELTY_UNRESOLVED":
            n10_diagnostic = "N10_REFERENCE_UNRESOLVED"
        elif status == "NOVELTY_REJECTED":
            n10_diagnostic = "N10_REFERENCE_NEGATIVE"
        else:
            n10_diagnostic = "N10_REFERENCE_UNKNOWN"

        final_rows.append(
            {
                "hypothesis_id": hid,
                "title": card.title,
                "research_candidate_state": (
                    "EFFECTIVE_GEN1_RESEARCH_CANDIDATE"
                ),
                "n10_reference_state": n10_diagnostic,
                "certification_status": status,
                "positive_nonobviousness_authority": row[
                    "positive_nonobviousness_authority"
                ],
                "selection_class": row["selection_class"],
                "action": row["action"],
                "unresolved_dimensions": row[
                    "unresolved_dimensions"
                ],
                "reason_codes": row["reason_codes"],
                "in_certified_subset": hid in certified_ids,
                "n10_can_remove_research_candidate": False,
                "n10_can_upgrade_research_candidate": False,
            }
        )

    counts = Counter(
        row["n10_reference_state"]
        for row in final_rows
    )

    return {
        "schema_version":
            "sers-effective-gen1-certification-reference-closeout-v1",
        "source_effective_gen1_portfolio_id":
            effective_portfolio.portfolio_id,
        "strict_n9_n10_chain_completed":
            manifest.get("status") == "complete",
        "authority_mode":
            manifest.get("authority_mode")
            or "certification_only",
        "n9_intake":
            _n9_intake_summary(n9_intake, ids),
        "n9_full":
            _n9_full_summary(n9_full, ids),
        "candidate_gate_rows":
            _gate_rows(candidate_gate, ids),
        "production_gate_rows":
            _gate_rows(production_gate, ids),
        "n10_certification":
            cert,
        "n10_reference_state_counts":
            dict(sorted(counts.items())),
        "final_candidates":
            final_rows,
        "certified_hypothesis_ids":
            sorted(certified_ids),
        "candidate_portfolio_preserved":
            True,
        "certification_only":
            True,
        "n10_is_reference_only_for_research_candidate_selection":
            True,
        "n10_grounding_bias_may_reduce_free_novelty_exploration":
            True,
        "candidate_retention_is_not_novelty_authority":
            True,
        "positive_n10_authority_created_for_certified_subset":
            bool(certified_ids),
        "n10_result_can_change_effective_gen1_membership":
            False,
        "production_selection_changed":
            False,
        "canonical_graph_mutated":
            False,
        "gen0_baseline":
            summarize_gen0_baseline(
                projection=gen0_projection,
                baseline_comparison=gen0_baseline_comparison,
            ),
        "comparison_scope": (
            "Gen0 persisted N9/N10 state versus effective Gen1 fresh "
            "certification outcome; candidate identities differ after "
            "closed-loop replacement, so comparison is stage/outcome-level "
            "rather than paired hypothesis-level causal attribution"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SERS Effective Gen1 Certification Closeout",
        "",
        "## Scope",
        "",
        "- Input: the two effective Gen1 residual candidates from the frozen SERS closed loop.",
        "- Evaluation: existing strict N9 intake/full chain followed by N10 certification-only authority.",
        "- Effective Gen1 research-candidate membership is frozen before N10.",
        "- N10 is reference-only here because its grounding-heavy design can suppress freer novelty exploration.",
        "- Candidate retention is independent of N10 certification.",
        "- This is search-bounded diagnostic certification, not literature-wide novelty proof, scientific truth, or candidate-selection authority.",
        "",
        "## Effective Gen1 outcomes",
        "",
        "| Hypothesis | Title | N10 class | Positive authority | Certification | Final state |",
        "|---|---|---|---:|---|---|",
    ]
    for row in report.get("final_candidates", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["hypothesis_id"]),
                    str(row["title"]).replace("|", "\\|"),
                    str(row["selection_class"]),
                    str(
                        row["positive_nonobviousness_authority"]
                    ),
                    str(row["certification_status"]),
                    str(row["n10_reference_state"]),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## N9 intake",
        "",
        f"- Claim states: `{report['n9_intake']['claim_state_counts']}`",
        "",
        "## N9 full review",
        "",
        f"- Claim states: `{report['n9_full']['claim_state_counts']}`",
        "",
        "## N10 certification",
        "",
        f"- Diagnostic counts: `{report['n10_reference_state_counts']}`",
        f"- N10-positive reference IDs: `{report['certified_hypothesis_ids']}`",
        "- These IDs are not a stricter replacement for the effective Gen1 research-candidate set.",
        "",
        "## Gen0 comparison",
        "",
        (
            "The historical Gen0 comparison uses persisted certification "
            "artifacts identified during the V8 closeout. Gen0 and Gen1 are "
            "different hypotheses after re-axis/replacement, so these records "
            "show stage/outcome differences rather than a paired evaluation "
            "of identical candidates."
        ),
        "",
    ]

    for artifact in report["gen0_baseline"]["artifacts"]:
        lines.append(
            f"### {artifact.get('name') or 'artifact'}"
        )
        lines.append("")
        lines.append(
            f"- Path: `{artifact.get('path')}`"
        )
        lines.append(
            f"- Status: `{artifact.get('load_status')}`"
        )
        for row in artifact.get("matching_rows", []):
            lines.append(
                "- `"
                + str(row.get("hypothesis_id"))
                + "` "
                + json.dumps(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {
                            "hypothesis_id",
                            "_json_path",
                        }
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        lines.append("")

    lines += [
        "## Authority semantics",
        "",
        "- N10 is retained as a diagnostic reference, not as the final novelty/research-candidate metric.",
        "- `NOVELTY_CERTIFIED` means the candidate also satisfies the existing grounding-heavy positive N10 authority contract.",
        "- `NOVELTY_UNRESOLVED` and `NOVELTY_REJECTED` do not remove an already-frozen effective Gen1 research candidate in this closeout.",
        "- Certification-only mode does not mutate effective Gen1 membership, production selection, or the canonical graph.",
        "",
    ]
    return "\n".join(lines)
