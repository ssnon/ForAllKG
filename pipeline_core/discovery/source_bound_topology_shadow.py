from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


RELATION_BACKED_STATUSES = {
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
}


def _norm(text: object) -> str:
    value = str(text or "").casefold()
    value = re.sub(r"[‐‑‒–—−_/]+", " ", value)
    value = re.sub(r"[^\w\s+*.]", " ", value)
    return " ".join(value.split())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_source_binding_bundle(
    root: Path,
    *,
    source_query_plan_id: str,
) -> tuple[Path | None, dict[str, Any] | None]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("schema_version") != "atomic-scientific-source-binding-bundle-v1":
            continue
        if payload.get("source_query_plan_id") != source_query_plan_id:
            continue
        candidates.append((path, payload))

    if not candidates:
        return None, None

    # Prefer the shortest/closest artifact path deterministically.
    candidates.sort(key=lambda row: (len(row[0].parts), str(row[0])))
    return candidates[0]


def _claim_index(query_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for group in query_plan.get("claims", []):
        for claim in group.get("claims", []):
            cid = str(claim.get("claim_id") or "")
            if cid:
                out[cid] = claim
    return out


def _claims_by_hypothesis(
    query_plan: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for group in query_plan.get("claims", []):
        hid = str(group.get("hypothesis_id") or "")
        out[hid] = list(group.get("claims", []))
    return out


def _binding_index(
    bundle: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not bundle:
        return {}
    return {
        str(row.get("claim_id")): row
        for row in bundle.get("records", [])
        if row.get("claim_id")
    }


def source_bound_recoverable_components(
    *,
    composite: dict[str, Any],
    siblings: list[dict[str, Any]],
    binding_by_claim: dict[str, dict[str, Any]],
) -> list[str]:
    """Recover only components whose exact source proposition is contained
    in an explicit higher-order source span.

    This is intentionally conservative and diagnostic-only. It never infers
    a component from lexical similarity, shared entities, or scientific logic.
    """
    bases = [
        _norm(x)
        for x in composite.get("higher_order_relation_basis", [])
        if _norm(x)
    ]
    if not bases:
        return []

    recovered: list[str] = []
    for claim in siblings:
        cid = str(claim.get("claim_id") or "")
        if not cid or cid == composite.get("claim_id"):
            continue
        if claim.get("kind") == "composite":
            continue
        binding = binding_by_claim.get(cid)
        if not binding:
            continue
        proposition = _norm(binding.get("proposition_basis"))
        if not proposition:
            continue
        if any(proposition in basis for basis in bases):
            recovered.append(cid)
    return recovered


def _saturation_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("claim_id")): row
        for row in report.get("claim_saturation", [])
        if row.get("claim_id")
    }


def _external_review_index(
    external_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for card in external_report.get("cards", []):
        for review in card.get("claim_reviews", []):
            cid = str(review.get("claim_id") or "")
            if cid:
                out[cid] = review
    return out


def component_closure(
    component_ids: list[str],
    saturation_by_claim: dict[str, dict[str, Any]],
) -> tuple[str, list[str], list[str], list[str]]:
    if not component_ids:
        return "NOT_APPLICABLE", [], [], []

    backed_ids: list[str] = []
    unresolved_ids: list[str] = []
    work_sets: list[set[str]] = []

    for cid in component_ids:
        row = saturation_by_claim.get(cid)
        if row is None:
            unresolved_ids.append(cid)
            continue

        status = str(row.get("saturation_claim_status") or "")
        work_ids = {
            str(x)
            for x in row.get("relation_backed_work_ids", [])
            if str(x)
        }

        if status in RELATION_BACKED_STATUSES or work_ids:
            backed_ids.append(cid)
            work_sets.append(work_ids)
        elif status in {"INSUFFICIENT_METADATA", "TITLE_ONLY_NEIGHBORS"}:
            unresolved_ids.append(cid)

    if len(backed_ids) < len(component_ids):
        if backed_ids:
            return "PARTIAL", backed_ids, unresolved_ids, []
        return "NONE", backed_ids, unresolved_ids, []

    nonempty_sets = [s for s in work_sets if s]
    if len(nonempty_sets) != len(component_ids):
        return "ALL_BACKED_WORK_ID_INCOMPLETE", backed_ids, unresolved_ids, []

    shared = set.intersection(*nonempty_sets) if nonempty_sets else set()
    if shared:
        return "SAME_WORK", backed_ids, unresolved_ids, sorted(shared)
    return "DISTRIBUTED", backed_ids, unresolved_ids, []


def residual_state(
    *,
    full_relation_status: str,
    topology_state: str,
    component_ids: list[str],
    backed_component_ids: list[str],
    closure: str,
) -> str:
    if full_relation_status in RELATION_BACKED_STATUSES:
        return "NO_RESIDUAL_FULL_RELATION_BACKED"

    if topology_state == "NO_COMPONENT_TOPOLOGY":
        return "NOT_ASSESSABLE_NO_COMPONENT_TOPOLOGY"

    if not component_ids:
        return "NOT_ASSESSABLE_NO_COMPONENT_TOPOLOGY"

    if len(backed_component_ids) == len(component_ids):
        if closure == "SAME_WORK":
            return "HIGHER_ORDER_RESIDUAL_SAME_WORK_KNOWN_BASE"
        if closure == "DISTRIBUTED":
            return "HIGHER_ORDER_RESIDUAL_DISTRIBUTED_KNOWN_BASE"
        return "HIGHER_ORDER_RESIDUAL_KNOWN_BASE"

    if backed_component_ids:
        return "PARTIAL_BASE_SATURATION"

    return "UNSATURATED_BASE"


def build_topology_report(
    *,
    query_plan: dict[str, Any],
    saturation_report: dict[str, Any],
    external_report: dict[str, Any],
    source_binding_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    claims_by_h = _claims_by_hypothesis(query_plan)
    binding_by_claim = _binding_index(source_binding_bundle)
    saturation_by_claim = _saturation_index(saturation_report)
    external_by_claim = _external_review_index(external_report)

    composites: list[dict[str, Any]] = []
    for hid, claims in claims_by_h.items():
        for claim in claims:
            if claim.get("kind") != "composite":
                continue

            explicit = [
                str(x)
                for x in claim.get("higher_order_component_claim_ids", [])
                if str(x)
            ]

            recovered: list[str] = []
            if explicit:
                topology_state = "EXPLICIT"
                effective = explicit
            else:
                recovered = source_bound_recoverable_components(
                    composite=claim,
                    siblings=claims,
                    binding_by_claim=binding_by_claim,
                )
                if recovered:
                    topology_state = "SOURCE_BOUND_RECOVERABLE_SHADOW"
                    effective = recovered
                else:
                    topology_state = "NO_COMPONENT_TOPOLOGY"
                    effective = []

            closure, backed, unresolved, same_work_ids = component_closure(
                effective,
                saturation_by_claim,
            )

            full_review = external_by_claim.get(
                str(claim.get("claim_id") or ""),
                {},
            )
            full_status = str(full_review.get("status") or "UNKNOWN")

            composites.append(
                {
                    "hypothesis_id": hid,
                    "claim_id": claim.get("claim_id"),
                    "claim_text": claim.get("text"),
                    "full_relation_status": full_status,
                    "topology_state": topology_state,
                    "explicit_component_claim_ids": explicit,
                    "source_bound_recovered_component_claim_ids": recovered,
                    "effective_component_claim_ids_shadow": effective,
                    "component_count": len(effective),
                    "relation_backed_component_claim_ids": backed,
                    "relation_backed_component_count": len(backed),
                    "unresolved_component_claim_ids": unresolved,
                    "component_closure": closure,
                    "same_work_closure_work_ids": same_work_ids,
                    "residual_state": residual_state(
                        full_relation_status=full_status,
                        topology_state=topology_state,
                        component_ids=effective,
                        backed_component_ids=backed,
                        closure=closure,
                    ),
                }
            )

    counts: dict[str, int] = {}
    topology_counts: dict[str, int] = {}
    closure_counts: dict[str, int] = {}
    for row in composites:
        counts[row["residual_state"]] = counts.get(row["residual_state"], 0) + 1
        topology_counts[row["topology_state"]] = (
            topology_counts.get(row["topology_state"], 0) + 1
        )
        closure_counts[row["component_closure"]] = (
            closure_counts.get(row["component_closure"], 0) + 1
        )

    return {
        "schema_version": "source-bound-topology-residual-shadow-v2",
        "shadow_only": True,
        "production_authority": False,
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
        "source_bound_recovery_policy": (
            "exact atomic proposition_basis must be contained in an explicit "
            "composite higher_order_relation_basis span; no semantic inference"
        ),
        "source_binding_bundle_available": bool(source_binding_bundle),
        "topology_state_counts": dict(sorted(topology_counts.items())),
        "component_closure_counts": dict(sorted(closure_counts.items())),
        "residual_state_counts": dict(sorted(counts.items())),
        "composites": composites,
    }


def build_sentinel_audit(
    *,
    prior_art_packet: dict[str, Any],
    saturation_claim_reviews: dict[str, Any],
) -> dict[str, Any]:
    works = {
        str(w.get("work_id")): w
        for w in prior_art_packet.get("works", [])
        if w.get("work_id")
    }

    direct_or_partial = []
    for review in saturation_claim_reviews.get("reviews", []):
        for match in review.get("matches", []):
            if match.get("relationship") not in RELATION_BACKED_STATUSES:
                continue
            wid = str(match.get("work_id") or "")
            work = works.get(wid, {})
            direct_or_partial.append(
                {
                    "claim_id": review.get("claim_id"),
                    "work_id": wid,
                    "title": work.get("title") or match.get("title"),
                    "doi": work.get("doi") or match.get("doi"),
                    "relationship": match.get("relationship"),
                }
            )

    false_positive_analogues = []
    for row in direct_or_partial:
        title = _norm(row.get("title"))
        if (
            ("wse2" in title or "ws2" in title or "semiconductor" in title)
            and ("bilayer" in title or "interlayer" in title)
        ):
            false_positive_analogues.append(row)

    # Case-study sentinels are diagnostic only; they are not production logic.
    packet_dois = {
        _norm(w.get("doi")): w
        for w in prior_art_packet.get("works", [])
        if w.get("doi")
    }
    known_case_dois = {
        "h6_au_ag_au_film": "10.1088/1361-6528/ac0ddd",
        "h7_au_hmm": "10.3390/nano11030587",
    }
    known_case_presence = {}
    for name, doi in known_case_dois.items():
        work = packet_dois.get(_norm(doi))
        known_case_presence[name] = {
            "doi": doi,
            "retrieved": bool(work),
            "work_id": work.get("work_id") if work else None,
            "title": work.get("title") if work else None,
            "relation_backed_any_claim": bool(
                work and any(
                    row.get("work_id") == work.get("work_id")
                    for row in direct_or_partial
                )
            ),
        }

    return {
        "schema_version": "lower-order-saturation-sentinel-audit-v1",
        "diagnostic_only": True,
        "production_authority": False,
        "generic_semiconductor_bilayer_relation_backed_count": len(
            false_positive_analogues
        ),
        "generic_semiconductor_bilayer_relation_backed": false_positive_analogues,
        "known_case_presence": known_case_presence,
    }
