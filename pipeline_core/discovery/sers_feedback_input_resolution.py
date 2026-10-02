
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > 30_000_000:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _all_json(roots: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    seen = set()
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            payload = _load(path)
            if payload is not None:
                rows.append((path, payload))
    return rows


def _context_match(payload: dict[str, Any], cid: str, sha: str) -> bool:
    candidate = payload.get("grounded_context")
    if isinstance(candidate, dict):
        payload = candidate
    return (
        str(payload.get("context_id") or "") == cid
        and str(payload.get("context_sha256") or "") == sha
        and isinstance(payload.get("evidence_statements"), list)
    )


def _score_external(path: Path, payload: dict[str, Any]) -> tuple[int, int, float]:
    name = path.name.lower()
    score = 0
    for token, weight in (
        ("residual_replay_v2_resume", 80),
        ("residual_replay_v2", 60),
        ("topology_replay_v1_resume", 50),
        ("topology_replay_v1", 40),
        ("external_novelty", 20),
    ):
        if token in name:
            score += weight
    backed = 0
    for card in payload.get("cards", []):
        for review in card.get("claim_reviews", []):
            if review.get("status") in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}:
                backed += 1
    return score, backed, path.stat().st_mtime


def resolve_inputs(
    *,
    run_root: Path,
    abl: Path,
    portfolio_path: Path,
) -> dict[str, Any]:
    portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
    portfolio_id = str(portfolio.get("portfolio_id") or "")
    context_id = str(portfolio.get("source_context_id") or "")
    context_sha = str(portfolio.get("source_context_sha256") or "")
    hypothesis_ids = {
        str(row.get("hypothesis_id"))
        for row in portfolio.get("hypotheses", [])
        if row.get("hypothesis_id")
    }

    rows = _all_json([abl, run_root])

    contexts = [
        path
        for path, payload in rows
        if _context_match(payload, context_id, context_sha)
    ]
    if not contexts:
        raise RuntimeError(
            "No HypothesisContext/DualHypothesisContext matching the "
            "materialized portfolio lineage was found."
        )
    contexts.sort(
        key=lambda p: (
            0 if "dual" in p.name.lower() else 1,
            len(str(p)),
        )
    )
    context_source = contexts[0]

    external_candidates = []
    for path, payload in rows:
        if str(payload.get("source_portfolio_id") or "") != portfolio_id:
            continue
        cards = payload.get("cards")
        if not isinstance(cards, list):
            continue
        ids = {
            str(x.get("hypothesis_id"))
            for x in cards
            if isinstance(x, dict) and x.get("hypothesis_id")
        }
        if ids != hypothesis_ids:
            continue
        if not payload.get("report_id"):
            continue
        external_candidates.append((path, payload))

    if not external_candidates:
        raise RuntimeError(
            "No exact Gen0 ExternalNoveltyReport matching the SERS "
            "materialized portfolio was found."
        )

    external_candidates.sort(
        key=lambda row: _score_external(row[0], row[1]),
        reverse=True,
    )
    external_path, external = external_candidates[0]

    # ExternalNoveltyReport does not carry source_query_plan_id directly.
    # Canonical lineage is:
    #
    # ExternalNoveltyReport
    #   -> source_prior_art_packet_id
    # PriorArtPacket
    #   -> source_query_plan_id
    # LiteratureQueryPlan
    packet_id = str(
        external.get("source_prior_art_packet_id")
        or ""
    )
    if not packet_id:
        raise RuntimeError(
            "Selected Gen0 external report is missing "
            "source_prior_art_packet_id."
        )

    packet_matches = [
        (path, payload)
        for path, payload in rows
        if (
            str(payload.get("packet_id") or "") == packet_id
            and isinstance(payload.get("works"), list)
        )
    ]
    if not packet_matches:
        raise RuntimeError(
            "Could not resolve the prior-art packet referenced by the "
            "selected Gen0 external report."
        )

    packet_matches.sort(
        key=lambda row: (
            0
            if "residual_replay_v2" in row[0].name.lower()
            else 1,
            len(str(row[0])),
        )
    )
    packet_path, packet = packet_matches[0]

    if str(packet.get("source_portfolio_id") or "") != portfolio_id:
        raise RuntimeError(
            "Resolved prior-art packet does not match the Gen0 portfolio."
        )

    query_plan_id = str(
        packet.get("source_query_plan_id")
        or ""
    )
    if not query_plan_id:
        raise RuntimeError(
            "Resolved Gen0 prior-art packet is missing "
            "source_query_plan_id."
        )

    query_candidates = [
        path
        for path, payload in rows
        if (
            str(payload.get("plan_id") or "") == query_plan_id
            and str(payload.get("source_portfolio_id") or "")
            == portfolio_id
            and isinstance(payload.get("queries"), list)
        )
    ]
    if not query_candidates:
        raise RuntimeError(
            "Could not resolve the LiteratureQueryPlan referenced by the "
            "selected Gen0 prior-art packet. "
            f"query_plan_id={query_plan_id!r}"
        )
    query_candidates.sort(
        key=lambda p: (
            0 if "residual_replay_v2" in p.name.lower() else
            1 if "topology_replay_v1" in p.name.lower() else
            2,
            len(str(p)),
        )
    )
    query_path = query_candidates[0]

    materialization = [
        path
        for path, payload in rows
        if (
            payload.get("schema_version")
            == "scientific-portfolio-materialization-report-v1"
            and str(payload.get("output_portfolio_id") or "")
            == portfolio_id
        )
    ]
    materialization.sort(key=lambda p: len(str(p)))

    provider_candidates = [
        path
        for path, payload in rows
        if (
            "provider_plan" in path.name.lower()
            and isinstance(payload, dict)
        )
    ]
    if not provider_candidates:
        raise RuntimeError(
            "No literature provider plan was found in the SERS run."
        )
    provider_candidates.sort(
        key=lambda p: (
            0 if "residual_replay_v2" in p.name.lower() else
            1 if "topology_replay_v1" in p.name.lower() else
            2,
            -p.stat().st_mtime,
        )
    )
    provider_path = provider_candidates[0]

    return {
        "schema_version": "sers-closed-loop-input-resolution-v1",
        "portfolio": str(portfolio_path.resolve()),
        "portfolio_id": portfolio_id,
        "context_source": str(context_source.resolve()),
        "external_report": str(external_path.resolve()),
        "external_query_plan": str(query_path.resolve()),
        "external_prior_art": str(packet_path.resolve()),
        "provider_plan": str(provider_path.resolve()),
        "materialization_report": (
            str(materialization[0].resolve())
            if materialization
            else None
        ),
        "materialization_lineage_available": bool(materialization),
        "resolver_is_semantic_authority": False,
        "production_selection_changed": False,
    }
