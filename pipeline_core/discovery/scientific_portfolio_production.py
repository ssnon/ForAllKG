from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScientificPortfolioNoveltyCertificationRecord(StrictModel):
    hypothesis_id: str
    selection_class: str
    novelty_certified: bool
    positive_nonobviousness_authority: bool
    action: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    blocking_claim_ids: list[str] = Field(default_factory=list)
    unresolved_claim_ids: list[str] = Field(default_factory=list)
    resolution_requirements: list[dict[str, Any]] = Field(default_factory=list)


class ScientificPortfolioNoveltyCertificationReport(StrictModel):
    schema_version: str = "scientific-portfolio-n10-certification-v1"
    report_id: str
    report_sha256: str
    source_portfolio_id: str
    source_production_gate_schema: str
    source_production_gate_sha256: str
    source_query_plan_id: str | None = None
    scientific_candidate_portfolio_id: str
    certified_portfolio_id: str
    hypothesis_count: int = Field(ge=0)
    certified_count: int = Field(ge=0)
    conditional_count: int = Field(ge=0)
    ineligible_count: int = Field(ge=0)
    selection_counts: dict[str, int] = Field(default_factory=dict)
    records: list[ScientificPortfolioNoveltyCertificationRecord] = Field(default_factory=list)

    scientific_candidate_authority: bool = True
    novelty_certification_authority: bool = True
    n10_candidate_survival_authority: bool = False
    conditional_candidates_retained: bool = True
    ineligible_candidates_retained: bool = True

    scientific_reaggregation_performed: bool = False
    external_prior_art_as_positive_premise: bool = False
    positive_premise_authority_created: bool = False
    canonical_graph_mutated: bool = False
    stage8_input_changed: bool = False


class ScientificPortfolioProductionBindingRecord(StrictModel):
    hypothesis_id: str
    selection_class: str
    selected_for_production: bool
    fallback_allowed: bool
    positive_nonobviousness_authority: bool
    reason_codes: list[str] = Field(default_factory=list)


class ScientificPortfolioProductionBindingReport(StrictModel):
    schema_version: str = "scientific-portfolio-production-binding-v1"
    report_id: str
    report_sha256: str
    source_portfolio_id: str
    source_production_gate_schema: str
    source_production_gate_sha256: str
    source_production_gate_authority_scope: str
    source_query_plan_id: str | None = None
    output_portfolio_id: str
    source_hypothesis_count: int = Field(ge=0)
    production_selected_count: int = Field(ge=0)
    production_blocked_count: int = Field(ge=0)
    selection_counts: dict[str, int] = Field(default_factory=dict)
    records: list[ScientificPortfolioProductionBindingRecord] = Field(default_factory=list)
    n10_role_aware_authority_applied: bool = True
    authority_scope: str = "SCIENTIFIC_PORTFOLIO_FINAL_SELECTION"
    authority_rebinding_policy: str = "EXACT_N10_V2_BOOLEAN_PROMOTION_NO_REAGGREGATION"
    scientific_reaggregation_performed: bool = False
    external_prior_art_as_positive_premise: bool = False
    positive_premise_authority_created: bool = False
    canonical_graph_mutated: bool = False
    stage8_input_changed: bool = False
    production_selection_authority: bool = True


_ALLOWED = {"ELIGIBLE", "CONDITIONAL", "INELIGIBLE"}


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}:{_sha(parts)[:20]}"


def _validated_gate(
    portfolio: HypothesisPortfolio,
    gate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if gate.get("schema_version") != "scientific-novelty-fallback-gate-v2":
        raise ValueError("unexpected N10-v2 production gate schema")
    if gate.get("production_authority") is not True:
        raise ValueError("N10-v2 source gate lacks production authority")
    if gate.get("authority_source") != "n10_role_aware_nonobviousness_v2":
        raise ValueError("unexpected N10-v2 authority source")
    if gate.get("source_portfolio_id") != portfolio.portfolio_id:
        raise ValueError("N10-v2 gate/source portfolio lineage mismatch")

    rows = gate.get("gates")
    if not isinstance(rows, list) or gate.get("gate_count") != len(rows):
        raise ValueError("invalid N10-v2 production gate rows")

    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("N10-v2 production gate row must be an object")
        hypothesis_id = str(row.get("hypothesis_id") or "").strip()
        selection_class = str(row.get("selection_class") or "").strip()
        allowed = row.get("fallback_allowed")
        positive = row.get("positive_nonobviousness_authority")

        if not hypothesis_id or hypothesis_id in by_id:
            raise ValueError("invalid or duplicate N10 hypothesis_id")
        if selection_class not in _ALLOWED:
            raise ValueError("unsupported N10 selection class")
        if not isinstance(allowed, bool) or not isinstance(positive, bool):
            raise ValueError("N10 authority fields must be boolean")
        if allowed is not bool(selection_class == "ELIGIBLE" and positive):
            raise ValueError("N10 gate boolean inconsistent with role-aware authority")

        by_id[hypothesis_id] = row

    ids = [card.hypothesis_id for card in portfolio.hypotheses]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate hypothesis IDs in source portfolio")
    if set(ids) != set(by_id):
        raise ValueError("N10-v2 production gate must cover source portfolio exactly")

    return by_id


def certify_scientific_portfolio_novelty(
    *,
    portfolio: HypothesisPortfolio,
    n10_production_gate: dict[str, Any],
) -> tuple[
    ScientificPortfolioNoveltyCertificationReport,
    HypothesisPortfolio,
    HypothesisPortfolio,
]:
    """Preserve scientific candidates; certify only the N10-positive subset."""

    by_id = _validated_gate(portfolio, n10_production_gate)
    counts: Counter[str] = Counter()
    certified_cards = []
    records = []

    for card in portfolio.hypotheses:
        row = by_id[card.hypothesis_id]
        selection_class = str(row["selection_class"])
        counts[selection_class] += 1
        certified = bool(
            selection_class == "ELIGIBLE"
            and row["positive_nonobviousness_authority"] is True
            and row["fallback_allowed"] is True
        )
        if certified:
            certified_cards.append(card)

        records.append(
            ScientificPortfolioNoveltyCertificationRecord(
                hypothesis_id=card.hypothesis_id,
                selection_class=selection_class,
                novelty_certified=certified,
                positive_nonobviousness_authority=bool(
                    row["positive_nonobviousness_authority"]
                ),
                action=(
                    str(row.get("action"))
                    if row.get("action") is not None
                    else None
                ),
                reason_codes=[str(x) for x in row.get("reason_codes", [])],
                blocking_claim_ids=[str(x) for x in row.get("blocking_claim_ids", [])],
                unresolved_claim_ids=[str(x) for x in row.get("unresolved_claim_ids", [])],
                resolution_requirements=[
                    dict(x)
                    for x in row.get("resolution_requirements", [])
                    if isinstance(x, dict)
                ],
            )
        )

    candidate_portfolio = portfolio.model_copy(deep=True)
    certified_id = _stable_id(
        "scientific_portfolio_n10_certified",
        portfolio.portfolio_id,
        _sha(n10_production_gate),
        *(card.hypothesis_id for card in certified_cards),
    )
    certified_portfolio = portfolio.model_copy(
        update={
            "portfolio_id": certified_id,
            "hypotheses": certified_cards,
            "abstention_reason": (
                None
                if certified_cards
                else (
                    "No Scientific Portfolio hypothesis currently has positive "
                    "role-aware N10 novelty certification."
                )
            ),
        }
    )

    provisional = ScientificPortfolioNoveltyCertificationReport(
        report_id="pending",
        report_sha256="pending",
        source_portfolio_id=portfolio.portfolio_id,
        source_production_gate_schema=str(
            n10_production_gate.get("schema_version") or ""
        ),
        source_production_gate_sha256=_sha(n10_production_gate),
        source_query_plan_id=(
            str(n10_production_gate.get("source_query_plan_id"))
            if n10_production_gate.get("source_query_plan_id") is not None
            else None
        ),
        scientific_candidate_portfolio_id=candidate_portfolio.portfolio_id,
        certified_portfolio_id=certified_portfolio.portfolio_id,
        hypothesis_count=len(portfolio.hypotheses),
        certified_count=len(certified_cards),
        conditional_count=counts["CONDITIONAL"],
        ineligible_count=counts["INELIGIBLE"],
        selection_counts=dict(sorted(counts.items())),
        records=records,
    )

    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)

    return (
        provisional.model_copy(
            update={
                "report_id": f"scientific_portfolio_n10_certification:{digest[:20]}",
                "report_sha256": digest,
            }
        ),
        candidate_portfolio,
        certified_portfolio,
    )


def bind_scientific_portfolio_production(
    *,
    portfolio: HypothesisPortfolio,
    n10_production_gate: dict[str, Any],
) -> tuple[
    ScientificPortfolioProductionBindingReport,
    HypothesisPortfolio,
]:
    """Legacy hard-filter behavior retained for explicit comparison/debugging."""

    by_id = _validated_gate(portfolio, n10_production_gate)
    selected_cards = []
    records = []
    counts: Counter[str] = Counter()

    for card in portfolio.hypotheses:
        row = by_id[card.hypothesis_id]
        selection_class = str(row["selection_class"])
        allowed = bool(row["fallback_allowed"])
        counts[selection_class] += 1
        if allowed:
            selected_cards.append(card)

        records.append(
            ScientificPortfolioProductionBindingRecord(
                hypothesis_id=card.hypothesis_id,
                selection_class=selection_class,
                selected_for_production=allowed,
                fallback_allowed=allowed,
                positive_nonobviousness_authority=bool(
                    row["positive_nonobviousness_authority"]
                ),
                reason_codes=[str(x) for x in row.get("reason_codes", [])],
            )
        )

    output = portfolio.model_copy(
        update={
            "portfolio_id": _stable_id(
                "scientific_portfolio_production",
                portfolio.portfolio_id,
                _sha(n10_production_gate),
                *(card.hypothesis_id for card in selected_cards),
            ),
            "hypotheses": selected_cards,
            "abstention_reason": (
                None
                if selected_cards
                else (
                    "No Scientific Portfolio hypothesis received positive "
                    "role-aware N10 production authority."
                )
            ),
        }
    )

    provisional = ScientificPortfolioProductionBindingReport(
        report_id="pending",
        report_sha256="pending",
        source_portfolio_id=portfolio.portfolio_id,
        source_production_gate_schema=str(
            n10_production_gate.get("schema_version") or ""
        ),
        source_production_gate_sha256=_sha(n10_production_gate),
        source_production_gate_authority_scope=str(
            n10_production_gate.get("authority_scope") or ""
        ),
        source_query_plan_id=(
            str(n10_production_gate.get("source_query_plan_id"))
            if n10_production_gate.get("source_query_plan_id") is not None
            else None
        ),
        output_portfolio_id=output.portfolio_id,
        source_hypothesis_count=len(portfolio.hypotheses),
        production_selected_count=len(selected_cards),
        production_blocked_count=len(portfolio.hypotheses) - len(selected_cards),
        selection_counts=dict(sorted(counts.items())),
        records=records,
    )

    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)

    return (
        provisional.model_copy(
            update={
                "report_id": f"scientific_portfolio_production_binding:{digest[:20]}",
                "report_sha256": digest,
            }
        ),
        output,
    )
