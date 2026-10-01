from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReview,
    ExternalNoveltyReport,
    HypothesisNoveltyClaims,
    LiteratureQuery,
    LiteratureQueryPlan,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SaturationTargetRole = Literal[
    "CORE_ATOMIC",
    "COMPOSITE_COMPONENT",
]

ResidualClass = Literal[
    "FULL_RELATION_ALREADY_BACKED",
    "SAME_WORK_COMPONENT_CLOSURE_WITH_RESIDUAL_GAP",
    "DISTRIBUTED_KNOWN_COMPONENTS_WITH_RESIDUAL_GAP",
    "PARTIAL_COMPONENT_SATURATION",
    "UNSATURATED_COMPONENT_BASE",
    "UNRESOLVED_COMPONENT_BASE",
]


class LowerOrderSaturationTarget(StrictModel):
    hypothesis_id: str
    claim_id: str
    target_roles: list[SaturationTargetRole] = Field(default_factory=list)
    parent_composite_claim_ids: list[str] = Field(default_factory=list)
    query_ids: list[str] = Field(default_factory=list)
    query_texts: list[str] = Field(default_factory=list)


class LowerOrderClaimSaturation(StrictModel):
    hypothesis_id: str
    claim_id: str
    claim_text: str
    importance: str
    novelty_selection_role: str | None = None
    target_roles: list[SaturationTargetRole] = Field(default_factory=list)
    parent_composite_claim_ids: list[str] = Field(default_factory=list)

    original_claim_status: str | None = None
    saturation_claim_status: str

    direct_prior_art_work_ids: list[str] = Field(default_factory=list)
    partial_prior_art_work_ids: list[str] = Field(default_factory=list)
    relation_backed_work_ids: list[str] = Field(default_factory=list)

    query_count: int = 0
    successful_query_count: int = 0
    unique_work_count: int = 0
    abstract_work_count: int = 0

    relation_backed: bool = False
    unresolved: bool = False


class CompositeResidualSaturation(StrictModel):
    hypothesis_id: str
    composite_claim_id: str
    composite_claim_text: str

    component_claim_ids: list[str] = Field(default_factory=list)
    saturated_component_claim_ids: list[str] = Field(default_factory=list)
    unsaturated_component_claim_ids: list[str] = Field(default_factory=list)
    unresolved_component_claim_ids: list[str] = Field(default_factory=list)

    original_composite_claim_status: str | None = None

    all_components_relation_backed: bool = False
    same_work_component_closure_work_ids: list[str] = Field(default_factory=list)
    residual_class: ResidualClass

    residual_novelty_preserved: bool = False
    interpretation: str


class HypothesisLowerOrderSaturationSummary(StrictModel):
    hypothesis_id: str
    title: str
    source_external_status: str
    target_claim_count: int = 0
    relation_backed_target_count: int = 0
    unresolved_target_count: int = 0
    same_work_closure_composite_count: int = 0
    residual_gap_composite_count: int = 0


class LowerOrderPriorArtSaturationReport(StrictModel):
    schema_version: Literal[
        "lower-order-prior-art-saturation-shadow-v1"
    ] = "lower-order-prior-art-saturation-shadow-v1"

    report_id: str
    report_sha256: str

    source_portfolio_id: str
    source_query_plan_id: str
    source_external_report_id: str
    saturation_query_plan_id: str
    saturation_prior_art_packet_id: str

    targets: list[LowerOrderSaturationTarget] = Field(default_factory=list)
    claim_saturation: list[LowerOrderClaimSaturation] = Field(default_factory=list)
    composite_residuals: list[CompositeResidualSaturation] = Field(default_factory=list)
    hypothesis_summaries: list[HypothesisLowerOrderSaturationSummary] = Field(
        default_factory=list
    )

    saturation_status_counts: dict[str, int] = Field(default_factory=dict)
    residual_class_counts: dict[str, int] = Field(default_factory=dict)

    exact_verification_query_count: int = 0
    relation_backed_target_count: int = 0
    same_work_closure_composite_count: int = 0
    residual_gap_composite_count: int = 0

    shadow_only: Literal[True] = True
    external_prior_art_as_positive_premise: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    n9_authority_created: Literal[False] = False
    n10_authority_created: Literal[False] = False
    production_selection_changed: Literal[False] = False

    interpretation: str


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _clean_terms(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value or "").split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


_QUERY_COMPACTION_STOP = {
    "a", "an", "and", "among", "as", "at", "be", "between", "beyond",
    "by", "can", "changing", "change", "changes", "comparison",
    "comparable", "configuration", "dependence", "different", "for",
    "from", "in", "independently", "is", "it", "may", "mean",
    "measurement", "number", "of", "on", "or", "provides", "relative",
    "relates", "relationship", "structural", "that", "the", "their",
    "this", "to", "under", "value", "with", "within",
}


def _compact_query(value: str, *, max_tokens: int = 10) -> str:
    import re
    text = str(value or "")
    text = re.sub(r"[/_]+", " ", text)
    tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text)
    output: list[str] = []
    for token in tokens:
        if token.lower() in _QUERY_COMPACTION_STOP:
            continue
        output.append(token)
        if len(output) >= max_tokens:
            break
    return " ".join(output)


def _query_texts_for_claim(claim: Any) -> list[str]:
    candidates: list[str] = []

    identity = list(getattr(claim, "prior_art_identity_terms", []) or [])
    structural = list(getattr(claim, "diagnostic_structural_terms", []) or [])
    nucleus = list(getattr(claim, "relation_nucleus_terms", []) or [])
    concepts = list(getattr(claim, "search_concepts", []) or [])

    literal = _compact_query(
        str(getattr(claim, "text", "") or ""),
        max_tokens=10,
    )
    if literal:
        candidates.append(literal)

    for parts in (
        [*identity, *nucleus],
        [*structural, *identity, *nucleus],
        [*identity, *structural, *concepts],
    ):
        q = _compact_query(" ".join(parts), max_tokens=10)
        if q:
            candidates.append(q)

    for value in list(getattr(claim, "search_queries", []) or []):
        compact = _compact_query(value, max_tokens=10)
        if compact:
            candidates.append(compact)

    diagnostic = _compact_query(
        str(getattr(claim, "diagnostic_execution_query", "") or ""),
        max_tokens=10,
    )
    if diagnostic:
        candidates.append(diagnostic)

    for value in [*identity, *structural]:
        compact = _compact_query(value, max_tokens=10)
        if len(compact.split()) >= 2:
            candidates.append(compact)

    output: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        cleaned = " ".join(value.split())
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        if len(cleaned.split()) < 2:
            continue
        seen.add(key)
        output.append(cleaned)
        if len(output) >= 8:
            break

    return output

def build_lower_order_saturation_query_plan(
    base: LiteratureQueryPlan,
) -> tuple[LiteratureQueryPlan, list[LowerOrderSaturationTarget]]:
    """Create a shadow exact-verification plan from existing atomic claims only.

    Target population:
      * non-composite core atomic claims;
      * non-composite claims explicitly referenced by a composite claim's
        higher_order_component_claim_ids.

    This function never decomposes text itself and never invents an edge.
    """
    component_parents: dict[str, set[str]] = {}
    by_id: dict[str, Any] = {}

    for group in base.claims:
        for claim in group.claims:
            by_id[claim.claim_id] = claim
            if claim.kind != "composite":
                continue
            for component_id in claim.higher_order_component_claim_ids:
                component_parents.setdefault(component_id, set()).add(
                    claim.claim_id
                )

    targets: list[LowerOrderSaturationTarget] = []
    queries: list[LiteratureQuery] = []
    seen_query: set[tuple[str, str, str]] = set()

    for group in base.claims:
        for claim in group.claims:
            if claim.kind == "composite":
                continue

            roles: list[SaturationTargetRole] = []
            if claim.importance == "core":
                roles.append("CORE_ATOMIC")
            parents = sorted(component_parents.get(claim.claim_id, set()))
            if parents:
                roles.append("COMPOSITE_COMPONENT")
            if not roles:
                continue

            query_texts = _query_texts_for_claim(claim)
            query_ids: list[str] = []
            for query_text in query_texts:
                key = (
                    claim.hypothesis_id,
                    claim.claim_id,
                    query_text.lower(),
                )
                if key in seen_query:
                    continue
                seen_query.add(key)
                query_id = _stable_id(
                    "literature_query",
                    base.plan_id,
                    claim.hypothesis_id,
                    claim.claim_id,
                    "claim_exact_verification",
                    query_text,
                )
                query_ids.append(query_id)
                queries.append(
                    LiteratureQuery(
                        query_id=query_id,
                        hypothesis_id=claim.hypothesis_id,
                        claim_id=claim.claim_id,
                        query_kind="claim_exact_verification",
                        query_text=query_text,
                    )
                )

            targets.append(
                LowerOrderSaturationTarget(
                    hypothesis_id=claim.hypothesis_id,
                    claim_id=claim.claim_id,
                    target_roles=roles,
                    parent_composite_claim_ids=parents,
                    query_ids=query_ids,
                    query_texts=query_texts,
                )
            )

    plan_id = _stable_id(
        "literature_query_plan",
        base.source_portfolio_id,
        base.plan_id,
        "lower_order_saturation_shadow",
        *[row.query_id for row in queries],
    )
    body = {
        "schema_version": "literature-query-plan-v1",
        "plan_id": plan_id,
        "source_portfolio_id": base.source_portfolio_id,
        "queries": [row.model_dump(mode="json") for row in queries],
        "claims": [
            group.model_dump(mode="json")
            for group in base.claims
        ],
        "policy_version": "external-novelty-query-policy-v1",
    }
    return (
        LiteratureQueryPlan(
            **body,
            plan_sha256=_sha256_json(body),
        ),
        targets,
    )


def _relation_backed_ids(review: ClaimPriorArtReview) -> tuple[list[str], list[str]]:
    direct = sorted(
        {
            row.work_id
            for row in review.matches
            if row.relationship == "DIRECT_PRIOR_ART"
        }
    )
    partial = sorted(
        {
            row.work_id
            for row in review.matches
            if row.relationship == "PARTIAL_PRIOR_ART"
        }
    )
    return direct, partial


def _residual_class(
    *,
    original_composite_status: str | None,
    component_ids: list[str],
    relation_backed_by_component: dict[str, set[str]],
    unresolved_component_ids: set[str],
) -> tuple[ResidualClass, list[str], bool, str]:
    if original_composite_status in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}:
        return (
            "FULL_RELATION_ALREADY_BACKED",
            [],
            False,
            "The composite relation itself already has direct or partial prior-art coverage.",
        )

    unresolved = [cid for cid in component_ids if cid in unresolved_component_ids]
    if unresolved:
        return (
            "UNRESOLVED_COMPONENT_BASE",
            [],
            False,
            "At least one explicit component relation remains unresolved under exact-verification retrieval.",
        )

    saturated = [
        cid
        for cid in component_ids
        if relation_backed_by_component.get(cid)
    ]
    if not component_ids:
        return (
            "UNSATURATED_COMPONENT_BASE",
            [],
            False,
            "The composite claim has no explicit component topology to saturate.",
        )

    if len(saturated) == len(component_ids):
        work_sets = [
            relation_backed_by_component[cid]
            for cid in component_ids
        ]
        closure = sorted(set.intersection(*work_sets)) if work_sets else []
        if closure:
            return (
                "SAME_WORK_COMPONENT_CLOSURE_WITH_RESIDUAL_GAP",
                closure,
                True,
                "All explicit lower-order component relations are prior-art-backed and at least one work covers every component; only the higher-order residual relation remains candidate novelty.",
            )
        return (
            "DISTRIBUTED_KNOWN_COMPONENTS_WITH_RESIDUAL_GAP",
            [],
            True,
            "All explicit lower-order component relations are prior-art-backed, but not by one common work; the higher-order residual relation remains candidate novelty.",
        )

    if saturated:
        return (
            "PARTIAL_COMPONENT_SATURATION",
            [],
            False,
            "Only a subset of the explicit lower-order component relations is prior-art-backed.",
        )

    return (
        "UNSATURATED_COMPONENT_BASE",
        [],
        False,
        "No explicit lower-order component relation received direct or partial prior-art coverage in the saturation pass.",
    )


def build_lower_order_saturation_report(
    *,
    base_plan: LiteratureQueryPlan,
    source_external_report: ExternalNoveltyReport,
    saturation_plan: LiteratureQueryPlan,
    saturation_packet_id: str,
    targets: list[LowerOrderSaturationTarget],
    saturation_reviews: list[ClaimPriorArtReview],
) -> LowerOrderPriorArtSaturationReport:
    claim_by_id = {
        claim.claim_id: claim
        for group in base_plan.claims
        for claim in group.claims
    }
    original_review_by_id = {
        review.claim_id: review
        for card in source_external_report.cards
        for review in card.claim_reviews
    }
    saturation_review_by_id = {
        review.claim_id: review
        for review in saturation_reviews
    }
    target_by_id = {row.claim_id: row for row in targets}

    claim_rows: list[LowerOrderClaimSaturation] = []
    relation_backed_by_component: dict[str, set[str]] = {}
    unresolved_component_ids: set[str] = set()

    for target in targets:
        claim = claim_by_id[target.claim_id]
        review = saturation_review_by_id.get(target.claim_id)
        if review is None:
            raise ValueError(
                "missing saturation review for target claim: "
                + target.claim_id
            )
        direct_ids, partial_ids = _relation_backed_ids(review)
        relation_ids = sorted(set(direct_ids) | set(partial_ids))
        original_review = original_review_by_id.get(target.claim_id)

        unresolved = review.status in {
            "INSUFFICIENT_METADATA",
            "TITLE_ONLY_NEIGHBORS",
        }
        if unresolved:
            unresolved_component_ids.add(target.claim_id)
        if relation_ids:
            relation_backed_by_component[target.claim_id] = set(relation_ids)

        claim_rows.append(
            LowerOrderClaimSaturation(
                hypothesis_id=claim.hypothesis_id,
                claim_id=claim.claim_id,
                claim_text=claim.text,
                importance=claim.importance,
                novelty_selection_role=claim.novelty_selection_role,
                target_roles=target.target_roles,
                parent_composite_claim_ids=target.parent_composite_claim_ids,
                original_claim_status=(
                    original_review.status
                    if original_review is not None
                    else None
                ),
                saturation_claim_status=review.status,
                direct_prior_art_work_ids=direct_ids,
                partial_prior_art_work_ids=partial_ids,
                relation_backed_work_ids=relation_ids,
                query_count=review.coverage.query_count,
                successful_query_count=review.coverage.successful_query_count,
                unique_work_count=review.coverage.unique_work_count,
                abstract_work_count=review.coverage.abstract_work_count,
                relation_backed=bool(relation_ids),
                unresolved=unresolved,
            )
        )

    composite_rows: list[CompositeResidualSaturation] = []
    for group in base_plan.claims:
        for claim in group.claims:
            if claim.kind != "composite":
                continue

            component_ids = list(claim.higher_order_component_claim_ids)
            original = original_review_by_id.get(claim.claim_id)

            saturated = [
                cid
                for cid in component_ids
                if relation_backed_by_component.get(cid)
            ]
            unsaturated = [
                cid
                for cid in component_ids
                if cid not in saturated
                and cid not in unresolved_component_ids
            ]
            unresolved = [
                cid
                for cid in component_ids
                if cid in unresolved_component_ids
            ]

            residual_class, closure, residual_preserved, interpretation = (
                _residual_class(
                    original_composite_status=(
                        original.status if original is not None else None
                    ),
                    component_ids=component_ids,
                    relation_backed_by_component=relation_backed_by_component,
                    unresolved_component_ids=unresolved_component_ids,
                )
            )

            composite_rows.append(
                CompositeResidualSaturation(
                    hypothesis_id=claim.hypothesis_id,
                    composite_claim_id=claim.claim_id,
                    composite_claim_text=claim.text,
                    component_claim_ids=component_ids,
                    saturated_component_claim_ids=saturated,
                    unsaturated_component_claim_ids=unsaturated,
                    unresolved_component_claim_ids=unresolved,
                    original_composite_claim_status=(
                        original.status if original is not None else None
                    ),
                    all_components_relation_backed=bool(
                        component_ids
                        and len(saturated) == len(component_ids)
                    ),
                    same_work_component_closure_work_ids=closure,
                    residual_class=residual_class,
                    residual_novelty_preserved=residual_preserved,
                    interpretation=interpretation,
                )
            )

    cards = {
        card.hypothesis_id: card
        for card in source_external_report.cards
    }
    claim_rows_by_hypothesis: dict[str, list[LowerOrderClaimSaturation]] = {}
    composite_by_hypothesis: dict[str, list[CompositeResidualSaturation]] = {}
    for row in claim_rows:
        claim_rows_by_hypothesis.setdefault(row.hypothesis_id, []).append(row)
    for row in composite_rows:
        composite_by_hypothesis.setdefault(row.hypothesis_id, []).append(row)

    summaries: list[HypothesisLowerOrderSaturationSummary] = []
    for group in base_plan.claims:
        card = cards.get(group.hypothesis_id)
        claim_group = claim_rows_by_hypothesis.get(group.hypothesis_id, [])
        composites = composite_by_hypothesis.get(group.hypothesis_id, [])
        summaries.append(
            HypothesisLowerOrderSaturationSummary(
                hypothesis_id=group.hypothesis_id,
                title=(card.title if card is not None else group.title),
                source_external_status=(
                    card.status if card is not None else "UNKNOWN"
                ),
                target_claim_count=len(claim_group),
                relation_backed_target_count=sum(
                    row.relation_backed for row in claim_group
                ),
                unresolved_target_count=sum(
                    row.unresolved for row in claim_group
                ),
                same_work_closure_composite_count=sum(
                    bool(row.same_work_component_closure_work_ids)
                    for row in composites
                ),
                residual_gap_composite_count=sum(
                    row.residual_novelty_preserved
                    for row in composites
                ),
            )
        )

    saturation_counts = Counter(
        row.saturation_claim_status
        for row in claim_rows
    )
    residual_counts = Counter(
        row.residual_class
        for row in composite_rows
    )

    body = {
        "schema_version": "lower-order-prior-art-saturation-shadow-v1",
        "source_portfolio_id": base_plan.source_portfolio_id,
        "source_query_plan_id": base_plan.plan_id,
        "source_external_report_id": source_external_report.report_id,
        "saturation_query_plan_id": saturation_plan.plan_id,
        "saturation_prior_art_packet_id": saturation_packet_id,
        "targets": [row.model_dump(mode="json") for row in targets],
        "claim_saturation": [row.model_dump(mode="json") for row in claim_rows],
        "composite_residuals": [
            row.model_dump(mode="json")
            for row in composite_rows
        ],
        "hypothesis_summaries": [
            row.model_dump(mode="json")
            for row in summaries
        ],
        "saturation_status_counts": dict(sorted(saturation_counts.items())),
        "residual_class_counts": dict(sorted(residual_counts.items())),
        "exact_verification_query_count": len(saturation_plan.queries),
        "relation_backed_target_count": sum(
            row.relation_backed for row in claim_rows
        ),
        "same_work_closure_composite_count": sum(
            bool(row.same_work_component_closure_work_ids)
            for row in composite_rows
        ),
        "residual_gap_composite_count": sum(
            row.residual_novelty_preserved
            for row in composite_rows
        ),
        "shadow_only": True,
        "external_prior_art_as_positive_premise": False,
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
        "interpretation": (
            "Exact-verification retrieval saturates only canonical atomic "
            "claims already present in the novelty decomposition. Composite "
            "same-work closure records whether all explicit components are "
            "co-located in one prior-art work. Any residual classification is "
            "diagnostic only and does not change external novelty, N9, N10, "
            "or production selection."
        ),
    }
    report_id = _stable_id(
        "lower_order_prior_art_saturation",
        base_plan.plan_id,
        saturation_plan.plan_id,
        saturation_packet_id,
        *[
            f"{row.claim_id}:{row.saturation_claim_status}"
            for row in claim_rows
        ],
        *[
            f"{row.composite_claim_id}:{row.residual_class}"
            for row in composite_rows
        ],
    )
    hashed = {**body, "report_id": report_id}
    return LowerOrderPriorArtSaturationReport(
        **hashed,
        report_sha256=_sha256_json(hashed),
    )


__all__ = [
    "LowerOrderPriorArtSaturationReport",
    "LowerOrderSaturationTarget",
    "LowerOrderClaimSaturation",
    "CompositeResidualSaturation",
    "build_lower_order_saturation_query_plan",
    "build_lower_order_saturation_report",
]
