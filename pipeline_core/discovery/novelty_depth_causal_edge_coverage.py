from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


EdgeCoverageClass = Literal[
    "DIRECTLY_KNOWN",
    "PARTIALLY_KNOWN",
    "HIGHER_ORDER_GAP_OVER_KNOWN_BASE",
    "LOCAL_RELATIONAL_GAP",
    "WEAKLY_GROUNDED_BRIDGE_GAP",
    "GAP_WITHOUT_RELATION_BACKBONE",
    "CONFLICTED",
    "UNRESOLVED",
]

NoveltyDepthClass = Literal[
    "KNOWN_RELATION",
    "SHALLOW_LOCAL_EXTENSION",
    "HIGHER_ORDER_INTERACTION_GAP",
    "UNSUPPORTED_BRIDGE_DEPTH_UNKNOWN",
    "UNRESOLVED_DEPTH",
    "CONFLICTED_DEPTH",
]

PlannerAdvisory = Literal[
    "KEEP_TESTABLE_GAP",
    "SHARPEN_LOCAL_EXTENSION",
    "RETRIEVE_MECHANISM_SUPPORT",
    "REAXIS_OR_ABSTAIN",
    "HOLD_UNRESOLVED",
    "KNOWN_RELATION",
    "CONFLICT_REVIEW",
]


RELATION_BACKED = {
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
    "LOWER_ORDER_RELATION_PRIOR_ART",
}

GAP_LIKE = {
    "COMPONENTS_ONLY",
    "NO_DIRECT_MATCH_FOUND",
}

UNRESOLVED = {
    "TITLE_ONLY_NEIGHBORS",
    "INSUFFICIENT_METADATA",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)?")

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "how", "in", "into", "is", "it", "of", "on", "or",
    "that", "the", "their", "this", "to", "under", "with", "without",
    "when", "which", "while", "different", "specific", "relative",
    "comparable", "conditions",
}


class ClaimEdgeCoverage(StrictModel):
    claim_id: str
    importance: str
    claim_text: str
    claim_status: str

    edge_unit_kind: Literal[
        "EXTERNAL_NOVELTY_CLAIM_RELATION_NUCLEUS"
    ] = "EXTERNAL_NOVELTY_CLAIM_RELATION_NUCLEUS"

    direct_prior_art_count: int = 0
    partial_prior_art_count: int = 0
    lower_order_relation_count: int = 0
    component_only_count: int = 0
    conflicting_prior_art_count: int = 0
    unresolved_match_count: int = 0

    relation_backed_work_ids: list[str] = Field(default_factory=list)
    lower_order_work_ids: list[str] = Field(default_factory=list)
    component_only_work_ids: list[str] = Field(default_factory=list)

    premise_lexical_coverage: float = 0.0
    premise_support_thin: bool = False

    coverage_class: EdgeCoverageClass
    novelty_bearing: bool = False
    unsupported_bridge_risk: bool = False

    interpretation: str


class HypothesisNoveltyDepthProfile(StrictModel):
    hypothesis_id: str
    title: str
    source_external_status: str

    claim_edges: list[ClaimEdgeCoverage] = Field(default_factory=list)

    core_claim_count: int = 0
    core_gap_like_count: int = 0
    core_relation_backed_count: int = 0
    core_lower_order_relation_count: int = 0
    core_unsupported_bridge_count: int = 0
    core_unresolved_count: int = 0
    core_conflicted_count: int = 0

    relation_backed_core_fraction: float | None = None
    minimum_core_premise_lexical_coverage: float | None = None

    novelty_depth_class: NoveltyDepthClass
    planner_advisory: PlannerAdvisory

    novelty_bearing_claim_ids: list[str] = Field(default_factory=list)
    weak_bridge_claim_ids: list[str] = Field(default_factory=list)

    foundational_knownness_checked: Literal[False] = False
    conceptual_l1_l2_l3_integrated: Literal[False] = False
    causal_edge_extraction_claims_actual_graph_edges: Literal[False] = False

    reason_codes: list[str] = Field(default_factory=list)
    interpretation: str


class NoveltyDepthCausalEdgeCoverageReport(StrictModel):
    schema_version: Literal[
        "novelty-depth-causal-edge-coverage-v1"
    ] = "novelty-depth-causal-edge-coverage-v1"

    report_id: str
    report_sha256: str

    source_context_id: str
    source_portfolio_id: str
    source_external_report_id: str

    profiles: list[HypothesisNoveltyDepthProfile] = Field(default_factory=list)

    hypothesis_count: int = 0
    novelty_depth_counts: dict[str, int] = Field(default_factory=dict)
    planner_advisory_counts: dict[str, int] = Field(default_factory=dict)
    weak_bridge_hypothesis_count: int = 0
    higher_order_gap_hypothesis_count: int = 0
    shallow_local_extension_count: int = 0

    foundational_knownness_checked: Literal[False] = False
    conceptual_l1_l2_l3_integrated: Literal[False] = False
    external_prior_art_as_positive_premise: Literal[False] = False
    ranking_computed: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    production_selection_changed: Literal[False] = False


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(str(text))
        if len(token) > 1 and token.lower() not in _STOP
    }


def _premise_text_by_hypothesis(
    *,
    context: Any,
    portfolio: Any,
) -> dict[str, str]:
    statement_text = {
        str(row.statement_id): str(getattr(row, "text", "") or "")
        for row in getattr(context, "evidence_statements", [])
        if getattr(row, "statement_id", None)
    }

    result: dict[str, str] = {}
    for card in getattr(portfolio, "hypotheses", []):
        texts = [
            statement_text.get(str(sid), "")
            for sid in getattr(card, "premise_statement_ids", [])
        ]
        result[str(card.hypothesis_id)] = " ".join(
            text for text in texts if text
        )
    return result


def _lexical_coverage(claim_text: str, premise_text: str) -> float:
    claim_tokens = _tokens(claim_text)
    premise_tokens = _tokens(premise_text)

    if not claim_tokens:
        return 0.0

    return round(
        len(claim_tokens & premise_tokens)
        / len(claim_tokens),
        6,
    )


def _relationship(match: Any) -> str:
    return str(getattr(match, "relationship", "") or "")


def _work_id(match: Any) -> str:
    return str(getattr(match, "work_id", "") or "")


def _classify_claim_edge(
    *,
    review: Any,
    premise_text: str,
) -> ClaimEdgeCoverage:
    relationships = [
        _relationship(match)
        for match in getattr(review, "matches", [])
    ]

    by_rel = Counter(relationships)

    relation_backed_work_ids = [
        _work_id(match)
        for match in getattr(review, "matches", [])
        if _relationship(match) in RELATION_BACKED
    ]
    lower_order_work_ids = [
        _work_id(match)
        for match in getattr(review, "matches", [])
        if _relationship(match) == "LOWER_ORDER_RELATION_PRIOR_ART"
    ]
    component_only_work_ids = [
        _work_id(match)
        for match in getattr(review, "matches", [])
        if _relationship(match) == "COMPONENT_ONLY"
    ]

    claim_status = str(review.status)
    lexical = _lexical_coverage(
        str(review.claim_text),
        premise_text,
    )
    premise_support_thin = lexical < 0.30

    direct = by_rel["DIRECT_PRIOR_ART"]
    partial = by_rel["PARTIAL_PRIOR_ART"]
    lower = by_rel["LOWER_ORDER_RELATION_PRIOR_ART"]
    component = by_rel["COMPONENT_ONLY"]
    conflict = (
        by_rel["CONFLICTING_PRIOR_ART"]
        + by_rel["DIRECTIONAL_COUNTEREVIDENCE"]
    )
    unresolved_matches = (
        by_rel["TITLE_ONLY_NEIGHBOR"]
        + by_rel["INSUFFICIENT_METADATA"]
    )

    novelty_bearing = (
        claim_status in GAP_LIKE
        and str(review.importance) == "core"
    )

    unsupported_bridge = bool(
        novelty_bearing
        and lower == 0
        and partial == 0
        and direct == 0
        and (
            premise_support_thin
            or component > 0
        )
    )

    if claim_status == "DIRECT_PRIOR_ART" or direct > 0:
        coverage_class: EdgeCoverageClass = "DIRECTLY_KNOWN"
        interpretation = (
            "The claim relation nucleus has direct prior-art coverage."
        )
    elif claim_status == "PARTIAL_PRIOR_ART" or partial > 0:
        coverage_class = "PARTIALLY_KNOWN"
        interpretation = (
            "The claim relation nucleus is materially covered but not "
            "fully matched in the bounded prior-art review."
        )
    elif claim_status == "CONFLICTING_PRIOR_ART" or conflict > 0:
        coverage_class = "CONFLICTED"
        interpretation = (
            "Relevant prior art materially conflicts with the claim relation "
            "or its asserted direction."
        )
    elif claim_status in UNRESOLVED:
        coverage_class = "UNRESOLVED"
        interpretation = (
            "Metadata or title-only evidence prevents a reliable relation "
            "coverage judgment."
        )
    elif claim_status in GAP_LIKE and lower >= 2:
        coverage_class = "HIGHER_ORDER_GAP_OVER_KNOWN_BASE"
        interpretation = (
            "The exact/core relation remains gap-like while multiple explicit "
            "lower-order relations are already covered. This is consistent "
            "with a higher-order interaction gap rather than an entirely new "
            "mechanistic bridge."
        )
    elif claim_status in GAP_LIKE and lower == 1:
        coverage_class = "LOCAL_RELATIONAL_GAP"
        interpretation = (
            "A meaningful lower-order relation is known and the remaining "
            "claim novelty is localized to an added condition, moderator, "
            "or extension."
        )
    elif unsupported_bridge:
        coverage_class = "WEAKLY_GROUNDED_BRIDGE_GAP"
        interpretation = (
            "The core claim is gap-like but lacks relation-backed lower-order "
            "support in the retrieved evidence; its novelty-bearing bridge is "
            "therefore weakly grounded rather than positively established as "
            "a deep novel mechanism."
        )
    else:
        coverage_class = "GAP_WITHOUT_RELATION_BACKBONE"
        interpretation = (
            "The core claim remains gap-like and the bounded review does not "
            "establish a strong lower-order relation backbone."
        )

    return ClaimEdgeCoverage(
        claim_id=str(review.claim_id),
        importance=str(review.importance),
        claim_text=str(review.claim_text),
        claim_status=claim_status,
        direct_prior_art_count=direct,
        partial_prior_art_count=partial,
        lower_order_relation_count=lower,
        component_only_count=component,
        conflicting_prior_art_count=conflict,
        unresolved_match_count=unresolved_matches,
        relation_backed_work_ids=sorted(set(relation_backed_work_ids)),
        lower_order_work_ids=sorted(set(lower_order_work_ids)),
        component_only_work_ids=sorted(set(component_only_work_ids)),
        premise_lexical_coverage=lexical,
        premise_support_thin=premise_support_thin,
        coverage_class=coverage_class,
        novelty_bearing=novelty_bearing,
        unsupported_bridge_risk=unsupported_bridge,
        interpretation=interpretation,
    )


def _profile_from_edges(
    *,
    external_card: Any,
    claim_edges: list[ClaimEdgeCoverage],
) -> HypothesisNoveltyDepthProfile:
    core = [
        row
        for row in claim_edges
        if row.importance == "core"
    ]
    if not core:
        core = list(claim_edges)

    core_count = len(core)
    core_gap = sum(row.claim_status in GAP_LIKE for row in core)
    core_relation_backed = sum(
        row.coverage_class
        in {
            "DIRECTLY_KNOWN",
            "PARTIALLY_KNOWN",
            "HIGHER_ORDER_GAP_OVER_KNOWN_BASE",
            "LOCAL_RELATIONAL_GAP",
        }
        for row in core
    )
    core_lower = sum(row.lower_order_relation_count for row in core)
    core_weak = sum(row.unsupported_bridge_risk for row in core)
    core_unresolved = sum(row.coverage_class == "UNRESOLVED" for row in core)
    core_conflicted = sum(row.coverage_class == "CONFLICTED" for row in core)

    reason_codes: list[str] = []

    if core_conflicted:
        depth: NoveltyDepthClass = "CONFLICTED_DEPTH"
        advisory: PlannerAdvisory = "CONFLICT_REVIEW"
        reason_codes.append("core_conflicting_prior_art")
        interpretation = (
            "At least one core relation is conflicted; novelty depth should "
            "not be promoted until the conflict is resolved."
        )
    elif core_unresolved:
        depth = "UNRESOLVED_DEPTH"
        advisory = "HOLD_UNRESOLVED"
        reason_codes.append("core_relation_coverage_unresolved")
        interpretation = (
            "Core relation coverage is unresolved because metadata or "
            "evidence coverage is insufficient."
        )
    elif any(
        row.coverage_class == "DIRECTLY_KNOWN"
        for row in core
    ) and core_gap == 0:
        depth = "KNOWN_RELATION"
        advisory = "KNOWN_RELATION"
        reason_codes.append("core_relation_directly_known")
        interpretation = (
            "The core relation is directly covered by prior art in the "
            "bounded review."
        )
    elif core_weak:
        depth = "UNSUPPORTED_BRIDGE_DEPTH_UNKNOWN"
        advisory = "REAXIS_OR_ABSTAIN"
        reason_codes.extend(
            [
                "novelty_bearing_bridge_weakly_grounded",
                "do_not_reward_absence_as_deep_novelty",
            ]
        )
        interpretation = (
            "The apparent novelty is carried by one or more weakly grounded "
            "bridges. Exact-match absence should not be interpreted as deep "
            "novelty; re-axis, retrieve mechanism support, or abstain."
        )
    elif any(
        row.coverage_class == "HIGHER_ORDER_GAP_OVER_KNOWN_BASE"
        for row in core
    ):
        depth = "HIGHER_ORDER_INTERACTION_GAP"
        advisory = "KEEP_TESTABLE_GAP"
        reason_codes.extend(
            [
                "multiple_lower_order_relations_known",
                "exact_higher_order_relation_remains_gap_like",
            ]
        )
        interpretation = (
            "Multiple lower-order relations are already covered while the "
            "combined/higher-order relation remains gap-like. This is a "
            "testable higher-order gap, not evidence of a wholly new "
            "mechanism."
        )
    elif any(
        row.coverage_class == "LOCAL_RELATIONAL_GAP"
        for row in core
    ):
        depth = "SHALLOW_LOCAL_EXTENSION"
        advisory = "SHARPEN_LOCAL_EXTENSION"
        reason_codes.extend(
            [
                "known_lower_order_relation_present",
                "novelty_localized_to_added_condition_or_extension",
            ]
        )
        interpretation = (
            "The novelty-bearing difference is localized around an already "
            "known relation. Residual, boundary, decoupling, or other "
            "structure-changing refinement is preferable to treating the "
            "whole hypothesis as deeply novel."
        )
    elif core_gap:
        depth = "UNSUPPORTED_BRIDGE_DEPTH_UNKNOWN"
        advisory = "RETRIEVE_MECHANISM_SUPPORT"
        reason_codes.append("gap_without_relation_backbone")
        interpretation = (
            "The claim is gap-like but lacks enough relation-backed structure "
            "to determine whether this is a meaningful mechanistic gap or "
            "merely an unsupported specification."
        )
    else:
        depth = "KNOWN_RELATION"
        advisory = "KNOWN_RELATION"
        reason_codes.append("no_core_gap_detected")
        interpretation = (
            "No core novelty-bearing relation gap is established in the "
            "current bounded review."
        )

    fraction = (
        round(core_relation_backed / core_count, 6)
        if core_count
        else None
    )
    min_lexical = (
        min(row.premise_lexical_coverage for row in core)
        if core
        else None
    )

    return HypothesisNoveltyDepthProfile(
        hypothesis_id=str(external_card.hypothesis_id),
        title=str(external_card.title),
        source_external_status=str(external_card.status),
        claim_edges=claim_edges,
        core_claim_count=core_count,
        core_gap_like_count=core_gap,
        core_relation_backed_count=core_relation_backed,
        core_lower_order_relation_count=core_lower,
        core_unsupported_bridge_count=core_weak,
        core_unresolved_count=core_unresolved,
        core_conflicted_count=core_conflicted,
        relation_backed_core_fraction=fraction,
        minimum_core_premise_lexical_coverage=min_lexical,
        novelty_depth_class=depth,
        planner_advisory=advisory,
        novelty_bearing_claim_ids=[
            row.claim_id for row in core if row.novelty_bearing
        ],
        weak_bridge_claim_ids=[
            row.claim_id for row in core if row.unsupported_bridge_risk
        ],
        reason_codes=sorted(set(reason_codes)),
        interpretation=interpretation,
    )


def build_novelty_depth_causal_edge_coverage(
    *,
    context: Any,
    portfolio: Any,
    external_report: Any,
) -> NoveltyDepthCausalEdgeCoverageReport:
    premise_text = _premise_text_by_hypothesis(
        context=context,
        portfolio=portfolio,
    )

    portfolio_ids = {
        str(row.hypothesis_id)
        for row in getattr(portfolio, "hypotheses", [])
    }

    profiles: list[HypothesisNoveltyDepthProfile] = []

    for external_card in getattr(external_report, "cards", []):
        hypothesis_id = str(external_card.hypothesis_id)
        if hypothesis_id not in portfolio_ids:
            raise ValueError(
                "external novelty card references hypothesis outside "
                f"source portfolio: {hypothesis_id}"
            )

        edges = [
            _classify_claim_edge(
                review=review,
                premise_text=premise_text.get(hypothesis_id, ""),
            )
            for review in getattr(external_card, "claim_reviews", [])
        ]

        profiles.append(
            _profile_from_edges(
                external_card=external_card,
                claim_edges=edges,
            )
        )

    depth_counts = Counter(row.novelty_depth_class for row in profiles)
    advisory_counts = Counter(row.planner_advisory for row in profiles)

    body = {
        "schema_version": "novelty-depth-causal-edge-coverage-v1",
        "source_context_id": str(context.context_id),
        "source_portfolio_id": str(portfolio.portfolio_id),
        "source_external_report_id": str(external_report.report_id),
        "profiles": [
            row.model_dump(mode="json") for row in profiles
        ],
        "hypothesis_count": len(profiles),
        "novelty_depth_counts": dict(sorted(depth_counts.items())),
        "planner_advisory_counts": dict(sorted(advisory_counts.items())),
        "weak_bridge_hypothesis_count": sum(
            row.core_unsupported_bridge_count > 0 for row in profiles
        ),
        "higher_order_gap_hypothesis_count": sum(
            row.novelty_depth_class == "HIGHER_ORDER_INTERACTION_GAP"
            for row in profiles
        ),
        "shallow_local_extension_count": sum(
            row.novelty_depth_class == "SHALLOW_LOCAL_EXTENSION"
            for row in profiles
        ),
        "foundational_knownness_checked": False,
        "conceptual_l1_l2_l3_integrated": False,
        "external_prior_art_as_positive_premise": False,
        "ranking_computed": False,
        "novelty_authority_created": False,
        "production_selection_changed": False,
    }

    report_id = _stable_id(
        "novelty_depth_causal_edge_coverage",
        body["source_context_id"],
        body["source_portfolio_id"],
        body["source_external_report_id"],
        *[
            (
                f"{row.hypothesis_id}:"
                f"{row.novelty_depth_class}:"
                f"{row.planner_advisory}"
            )
            for row in profiles
        ],
    )

    return NoveltyDepthCausalEdgeCoverageReport(
        **body,
        report_id=report_id,
        report_sha256=_sha256({**body, "report_id": report_id}),
    )


__all__ = [
    "ClaimEdgeCoverage",
    "HypothesisNoveltyDepthProfile",
    "NoveltyDepthCausalEdgeCoverageReport",
    "build_novelty_depth_causal_edge_coverage",
]
