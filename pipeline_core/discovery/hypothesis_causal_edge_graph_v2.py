from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.novelty_depth_causal_edge_coverage import (
    ClaimEdgeCoverage,
    NoveltyDepthCausalEdgeCoverageReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


BackboneStrength = Literal[
    "NONE",
    "SINGLE_SIGNAL",
    "MULTIPLE_INDEPENDENT_SIGNALS",
]

RelationGraphEdgeKind = Literal[
    "SHARED_PRIOR_ART_WORK",
    "RELATION_NUCLEUS_OVERLAP",
    "SHARED_RELATION_BACKED_WORK",
]


class RelationCoverageNode(StrictModel):
    claim_id: str
    importance: str
    claim_text: str
    claim_status: str
    coverage_class_v1: str
    novelty_bearing: bool

    match_work_ids: list[str] = Field(default_factory=list)
    relation_backed_work_ids: list[str] = Field(default_factory=list)
    lower_order_work_ids: list[str] = Field(default_factory=list)

    relation_backed: bool = False
    relation_backed_signal_count: int = 0
    component_only_count: int = 0
    premise_lexical_coverage: float = 0.0


class RelationCoverageLink(StrictModel):
    left_claim_id: str
    right_claim_id: str

    shared_work_ids: list[str] = Field(default_factory=list)
    shared_relation_backed_work_ids: list[str] = Field(default_factory=list)
    shared_content_tokens: list[str] = Field(default_factory=list)
    lexical_jaccard: float = 0.0

    edge_kinds: list[RelationGraphEdgeKind] = Field(default_factory=list)
    qualifies_as_backbone_link: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class NoveltyBearingRelationAssessment(StrictModel):
    claim_id: str

    own_lower_order_work_ids: list[str] = Field(default_factory=list)
    linked_backbone_claim_ids: list[str] = Field(default_factory=list)
    linked_relation_backed_work_ids: list[str] = Field(default_factory=list)

    independent_backbone_signal_count: int = 0
    backbone_strength: BackboneStrength = "NONE"

    v1_coverage_class: str
    v2_coverage_class: str

    unsupported_bridge_risk_v1: bool = False
    unsupported_bridge_risk_v2: bool = False

    reason_codes: list[str] = Field(default_factory=list)


class ConceptualKnownnessOverlay(StrictModel):
    available: bool = False
    first_gap_level: str | None = None
    disposition: str | None = None
    coverage_sufficient: bool | None = None
    source_shape: str | None = None

    changes_v2_depth: Literal[False] = False
    authority_created: Literal[False] = False


class HypothesisRelationCoverageGraphV2(StrictModel):
    hypothesis_id: str
    title: str
    source_external_status: str

    nodes: list[RelationCoverageNode] = Field(default_factory=list)
    links: list[RelationCoverageLink] = Field(default_factory=list)
    novelty_bearing_relations: list[
        NoveltyBearingRelationAssessment
    ] = Field(default_factory=list)

    known_backbone_claim_ids: list[str] = Field(default_factory=list)
    novelty_bearing_claim_ids: list[str] = Field(default_factory=list)

    v1_novelty_depth_class: str
    novelty_depth_class: str
    planner_advisory: str

    conceptual_knownness: ConceptualKnownnessOverlay = Field(
        default_factory=ConceptualKnownnessOverlay
    )

    relation_graph_semantics: Literal[
        "claim_relation_nucleus_connectivity_not_causal_truth"
    ] = "claim_relation_nucleus_connectivity_not_causal_truth"

    cross_claim_backbone_used: bool = False
    weak_bridge_claim_ids: list[str] = Field(default_factory=list)
    higher_order_claim_ids: list[str] = Field(default_factory=list)
    local_extension_claim_ids: list[str] = Field(default_factory=list)

    reason_codes: list[str] = Field(default_factory=list)
    interpretation: str


class HypothesisCausalEdgeGraphNoveltyDepthV2Report(StrictModel):
    schema_version: Literal[
        "hypothesis-causal-edge-graph-novelty-depth-v2"
    ] = "hypothesis-causal-edge-graph-novelty-depth-v2"

    report_id: str
    report_sha256: str

    source_context_id: str
    source_portfolio_id: str
    source_external_report_id: str
    source_v1_report_id: str

    profiles: list[HypothesisRelationCoverageGraphV2] = Field(
        default_factory=list
    )

    hypothesis_count: int = 0
    novelty_depth_counts: dict[str, int] = Field(default_factory=dict)
    planner_advisory_counts: dict[str, int] = Field(default_factory=dict)

    cross_claim_backbone_hypothesis_count: int = 0
    higher_order_gap_hypothesis_count: int = 0
    shallow_local_extension_count: int = 0
    weak_bridge_hypothesis_count: int = 0

    conceptual_knownness_signal_count: int = 0
    conceptual_knownness_sufficient_count: int = 0
    conceptual_knownness_changed_depth_count: Literal[0] = 0

    foundational_knownness_checked: Literal[False] = False
    conceptual_knownness_authority_created: Literal[False] = False
    external_prior_art_as_positive_premise: Literal[False] = False
    ranking_computed: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    production_selection_changed: Literal[False] = False


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)?")

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "how", "in", "into", "is", "it", "of", "on", "or",
    "that", "the", "their", "this", "to", "under", "with", "without",
    "when", "which", "while", "different", "specific", "relative",
    "comparable", "conditions", "may", "can", "could", "should", "would",
}

RELATION_BACKED_RELATIONSHIPS = {
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
    "LOWER_ORDER_RELATION_PRIOR_ART",
}


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


def _match_work_ids(review: Any) -> list[str]:
    return sorted(
        {
            str(match.work_id)
            for match in getattr(review, "matches", [])
            if getattr(match, "work_id", None)
        }
    )


def _relation_backed_work_ids(review: Any) -> list[str]:
    return sorted(
        {
            str(match.work_id)
            for match in getattr(review, "matches", [])
            if str(getattr(match, "relationship", ""))
            in RELATION_BACKED_RELATIONSHIPS
            and getattr(match, "work_id", None)
        }
    )


def _lower_order_work_ids(review: Any) -> list[str]:
    return sorted(
        {
            str(match.work_id)
            for match in getattr(review, "matches", [])
            if str(getattr(match, "relationship", ""))
            == "LOWER_ORDER_RELATION_PRIOR_ART"
            and getattr(match, "work_id", None)
        }
    )


def _node(
    *,
    review: Any,
    v1_edge: ClaimEdgeCoverage,
) -> RelationCoverageNode:
    rel_backed = _relation_backed_work_ids(review)
    lower = _lower_order_work_ids(review)

    relation_signal_count = len(
        set(rel_backed) | set(lower)
    )

    return RelationCoverageNode(
        claim_id=str(review.claim_id),
        importance=str(review.importance),
        claim_text=str(review.claim_text),
        claim_status=str(review.status),
        coverage_class_v1=str(v1_edge.coverage_class),
        novelty_bearing=bool(v1_edge.novelty_bearing),
        match_work_ids=_match_work_ids(review),
        relation_backed_work_ids=rel_backed,
        lower_order_work_ids=lower,
        relation_backed=bool(
            rel_backed
            or str(v1_edge.coverage_class)
            in {
                "DIRECTLY_KNOWN",
                "PARTIALLY_KNOWN",
                "LOCAL_RELATIONAL_GAP",
                "HIGHER_ORDER_GAP_OVER_KNOWN_BASE",
            }
        ),
        relation_backed_signal_count=relation_signal_count,
        component_only_count=int(v1_edge.component_only_count),
        premise_lexical_coverage=float(
            v1_edge.premise_lexical_coverage
        ),
    )


def _link(
    left: RelationCoverageNode,
    right: RelationCoverageNode,
) -> RelationCoverageLink:
    left_tokens = _tokens(left.claim_text)
    right_tokens = _tokens(right.claim_text)

    shared_tokens = sorted(left_tokens & right_tokens)
    union = left_tokens | right_tokens
    jaccard = (
        round(len(shared_tokens) / len(union), 6)
        if union
        else 0.0
    )

    shared_work = sorted(
        set(left.match_work_ids)
        & set(right.match_work_ids)
    )
    shared_relation = sorted(
        set(left.relation_backed_work_ids)
        & set(right.relation_backed_work_ids)
    )

    kinds: list[RelationGraphEdgeKind] = []
    reasons: list[str] = []

    if shared_work:
        kinds.append("SHARED_PRIOR_ART_WORK")
    if shared_relation:
        kinds.append("SHARED_RELATION_BACKED_WORK")
    if len(shared_tokens) >= 2 and jaccard >= 0.14:
        kinds.append("RELATION_NUCLEUS_OVERLAP")

    relation_backed_other = left.relation_backed or right.relation_backed

    qualifies = bool(
        shared_relation
        or (
            relation_backed_other
            and len(shared_tokens) >= 2
            and jaccard >= 0.18
        )
    )

    if shared_relation:
        reasons.append("shared_relation_backed_prior_art_work")
    if (
        relation_backed_other
        and len(shared_tokens) >= 2
        and jaccard >= 0.18
    ):
        reasons.append(
            "relation_backed_claim_with_relation_nucleus_overlap"
        )

    return RelationCoverageLink(
        left_claim_id=left.claim_id,
        right_claim_id=right.claim_id,
        shared_work_ids=shared_work,
        shared_relation_backed_work_ids=shared_relation,
        shared_content_tokens=shared_tokens,
        lexical_jaccard=jaccard,
        edge_kinds=kinds,
        qualifies_as_backbone_link=qualifies,
        reason_codes=reasons,
    )


def _conceptual_map(
    conceptual_knownness: Any | None,
) -> dict[str, ConceptualKnownnessOverlay]:
    if conceptual_knownness is None:
        return {}

    result: dict[str, ConceptualKnownnessOverlay] = {}

    records = getattr(conceptual_knownness, "records", None)
    if records is None and isinstance(conceptual_knownness, dict):
        records = conceptual_knownness.get("records")

    if records:
        for row in records:
            if isinstance(row, dict):
                hid = str(row.get("hypothesis_id", ""))
                first_gap = row.get("first_gap_level")
                levels = row.get("levels") or []
                sufficient_values = [
                    level.get("sufficient_coverage")
                    for level in levels
                    if isinstance(level, dict)
                    and level.get("sufficient_coverage") is not None
                ]
            else:
                hid = str(getattr(row, "hypothesis_id", ""))
                first_gap = getattr(row, "first_gap_level", None)
                levels = list(getattr(row, "levels", []) or [])
                sufficient_values = [
                    getattr(level, "sufficient_coverage", None)
                    for level in levels
                    if getattr(level, "sufficient_coverage", None)
                    is not None
                ]

            if not hid:
                continue

            sufficient = (
                all(bool(x) for x in sufficient_values)
                if sufficient_values
                else None
            )

            result[hid] = ConceptualKnownnessOverlay(
                available=True,
                first_gap_level=(
                    None if first_gap is None else str(first_gap)
                ),
                coverage_sufficient=sufficient,
                source_shape="records",
            )

    claim_profiles = getattr(
        conceptual_knownness,
        "claim_profiles",
        None,
    )
    if claim_profiles is None and isinstance(conceptual_knownness, dict):
        claim_profiles = conceptual_knownness.get("claim_profiles")

    if claim_profiles:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in claim_profiles:
            if not isinstance(row, dict):
                continue
            hid = str(row.get("hypothesis_id", ""))
            if hid:
                grouped.setdefault(hid, []).append(row)

        for hid, rows in grouped.items():
            gap_levels = [
                row.get("first_search_bounded_gap_level")
                for row in rows
                if row.get("first_search_bounded_gap_level")
            ]
            first_gap = gap_levels[0] if gap_levels else None

            result[hid] = ConceptualKnownnessOverlay(
                available=True,
                first_gap_level=(
                    None if first_gap is None else str(first_gap)
                ),
                coverage_sufficient=None,
                source_shape="claim_profiles",
            )

    return result


def _backbone_assessment(
    *,
    novelty_node: RelationCoverageNode,
    all_nodes: list[RelationCoverageNode],
    links: list[RelationCoverageLink],
    v1_edge: ClaimEdgeCoverage,
) -> NoveltyBearingRelationAssessment:
    linked_claim_ids: list[str] = []
    linked_work_ids: set[str] = set()
    reasons: list[str] = []

    node_by_id = {
        node.claim_id: node
        for node in all_nodes
    }

    for link in links:
        if not link.qualifies_as_backbone_link:
            continue

        other_id = None
        if link.left_claim_id == novelty_node.claim_id:
            other_id = link.right_claim_id
        elif link.right_claim_id == novelty_node.claim_id:
            other_id = link.left_claim_id

        if other_id is None:
            continue

        other = node_by_id[other_id]

        if not other.relation_backed:
            continue

        if other.novelty_bearing and other.importance == "core":
            # Another unresolved core claim is not allowed to silently
            # become the known backbone of this one.
            continue

        linked_claim_ids.append(other_id)
        linked_work_ids.update(
            other.relation_backed_work_ids
        )
        linked_work_ids.update(
            link.shared_relation_backed_work_ids
        )

    own_lower = sorted(
        set(novelty_node.lower_order_work_ids)
    )

    # Count independent backbone *routes*, not raw evidence rows.
    # One supporting relation claim may cite several works; that remains
    # one cross-claim backbone route for depth classification.
    independent_signals = (
        len(set(own_lower))
        + len(set(linked_claim_ids))
    )

    if independent_signals >= 2:
        strength: BackboneStrength = (
            "MULTIPLE_INDEPENDENT_SIGNALS"
        )
    elif independent_signals == 1:
        strength = "SINGLE_SIGNAL"
    else:
        strength = "NONE"

    if strength == "MULTIPLE_INDEPENDENT_SIGNALS":
        v2_class = "HIGHER_ORDER_GAP_OVER_COMPOSED_BACKBONE"
        unsupported_v2 = False
        reasons.append(
            "multiple_independent_lower_order_backbone_signals"
        )
    elif strength == "SINGLE_SIGNAL":
        v2_class = "LOCAL_RELATIONAL_GAP_OVER_COMPOSED_BACKBONE"
        unsupported_v2 = False
        reasons.append(
            "single_lower_order_backbone_signal"
        )
    else:
        v2_class = str(v1_edge.coverage_class)
        unsupported_v2 = bool(
            v1_edge.unsupported_bridge_risk
        )
        reasons.append(
            "no_cross_claim_relation_backbone_established"
        )

    if linked_claim_ids:
        reasons.append(
            "cross_claim_backbone_used"
        )

    return NoveltyBearingRelationAssessment(
        claim_id=novelty_node.claim_id,
        own_lower_order_work_ids=own_lower,
        linked_backbone_claim_ids=sorted(
            set(linked_claim_ids)
        ),
        linked_relation_backed_work_ids=sorted(
            linked_work_ids
        ),
        independent_backbone_signal_count=independent_signals,
        backbone_strength=strength,
        v1_coverage_class=str(v1_edge.coverage_class),
        v2_coverage_class=v2_class,
        unsupported_bridge_risk_v1=bool(
            v1_edge.unsupported_bridge_risk
        ),
        unsupported_bridge_risk_v2=unsupported_v2,
        reason_codes=reasons,
    )


def _depth_from_assessments(
    *,
    v1_profile: Any,
    assessments: list[NoveltyBearingRelationAssessment],
) -> tuple[str, str, list[str], str]:
    reasons: list[str] = []

    if str(v1_profile.novelty_depth_class) in {
        "CONFLICTED_DEPTH",
        "UNRESOLVED_DEPTH",
        "KNOWN_RELATION",
    }:
        return (
            str(v1_profile.novelty_depth_class),
            str(v1_profile.planner_advisory),
            ["v1_terminal_depth_preserved"],
            str(v1_profile.interpretation),
        )

    if any(
        row.backbone_strength == "MULTIPLE_INDEPENDENT_SIGNALS"
        for row in assessments
    ):
        reasons.append(
            "composed_known_backbone_supports_higher_order_gap"
        )
        return (
            "HIGHER_ORDER_INTERACTION_GAP",
            "KEEP_TESTABLE_GAP",
            reasons,
            (
                "The novelty-bearing relation is connected to multiple "
                "independent relation-backed lower-order signals across "
                "the hypothesis claim set. This supports treating the "
                "candidate as a bounded higher-order gap rather than as "
                "an unsupported bridge."
            ),
        )

    if any(
        row.backbone_strength == "SINGLE_SIGNAL"
        for row in assessments
    ):
        reasons.append(
            "composed_known_backbone_supports_local_extension"
        )
        return (
            "SHALLOW_LOCAL_EXTENSION",
            "SHARPEN_LOCAL_EXTENSION",
            reasons,
            (
                "The novelty-bearing relation has one established "
                "lower-order backbone signal across the hypothesis claim "
                "set. Novelty is therefore localized and should be "
                "sharpened rather than interpreted as a deep new mechanism."
            ),
        )

    if assessments and all(
        row.unsupported_bridge_risk_v2
        for row in assessments
    ):
        reasons.append(
            "novelty_bearing_relations_lack_composed_backbone"
        )
        return (
            "UNSUPPORTED_BRIDGE_DEPTH_UNKNOWN",
            "REAXIS_OR_ABSTAIN",
            reasons,
            (
                "The novelty-bearing relations remain weakly grounded after "
                "cross-claim graph assembly. Absence of direct prior art is "
                "not sufficient to claim deep novelty."
            ),
        )

    return (
        str(v1_profile.novelty_depth_class),
        str(v1_profile.planner_advisory),
        ["v1_depth_retained_after_graph_assembly"],
        str(v1_profile.interpretation),
    )


def build_hypothesis_causal_edge_graph_novelty_depth_v2(
    *,
    context: Any,
    portfolio: Any,
    external_report: Any,
    v1_report: NoveltyDepthCausalEdgeCoverageReport,
    conceptual_knownness: Any | None = None,
) -> HypothesisCausalEdgeGraphNoveltyDepthV2Report:
    del context  # lineage retained through v1; no new premise promotion here.

    v1_by_hypothesis = {
        str(row.hypothesis_id): row
        for row in v1_report.profiles
    }
    external_by_hypothesis = {
        str(row.hypothesis_id): row
        for row in getattr(external_report, "cards", [])
    }
    conceptual = _conceptual_map(conceptual_knownness)

    portfolio_ids = {
        str(row.hypothesis_id)
        for row in getattr(portfolio, "hypotheses", [])
    }

    profiles: list[HypothesisRelationCoverageGraphV2] = []

    for hypothesis_id in sorted(portfolio_ids):
        external_card = external_by_hypothesis.get(hypothesis_id)
        v1_profile = v1_by_hypothesis.get(hypothesis_id)

        if external_card is None or v1_profile is None:
            continue

        v1_edges = {
            str(row.claim_id): row
            for row in v1_profile.claim_edges
        }

        reviews = list(
            getattr(external_card, "claim_reviews", [])
        )

        nodes = [
            _node(
                review=review,
                v1_edge=v1_edges[str(review.claim_id)],
            )
            for review in reviews
            if str(review.claim_id) in v1_edges
        ]

        links: list[RelationCoverageLink] = []
        for index, left in enumerate(nodes):
            for right in nodes[index + 1:]:
                links.append(_link(left, right))

        novelty_nodes = [
            node
            for node in nodes
            if node.novelty_bearing
        ]

        assessments = [
            _backbone_assessment(
                novelty_node=node,
                all_nodes=nodes,
                links=links,
                v1_edge=v1_edges[node.claim_id],
            )
            for node in novelty_nodes
        ]

        depth, advisory, reasons, interpretation = (
            _depth_from_assessments(
                v1_profile=v1_profile,
                assessments=assessments,
            )
        )

        overlay = conceptual.get(
            hypothesis_id,
            ConceptualKnownnessOverlay(),
        )

        known_backbone_claim_ids = sorted(
            {
                claim_id
                for assessment in assessments
                for claim_id in assessment.linked_backbone_claim_ids
            }
        )

        higher_order_ids = [
            row.claim_id
            for row in assessments
            if row.backbone_strength
            == "MULTIPLE_INDEPENDENT_SIGNALS"
        ]
        local_ids = [
            row.claim_id
            for row in assessments
            if row.backbone_strength == "SINGLE_SIGNAL"
        ]
        weak_ids = [
            row.claim_id
            for row in assessments
            if row.unsupported_bridge_risk_v2
        ]

        cross_claim = any(
            row.linked_backbone_claim_ids
            for row in assessments
        )

        if overlay.available:
            reasons.append(
                "conceptual_knownness_overlay_available_diagnostic_only"
            )
            if overlay.coverage_sufficient is False:
                reasons.append(
                    "conceptual_knownness_overlay_coverage_insufficient"
                )
            elif overlay.coverage_sufficient is True:
                reasons.append(
                    "conceptual_knownness_overlay_coverage_sufficient"
                )

        profiles.append(
            HypothesisRelationCoverageGraphV2(
                hypothesis_id=hypothesis_id,
                title=str(external_card.title),
                source_external_status=str(external_card.status),
                nodes=nodes,
                links=links,
                novelty_bearing_relations=assessments,
                known_backbone_claim_ids=known_backbone_claim_ids,
                novelty_bearing_claim_ids=[
                    row.claim_id
                    for row in novelty_nodes
                ],
                v1_novelty_depth_class=str(
                    v1_profile.novelty_depth_class
                ),
                novelty_depth_class=depth,
                planner_advisory=advisory,
                conceptual_knownness=overlay,
                cross_claim_backbone_used=cross_claim,
                weak_bridge_claim_ids=weak_ids,
                higher_order_claim_ids=higher_order_ids,
                local_extension_claim_ids=local_ids,
                reason_codes=sorted(set(reasons)),
                interpretation=interpretation,
            )
        )

    depth_counts = Counter(
        row.novelty_depth_class
        for row in profiles
    )
    advisory_counts = Counter(
        row.planner_advisory
        for row in profiles
    )

    body = {
        "schema_version":
            "hypothesis-causal-edge-graph-novelty-depth-v2",
        "source_context_id": str(v1_report.source_context_id),
        "source_portfolio_id": str(portfolio.portfolio_id),
        "source_external_report_id": str(external_report.report_id),
        "source_v1_report_id": str(v1_report.report_id),
        "profiles": [
            row.model_dump(mode="json")
            for row in profiles
        ],
        "hypothesis_count": len(profiles),
        "novelty_depth_counts": dict(sorted(depth_counts.items())),
        "planner_advisory_counts": dict(
            sorted(advisory_counts.items())
        ),
        "cross_claim_backbone_hypothesis_count": sum(
            row.cross_claim_backbone_used
            for row in profiles
        ),
        "higher_order_gap_hypothesis_count": sum(
            row.novelty_depth_class
            == "HIGHER_ORDER_INTERACTION_GAP"
            for row in profiles
        ),
        "shallow_local_extension_count": sum(
            row.novelty_depth_class
            == "SHALLOW_LOCAL_EXTENSION"
            for row in profiles
        ),
        "weak_bridge_hypothesis_count": sum(
            row.novelty_depth_class
            == "UNSUPPORTED_BRIDGE_DEPTH_UNKNOWN"
            for row in profiles
        ),
        "conceptual_knownness_signal_count": sum(
            row.conceptual_knownness.available
            for row in profiles
        ),
        "conceptual_knownness_sufficient_count": sum(
            row.conceptual_knownness.coverage_sufficient is True
            for row in profiles
        ),
        "conceptual_knownness_changed_depth_count": 0,
        "foundational_knownness_checked": False,
        "conceptual_knownness_authority_created": False,
        "external_prior_art_as_positive_premise": False,
        "ranking_computed": False,
        "novelty_authority_created": False,
        "production_selection_changed": False,
    }

    report_id = _stable_id(
        "hypothesis_causal_edge_graph_novelty_depth_v2",
        body["source_context_id"],
        body["source_portfolio_id"],
        body["source_external_report_id"],
        body["source_v1_report_id"],
        *[
            (
                f"{row.hypothesis_id}:"
                f"{row.v1_novelty_depth_class}:"
                f"{row.novelty_depth_class}:"
                f"{row.planner_advisory}"
            )
            for row in profiles
        ],
    )

    return HypothesisCausalEdgeGraphNoveltyDepthV2Report(
        **body,
        report_id=report_id,
        report_sha256=_sha256(
            {
                **body,
                "report_id": report_id,
            }
        ),
    )


__all__ = [
    "RelationCoverageNode",
    "RelationCoverageLink",
    "NoveltyBearingRelationAssessment",
    "ConceptualKnownnessOverlay",
    "HypothesisRelationCoverageGraphV2",
    "HypothesisCausalEdgeGraphNoveltyDepthV2Report",
    "build_hypothesis_causal_edge_graph_novelty_depth_v2",
]
