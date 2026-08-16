from __future__ import annotations

import hashlib
import json
import re

from dac_her.domain_profile import ScientificDomainProfile
from dac_her.domains.registry import get_domain_profile
from dac_her.external_novelty_contracts import (
    ClaimPriorArtReview,
    ExternalNoveltyCard,
    ExternalNoveltyReport,
    LiteratureQueryPlan,
)
from dac_her.hypothesis_contracts import HypothesisPortfolio
from dac_her.novelty_refinement_contracts import (
    NoveltyGap,
    NoveltyGapPlan,
    TargetedGapQuery,
)


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _compact(text: str, limit: int = 210) -> str:
    value = " ".join(str(text).split())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _claim_priority(review: ClaimPriorArtReview) -> tuple[int, int, float, str]:
    # Lower is more useful as a refinement differentiator.
    order = {
        "NO_DIRECT_MATCH_FOUND": 0,
        "COMPONENTS_ONLY": 1,
        "TITLE_ONLY_NEIGHBORS": 2,
        "INSUFFICIENT_METADATA": 3,
        "PARTIAL_PRIOR_ART": 4,
        "DIRECT_PRIOR_ART": 5,
        "CONFLICTING_PRIOR_ART": 6,
    }
    best_scope = max(
        (x.catalyst_scope_relevance for x in review.matches),
        default=0.0,
    )
    return (
        order.get(review.status, 9),
        -int(review.importance == "core"),
        best_scope,
        review.claim_id,
    )


def _query_terms(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9 Δδ\-]+", " ", text)
    words = [x for x in text.split() if len(x) > 2]
    stop = {
        "the", "and", "that", "with", "from", "when", "into", "may", "can",
        "should", "different", "effect", "through", "rather", "than", "only",
        "between", "across", "sites", "site",
    }
    out = []
    seen = set()
    for word in words:
        key = word.lower()
        if key in stop or key in seen:
            continue
        seen.add(key)
        out.append(word)
        if len(out) >= 14:
            break
    return " ".join(out)


class NoveltyGapAnalyzer:
    """Deterministic external-novelty gap planner.

    It never turns external prior art into scientific premises. It only identifies
    what the next bounded search/refinement should discriminate. Targeted query
    expansion is owned by the selected ScientificDomainProfile.
    """

    ACTION_BY_STATUS = {
        "PLAUSIBLY_NOVEL": "keep",
        "NEW_COMBINATION_OF_KNOWN_EFFECTS": "keep",
        "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP": "targeted_search_only",
        "INSUFFICIENT_SEARCH_EVIDENCE": "targeted_search_then_refine",
        "CONFLICTING_PRIOR_ART": "refine_away_from_conflict",
        "WELL_ESTABLISHED": "targeted_search_then_refine",
        "LITERATURE_SUPPORTED_EXTENSION": "targeted_search_then_refine",
    }

    def __init__(
        self,
        *,
        max_target_claims: int = 2,
        queries_per_gap: int = 3,
        domain_profile: ScientificDomainProfile | None = None,
    ) -> None:
        self.max_target_claims = max(1, int(max_target_claims))
        self.queries_per_gap = max(1, int(queries_per_gap))
        self.domain_profile = domain_profile or get_domain_profile("dac_her")

    def _action(self, card: ExternalNoveltyCard) -> str:
        try:
            return self.ACTION_BY_STATUS[card.status]
        except KeyError as exc:
            raise ValueError(
                f"unsupported external novelty status for gap planning: {card.status}"
            ) from exc

    def _queries(
        self,
        card: ExternalNoveltyCard,
        targets: list[ClaimPriorArtReview],
        existing_plan: LiteratureQueryPlan,
    ) -> list[TargetedGapQuery]:
        existing = {
            (
                q.claim_id,
                q.query_text.strip().lower(),
            )
            for q in existing_plan.queries
            if (
                q.hypothesis_id == card.hypothesis_id
                and q.claim_id is not None
            )
        }
        novelty = self.domain_profile.novelty
        per_claim: list[list[TargetedGapQuery]] = []
        for review in targets:
            core = _query_terms(review.claim_text)
            rows: list[TargetedGapQuery] = []
            if core:
                variants = novelty.targeted_query_variants(core)
                for index, query in enumerate(variants):
                    rows.append(
                        TargetedGapQuery(
                            claim_id=review.claim_id,
                            query_role=(
                                "relation_primary"
                                if index == 0
                                else "relation_variant"
                            ),
                            query_text=query,
                        )
                    )
            per_claim.append(rows)

        candidates: list[TargetedGapQuery] = []

        # Phase 1: every target claim gets a primary relation query before
        # any claim consumes budget on secondary variants.
        for rows in per_claim:
            if rows:
                candidates.append(rows[0])

        # Phase 2: contextual-conflict scope checks are bound only to a target
        # claim whose frozen review actually contains that contextual conflict.
        contextual_ids = set(card.contextual_conflict_work_ids)
        if contextual_ids:
            for review in targets:
                has_contextual_conflict = any(
                    (
                        match.relationship == "CONTEXTUAL_CONFLICT"
                        and match.work_id in contextual_ids
                    )
                    for match in review.matches
                )
                if not has_contextual_conflict:
                    continue
                core = _query_terms(review.claim_text)
                if not core:
                    continue
                for query in novelty.contextual_conflict_query_variants(core):
                    candidates.append(
                        TargetedGapQuery(
                            claim_id=review.claim_id,
                            query_role="scope_check",
                            query_text=query,
                        )
                    )

        # Phase 3: secondary relation variants are allocated round-robin
        # across target claims so one claim cannot monopolize the query budget.
        max_variants = max((len(rows) for rows in per_claim), default=0)
        for variant_index in range(1, max_variants):
            for rows in per_claim:
                if variant_index < len(rows):
                    candidates.append(rows[variant_index])

        result: list[TargetedGapQuery] = []
        seen = set(existing)
        for query in candidates:
            normalized = " ".join(query.query_text.split())
            key = (query.claim_id, normalized.lower())
            if not normalized or key in seen:
                continue
            seen.add(key)
            result.append(
                query.model_copy(update={"query_text": normalized[:300]})
            )
            if len(result) >= self.queries_per_gap:
                break
        return result

    def build(
        self,
        portfolio: HypothesisPortfolio,
        external: ExternalNoveltyReport,
        existing_plan: LiteratureQueryPlan,
    ) -> NoveltyGapPlan:
        if external.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("external novelty report source_portfolio_id mismatch")
        if existing_plan.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("literature query plan source_portfolio_id mismatch")

        card_by_id = {x.hypothesis_id: x for x in external.cards}
        gaps: list[NoveltyGap] = []
        for hypothesis in portfolio.hypotheses:
            card = card_by_id.get(hypothesis.hypothesis_id)
            if card is None:
                raise ValueError(
                    f"external report lacks hypothesis {hypothesis.hypothesis_id}"
                )
            action = self._action(card)
            ordered = sorted(card.claim_reviews, key=_claim_priority)
            if action == "refine_away_from_conflict":
                conflicting = [
                    row
                    for row in ordered
                    if row.status == "CONFLICTING_PRIOR_ART"
                ]
                non_conflicting = [
                    row
                    for row in ordered
                    if row.status != "CONFLICTING_PRIOR_ART"
                ]
                ordered = conflicting + non_conflicting
            targets = ordered[: self.max_target_claims]
            known = []
            unresolved = []
            for review in card.claim_reviews:
                line = f"{review.status}: {_compact(review.claim_text)}"
                if review.status in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}:
                    known.append(line)
                else:
                    unresolved.append(line)

            differentiator = (
                _compact(targets[0].claim_text, 320)
                if targets
                else _compact(hypothesis.hypothesis_statement, 320)
            )
            reasons = [f"source_status:{card.status}"]
            if card.contextual_conflict_work_ids:
                reasons.append("contextual_conflict_requires_target_scope_check")
            if not card.coverage.sufficient_for_absence_based_novelty:
                reasons.append("absence_coverage_incomplete")
            if any(x.status == "PARTIAL_PRIOR_ART" for x in targets):
                reasons.append("refine_beyond_partial_prior_art")
            if any(
                x.status in {
                    "COMPONENTS_ONLY",
                    "TITLE_ONLY_NEIGHBORS",
                    "NO_DIRECT_MATCH_FOUND",
                    "INSUFFICIENT_METADATA",
                }
                for x in targets
            ):
                reasons.append("unresolved_claim_boundary")

            gap_id = _stable_id(
                "novelty_gap",
                hypothesis.hypothesis_id,
                card.status,
                *(x.claim_id for x in targets),
            )
            gaps.append(
                NoveltyGap(
                    gap_id=gap_id,
                    hypothesis_id=hypothesis.hypothesis_id,
                    source_external_status=card.status,
                    action=action,
                    target_claim_ids=[x.claim_id for x in targets],
                    differentiator=differentiator,
                    already_known_boundary=known[:6],
                    unresolved_boundary=unresolved[:6],
                    contextual_conflict_work_ids=list(
                        card.contextual_conflict_work_ids[:5]
                    ),
                    targeted_queries=(
                        []
                        if action == "keep"
                        else self._queries(card, targets, existing_plan)
                    ),
                    reason_codes=sorted(set(reasons)),
                )
            )

        plan_id = _stable_id(
            "novelty_gap_plan",
            portfolio.portfolio_id,
            external.report_id,
            *(f"{x.gap_id}:{x.action}" for x in gaps),
        )
        body = {
            "schema_version": "novelty-gap-plan-v2",
            "plan_id": plan_id,
            "source_portfolio_id": portfolio.portfolio_id,
            "source_external_report_id": external.report_id,
            "gaps": [x.model_dump(mode="json") for x in gaps],
            "policy_version": "novelty-gap-policy-v2",
        }
        return NoveltyGapPlan(**body, plan_sha256=_sha256_json(body))
