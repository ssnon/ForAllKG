from __future__ import annotations

import hashlib
import json
import re

from dac_her.external_novelty_contracts import (
    ClaimPriorArtReview,
    ExternalNoveltyCard,
    ExternalNoveltyReport,
    LiteratureQueryPlan,
)
from dac_her.hypothesis_contracts import HypothesisPortfolio
from dac_her.novelty_refinement_contracts import NoveltyGap, NoveltyGapPlan


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
    abstract_count = review.coverage.abstract_work_count
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
    what the next bounded search/refinement should discriminate.
    """

    def __init__(self, *, max_target_claims: int = 2, queries_per_gap: int = 3) -> None:
        self.max_target_claims = max(1, int(max_target_claims))
        self.queries_per_gap = max(1, int(queries_per_gap))

    def _action(self, card: ExternalNoveltyCard) -> str:
        if card.status in {"PLAUSIBLY_NOVEL", "NEW_COMBINATION_OF_KNOWN_EFFECTS"}:
            return "keep"
        if card.status == "INSUFFICIENT_SEARCH_EVIDENCE":
            return "targeted_search_then_refine"
        if card.status == "CONFLICTING_PRIOR_ART":
            return "refine_away_from_conflict"
        if card.status == "WELL_ESTABLISHED":
            return "targeted_search_then_refine"
        if card.status == "LITERATURE_SUPPORTED_EXTENSION":
            return "targeted_search_then_refine"
        return "targeted_search_only"

    def _queries(
        self,
        card: ExternalNoveltyCard,
        targets: list[ClaimPriorArtReview],
        existing_plan: LiteratureQueryPlan,
    ) -> list[str]:
        existing = {
            q.query_text.strip().lower()
            for q in existing_plan.queries
            if q.hypothesis_id == card.hypothesis_id
        }
        candidates: list[str] = []
        for review in targets:
            core = _query_terms(review.claim_text)
            if core:
                candidates.extend(
                    [
                        core,
                        f"{core} hydrogen evolution reaction mechanism",
                        f"{core} dual atom catalyst nitrogen coordination",
                    ]
                )
        # A contextual conflict should be tested in the actual target scope.
        if card.contextual_conflict_work_ids and targets:
            core = _query_terms(targets[0].claim_text)
            candidates.append(f"{core} nitrogen coordinated dual atom HER")
        result: list[str] = []
        seen = set(existing)
        for query in candidates:
            query = " ".join(query.split())
            key = query.lower()
            if not query or key in seen:
                continue
            seen.add(key)
            result.append(query[:300])
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
            ordered = sorted(card.claim_reviews, key=_claim_priority)
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
            action = self._action(card)
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
                    targeted_queries=self._queries(
                        card, targets, existing_plan
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
            "schema_version": "novelty-gap-plan-v1",
            "plan_id": plan_id,
            "source_portfolio_id": portfolio.portfolio_id,
            "source_external_report_id": external.report_id,
            "gaps": [x.model_dump(mode="json") for x in gaps],
            "policy_version": "novelty-gap-policy-v1",
        }
        return NoveltyGapPlan(**body, plan_sha256=_sha256_json(body))
