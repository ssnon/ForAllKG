from __future__ import annotations

import re
from typing import Any, Protocol

import numpy as np

from pipeline_core.domain_profile import ScientificDomainProfile
from domains.registry import get_domain_profile
from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtCandidateSet,
    ClaimPriorArtReview,
    ClaimPriorArtReviewDraft,
    ClaimSearchCoverage,
    LiteratureQueryPlan,
    NoveltyClaim,
    PriorArtMatch,
    PriorArtPacket,
    PriorArtRelationship,
    RankedPriorArtWork,
)


def _normalize_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"expected 1D vector, got {array.shape}")
    norm = float(np.linalg.norm(array))
    return array if norm <= 0.0 else array / norm


def _norm_text(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[‐‑‒–—−-]+", " ", text)
    text = re.sub(r"[^a-z0-9α-ω가-힣]+", " ", text)
    return " ".join(text.split())


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "in", "is", "it", "may", "of", "on", "or", "that", "the", "their", "this",
    "to", "with", "within", "via", "effect", "effects", "activity", "site", "sites",
    "metal", "metals", "catalyst", "catalysts", "reaction", "response", "different",
}


def _tokens(value: str) -> set[str]:
    return {
        tok
        for tok in _norm_text(value).split()
        if len(tok) >= 2 and tok not in _STOPWORDS
    }


def _lexical_coverage(claim: NoveltyClaim, document: str) -> float:
    """Token/concept coverage rather than impossible whole-phrase containment.

    Alpha5 treated every search_concept string as one literal phrase, which made
    coverage almost always zero. Alpha5.1 uses the union of distinctive tokens
    from search_concepts (falling back to the claim text).
    """
    concept_tokens: set[str] = set()
    for row in claim.search_concepts:
        concept_tokens |= _tokens(row)
    if not concept_tokens:
        concept_tokens = _tokens(claim.text)
    if not concept_tokens:
        return 0.0
    haystack = _tokens(document)
    return len(concept_tokens & haystack) / len(concept_tokens)


_DEFAULT_DOMAIN_PROFILE = get_domain_profile("dac_her")
_REACTION_PATTERNS = _DEFAULT_DOMAIN_PROFILE.novelty.compiled_domain_patterns()


def _reaction_domains(text: str) -> set[str]:
    return {
        name
        for name, patterns in _REACTION_PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    }


def _reaction_domain_relevance(claim_text: str, document: str) -> float:
    return _DEFAULT_DOMAIN_PROFILE.novelty.domain_relevance(claim_text, document)


_SCOPE_PATTERNS = _DEFAULT_DOMAIN_PROFILE.novelty.compiled_scope_patterns()


def _scope_features(text: str) -> set[str]:
    return {
        name
        for name, patterns in _SCOPE_PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    }


def _catalyst_scope_relevance(claim_text: str, document: str) -> float:
    return _DEFAULT_DOMAIN_PROFILE.novelty.scope_relevance(claim_text, document)


def _strong_scope_compatibility(
    claim_text: str,
    document: str,
    *,
    min_reaction: float,
    min_scope: float,
) -> tuple[bool, float, float, list[str]]:
    return _DEFAULT_DOMAIN_PROFILE.novelty.strong_scope_compatibility(
        claim_text, document, min_domain=min_reaction, min_scope=min_scope
    )


class EncoderProtocol(Protocol):
    def encode_query(self, text: str) -> np.ndarray: ...
    def encode_documents(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray: ...


class ClaimReviewBackend(Protocol):
    def review_claim(
        self,
        claim: NoveltyClaim,
        works: list[dict[str, Any]],
    ) -> ClaimPriorArtReviewDraft: ...


class PriorArtRanker:
    def __init__(
        self,
        encoder: EncoderProtocol,
        *,
        max_ranked_works_per_claim: int = 8,
        domain_profile: ScientificDomainProfile | None = None,
    ) -> None:
        self.encoder = encoder
        self.max_ranked = int(max_ranked_works_per_claim)
        self.domain_profile = domain_profile or _DEFAULT_DOMAIN_PROFILE

    def rank(
        self,
        claim: NoveltyClaim,
        packet: PriorArtPacket,
        plan: LiteratureQueryPlan,
    ) -> ClaimPriorArtCandidateSet:
        global_query_ids = {
            row.query_id
            for row in plan.queries
            if row.hypothesis_id == claim.hypothesis_id and row.claim_id is None
        }
        candidate_rows = [
            work
            for work in packet.works
            if claim.claim_id in work.retrieval_claim_ids
            or bool(global_query_ids & set(work.retrieval_query_ids))
        ]
        if not candidate_rows:
            return ClaimPriorArtCandidateSet(
                hypothesis_id=claim.hypothesis_id,
                claim_id=claim.claim_id,
                ranked_works=[],
            )

        query = _normalize_vector(self.encoder.encode_query(claim.text))
        documents = [
            "\n".join(x for x in [work.title, work.abstract or ""] if x)
            for work in candidate_rows
        ]
        encoded = np.asarray(
            self.encoder.encode_documents(
                documents,
                batch_size=max(1, min(32, len(documents))),
            ),
            dtype=np.float32,
        )
        ranked: list[RankedPriorArtWork] = []
        for work, document, vector in zip(candidate_rows, documents, encoded, strict=True):
            semantic = max(0.0, min(1.0, float(np.dot(query, _normalize_vector(vector)))))
            lexical = _lexical_coverage(claim, document)
            reaction = self.domain_profile.novelty.domain_relevance(claim.text, document)
            scope = self.domain_profile.novelty.scope_relevance(claim.text, document)
            score = max(
                0.0,
                min(
                    1.0,
                    0.62 * semantic
                    + 0.18 * lexical
                    + 0.12 * reaction
                    + 0.08 * scope,
                ),
            )
            ranked.append(
                RankedPriorArtWork(
                    work_id=work.work_id,
                    relevance_score=score,
                    semantic_similarity=semantic,
                    lexical_coverage=lexical,
                    reaction_domain_relevance=reaction,
                    catalyst_scope_relevance=scope,
                    abstract_available=bool(work.abstract),
                )
            )
        ranked.sort(
            key=lambda row: (
                -row.relevance_score,
                -row.reaction_domain_relevance,
                -row.catalyst_scope_relevance,
                -row.semantic_similarity,
                row.work_id,
            )
        )
        return ClaimPriorArtCandidateSet(
            hypothesis_id=claim.hypothesis_id,
            claim_id=claim.claim_id,
            ranked_works=ranked[: self.max_ranked],
        )


class ClaimPriorArtCompiler:
    def __init__(
        self,
        *,
        min_match_confidence: float = 0.65,
        direct_match_confidence: float = 0.70,
        require_abstract_for_strong_match: bool = True,
        require_abstract_for_partial_match: bool = True,
        min_reaction_domain_for_conflict: float = 0.75,
        min_catalyst_scope_for_conflict: float = 0.75,
        domain_profile: ScientificDomainProfile | None = None,
    ) -> None:
        self.min_match_confidence = float(min_match_confidence)
        self.domain_profile = domain_profile or _DEFAULT_DOMAIN_PROFILE
        self.direct_match_confidence = float(direct_match_confidence)
        self.require_abstract = bool(require_abstract_for_strong_match)
        self.require_abstract_for_partial = bool(require_abstract_for_partial_match)
        self.min_reaction_for_conflict = float(min_reaction_domain_for_conflict)
        self.min_scope_for_conflict = float(min_catalyst_scope_for_conflict)

    def compile(
        self,
        claim: NoveltyClaim,
        candidates: ClaimPriorArtCandidateSet,
        draft: ClaimPriorArtReviewDraft,
        packet: PriorArtPacket,
        plan: LiteratureQueryPlan,
    ) -> ClaimPriorArtReview:
        works = {row.work_id: row for row in packet.works}
        ranking = {row.work_id: row for row in candidates.ranked_works}
        allowed = set(ranking)
        unknown = sorted({row.work_id for row in draft.matches} - allowed)
        valid_draft_matches = [
            row
            for row in draft.matches
            if row.work_id in allowed
        ]

        matches: list[PriorArtMatch] = []
        reason_codes: list[str] = []
        if unknown:
            # Never fuzzy-match an unknown model-returned ID to a real paper.
            # The invalid match is unusable evidence, but it should not abort
            # the entire external-novelty assessment.
            reason_codes.append("reviewer_unknown_work_id_dropped")
            reason_codes.extend(
                f"reviewer_unknown_work_id_dropped:{work_id}"
                for work_id in unknown
            )

        for row in valid_draft_matches:
            work = works[row.work_id]
            ranked = ranking[row.work_id]
            relationship: PriorArtRelationship = row.relationship
            document = "\n".join(x for x in [work.title, work.abstract or ""] if x)
            compatible, reaction, scope, scope_reasons = (
                self.domain_profile.novelty.strong_scope_compatibility(
                    claim.text,
                    document,
                    min_domain=self.min_reaction_for_conflict,
                    min_scope=self.min_scope_for_conflict,
                )
            )

            if not work.abstract:
                if relationship in {"DIRECT_PRIOR_ART", "CONFLICTING_PRIOR_ART"}:
                    relationship = "TITLE_ONLY_NEIGHBOR"
                    reason_codes.append("strong_match_downgraded_without_abstract")
                elif self.require_abstract_for_partial and relationship == "PARTIAL_PRIOR_ART":
                    relationship = "TITLE_ONLY_NEIGHBOR"
                    reason_codes.append("partial_match_downgraded_without_abstract")
            elif relationship == "CONFLICTING_PRIOR_ART" and not compatible:
                relationship = "CONTEXTUAL_CONFLICT"
                reason_codes.append("conflict_downgraded_for_scope_mismatch")
            elif relationship == "DIRECT_PRIOR_ART" and not compatible:
                relationship = "PARTIAL_PRIOR_ART"
                reason_codes.append("direct_match_downgraded_for_scope_mismatch")

            matches.append(
                PriorArtMatch(
                    work_id=work.work_id,
                    relationship=relationship,
                    confidence=row.confidence,
                    rationale=row.rationale,
                    relevance_score=ranked.relevance_score,
                    semantic_similarity=ranked.semantic_similarity,
                    lexical_coverage=ranked.lexical_coverage,
                    reaction_domain_relevance=reaction,
                    catalyst_scope_relevance=scope,
                    scope_compatible_for_conflict=compatible,
                    scope_reason_codes=scope_reasons,
                    title=work.title,
                    year=work.year,
                    doi=work.doi,
                    url=work.url,
                    abstract_available=bool(work.abstract),
                )
            )

        direct = [
            row for row in matches
            if row.relationship == "DIRECT_PRIOR_ART"
            and row.confidence >= self.direct_match_confidence
        ]
        conflicting = [
            row for row in matches
            if row.relationship == "CONFLICTING_PRIOR_ART"
            and row.confidence >= self.direct_match_confidence
        ]
        partial = [
            row for row in matches
            if row.relationship == "PARTIAL_PRIOR_ART"
            and row.confidence >= self.min_match_confidence
        ]
        components = [
            row for row in matches
            if row.relationship in {"COMPONENT_ONLY", "CONTEXTUAL_CONFLICT"}
            and row.confidence >= self.min_match_confidence
        ]
        title_only = [
            row for row in matches
            if row.relationship == "TITLE_ONLY_NEIGHBOR"
            and row.confidence >= self.min_match_confidence
        ]
        contextual_conflicts = [
            row for row in matches
            if row.relationship == "CONTEXTUAL_CONFLICT"
            and row.confidence >= self.min_match_confidence
        ]
        if contextual_conflicts:
            reason_codes.append("contextual_conflict_present")

        claim_query_ids = {
            row.query_id
            for row in plan.queries
            if row.claim_id == claim.claim_id
        }
        successful_query_ids = {
            row.query_id
            for row in packet.executions
            if row.success and row.query_id in claim_query_ids
        }
        candidate_work_ids = {row.work_id for row in candidates.ranked_works}
        abstract_count = sum(bool(works[row].abstract) for row in candidate_work_ids)
        coverage = ClaimSearchCoverage(
            claim_id=claim.claim_id,
            query_count=len(claim_query_ids),
            successful_query_count=len(successful_query_ids),
            unique_work_count=len(candidate_work_ids),
            abstract_work_count=abstract_count,
            reviewed_work_count=len(matches),
        )

        if conflicting:
            status = "CONFLICTING_PRIOR_ART"
            reason_codes.append("high_confidence_scope_matched_conflicting_prior_art")
        elif direct:
            status = "DIRECT_PRIOR_ART"
            reason_codes.append("high_confidence_direct_prior_art")
        elif partial:
            status = "PARTIAL_PRIOR_ART"
            reason_codes.append("abstract_backed_partial_prior_art")
        elif components:
            status = "COMPONENTS_ONLY"
            reason_codes.append("known_components_without_direct_match")
        elif title_only:
            status = "TITLE_ONLY_NEIGHBORS"
            reason_codes.append("title_only_neighboring_prior_art")
        elif unknown:
            # Because the reviewer produced unusable IDs, absence of a valid
            # match cannot be interpreted as "no direct prior art found".
            status = "INSUFFICIENT_METADATA"
            reason_codes.append(
                "reviewer_unknown_work_id_prevents_absence_inference"
            )
        elif not candidates.ranked_works or abstract_count == 0:
            status = "INSUFFICIENT_METADATA"
            reason_codes.append("insufficient_ranked_abstract_evidence")
        else:
            status = "NO_DIRECT_MATCH_FOUND"
            reason_codes.append("no_direct_match_in_reviewed_candidates")

        matches.sort(
            key=lambda row: (
                -row.confidence,
                -row.relevance_score,
                row.work_id,
            )
        )
        return ClaimPriorArtReview(
            hypothesis_id=claim.hypothesis_id,
            claim_id=claim.claim_id,
            claim_text=claim.text,
            importance=claim.importance,
            status=status,
            matches=matches,
            coverage=coverage,
            reason_codes=sorted(set(reason_codes)),
            reviewer_unknown_work_ids=unknown,
            interpretation=draft.interpretation,
        )
