from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
    NoveltyClaim,
    PriorArtPacket,
)
from pipeline_core.discovery.prior_art_retrieval import (
    canonicalize_prior_art_packet,
)


_POSITIVE_RELATIONSHIPS = {
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by",
    "for", "from", "in", "is", "of", "on", "or",
    "that", "the", "their", "this", "to", "with",
    "across", "different", "relationship",
}


class EncoderProtocol(Protocol):
    def encode_query(self, text: str) -> np.ndarray: ...

    def encode_documents(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class PriorArtMemoryEntry:
    claim: NoveltyClaim
    work_ids: tuple[str, ...]


@dataclass(frozen=True)
class PriorArtMemoryMatch:
    new_claim_id: str
    memory_claim_id: str
    semantic_similarity: float
    relation_overlap: float
    shared_relation_token_count: int
    distinguishing_terms: tuple[str, ...]
    work_ids: tuple[str, ...]


def _norm_text(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[‐-‒–—−-]+", " ", text)
    text = re.sub(
        r"[^a-z0-9α-ω가-힣]+",
        " ",
        text,
    )
    return " ".join(text.split())


def _tokens(values: list[str]) -> set[str]:
    result: set[str] = set()

    for value in values:
        for token in _norm_text(value).split():
            if (
                len(token) >= 2
                and token not in _STOPWORDS
            ):
                result.add(token)

    return result


def _normalize_vector(value) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float32,
    )

    if array.ndim != 1:
        raise ValueError(
            f"expected 1D vector, got {array.shape}"
        )

    norm = float(np.linalg.norm(array))

    if norm <= 0.0:
        return array

    return array / norm


def _facet_corpus(claim: NoveltyClaim) -> list[str]:
    return [
        *claim.distinguishing_terms,
        *claim.search_concepts,
        claim.text,
    ]


def _distinguishing_facets_match(
    new_claim: NoveltyClaim,
    memory_claim: NoveltyClaim,
) -> bool:
    # Memory propagation is intentionally conservative.
    # Claims without an explicit new distinguishing facet do not
    # receive cross-claim memory.
    if not new_claim.distinguishing_terms:
        return False

    corpus = [
        _norm_text(value)
        for value in _facet_corpus(
            memory_claim
        )
        if _norm_text(value)
    ]

    for raw_term in new_claim.distinguishing_terms:
        term = _norm_text(raw_term)

        if not term:
            return False

        if not any(
            term == value
            or term in value
            for value in corpus
        ):
            return False

    return True


def _relation_tokens(
    claim: NoveltyClaim,
) -> set[str]:
    # diagnostic_relation_terms intentionally omit the
    # higher-order moderator, so they are preferred as the
    # base-relation signature.
    source = list(
        claim.diagnostic_relation_terms
    )

    if not source:
        source = list(
            claim.search_concepts
        )

    tokens = _tokens(source)

    # Do not allow the distinguishing facet itself to make
    # two otherwise unrelated relations look compatible.
    tokens -= _tokens(
        claim.distinguishing_terms
    )

    return tokens


def _relation_compatibility(
    left: NoveltyClaim,
    right: NoveltyClaim,
) -> tuple[float, int]:
    a = _relation_tokens(left)
    b = _relation_tokens(right)

    if not a or not b:
        return 0.0, 0

    shared = a & b

    overlap = (
        len(shared)
        / min(len(a), len(b))
    )

    return overlap, len(shared)


def build_prior_art_memory(
    plan: LiteratureQueryPlan,
    report: ExternalNoveltyReport,
) -> list[PriorArtMemoryEntry]:
    claims = {
        claim.claim_id: claim
        for group in plan.claims
        for claim in group.claims
    }

    entries: list[PriorArtMemoryEntry] = []

    for card in report.cards:
        for review in card.claim_reviews:
            claim = claims.get(
                review.claim_id
            )

            if claim is None:
                continue

            work_ids = sorted({
                match.work_id
                for match in review.matches
                if (
                    match.relationship
                    in _POSITIVE_RELATIONSHIPS
                )
            })

            if not work_ids:
                continue

            entries.append(
                PriorArtMemoryEntry(
                    claim=claim,
                    work_ids=tuple(work_ids),
                )
            )

    entries.sort(
        key=lambda row: row.claim.claim_id
    )

    return entries


class PriorArtMemoryMatcher:
    """Recall reviewed prior art without inheriting its old judgment.

    Matching only decides which previously reviewed papers deserve
    re-exposure to the current claim reviewer. DIRECT/PARTIAL status
    is never copied to the new claim.
    """

    def __init__(
        self,
        encoder: EncoderProtocol,
        *,
        min_semantic_similarity: float = 0.75,
        min_relation_overlap: float = 0.25,
        min_shared_relation_tokens: int = 2,
    ) -> None:
        self.encoder = encoder
        self.min_semantic_similarity = float(
            min_semantic_similarity
        )
        self.min_relation_overlap = float(
            min_relation_overlap
        )
        self.min_shared_relation_tokens = int(
            min_shared_relation_tokens
        )

    def match(
        self,
        claim: NoveltyClaim,
        memory: list[PriorArtMemoryEntry],
    ) -> list[PriorArtMemoryMatch]:
        gated: list[
            tuple[
                PriorArtMemoryEntry,
                float,
                int,
            ]
        ] = []

        for entry in memory:
            if not _distinguishing_facets_match(
                claim,
                entry.claim,
            ):
                continue

            overlap, shared = (
                _relation_compatibility(
                    claim,
                    entry.claim,
                )
            )

            if (
                shared
                < self.min_shared_relation_tokens
            ):
                continue

            if (
                overlap
                < self.min_relation_overlap
            ):
                continue

            gated.append(
                (
                    entry,
                    overlap,
                    shared,
                )
            )

        if not gated:
            return []

        query = _normalize_vector(
            self.encoder.encode_query(
                claim.text
            )
        )

        documents = [
            entry.claim.text
            for entry, _, _ in gated
        ]

        vectors = np.asarray(
            self.encoder.encode_documents(
                documents,
                batch_size=max(
                    1,
                    min(32, len(documents)),
                ),
            ),
            dtype=np.float32,
        )

        matches: list[
            PriorArtMemoryMatch
        ] = []

        for (
            entry,
            overlap,
            shared,
        ), vector in zip(
            gated,
            vectors,
            strict=True,
        ):
            similarity = float(
                np.dot(
                    query,
                    _normalize_vector(
                        vector
                    ),
                )
            )

            if (
                similarity
                < self.min_semantic_similarity
            ):
                continue

            matches.append(
                PriorArtMemoryMatch(
                    new_claim_id=claim.claim_id,
                    memory_claim_id=(
                        entry.claim.claim_id
                    ),
                    semantic_similarity=(
                        similarity
                    ),
                    relation_overlap=overlap,
                    shared_relation_token_count=(
                        shared
                    ),
                    distinguishing_terms=tuple(
                        claim.distinguishing_terms
                    ),
                    work_ids=entry.work_ids,
                )
            )

        matches.sort(
            key=lambda row: (
                -row.semantic_similarity,
                -row.relation_overlap,
                row.memory_claim_id,
            )
        )

        return matches



def augment_prior_art_packet_with_memory(
    *,
    current_plan: LiteratureQueryPlan,
    current_packet: PriorArtPacket,
    memory: list[PriorArtMemoryEntry],
    memory_packet: PriorArtPacket,
    matcher: PriorArtMemoryMatcher,
) -> tuple[
    PriorArtPacket,
    dict[str, tuple[str, ...]],
]:
    """Re-expose previously reviewed works to compatible current claims.

    The previous DIRECT/PARTIAL judgment is never inherited.

    Memory works:
    - are linked to the current claim for candidate ranking;
    - carry no current retrieval_query_id, because they were not
      discovered by the current search;
    - therefore do not inflate current search/absence coverage.

    The ordinary current reviewer must classify them again.
    """

    if (
        current_packet.source_query_plan_id
        != current_plan.plan_id
    ):
        raise ValueError(
            "current packet/query plan mismatch"
        )

    memory_works = {
        row.work_id: row
        for row in memory_packet.works
    }

    raw = list(current_packet.works)
    audit: dict[str, tuple[str, ...]] = {}

    for group in current_plan.claims:
        for claim in group.claims:
            matches = matcher.match(
                claim,
                memory,
            )

            selected_ids: list[str] = []
            seen: set[str] = set()

            for match in matches:
                for work_id in match.work_ids:
                    if work_id in seen:
                        continue

                    work = memory_works.get(
                        work_id
                    )

                    if work is None:
                        continue

                    seen.add(work_id)
                    selected_ids.append(work_id)

                    # Explicitly detach the old query lineage.
                    # This is prior-art memory, not a result of
                    # the current retrieval query.
                    raw.append(
                        work.model_copy(
                            update={
                                "retrieval_query_ids": [],
                                "retrieval_claim_ids": [
                                    claim.claim_id
                                ],
                            },
                            deep=True,
                        )
                    )

            audit[claim.claim_id] = tuple(
                selected_ids
            )

    augmented = current_packet.model_copy(
        update={
            "works": raw,
        },
        deep=True,
    )

    augmented = (
        canonicalize_prior_art_packet(
            augmented
        )
    )

    return augmented, audit
