from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtCandidateSet,
    NoveltyClaim,
    PriorArtPacket,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )


DiagnosticPriorArtRelationship = Literal[
    "LOWER_ORDER_RELATION_PRIOR_ART",
    "DIRECTIONAL_COUNTEREVIDENCE",
    "COMPONENT_ONLY",
    "TITLE_ONLY_NEIGHBOR",
    "UNRELATED",
    "INSUFFICIENT_METADATA",
]


class DiagnosticPriorArtMatchDraft(
    _StrictModel
):
    work_id: str
    relationship: (
        DiagnosticPriorArtRelationship
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    rationale: str = Field(
        min_length=1,
    )


class DiagnosticPriorArtReviewDraft(
    _StrictModel
):
    matches: list[
        DiagnosticPriorArtMatchDraft
    ] = Field(
        default_factory=list
    )
    interpretation: str = Field(
        min_length=1
    )


class DiagnosticPriorArtMatch(
    _StrictModel
):
    work_id: str
    relationship: (
        DiagnosticPriorArtRelationship
    )
    confidence: float
    rationale: str

    relevance_score: float
    semantic_similarity: float
    lexical_coverage: float
    reaction_domain_relevance: float
    catalyst_scope_relevance: float

    title: str
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    abstract_available: bool = False


class DiagnosticClaimPriorArtReview(
    _StrictModel
):
    schema_version: Literal[
        "diagnostic-claim-prior-art-review-v1"
    ] = (
        "diagnostic-claim-prior-art-review-v1"
    )

    hypothesis_id: str
    claim_id: str
    claim_text: str

    diagnostic_query_kind: str
    diagnostic_execution_query: str

    matches: list[
        DiagnosticPriorArtMatch
    ] = Field(
        default_factory=list
    )

    signal_work_ids: list[str] = Field(
        default_factory=list
    )

    reason_codes: list[str] = Field(
        default_factory=list
    )

    reviewer_unknown_work_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    interpretation: str

    epistemic_usage: Literal[
        "diagnostic_prior_art_only_not_full_claim_status_authority"
    ] = (
        "diagnostic_prior_art_only_not_full_claim_status_authority"
    )


_EXPECTED_SIGNAL = {
    "LOWER_ORDER_RELATION":
        "LOWER_ORDER_RELATION_PRIOR_ART",

    "DIRECTIONAL_BOUNDARY":
        "DIRECTIONAL_COUNTEREVIDENCE",
}


def compile_diagnostic_prior_art_review(
    *,
    claim: NoveltyClaim,
    candidates: ClaimPriorArtCandidateSet,
    draft: DiagnosticPriorArtReviewDraft,
    packet: PriorArtPacket,
    min_match_confidence: float = 0.65,
) -> DiagnosticClaimPriorArtReview:
    """Compile diagnostic-only reviewer output fail-closed.

    This compiler has no authority to change the ordinary full-claim
    DIRECT/PARTIAL status.

    A diagnostic signal is eligible only when:
      * its work ID was actually ranked for this claim,
      * its relationship matches the claim's diagnostic query kind,
      * the record has abstract metadata for a strong diagnostic signal,
      * confidence reaches the diagnostic threshold.

    Wrong diagnostic signal types are dropped rather than remapped.
    """

    expected = _EXPECTED_SIGNAL.get(
        claim.diagnostic_query_kind
    )

    if expected is None:
        raise ValueError(
            "diagnostic review requires "
            "diagnostic_query_kind != NONE"
        )

    execution_query = str(
        claim.diagnostic_execution_query
        or ""
    ).strip()

    if not execution_query:
        raise ValueError(
            "diagnostic review requires a "
            "diagnostic_execution_query"
        )

    works = {
        row.work_id: row
        for row in packet.works
    }

    ranking = {
        row.work_id: row
        for row in candidates.ranked_works
    }

    allowed = set(ranking)

    unknown = sorted(
        {
            row.work_id
            for row in draft.matches
            if row.work_id not in allowed
        }
    )

    reason_codes: list[str] = []

    if unknown:
        reason_codes.append(
            "diagnostic_reviewer_unknown_work_id_dropped"
        )

    compiled: list[
        DiagnosticPriorArtMatch
    ] = []

    signals: list[str] = []

    seen_signal_ids: set[str] = set()

    for row in draft.matches:
        if row.work_id not in allowed:
            continue

        work = works.get(
            row.work_id
        )

        ranked = ranking.get(
            row.work_id
        )

        if (
            work is None
            or ranked is None
        ):
            reason_codes.append(
                "diagnostic_candidate_missing_from_packet"
            )
            continue

        relationship = row.relationship

        if relationship in {
            "LOWER_ORDER_RELATION_PRIOR_ART",
            "DIRECTIONAL_COUNTEREVIDENCE",
        }:
            if relationship != expected:
                reason_codes.append(
                    "diagnostic_signal_kind_mismatch_dropped"
                )
                continue

            if not work.abstract:
                relationship = (
                    "TITLE_ONLY_NEIGHBOR"
                )

                reason_codes.append(
                    "diagnostic_strong_match_downgraded_without_abstract"
                )

        match = DiagnosticPriorArtMatch(
            work_id=work.work_id,
            relationship=relationship,
            confidence=row.confidence,
            rationale=row.rationale,
            relevance_score=(
                ranked.relevance_score
            ),
            semantic_similarity=(
                ranked.semantic_similarity
            ),
            lexical_coverage=(
                ranked.lexical_coverage
            ),
            reaction_domain_relevance=(
                ranked.reaction_domain_relevance
            ),
            catalyst_scope_relevance=(
                ranked.catalyst_scope_relevance
            ),
            title=work.title,
            year=work.year,
            doi=work.doi,
            url=work.url,
            abstract_available=bool(
                work.abstract
            ),
        )

        compiled.append(
            match
        )

        if (
            relationship == expected
            and row.confidence
            >= float(
                min_match_confidence
            )
            and work.work_id
            not in seen_signal_ids
        ):
            seen_signal_ids.add(
                work.work_id
            )

            signals.append(
                work.work_id
            )

    if signals:
        reason_codes.append(
            "diagnostic_prior_art_signal_present"
        )
    else:
        reason_codes.append(
            "no_eligible_diagnostic_prior_art_signal"
        )

    compiled.sort(
        key=lambda row: (
            -row.confidence,
            -row.relevance_score,
            row.work_id,
        )
    )

    return DiagnosticClaimPriorArtReview(
        hypothesis_id=claim.hypothesis_id,
        claim_id=claim.claim_id,
        claim_text=claim.text,
        diagnostic_query_kind=(
            claim.diagnostic_query_kind
        ),
        diagnostic_execution_query=(
            execution_query
        ),
        matches=compiled,
        signal_work_ids=signals,
        reason_codes=sorted(
            set(reason_codes)
        ),
        reviewer_unknown_work_ids=(
            unknown
        ),
        interpretation=(
            draft.interpretation
        ),
    )
