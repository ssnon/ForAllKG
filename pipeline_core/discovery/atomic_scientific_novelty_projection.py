from __future__ import annotations

from pipeline_core.discovery.atomic_scientific_specification import (
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.external_novelty_contracts import NoveltyClaim


def project_compiled_atomic_specification_to_novelty_claim(
    *,
    hypothesis_id: str,
    claim_rank: int,
    specification: CompiledAtomicSpecification,
) -> NoveltyClaim:
    """Project canonical atomic scientific structure into a novelty/search view.

    CompiledAtomicSpecification remains the owner of scientific source identity,
    including prediction/falsifier IDs and endpoint/scope provenance. This
    projection intentionally carries only the fields NoveltyClaim needs for
    novelty/search processing.

    The function grants no novelty, semantic, or production authority.
    """

    if not str(hypothesis_id).strip():
        raise ValueError("novelty projection requires hypothesis_id")
    if claim_rank < 1:
        raise ValueError("novelty projection requires positive claim_rank")

    return NoveltyClaim(
        claim_id=specification.claim_id,
        hypothesis_id=hypothesis_id,
        claim_rank=claim_rank,
        kind=specification.kind,
        importance=specification.importance,
        novelty_selection_role=specification.novelty_selection_role,
        text=specification.text,
        rationale=specification.rationale,
        search_concepts=list(specification.search_concepts),
        search_queries=list(specification.search_queries),
        distinguishing_terms=list(specification.distinguishing_terms),
        prior_art_identity_terms=list(specification.prior_art_identity_terms),
        relation_nucleus_terms=list(specification.relation_nucleus_terms),
        required_bridge=specification.required_bridge,
        predicted_observation=specification.predicted_observation,
        falsification_condition=specification.falsification_condition,
        scientific_structure=specification.scientific_structure,
        scientific_structure_reason_codes=list(
            specification.scientific_structure_reason_codes
        ),
    )


_SCIENTIFIC_PROJECTION_FIELDS = (
    "claim_id",
    "kind",
    "importance",
    "novelty_selection_role",
    "text",
    "rationale",
    "search_concepts",
    "search_queries",
    "distinguishing_terms",
    "prior_art_identity_terms",
    "relation_nucleus_terms",
    "required_bridge",
    "predicted_observation",
    "falsification_condition",
    "scientific_structure",
    "scientific_structure_reason_codes",
)


def novelty_projection_scientific_mismatch_reason_codes(
    *,
    hypothesis_id: str,
    claim_rank: int,
    specification: CompiledAtomicSpecification,
    claim: NoveltyClaim,
) -> list[str]:
    """Audit one novelty projection against its canonical scientific source."""

    projected = project_compiled_atomic_specification_to_novelty_claim(
        hypothesis_id=hypothesis_id,
        claim_rank=claim_rank,
        specification=specification,
    )
    reasons: list[str] = []

    if claim.hypothesis_id != hypothesis_id:
        reasons.append(
            "canonical_specification_projection_mismatch:hypothesis_id"
        )
    if claim.claim_rank != claim_rank:
        reasons.append(
            "canonical_specification_projection_mismatch:claim_rank"
        )

    for field in _SCIENTIFIC_PROJECTION_FIELDS:
        left = getattr(projected, field)
        right = getattr(claim, field)
        if hasattr(left, "model_dump"):
            left = left.model_dump(mode="json")
        if hasattr(right, "model_dump"):
            right = right.model_dump(mode="json")
        if left != right:
            reasons.append(
                "canonical_specification_projection_mismatch:" + field
            )

    return reasons


__all__ = [
    "novelty_projection_scientific_mismatch_reason_codes",
    "project_compiled_atomic_specification_to_novelty_claim",
]
