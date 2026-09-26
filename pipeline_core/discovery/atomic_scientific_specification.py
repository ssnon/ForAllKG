from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AtomicClaimKind = Literal[
    "mediator",
    "moderator_interaction",
    "context_condition",
    "pathway_competition",
    "descriptor_interaction",
    "distinctive_prediction",
    "mechanistic_link",
]

AtomicSelectionRole = Literal[
    "NOVELTY_BEARING",
    "REQUIRED_ENABLING_RELATION",
    "TESTING_PREDICTION",
    "AUXILIARY",
]


class CompiledAtomicSpecification(StrictModel):
    """Neutral compiled scientific specification shared across discovery lanes.

    This contract owns stable atomic scientific structure and source identity.
    It does not itself grant novelty, semantic-admissibility, or production
    authority.
    """

    local_id: str
    claim_id: str
    kind: AtomicClaimKind
    importance: Literal["core", "supporting"]
    novelty_selection_role: AtomicSelectionRole
    text: str
    rationale: str

    source_candidate_ids: list[str]
    premise_statement_ids: list[str]
    gap_statement_ids: list[str]

    prior_art_identity_terms: list[str]
    relation_endpoint_anchors: list[str]
    scope_qualifier_spans: list[str]
    directional_qualifier_spans: list[str]
    relation_nucleus_terms: list[str]
    distinguishing_terms: list[str]

    required_bridge: str
    observable: str
    predicted_observation: str
    falsification_condition: str
    prediction_observation_id: str
    falsification_criterion_id: str

    search_concepts: list[str]
    search_queries: list[str]
    scientific_structure: NoveltyClaimScientificStructure
    scientific_structure_reason_codes: list[str]


__all__ = [
    "AtomicClaimKind",
    "AtomicSelectionRole",
    "CompiledAtomicSpecification",
]
