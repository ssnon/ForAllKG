from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from dac_her.draft_schema import KnowledgeGraphDraft
from dac_her.graph_domain import RelationConstraint


@dataclass(frozen=True)
class ExtractionDomainAdapter:
    """Domain-owned strict-extraction policy."""

    adapter_id: str
    domain_profile_id: str
    prompt_version: str
    system_prompt: str
    patch_system_prompt: str
    micro_reextract_system_prompt: str
    default_data_root: str
    allowed_entity_types: frozenset[str]
    allowed_relation_types: frozenset[str]
    relation_aliases: tuple[tuple[str, str], ...] = ()
    strict_relation_constraints: tuple[RelationConstraint, ...] = ()
    compact_generation_response_model: type[BaseModel] | None = None

    def canonical_relation(self, relation: str) -> str:
        aliases = dict(self.relation_aliases)
        return aliases.get(str(relation), str(relation))

    def generation_response_model(
        self,
        *,
        compact: bool = False,
    ) -> type[BaseModel]:
        if not compact:
            return KnowledgeGraphDraft
        if self.compact_generation_response_model is None:
            raise ValueError(
                "compact generation schema is not configured for "
                f"domain {self.domain_profile_id!r}"
            )
        return self.compact_generation_response_model

    def canonicalize_generation_output(
        self,
        generated: BaseModel,
    ) -> KnowledgeGraphDraft:
        if isinstance(generated, KnowledgeGraphDraft):
            return generated
        converter = getattr(
            generated,
            "to_knowledge_graph_draft",
            None,
        )
        if callable(converter):
            canonical = converter()
            if isinstance(canonical, KnowledgeGraphDraft):
                return canonical
        raise TypeError(
            "generation response cannot be converted to "
            f"KnowledgeGraphDraft for domain {self.domain_profile_id!r}: "
            f"{type(generated).__name__}"
        )

    def normalize_draft_vocabulary(self, draft: Any) -> Any:
        """Apply only explicitly declared, semantics-preserving aliases."""
        for edge in getattr(draft, "edges", []):
            current = str(edge.relation)
            canonical = self.canonical_relation(current)
            if canonical == current:
                continue
            try:
                edge.relation = canonical
            except Exception:
                object.__setattr__(edge, "relation", canonical)
        return draft

    def validate_draft_vocabulary(self, draft: Any) -> None:
        invalid_entities = sorted({
            str(node.type)
            for node in getattr(draft, "entities", [])
            if str(node.type) not in self.allowed_entity_types
        })
        invalid_relations = sorted({
            str(edge.relation)
            for edge in getattr(draft, "edges", [])
            if str(edge.relation) not in self.allowed_relation_types
        })
        problems: list[str] = []
        if invalid_entities:
            reserved = {
                "Experiment",
                "Calculation",
                "Measurement",
                "MeasurementGroup",
                "ObservationClaim",
                "MechanismClaim",
            }
            misplaced = sorted(set(invalid_entities) & reserved)
            message = (
                "entity types outside domain vocabulary: "
                + ", ".join(invalid_entities)
            )
            if misplaced:
                message += (
                    "; reserved structured node type(s) "
                    + ", ".join(misplaced)
                    + " must use their dedicated top-level collection, "
                    "not entities[]"
                )
            problems.append(message)
        if invalid_relations:
            problems.append(
                "relations outside domain vocabulary: "
                + ", ".join(invalid_relations)
            )
        if problems:
            raise ValueError(
                f"Extraction-domain vocabulary violation "
                f"[{self.domain_profile_id}]: "
                + "; ".join(problems)
            )
