from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
            problems.append(
                "entity types outside domain vocabulary: "
                + ", ".join(invalid_entities)
            )
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
