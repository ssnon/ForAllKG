from __future__ import annotations

import inspect

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from pipeline_core.corpus.extraction.draft_schema import KnowledgeGraphDraft
from pipeline_core.corpus.graph.graph_domain import RelationConstraint


@dataclass(frozen=True)
class ExtractionDomainAdapter:
    """Domain-owned strict-extraction policy."""

    adapter_id: str
    domain_profile_id: str
    prompt_version: str
    system_prompt: str
    patch_system_prompt: str
    micro_reextract_system_prompt: str
    generation_prompt_builder: Callable[..., str]
    semantic_patch_prompt_builder: Callable[..., str]
    patch_rejection_feedback_builder: Callable[[Exception], str]
    micro_reextract_prompt_builder: Callable[..., str]
    domain_gate_recovery_prompt_builder: Callable[..., str]
    default_data_root: str
    allowed_entity_types: frozenset[str]
    allowed_relation_types: frozenset[str]
    relation_aliases: tuple[tuple[str, str], ...] = ()
    strict_relation_constraints: tuple[RelationConstraint, ...] = ()
    compact_generation_response_model: type[BaseModel] | None = None
    compact_domain_gate_recovery_response_model: (
        type[BaseModel] | None
    ) = None
    compact_generation_schema_id: str | None = None
    compact_domain_gate_recovery_schema_id: str | None = None
    strict_semantic_contract_id: str | None = None
    strict_semantic_contract_rules: tuple[str, ...] = ()
    strict_semantic_issue_collector: Callable[[Any], list[Any]] | None = None

    def prompt_builder_implementation_paths(
        self,
    ) -> tuple[str, ...]:
        """Return deterministic source files for active user-prompt builders."""

        builders = (
            self.generation_prompt_builder,
            self.semantic_patch_prompt_builder,
            self.patch_rejection_feedback_builder,
            self.micro_reextract_prompt_builder,
            self.domain_gate_recovery_prompt_builder,
        )

        paths: list[str] = []

        for builder in builders:
            source_path = inspect.getsourcefile(builder)

            if source_path is None:
                raise RuntimeError(
                    "Could not resolve prompt-builder implementation "
                    f"source for {builder!r}"
                )

            if source_path not in paths:
                paths.append(source_path)

        return tuple(paths)

    def compact_response_model_implementation_paths(
        self,
    ) -> tuple[str, ...]:
        """Return deterministic source files for active compact response models."""

        models = (
            self.compact_generation_response_model,
            self.compact_domain_gate_recovery_response_model,
        )

        paths: list[str] = []

        for model in models:
            if model is None:
                continue

            source_path = inspect.getsourcefile(model)

            if source_path is None:
                raise RuntimeError(
                    "Could not resolve compact response-model "
                    f"implementation source for {model!r}"
                )

            if source_path not in paths:
                paths.append(source_path)

        return tuple(paths)


    def strict_relation_contract_payload(
        self,
    ) -> list[dict[str, Any]]:
        """Return deterministic strict-validation semantics for run provenance."""
        return [
            {
                "relation": constraint.relation,
                "source_types": sorted(
                    constraint.source_types
                ),
                "target_types": sorted(
                    constraint.target_types
                ),
                "severity": constraint.severity,
            }
            for constraint
            in self.strict_relation_constraints
        ]

    def strict_semantic_contract_payload(self) -> dict[str, Any] | None:
        """Return deterministic domain-semantic validation provenance."""
        collector = self.strict_semantic_issue_collector
        if collector is None:
            return None
        if not self.strict_semantic_contract_id:
            raise ValueError(
                "strict semantic collector requires "
                "strict_semantic_contract_id"
            )
        return {
            "contract_id": self.strict_semantic_contract_id,
            "rules": list(self.strict_semantic_contract_rules),
            "collector": (
                f"{collector.__module__}.{collector.__qualname__}"
            ),
        }

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

    def domain_gate_recovery_response_model(
        self,
        *,
        compact: bool = False,
    ) -> type[BaseModel]:
        if not compact:
            return KnowledgeGraphDraft
        if self.compact_domain_gate_recovery_response_model is None:
            raise ValueError(
                "compact domain-gate recovery schema is not configured for "
                f"domain {self.domain_profile_id!r}"
            )
        return self.compact_domain_gate_recovery_response_model

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
