from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


BridgePromptBuilder = Callable[..., str]
BridgeCatalogBuilder = Callable[[Any], list[dict[str, Any]]]
BridgeValidationIssues = Callable[..., list[str]]
BridgeValidator = Callable[..., None]
BridgePartitioner = Callable[..., Any]


@dataclass(frozen=True)
class BridgeDomainAdapter:
    """Domain-owned semantics used by the shared Bridge runtime.

    alpha4b.1 deliberately keeps the mature HER implementations intact and
    routes them through this adapter boundary. SERS and broad-catalysis remain
    fail-closed until dedicated adapters are registered in later phases.
    """

    adapter_id: str
    domain_profile_id: str

    prompt_version: str
    system_prompt: str
    build_prompt: BridgePromptBuilder

    recovery_prompt_version: str
    recovery_system_prompt: str
    build_candidate_repair_prompt: BridgePromptBuilder

    policy_version: str
    strict_node_catalog_builder: BridgeCatalogBuilder
    validation_issues: BridgeValidationIssues
    validate_chunk: BridgeValidator
    partition_result: BridgePartitioner

    def __post_init__(self) -> None:
        adapter_id = self.adapter_id.strip().lower()
        domain_id = self.domain_profile_id.strip().lower()
        if not adapter_id:
            raise ValueError("Bridge adapter_id must not be empty.")
        if not domain_id:
            raise ValueError("Bridge domain_profile_id must not be empty.")
        if not self.prompt_version.strip():
            raise ValueError("Bridge prompt_version must not be empty.")
        if not self.recovery_prompt_version.strip():
            raise ValueError("Bridge recovery_prompt_version must not be empty.")
        if not self.policy_version.strip():
            raise ValueError("Bridge policy_version must not be empty.")
