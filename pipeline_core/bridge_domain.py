from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


BridgePromptBuilder = Callable[..., str]
BridgeCatalogBuilder = Callable[[Any], list[dict[str, Any]]]
BridgeAnchorContextIssues = Callable[..., list[str]]
BridgeValidationIssues = Callable[..., list[str]]
BridgeValidator = Callable[..., None]
BridgePartitioner = Callable[..., Any]
BridgeImplementationStage = Literal["extraction", "policy"]


@dataclass(frozen=True)
class BridgeImplementationFiles:
    """Domain-owned implementation files that participate in run identity."""

    extraction: tuple[str, ...] = ()
    policy: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for stage, paths in (
            ("extraction", self.extraction),
            ("policy", self.policy),
        ):
            if any(not str(path).strip() for path in paths):
                raise ValueError(
                    f"Bridge {stage} implementation paths must not contain empty values."
                )
            if len(set(map(str, paths))) != len(paths):
                raise ValueError(
                    f"Bridge {stage} implementation paths must be unique."
                )

    def for_stage(
        self,
        stage: BridgeImplementationStage,
    ) -> tuple[str, ...]:
        if stage == "extraction":
            return self.extraction
        if stage == "policy":
            return self.policy
        raise ValueError(f"Unknown Bridge implementation stage: {stage!r}")


@dataclass(frozen=True)
class BridgeSignatureCapabilities:
    """Scientific signature/canonical-anchor capabilities of one domain."""

    catalog_builder: BridgeCatalogBuilder
    anchor_context_issues: BridgeAnchorContextIssues | None = None


@dataclass(frozen=True)
class BridgeValidationCapabilities:
    """Validation callbacks consumed by the shared Bridge runtime."""

    issues: BridgeValidationIssues
    validate: BridgeValidator


@dataclass(frozen=True)
class BridgePolicyCapabilities:
    """Deterministic policy boundary for accepted/candidate/rejected lanes."""

    version: str
    partition: BridgePartitioner


@dataclass(frozen=True)
class BridgeDomainAdapter:
    """Domain-owned semantics used by the shared Bridge runtime.

    alpha4b.2a preserves the mature HER call surface while making the reusable
    capability boundaries explicit. Future domains should provide scientific
    prompts/signatures/policy through this adapter rather than adding domain
    conditionals to the shared runtime.
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

    anchor_context_issues: BridgeAnchorContextIssues | None = None
    implementation_files: BridgeImplementationFiles = field(
        default_factory=BridgeImplementationFiles
    )

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

    @property
    def signatures(self) -> BridgeSignatureCapabilities:
        return BridgeSignatureCapabilities(
            catalog_builder=self.strict_node_catalog_builder,
            anchor_context_issues=self.anchor_context_issues,
        )

    @property
    def validation(self) -> BridgeValidationCapabilities:
        return BridgeValidationCapabilities(
            issues=self.validation_issues,
            validate=self.validate_chunk,
        )

    @property
    def policy(self) -> BridgePolicyCapabilities:
        return BridgePolicyCapabilities(
            version=self.policy_version,
            partition=self.partition_result,
        )
