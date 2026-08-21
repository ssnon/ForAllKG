"""Temporary H1a compatibility wiring for extraction user prompts.

Before H1a, the shared strict extraction runtime directly invoked DAC-HER
user-prompt builders for every extraction domain. H1a moves those callables
behind ExtractionDomainAdapter while preserving the exact historical callable
identity for SERS and broad catalysis.

This module is intentionally transitional. H1b replaces SERS/Broad compatibility
wiring with domain-native user-prompt builders.
"""

from __future__ import annotations

from domains.dac_her.micro_reextract_prompts import (
    build_domain_gate_recovery_prompt as LEGACY_DOMAIN_GATE_RECOVERY_PROMPT_BUILDER,
)
from domains.dac_her.micro_reextract_prompts import (
    build_micro_reextract_prompt as LEGACY_MICRO_REEXTRACT_PROMPT_BUILDER,
)
from domains.dac_her.prompts import (
    build_extraction_prompt as LEGACY_GENERATION_PROMPT_BUILDER,
)
from domains.dac_her.semantic_patch_prompts import (
    build_patch_rejection_feedback as LEGACY_PATCH_REJECTION_FEEDBACK_BUILDER,
)
from domains.dac_her.semantic_patch_prompts import (
    build_semantic_patch_prompt as LEGACY_SEMANTIC_PATCH_PROMPT_BUILDER,
)


__all__ = (
    "LEGACY_GENERATION_PROMPT_BUILDER",
    "LEGACY_SEMANTIC_PATCH_PROMPT_BUILDER",
    "LEGACY_PATCH_REJECTION_FEEDBACK_BUILDER",
    "LEGACY_MICRO_REEXTRACT_PROMPT_BUILDER",
    "LEGACY_DOMAIN_GATE_RECOVERY_PROMPT_BUILDER",
)
