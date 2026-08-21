from __future__ import annotations

from domains.catalysis_mechanism.prompts import (
    CATALYSIS_MECHANISM_MICRO_REEXTRACT_SYSTEM_PROMPT,
    CATALYSIS_MECHANISM_PATCH_SYSTEM_PROMPT,
    CATALYSIS_MECHANISM_PROMPT_VERSION,
    CATALYSIS_MECHANISM_SYSTEM_PROMPT,
)
from pipeline_core.domain.extraction_domain import ExtractionDomainAdapter
from domains.catalysis_mechanism.prompt_builders import (
    build_domain_gate_recovery_prompt,
    build_extraction_prompt,
    build_micro_reextract_prompt,
    build_patch_rejection_feedback,
    build_semantic_patch_prompt,
)
from domains.catalysis_mechanism.compact_schema import (
    BROAD_COMPACT_SCHEMA_ID,
    BroadMechanismGraphDraft,
)
from domains.catalysis_mechanism.extraction_policy import (
    BROAD_ABSTRACT_RECOVERY_POLICY_ID,
    broad_abstract_extraction_policy,
)
from domains.catalysis_mechanism.vocabulary_context import (
    BROAD_METHODS_ONLY_CONTEXT_ID,
    build_broad_experiment_methods_vocabulary_context,
)
from domains.dac_her.relation_constraints import DAC_LEGACY_STRICT_RELATION_CONSTRAINTS as CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS


CATALYSIS_MECHANISM_EXTRACTION_ADAPTER = ExtractionDomainAdapter(
    adapter_id="catalysis_mechanism",
    domain_profile_id="catalysis_mechanism",
    prompt_version=CATALYSIS_MECHANISM_PROMPT_VERSION,
    system_prompt=CATALYSIS_MECHANISM_SYSTEM_PROMPT,
    patch_system_prompt=CATALYSIS_MECHANISM_PATCH_SYSTEM_PROMPT,
    micro_reextract_system_prompt=(
        CATALYSIS_MECHANISM_MICRO_REEXTRACT_SYSTEM_PROMPT
    ),
    generation_prompt_builder=build_extraction_prompt,
    semantic_patch_prompt_builder=build_semantic_patch_prompt,
    patch_rejection_feedback_builder=build_patch_rejection_feedback,
    micro_reextract_prompt_builder=build_micro_reextract_prompt,
    domain_gate_recovery_prompt_builder=build_domain_gate_recovery_prompt,
    default_data_root="data_broad",
    strict_relation_constraints=(
        CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS
    ),
    compact_generation_response_model=BroadMechanismGraphDraft,
    compact_domain_gate_recovery_response_model=(
        BroadMechanismGraphDraft
    ),
    compact_generation_schema_id=BROAD_COMPACT_SCHEMA_ID,
    compact_domain_gate_recovery_schema_id=BROAD_COMPACT_SCHEMA_ID,
    extraction_policy_id=BROAD_ABSTRACT_RECOVERY_POLICY_ID,
    extraction_policy_transform=broad_abstract_extraction_policy,
    reduced_vocabulary_context_id=BROAD_METHODS_ONLY_CONTEXT_ID,
    reduced_vocabulary_context_builder=(
        build_broad_experiment_methods_vocabulary_context
    ),
    allowed_entity_types=frozenset({
        "Paper",
        "Catalyst",
        "CatalystModel",
        "Metal",
        "Support",
        "CoordinationMotif",
        "Reaction",
        "ReactionStep",
        "Intermediate",
        "Material",
        "SynthesisMethod",
        "Precursor",
        "ActiveSite",
        "StructuralState",
        "AdsorbateState",
        "InterfacialEnvironment",
        "MechanisticFactor",
        "Descriptor",
    }),
    allowed_relation_types=frozenset({
        "STUDIES",
        "HAS_METAL",
        "SUPPORTED_ON",
        "HAS_MOTIF",
        "CATALYZES",
        "MODEL_OF",
        "MODELED_BY",
        "CALCULATES",
        "HAS_ACTIVE_SITE",
        "HAS_STRUCTURAL_STATE",
        "HAS_ADSORBATE_STATE",
        "HAS_ENVIRONMENT",
        "HAS_DESCRIPTOR",
        "INVOLVES_STEP",
        "INVOLVES_INTERMEDIATE",
        "ADSORBS",
        "INDUCES",
        "MODULATES",
        "STABILIZES",
        "DESTABILIZES",
        "PROMOTES",
        "SUPPRESSES",
        "FACILITATES_STEP",
        "INHIBITS_STEP",
        "RECONSTRUCTS_TO",
        "CHANGES_ACTIVE_SITE",
        "CHANGES_RDS",
        "DEPENDS_ON",
        "CORRELATES_WITH",
        "FAILS_WHEN",
        "EVALUATED_IN",
        "CHARACTERIZED_BY",
        "HAS_MEASUREMENT",
        "MEASURED_FOR",
        "IN_MEASUREMENT_GROUP",
        "SUPPORTS_CLAIM",
        "INTERPRETED_AS",
        "PROPOSES_CLAIM",
        "APPLIES_TO",
        "COMPARED_WITH",
        "DERIVED_FROM",
    }),
    # Keep relation aliases empty in v1. A broad cross-domain graph should not
    # silently collapse scientifically distinct mechanism verbs.
    relation_aliases=(),
)
