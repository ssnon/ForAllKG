from __future__ import annotations

from domains.catalysis_mechanism.prompts import (
    CATALYSIS_MECHANISM_MICRO_REEXTRACT_SYSTEM_PROMPT,
    CATALYSIS_MECHANISM_PATCH_SYSTEM_PROMPT,
    CATALYSIS_MECHANISM_PROMPT_VERSION,
    CATALYSIS_MECHANISM_SYSTEM_PROMPT,
)
from pipeline_core.domain.extraction_domain import ExtractionDomainAdapter
from domains.extraction_prompt_compat import (
    LEGACY_DOMAIN_GATE_RECOVERY_PROMPT_BUILDER,
    LEGACY_GENERATION_PROMPT_BUILDER,
    LEGACY_MICRO_REEXTRACT_PROMPT_BUILDER,
    LEGACY_PATCH_REJECTION_FEEDBACK_BUILDER,
    LEGACY_SEMANTIC_PATCH_PROMPT_BUILDER,
)
from pipeline_core.corpus.broad_compact_schema import BroadMechanismGraphDraft
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
    generation_prompt_builder=LEGACY_GENERATION_PROMPT_BUILDER,
    semantic_patch_prompt_builder=LEGACY_SEMANTIC_PATCH_PROMPT_BUILDER,
    patch_rejection_feedback_builder=LEGACY_PATCH_REJECTION_FEEDBACK_BUILDER,
    micro_reextract_prompt_builder=LEGACY_MICRO_REEXTRACT_PROMPT_BUILDER,
    domain_gate_recovery_prompt_builder=LEGACY_DOMAIN_GATE_RECOVERY_PROMPT_BUILDER,
    default_data_root="data_broad",
    strict_relation_constraints=(
        CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS
    ),
    compact_generation_response_model=BroadMechanismGraphDraft,
    compact_domain_gate_recovery_response_model=(
        BroadMechanismGraphDraft
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
