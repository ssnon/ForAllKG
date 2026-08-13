from __future__ import annotations

from dac_her.catalysis_mechanism_prompts import (
    CATALYSIS_MECHANISM_MICRO_REEXTRACT_SYSTEM_PROMPT,
    CATALYSIS_MECHANISM_PATCH_SYSTEM_PROMPT,
    CATALYSIS_MECHANISM_PROMPT_VERSION,
    CATALYSIS_MECHANISM_SYSTEM_PROMPT,
)
from dac_her.extraction_domain import ExtractionDomainAdapter
from dac_her.broad_compact_schema import BroadMechanismGraphDraft
from dac_her.domains.strict_relation_contracts import (
    CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS,
)


CATALYSIS_MECHANISM_EXTRACTION_ADAPTER = ExtractionDomainAdapter(
    adapter_id="catalysis_mechanism",
    domain_profile_id="catalysis_mechanism",
    prompt_version=CATALYSIS_MECHANISM_PROMPT_VERSION,
    system_prompt=CATALYSIS_MECHANISM_SYSTEM_PROMPT,
    patch_system_prompt=CATALYSIS_MECHANISM_PATCH_SYSTEM_PROMPT,
    micro_reextract_system_prompt=(
        CATALYSIS_MECHANISM_MICRO_REEXTRACT_SYSTEM_PROMPT
    ),
    default_data_root="data_broad",
    strict_relation_constraints=(
        CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS
    ),
    compact_generation_response_model=BroadMechanismGraphDraft,
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
