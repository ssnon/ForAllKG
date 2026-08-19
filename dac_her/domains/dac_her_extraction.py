from __future__ import annotations

from dac_her.extraction_domain import ExtractionDomainAdapter
from domains.dac_her.relation_constraints import (
    DAC_HER_STRICT_RELATION_CONSTRAINTS,
)
from dac_her.micro_reextract_prompts import MICRO_REEXTRACT_SYSTEM_PROMPT
from dac_her.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from dac_her.semantic_patch_prompts import PATCH_SYSTEM_PROMPT


DAC_HER_EXTRACTION_ADAPTER = ExtractionDomainAdapter(
    adapter_id="dac_her",
    domain_profile_id="dac_her",
    prompt_version=PROMPT_VERSION,
    system_prompt=SYSTEM_PROMPT,
    patch_system_prompt=PATCH_SYSTEM_PROMPT,
    micro_reextract_system_prompt=MICRO_REEXTRACT_SYSTEM_PROMPT,
    default_data_root="data_dac",
    allowed_entity_types=frozenset({
        "Paper", "Catalyst", "CatalystModel", "Metal", "Support",
        "CoordinationMotif", "SynthesisMethod", "Precursor", "Reaction",
        "ReactionStep", "Intermediate", "Material",
    }),
    strict_relation_constraints=DAC_HER_STRICT_RELATION_CONSTRAINTS,
    allowed_relation_types=frozenset({
        "STUDIES", "HAS_METAL", "SUPPORTED_ON", "HAS_MOTIF",
        "SYNTHESIZED_BY", "USES_PRECURSOR", "CATALYZES", "EVALUATED_IN",
        "CHARACTERIZED_BY", "MODELED_BY", "HAS_MEASUREMENT", "MEASURED_FOR",
        "IN_MEASUREMENT_GROUP", "MODEL_OF", "HAS_DESCRIPTOR", "CALCULATES",
        "SUPPORTS_CLAIM", "INTERPRETED_AS", "PROPOSES_CLAIM", "APPLIES_TO",
        "INVOLVES_STEP", "INVOLVES_INTERMEDIATE", "ADSORBS",
        "FACILITATES_STEP", "COMPARED_WITH", "DERIVED_FROM",
    }),
)
