from __future__ import annotations

from pipeline_core.extraction_domain import ExtractionDomainAdapter
from dac_her.domains.strict_relation_contracts import (
    SERS_AU_AG_STRICT_RELATION_CONSTRAINTS,
)
from domains.sers.prompts import (
    SERS_MICRO_REEXTRACT_SYSTEM_PROMPT,
    SERS_PATCH_SYSTEM_PROMPT,
    SERS_PROMPT_VERSION,
    SERS_SYSTEM_PROMPT,
)


SERS_AU_AG_EXTRACTION_ADAPTER = ExtractionDomainAdapter(
    adapter_id="sers_au_ag",
    domain_profile_id="sers_au_ag",
    prompt_version=SERS_PROMPT_VERSION,
    system_prompt=SERS_SYSTEM_PROMPT,
    patch_system_prompt=SERS_PATCH_SYSTEM_PROMPT,
    micro_reextract_system_prompt=SERS_MICRO_REEXTRACT_SYSTEM_PROMPT,
    default_data_root="data_sers",
    allowed_entity_types=frozenset({
        "Paper", "PlasmonicSubstrate", "Nanostructure", "Metal", "Material",
        "Support", "StructuralMotif", "Morphology", "Analyte", "RamanReporter",
        "OpticalCondition", "SynthesisMethod", "Precursor",
    }),
    strict_relation_constraints=SERS_AU_AG_STRICT_RELATION_CONSTRAINTS,
    allowed_relation_types=frozenset({
        "STUDIES", "HAS_COMPONENT", "HAS_ARCHITECTURE",
        "HAS_STRUCTURAL_MOTIF", "HAS_MORPHOLOGY", "HAS_SUPPORT",
        "PREPARED_BY", "USES_PRECURSOR", "USES_MATERIAL", "TESTED_IN",
        "CHARACTERIZED_IN", "SIMULATED_BY", "USES_ANALYTE", "USES_REPORTER",
        "USES_OPTICAL_CONDITION", "HAS_MEASUREMENT",
        "MEASURED_FOR", "IN_MEASUREMENT_GROUP", "HAS_DESCRIPTOR",
        "SUPPORTS_CLAIM", "INTERPRETED_AS", "PROPOSES_CLAIM", "APPLIES_TO",
        "COMPARED_WITH", "DERIVED_FROM",
    }),
    relation_aliases=(
        ("COMPOSED_OF", "HAS_COMPONENT"),
        ("HAS_ANALYTE", "USES_ANALYTE"),
        ("INVOLVES_ANALYTE", "USES_ANALYTE"),
        ("HAS_REPORTER", "USES_REPORTER"),
        ("INVOLVES_REPORTER", "USES_REPORTER"),
        ("HAS_OPTICAL_CONDITION", "USES_OPTICAL_CONDITION"),
        ("EVALUATED_BY", "TESTED_IN"),
    ),
)
