from __future__ import annotations

from pipeline_core.domain.extraction_domain import ExtractionDomainAdapter
from domains.sers.prompt_builders import (
    build_domain_gate_recovery_prompt,
    build_extraction_prompt,
    build_micro_reextract_prompt,
    build_patch_rejection_feedback,
    build_semantic_patch_prompt,
)
from domains.sers.extraction_semantics import (
    SERS_STRICT_SEMANTIC_CONTRACT_ID,
    SERS_STRICT_SEMANTIC_CONTRACT_RULES,
    collect_sers_strict_semantic_issues,
)
from pipeline_core.corpus.extraction.evidence_relation_constraints import (
    COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS,
)
from pipeline_core.corpus.graph.graph_domain import RelationConstraint
from domains.sers.prompts import (
    SERS_MICRO_REEXTRACT_SYSTEM_PROMPT,
    SERS_PATCH_SYSTEM_PROMPT,
    SERS_PROMPT_VERSION,
    SERS_SYSTEM_PROMPT,
)


# Extraction-hard SERS invariant.
#
# Keep this deliberately narrower than the paper-graph diagnostic contract.
# C24 frozen-corpus replay showed that USES_PRECURSOR alone reproduces the
# historical 10-paper / 15-chunk failure boundary without the false-positive
# blast radius caused by promoting the full graph relation contract.
SERS_AU_AG_STRICT_RELATION_CONSTRAINTS = (
    *COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS,
    RelationConstraint(
        "USES_PRECURSOR",
        source_types=frozenset({"SynthesisMethod"}),
        target_types=frozenset({"Precursor"}),
    ),
)


SERS_AU_AG_EXTRACTION_ADAPTER = ExtractionDomainAdapter(
    adapter_id="sers_au_ag",
    domain_profile_id="sers_au_ag",
    prompt_version=SERS_PROMPT_VERSION,
    system_prompt=SERS_SYSTEM_PROMPT,
    patch_system_prompt=SERS_PATCH_SYSTEM_PROMPT,
    micro_reextract_system_prompt=SERS_MICRO_REEXTRACT_SYSTEM_PROMPT,
    generation_prompt_builder=build_extraction_prompt,
    semantic_patch_prompt_builder=build_semantic_patch_prompt,
    patch_rejection_feedback_builder=build_patch_rejection_feedback,
    micro_reextract_prompt_builder=build_micro_reextract_prompt,
    domain_gate_recovery_prompt_builder=build_domain_gate_recovery_prompt,
    default_data_root="data_sers",
    allowed_entity_types=frozenset({
        "Paper", "PlasmonicSubstrate", "Nanostructure", "Metal", "Material",
        "Support", "StructuralMotif", "Morphology", "Analyte", "RamanReporter",
        "OpticalCondition", "SynthesisMethod", "Precursor",
    }),
    strict_relation_constraints=SERS_AU_AG_STRICT_RELATION_CONSTRAINTS,
    strict_semantic_contract_id=SERS_STRICT_SEMANTIC_CONTRACT_ID,
    strict_semantic_contract_rules=SERS_STRICT_SEMANTIC_CONTRACT_RULES,
    strict_semantic_issue_collector=collect_sers_strict_semantic_issues,
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
