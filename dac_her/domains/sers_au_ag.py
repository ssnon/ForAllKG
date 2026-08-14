from __future__ import annotations

from dac_her.domain_profile import (
    CorpusSemantics,
    DEFAULT_STRONG_CAUSAL_TEXT_PATTERNS,
    DiscoverySemantics,
    NoveltySemantics,
    ProjectionBacktraceRule,
    ProjectionSemantics,
    ResolutionSemantics,
    ScientificDomainProfile,
)


SERS_AU_AG_PROFILE = ScientificDomainProfile(
    profile_id="sers_au_ag",
    description=(
        "Au-Ag bimetallic plasmonic-substrate and surface-enhanced Raman "
        "scattering discovery profile."
    ),
    resolution=ResolutionSemantics(
        resolvable_node_types=frozenset({
            "Paper", "PlasmonicSubstrate", "Nanostructure", "Metal",
            "Material", "Support", "StructuralMotif", "Morphology", "Analyte",
            "RamanReporter", "OpticalCondition", "SynthesisMethod", "Precursor",
            "Experiment", "Calculation",
        }),
        auto_merge_types=frozenset({"Metal"}),
        text_replacements=(
            ("surface-enhanced raman scattering", "sers"),
            ("surface enhanced raman scattering", "sers"),
            ("surface-enhanced raman spectroscopy", "sers"),
            ("surface enhanced raman spectroscopy", "sers"),
            ("core shell", "core-shell"),
            ("nanogaps", "nanogap"),
            ("nanoparticles", "nanoparticle"),
            ("nanocubes", "nanocube"),
            ("nanorods", "nanorod"),
        ),
        reaction_aliases=(),
        nanoparticle_terms=("nanoparticle", "nanostructure"),
        support_signature_tokens=frozenset({
            "silica", "sio2", "glass", "silicon", "film", "substrate",
        }),
        high_priority_review_types=frozenset({
            "PlasmonicSubstrate", "Nanostructure", "Support", "Material",
            "StructuralMotif", "Morphology", "Analyte", "RamanReporter",
        }),
    ),
    discovery=DiscoverySemantics(
        generic_entity_types=frozenset({
            "PLASMONICSUBSTRATE", "NANOSTRUCTURE", "MATERIAL", "SUPPORT",
            "METAL", "ANALYTE", "RAMANREPORTER", "OPTICALCONDITION",
        }),
        mechanism_node_markers=("MECHANISM", "MECHANISTIC"),
        mechanism_relation_markers=(
            "INTERPRETED_AS", "INFLUENC", "MODULAT", "FACILITAT", "PROMOT",
            "REGULAT", "CONTROL", "CORRELAT", "CAUSE", "ENABLE", "ENHANC",
            "LOWER", "STABIL", "TRANSFER", "COUPL", "FOCUS", "TUN",
        ),
        scaffold_relations=frozenset({
            "HAS_COMPONENT", "HAS_ARCHITECTURE", "HAS_STRUCTURAL_MOTIF",
            "HAS_MORPHOLOGY", "HAS_SUPPORT", "PREPARED_BY", "USES_PRECURSOR",
            "USES_MATERIAL", "TESTED_IN", "CHARACTERIZED_IN", "SIMULATED_BY",
            "USES_ANALYTE",
            "USES_REPORTER", "USES_OPTICAL_CONDITION", "APPLIES_TO",
            "ALIGNS_TO_REGISTRY_ENTITY", "HAS_PAPER_MENTION",
        }),
        context_node_types=frozenset({
            "ANALYTE", "RAMANREPORTER", "OPTICALCONDITION",
        }),
        shared_entity_types=frozenset({
            "PLASMONICSUBSTRATE",
            "NANOSTRUCTURE",
            "MATERIAL",
            "SUPPORT",
            "METAL",
            "STRUCTURALMOTIF",
            "MORPHOLOGY",
            "ANALYTE",
            "RAMANREPORTER",
        }),
        legacy_mechanism_id_prefixes=("mech_",),
        strong_causal_text_patterns=(
            DEFAULT_STRONG_CAUSAL_TEXT_PATTERNS
            + (
                r"\bfocus(?:es|ed|ing)?\b",
                r"\bcoupl(?:e|es|ed|ing)\b",
            )
        ),
    ),
    projection=ProjectionSemantics(
        semantics_id="sers_au_ag_projection_v2_alpha4b2c3",
        mechanism_node_types=frozenset({
            "PlasmonicSubstrate", "Nanostructure", "Metal", "Material",
            "Support", "StructuralMotif", "Morphology", "SynthesisMethod",
            "ObservationClaim", "MechanismClaim", "BridgeConcept",
        }),
        origin_node_types=frozenset({
            "PlasmonicSubstrate", "Nanostructure", "Metal", "Material",
            "Support", "StructuralMotif", "Morphology", "SynthesisMethod",
        }),
        backtrace_rules=(
            ProjectionBacktraceRule("HAS_MEASUREMENT", "incoming"),
            ProjectionBacktraceRule("IN_MEASUREMENT_GROUP", "incoming"),
            ProjectionBacktraceRule("TESTED_IN", "incoming"),
            ProjectionBacktraceRule("CHARACTERIZED_IN", "incoming"),
            ProjectionBacktraceRule("SIMULATED_BY", "incoming"),
            ProjectionBacktraceRule("USES_PRECURSOR", "incoming"),
            ProjectionBacktraceRule("MEASURED_FOR", "outgoing"),
            ProjectionBacktraceRule("APPLIES_TO", "outgoing"),
        ),
        max_backtrace_depth=3,
    ),
    corpus=CorpusSemantics(
        semantics_id="sers_au_ag_corpus_v1_alpha4b3a",
        review_candidate_types=frozenset({
            "PlasmonicSubstrate",
            "Nanostructure",
            "Support",
            "Material",
            "StructuralMotif",
            "Morphology",
            "Analyte",
            "RamanReporter",
        }),
        pattern_alignment_mode="confirmed_exact",
    ),
    novelty=NoveltySemantics(
        domain_patterns=(
            ("SERS", (
                r"\bsers\b",
                r"surface[- ]enhanced\s+raman",
                r"surface[- ]enhanced\s+raman\s+(?:scattering|spectroscopy)",
            )),
        ),
        scope_patterns=(
            ("au_ag", (
                r"\bau\s*[-/@]?\s*ag\b",
                r"\bag\s*[-/@]?\s*au\b",
                r"gold.{0,30}silver",
                r"silver.{0,30}gold",
                r"bimetallic.{0,30}(?:au|ag|gold|silver)",
            )),
            ("alloy", (r"\balloy", r"nanoalloy")),
            ("core_shell", (
                r"core[- ]?shell", r"\bau\s*@\s*ag\b", r"\bag\s*@\s*au\b",
            )),
            ("nanogap", (
                r"nanogap", r"nano[- ]?gap", r"interior\s+gap",
                r"interparticle\s+gap",
            )),
            ("surface_composition", (
                r"surface\s+(?:composition|segregation|enrichment)",
                r"surface[- ]segregat",
            )),
            ("lspr", (
                r"\blspr\b", r"localized\s+surface\s+plasmon",
                r"plasmon\s+resonan",
            )),
            ("electromagnetic_enhancement", (
                r"electromagnetic\s+enhancement",
                r"local\s+(?:electric\s+)?field",
                r"near[- ]field", r"hot\s*spot|hotspot",
            )),
            ("chemical_enhancement", (
                r"chemical\s+enhancement", r"chemical\s+mechanism",
            )),
            ("charge_transfer", (
                r"charge\s+transfer", r"charge\s+redistribution",
            )),
            ("reproducibility_stability", (
                r"reproducib", r"uniformity", r"\brsd\b", r"stabil",
            )),
        ),
        critical_scope_features=frozenset({"au_ag"}),
        claim_context_patterns=(
            r"\bsers\b", r"surface[- ]enhanced\s+raman", r"plasmon",
        ),
        document_mismatch_patterns=(
            r"\belectrocatal", r"hydrogen\s+evolution", r"\bher\b",
        ),
        document_compatible_patterns=(
            r"\bsers\b", r"surface[- ]enhanced\s+raman", r"raman",
        ),
        mismatch_multiplier=0.35,
        domain_mismatch_reason="sers_domain_mismatch",
        low_scope_reason="low_sers_system_scope_overlap",
    ),
    extraction_adapter_id="sers_au_ag",
    graph_adapter_id="sers_au_ag",
    bridge_adapter_id="sers_au_ag",
    comparison_adapter_id="sers_au_ag",
    reproducibility_adapter_id="sers_au_ag",
    metric_definition_adapter_id="sers_au_ag",
    trend_adapter_id="sers_au_ag",
    feasibility_adapter_id=None,
)
