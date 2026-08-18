from __future__ import annotations

from pipeline_core.domain_profile import (
    CorpusSemantics,
    DiscoverySemantics,
    NoveltySemantics,
    ProjectionBacktraceRule,
    ProjectionSemantics,
    ResolutionSemantics,
    ScientificDomainProfile,
)


DAC_HER_PROFILE = ScientificDomainProfile(
    profile_id='dac_her',
    description=(
        'Dual-/single-atomic-site electrocatalysis profile used by the existing '
        'GraphAgentsDAC HER pipeline.'
    ),
    resolution=ResolutionSemantics(
        resolvable_node_types=frozenset({
            'Paper', 'Catalyst', 'CatalystModel', 'Metal', 'Support',
            'CoordinationMotif', 'SynthesisMethod', 'Precursor', 'Reaction',
            'ReactionStep', 'Intermediate', 'Material', 'Experiment', 'Calculation',
        }),
        auto_merge_types=frozenset({'Metal', 'Reaction'}),
        text_replacements=(
            ('hydrogen evolution reaction', 'her'),
            ('oxygen evolution reaction', 'oer'),
            ('oxygen reduction reaction', 'orr'),
            ('nitrogen doped', 'n doped'),
            ('n-doped', 'n doped'),
            ('dual atoms', 'dual atom'),
            ('single atoms', 'single atom'),
            ('dimers', 'dimer'),
            ('nanotubes', 'nanotube'),
        ),
        reaction_aliases=(
            ('her', 'hydrogen_evolution_reaction'),
            ('hydrogen evolution', 'hydrogen_evolution_reaction'),
            ('hydrogen evolution reaction', 'hydrogen_evolution_reaction'),
            ('oer', 'oxygen_evolution_reaction'),
            ('oxygen evolution reaction', 'oxygen_evolution_reaction'),
            ('orr', 'oxygen_reduction_reaction'),
            ('oxygen reduction reaction', 'oxygen_reduction_reaction'),
            ('co2rr', 'carbon_dioxide_reduction_reaction'),
            ('co2 reduction reaction', 'carbon_dioxide_reduction_reaction'),
            ('nrr', 'nitrogen_reduction_reaction'),
            ('nitrogen reduction reaction', 'nitrogen_reduction_reaction'),
        ),
        support_signature_tokens=frozenset({
            'graphene', 'nanotube', 'ncnt', 'carbon', 'nitrogen', 'doped', 'ng',
        }),
        high_priority_review_types=frozenset({
            'Catalyst', 'CatalystModel', 'Support', 'Material',
        }),
    ),
    discovery=DiscoverySemantics(
        generic_entity_types=frozenset({
            'CATALYST', 'CATALYSTMODEL', 'MATERIAL', 'SUPPORT', 'METAL', 'REACTION',
        }),
        mechanism_node_markers=('MECHANISM', 'MECHANISTIC'),
        mechanism_relation_markers=(
            'MECHANISM', 'INTERPRETED_AS', 'INFLUENC', 'MODULAT', 'FACILITAT',
            'PROMOT', 'REGULAT', 'CONTROL', 'CORRELAT', 'CAUSE', 'ENABLE',
            'ENHANC', 'LOWER', 'STABIL', 'TRANSFER', 'SPILLOVER', 'TUN',
        ),
        scaffold_relations=frozenset({
            'APPLIES_TO', 'SUPPORTED_ON', 'CATALYZES', 'ALIGNS_TO_REGISTRY_ENTITY',
            'HAS_PAPER_MENTION', 'INVOLVES_INTERMEDIATE', 'PART_OF', 'HAS_COMPONENT',
            'HAS_SUPPORT', 'HAS_METAL', 'HAS_MOTIF', 'MODEL_OF',
        }),
        context_node_types=frozenset({'REACTION'}),
        shared_entity_types=frozenset({
            'CATALYST', 'CATALYSTMODEL', 'MATERIAL', 'SUPPORT',
            'METAL', 'COORDINATIONMOTIF',
        }),
        legacy_mechanism_id_prefixes=('mech_',),
    ),
    projection=ProjectionSemantics(
        semantics_id="dac_her_projection_v1_alpha4b2c",
        mechanism_node_types=frozenset({
            "Catalyst", "CatalystModel", "Metal", "Support",
            "CoordinationMotif", "SynthesisMethod", "Precursor",
            "Reaction", "ReactionStep", "Intermediate", "Material",
            "ObservationClaim", "MechanismClaim", "BridgeConcept",
        }),
        origin_node_types=frozenset({
            "Catalyst", "CatalystModel", "Support", "CoordinationMotif",
            "Material", "Reaction", "ReactionStep", "Intermediate",
        }),
        backtrace_rules=(
            ProjectionBacktraceRule("HAS_MEASUREMENT", "incoming"),
            ProjectionBacktraceRule("EVALUATED_IN", "incoming"),
            ProjectionBacktraceRule("CHARACTERIZED_BY", "incoming"),
            ProjectionBacktraceRule("MODELED_BY", "incoming"),
            ProjectionBacktraceRule("APPLIES_TO", "incoming"),
            ProjectionBacktraceRule("SUPPORTS_CLAIM", "incoming"),
        ),
        max_backtrace_depth=3,
    ),
    corpus=CorpusSemantics(
        semantics_id="dac_her_corpus_v1_alpha4b3a",
        review_candidate_types=frozenset({
            "Catalyst",
            "CatalystModel",
            "Support",
            "CoordinationMotif",
            "Material",
            "Intermediate",
        }),
        pattern_alignment_mode="confirmed_exact",
    ),
    novelty=NoveltySemantics(
        domain_patterns=(
            ('HER', (r'\bher\b', r'hydrogen\s+evolution')),
            ('OER', (r'\boer\b', r'oxygen\s+evolution')),
            ('ORR', (r'\borr\b', r'oxygen\s+reduction')),
            ('CO2RR', (
                r'\bco2rr\b', r'co2\s+(?:electro)?reduction',
                r'carbon\s+dioxide\s+(?:electro)?reduction',
            )),
            ('NRR', (r'\bnrr\b', r'nitrogen\s+reduction', r'nitrogen\s+fixation')),
        ),
        scope_patterns=(
            ('nitrogen_coordination', (
                r'nitrogen.{0,35}coordin', r'coordin.{0,35}nitrogen',
                r'\bmn\s*4\b', r'\bm[\-–—]?n\s*4\b', r'\bn\s*[23456]\b',
            )),
            ('dual_atom', (
                r'dual[ -]?atom', r'diatomic', r'\bdimer', r'metal[ -]?pair',
                r'paired\s+metal',
            )),
            ('d_band', (r'd[ -]?band',)),
            ('hydrogen_adsorption', (
                r'hydrogen\s+(?:adsorption|binding)', r'\bdelta\s*g.{0,8}h\b',
            )),
            ('exchange_current', (r'exchange\s+current',)),
        ),
        critical_scope_features=frozenset({'nitrogen_coordination', 'dual_atom'}),
        claim_context_patterns=(r'electrocatal', r'exchange\s+current', r'\bher\b'),
        document_mismatch_patterns=(r'photocatal', r'photoelectro'),
        document_compatible_patterns=(r'electrocatal', r'exchange\s+current'),
        mismatch_multiplier=0.55,
        domain_mismatch_reason='reaction_domain_mismatch',
        low_scope_reason='low_catalyst_scope_overlap',
        targeted_query_templates=(
            "{core}",
            "{core} hydrogen evolution reaction mechanism",
            "{core} dual atom catalyst nitrogen coordination",
        ),
        contextual_conflict_query_templates=(
            "{core} nitrogen coordinated dual atom HER",
        ),
    ),
    extraction_adapter_id='dac_her',
    graph_adapter_id='dac_her',
    bridge_adapter_id='dac_her',
    feasibility_adapter_id='dac_her',
)
