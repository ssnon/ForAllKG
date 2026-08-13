from dac_her.bridge_schemas import BridgeConcept
from dac_her.sers_bridge_policy import (
    SERS_BRIDGE_POLICY_VERSION,
    concept_policy_issues,
)


def _pattern(**updates):
    data = dict(
        id='b1',
        concept_type='RelationPattern',
        label='SERS activity varies with nanogap size',
        source_phrase='SERS activity increased as the nanogap size decreased.',
        description=None,
        retention_lane='accepted_pattern',
        evidence_scope='paper_result',
        pattern_subject='SERS activity',
        pattern_relation='VARIES_WITH',
        pattern_object='nanogap size',
        relation_strength='correlational',
        qualifiers=[],
        pattern_support_mode='explicit_single_span',
        supporting_phrases=[
            'SERS activity increased as the nanogap size decreased.'
        ],
        subject_evidence_phrase='SERS activity',
        relation_evidence_phrase='increased as',
        object_evidence_phrase='nanogap size',
        comparison_items=[],
    )
    data.update(updates)
    return BridgeConcept(**data)


def _frontier(**updates):
    data = dict(
        id='f1',
        concept_type='Phenomenon',
        label='plasmon hybridization',
        source_phrase='The coupled particles exhibited plasmon hybridization.',
        description=None,
        retention_lane='paper_local_frontier',
        evidence_scope='author_interpretation',
        pattern_subject=None,
        pattern_relation=None,
        pattern_object=None,
        relation_strength=None,
        qualifiers=[],
        pattern_support_mode=None,
        supporting_phrases=[],
        subject_evidence_phrase=None,
        relation_evidence_phrase=None,
        object_evidence_phrase=None,
        comparison_items=[],
    )
    data.update(updates)
    return BridgeConcept(**data)


def _codes(concept):
    return {
        issue.code
        for issue in concept_policy_issues(
            concept,
            strict_nodes=[],
            core_text=concept.source_phrase,
            linked_links=[],
        )
    }


def test_alpha4b2b2_policy_version():
    assert SERS_BRIDGE_POLICY_VERSION == (
        'sers-au-ag-bridge-policy-v1-alpha4b2b2'
    )


def test_alpha4b2b2_explicit_control_and_linear_relation_cues_are_supported():
    controlled = _pattern(
        pattern_subject='Ag shell thickness',
        pattern_object='Ag source amount',
        source_phrase=(
            'By adjusting the Ag source amount, the growth of Ag shell '
            'thickness can be reliably controlled.'
        ),
        supporting_phrases=[
            'By adjusting the Ag source amount, the growth of Ag shell '
            'thickness can be reliably controlled.'
        ],
        subject_evidence_phrase='Ag shell thickness',
        relation_evidence_phrase='can be reliably controlled',
        object_evidence_phrase='Ag source amount',
    )
    proportional = _pattern(
        pattern_subject='SERS signal intensity',
        pattern_object='AgNO3 concentration',
        source_phrase=(
            'Signal intensity was proportional to AgNO3 concentration.'
        ),
        supporting_phrases=[
            'Signal intensity was proportional to AgNO3 concentration.'
        ],
        subject_evidence_phrase='Signal intensity',
        relation_evidence_phrase='was proportional to',
        object_evidence_phrase='AgNO3 concentration',
    )
    assert 'RELATION_CUE_MISMATCH' not in _codes(controlled)
    assert 'RELATION_CUE_MISMATCH' not in _codes(proportional)


def test_alpha4b2b2_protect_and_design_rule_cues_are_supported():
    protects = _pattern(
        pattern_subject='gold layer coating on silver',
        pattern_relation='SUPPRESSES',
        pattern_object='oxidation of inner silver atoms',
        relation_strength='causal_interpretive',
        source_phrase=(
            'A gold layer coating silver was created to protect inner '
            'silver atoms from being oxidized.'
        ),
        supporting_phrases=[
            'A gold layer coating silver was created to protect inner '
            'silver atoms from being oxidized.'
        ],
        subject_evidence_phrase='gold layer coating silver',
        relation_evidence_phrase='to protect',
        object_evidence_phrase='inner silver atoms from being oxidized',
    )
    design = _pattern(
        pattern_subject='Ag-Au structure stability',
        pattern_relation='SUGGESTS_DESIGN_RULE',
        pattern_object='limiting surface oxidation',
        relation_strength='causal_interpretive',
        source_phrase=(
            'The stability of Ag-Au structure can be improved by limiting '
            'its surface oxidation.'
        ),
        supporting_phrases=[
            'The stability of Ag-Au structure can be improved by limiting '
            'its surface oxidation.'
        ],
        subject_evidence_phrase='The stability of Ag-Au structure',
        relation_evidence_phrase='can be improved by',
        object_evidence_phrase='limiting its surface oxidation',
    )
    assert 'RELATION_CUE_MISMATCH' not in _codes(protects)
    assert 'RELATION_CUE_MISMATCH' not in _codes(design)


def test_alpha4b2b2_direct_analyte_calibration_is_not_a_discovery_bridge():
    concept = _pattern(
        pattern_subject='SERS signal intensity',
        pattern_object='methylene-blue concentration',
        source_phrase=(
            'A linear dependence between methylene-blue concentration and '
            'SERS signal intensity was observed.'
        ),
        supporting_phrases=[
            'A linear dependence between methylene-blue concentration and '
            'SERS signal intensity was observed.'
        ],
        subject_evidence_phrase='SERS signal intensity',
        relation_evidence_phrase='linear dependence between',
        object_evidence_phrase='methylene-blue concentration',
    )
    assert 'ANALYTICAL_CALIBRATION_PATTERN' in _codes(concept)


def test_alpha4b2b2_particle_concentration_is_not_mistaken_for_analyte_calibration():
    concept = _pattern(
        pattern_subject='SERS signal intensity',
        pattern_object='particle concentration',
        source_phrase=(
            'SERS signal intensity increased as particle concentration increased.'
        ),
        supporting_phrases=[
            'SERS signal intensity increased as particle concentration increased.'
        ],
        subject_evidence_phrase='SERS signal intensity',
        relation_evidence_phrase='increased as',
        object_evidence_phrase='particle concentration',
    )
    assert 'ANALYTICAL_CALIBRATION_PATTERN' not in _codes(concept)


def test_alpha4b2b2_passive_causal_reversal_is_review_candidate():
    generated_by = _pattern(
        pattern_subject='strong electromagnetic field',
        pattern_relation='PROMOTES',
        pattern_object='strong plasmonic coupling',
        relation_strength='causal_interpretive',
        source_phrase=(
            'The strong electromagnetic field generated in the nanogap by '
            'strong plasmonic coupling was calculated.'
        ),
        supporting_phrases=[
            'The strong electromagnetic field generated in the nanogap by '
            'strong plasmonic coupling was calculated.'
        ],
        subject_evidence_phrase='strong electromagnetic field',
        relation_evidence_phrase='generated in the nanogap by',
        object_evidence_phrase='strong plasmonic coupling',
    )
    restricted_by = _pattern(
        pattern_subject='SERS activity',
        pattern_relation='SUPPRESSES',
        pattern_object='outer Au shell layer',
        relation_strength='causal_interpretive',
        source_phrase=(
            'SERS activity in the core/shell structure might be restricted '
            'by the outer Au shell layer.'
        ),
        supporting_phrases=[
            'SERS activity in the core/shell structure might be restricted '
            'by the outer Au shell layer.'
        ],
        subject_evidence_phrase='SERS activity in the core/shell structure',
        relation_evidence_phrase='might be restricted by',
        object_evidence_phrase='outer Au shell layer',
    )
    assert 'CAUSAL_ARGUMENT_DIRECTION' in _codes(generated_by)
    assert 'CAUSAL_ARGUMENT_DIRECTION' in _codes(restricted_by)


def test_alpha4b2b2_correct_passive_cause_order_is_not_reversed():
    concept = _pattern(
        pattern_subject='silver oxidation',
        pattern_relation='SUPPRESSES',
        pattern_object='SERS performance',
        relation_strength='causal_interpretive',
        source_phrase=(
            'The SERS performance deteriorates slightly owing to the '
            'oxidation of silver.'
        ),
        supporting_phrases=[
            'The SERS performance deteriorates slightly owing to the '
            'oxidation of silver.'
        ],
        subject_evidence_phrase='oxidation of silver',
        relation_evidence_phrase='owing to',
        object_evidence_phrase='SERS performance deteriorates slightly',
    )
    assert 'CAUSAL_ARGUMENT_DIRECTION' not in _codes(concept)
    assert 'RELATION_CUE_MISMATCH' not in _codes(concept)


def test_alpha4b2b2_property_vs_axis_contrast_requires_review_but_peers_do_not():
    bad = _pattern(
        pattern_subject='electromagnetic enhancement',
        pattern_relation='CONTRASTS_WITH',
        pattern_object='metal identity',
        relation_strength='correlational',
        source_phrase='Ag and Au electromagnetic enhancement were contrasted.',
        supporting_phrases=[
            'Ag and Au electromagnetic enhancement were contrasted.'
        ],
        subject_evidence_phrase='electromagnetic enhancement',
        relation_evidence_phrase='contrasted',
        object_evidence_phrase='Ag and Au',
    )
    peer = _pattern(
        pattern_subject='Raman signal for an Ag-Au alloy architecture',
        pattern_relation='CONTRASTS_WITH',
        pattern_object='Raman signal for an Ag-Au core-shell architecture',
        relation_strength='correlational',
        source_phrase=(
            'The alloy sample showed one Raman response, whereas the '
            'core-shell sample showed another Raman response.'
        ),
        supporting_phrases=[
            'The alloy sample showed one Raman response, whereas the '
            'core-shell sample showed another Raman response.'
        ],
        subject_evidence_phrase='alloy sample',
        relation_evidence_phrase='whereas',
        object_evidence_phrase='core-shell sample',
    )
    assert 'RELATION_ARGUMENT_SCOPE_AMBIGUOUS' in _codes(bad)
    assert 'RELATION_ARGUMENT_SCOPE_AMBIGUOUS' not in _codes(peer)


def test_alpha4b2b2_obvious_relational_frontier_is_rejected():
    concept = _frontier(
        label='Ag-cluster diversity produces resonant multimodes',
        source_phrase='Ag-cluster diversity produces resonant multimodes.',
    )
    assert 'RELATIONAL_FRONTIER' in _codes(concept)


def test_alpha4b2b2_ambiguous_caption_and_mediation_remain_conservative():
    caption = _pattern(
        pattern_subject='maximum E-field intensity',
        pattern_object='particle size',
        source_phrase=(
            'Theoretically calculated maximum E-field intensity of Ag, Au '
            'and Au@Ag dimers with different particle sizes.'
        ),
        supporting_phrases=[
            'Theoretically calculated maximum E-field intensity of Ag, Au '
            'and Au@Ag dimers with different particle sizes.'
        ],
        subject_evidence_phrase='maximum E-field intensity',
        relation_evidence_phrase='with different',
        object_evidence_phrase='particle sizes',
    )
    mediation = _pattern(
        pattern_subject='Ag atom motion',
        pattern_relation='MEDIATES',
        pattern_object='Au diffusion into Ag seeds',
        relation_strength='causal_interpretive',
        source_phrase=(
            'The motion of Ag atoms facilitates the diffusion of Au atoms '
            'into the Ag seeds.'
        ),
        supporting_phrases=[
            'The motion of Ag atoms facilitates the diffusion of Au atoms '
            'into the Ag seeds.'
        ],
        subject_evidence_phrase='The motion of Ag atoms',
        relation_evidence_phrase='facilitates',
        object_evidence_phrase='the diffusion of Au atoms into the Ag seeds',
    )
    assert 'RELATION_CUE_MISMATCH' in _codes(caption)
    assert 'RELATION_CUE_MISMATCH' in _codes(mediation)
