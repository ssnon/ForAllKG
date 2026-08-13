from dac_her.bridge_schemas import BridgeConcept
from dac_her.sers_bridge_policy import concept_policy_issues


def _pattern(**updates):
    data = dict(
        id='b1',
        concept_type='RelationPattern',
        label='nanogap-dependent SERS intensity',
        source_phrase='The SERS intensity increased as the nanogap decreased.',
        description=None,
        retention_lane='accepted_pattern',
        evidence_scope='paper_result',
        pattern_subject='SERS intensity',
        pattern_relation='VARIES_WITH',
        pattern_object='nanogap size',
        relation_strength='correlational',
        qualifiers=[],
        pattern_support_mode='explicit_single_span',
        supporting_phrases=[
            'The SERS intensity increased as the nanogap decreased.'
        ],
        subject_evidence_phrase='SERS intensity',
        relation_evidence_phrase='increased as',
        object_evidence_phrase='nanogap',
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


def test_alpha4b2b_explicit_sers_relation_is_accepted_by_policy():
    assert _codes(_pattern()) == set()


def test_alpha4b2b_reversed_varies_with_orientation_is_rejected():
    concept = _pattern(
        pattern_subject='nanogap size',
        pattern_object='SERS intensity',
    )
    assert 'RELATION_ARGUMENT_DIRECTION' in _codes(concept)


def test_alpha4b2b_scalar_numeric_frontier_is_rejected():
    concept = _frontier(
        label='enhancement factor',
        source_phrase='The enhancement factor was 1.2 x 10^8.',
    )
    codes = _codes(concept)
    assert 'SCALAR_METRIC' in codes
    assert 'INSTANCE_ONLY' in codes


def test_alpha4b2b_mechanistic_frontier_is_allowed():
    assert _codes(_frontier()) == set()
