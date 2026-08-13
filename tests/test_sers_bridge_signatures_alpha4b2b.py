import json

import networkx as nx

from dac_her.sers_bridge_signatures import (
    strict_node_catalog,
    strong_anchor_context_issues,
)


def test_alpha4b2b_sers_catalog_exposes_plasmonic_signatures_and_context():
    graph = nx.MultiDiGraph()
    graph.add_node(
        'sub',
        type='PlasmonicSubstrate',
        label='Au@Ag core-shell nanocube substrate with an interparticle nanogap',
        description='Gold core and silver shell.',
    )
    graph.add_node(
        'exp',
        type='Experiment',
        label='R6G SERS measurement',
        conditions_json=json.dumps([
            {'name': 'analyte', 'value_text': 'R6G', 'value_numeric': None, 'unit': None},
            {'name': 'excitation wavelength', 'value_text': None, 'value_numeric': 633, 'unit': 'nm'},
        ]),
    )
    graph.add_edge('sub', 'exp', relation='TESTED_IN')

    rows = {row['id']: row for row in strict_node_catalog(graph)}
    assert set(rows['sub']['metal_signature']) == {'au', 'ag'}
    assert 'core_shell' in rows['sub']['architecture_signature']
    assert 'nanocube' in rows['sub']['morphology_signature']
    assert 'nanogap' in rows['sub']['structural_motif_signature']
    assert rows['exp']['measurement_context'][0]['name'] == 'analyte'


def test_alpha4b2b_anchor_metal_conflict_is_hard_but_missing_detail_is_not():
    au_anchor = {
        'id': 'au',
        'type': 'Nanostructure',
        'label': 'Gold nanocube',
        'metal_signature': ['au'],
    }
    assert strong_anchor_context_issues(
        concept_text='silver nanoparticle oxidation',
        anchor=au_anchor,
    )
    assert strong_anchor_context_issues(
        concept_text='nanogap-dependent local field enhancement',
        anchor=au_anchor,
    ) == []
