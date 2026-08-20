from pipeline_core.domain_profile import NoveltySemantics
from dac_her.domains import available_domain_profiles, get_domain_profile


def test_default_domain_profile_is_dac_her():
    profile = get_domain_profile()
    assert profile.profile_id == 'dac_her'
    assert 'REACTION' in profile.discovery.context_node_types
    assert profile.feasibility_adapter_id == 'dac_her'


def test_dac_her_scope_gate_is_preserved():
    novelty = get_domain_profile('dac_her').novelty
    compatible, domain, scope, reasons = novelty.strong_scope_compatibility(
        'nitrogen coordination in a dual-atom HER catalyst',
        'nitrogen-coordinated dual-atom catalyst for hydrogen evolution electrocatalysis',
        min_domain=0.75,
        min_scope=0.75,
    )
    assert compatible
    assert domain == 1.0
    assert scope == 1.0
    assert reasons == []


def test_custom_sers_like_novelty_semantics_are_not_her_bound():
    custom = NoveltySemantics(
        domain_patterns=(('SERS', (r'\bsers\b', r'surface enhanced raman')),),
        scope_patterns=(
            ('au_ag', (r'au.?ag', r'gold.{0,12}silver')),
            ('nanogap', (r'nanogap',)),
        ),
        critical_scope_features=frozenset({'au_ag'}),
    )
    assert custom.domain_relevance(
        'Au-Ag SERS enhancement',
        'AuAg nanogap substrate for SERS',
    ) == 1.0
    assert custom.scope_relevance(
        'Au-Ag SERS enhancement',
        'AuAg nanogap substrate for SERS',
    ) == 1.0


def test_registry_has_expected_builtin_profiles():
    assert available_domain_profiles() == (
        'catalysis_mechanism',
        'dac_her',
        'sers_au_ag',
    )
    assert get_domain_profile().profile_id == 'dac_her'
    assert get_domain_profile('default').profile_id == 'dac_her'
    assert get_domain_profile('sers').profile_id == 'sers_au_ag'
    assert get_domain_profile('broad').profile_id == 'catalysis_mechanism'
