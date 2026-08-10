import pytest

from dac_her.domain_profile import (
    DiscoverySemantics,
    NoveltySemantics,
    ResolutionSemantics,
    ScientificDomainProfile,
)
from dac_her.domains.feasibility_registry import (
    available_feasibility_adapters,
    get_feasibility_adapter,
    resolve_feasibility_adapter,
)
from dac_her.domains.dac_her_feasibility import DacHerFeasibilityAdapter


def _minimal_profile(
    profile_id: str,
    *,
    feasibility_adapter_id: str | None,
) -> ScientificDomainProfile:
    return ScientificDomainProfile(
        profile_id=profile_id,
        description="test",
        resolution=ResolutionSemantics(
            resolvable_node_types=frozenset(),
            auto_merge_types=frozenset(),
            text_replacements=(),
            reaction_aliases=(),
        ),
        discovery=DiscoverySemantics(
            generic_entity_types=frozenset(),
            mechanism_node_markers=(),
            mechanism_relation_markers=(),
            scaffold_relations=frozenset(),
            context_node_types=frozenset(),
        ),
        novelty=NoveltySemantics(
            domain_patterns=(),
            scope_patterns=(),
            critical_scope_features=frozenset(),
        ),
        feasibility_adapter_id=feasibility_adapter_id,
    )


def test_builtin_dac_her_profile_resolves_dac_her_adapter():
    adapter = get_feasibility_adapter("dac_her")
    assert isinstance(adapter, DacHerFeasibilityAdapter)
    assert adapter.adapter_id == "dac_her"
    assert adapter.domain_profile_id == "dac_her"
    assert available_feasibility_adapters() == ("dac_her",)


def test_profile_without_feasibility_adapter_fails_closed():
    profile = _minimal_profile(
        "synthetic_sers_without_feasibility",
        feasibility_adapter_id=None,
    )
    with pytest.raises(ValueError, match="has no feasibility adapter"):
        resolve_feasibility_adapter(profile)


def test_profile_cannot_silently_reuse_wrong_domain_adapter():
    profile = _minimal_profile(
        "synthetic_sers",
        feasibility_adapter_id="dac_her",
    )
    with pytest.raises(ValueError, match="adapter/domain mismatch"):
        resolve_feasibility_adapter(profile)
