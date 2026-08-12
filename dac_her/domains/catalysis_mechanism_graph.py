from __future__ import annotations

import networkx as nx

from dac_her.graph_domain import GraphDomainAdapter, RelationConstraint


BROAD_MECHANISM_NODE_TYPES = frozenset({
    "Catalyst",
    "CatalystModel",
    "Metal",
    "Support",
    "CoordinationMotif",
    "Reaction",
    "ReactionStep",
    "Intermediate",
    "Material",
    "ActiveSite",
    "StructuralState",
    "AdsorbateState",
    "InterfacialEnvironment",
    "MechanisticFactor",
    "Descriptor",
    "ObservationClaim",
    "MechanismClaim",
})

BROAD_MECHANISM_CORE_TYPES = frozenset({
    "ActiveSite",
    "StructuralState",
    "AdsorbateState",
    "InterfacialEnvironment",
    "MechanisticFactor",
    "Descriptor",
    "ReactionStep",
    "Intermediate",
    "MechanismClaim",
})

BROAD_DIRECT_MECHANISM_RELATIONS = frozenset({
    "ADSORBS",
    "INDUCES",
    "MODULATES",
    "STABILIZES",
    "DESTABILIZES",
    "PROMOTES",
    "SUPPRESSES",
    "FACILITATES_STEP",
    "INHIBITS_STEP",
    "RECONSTRUCTS_TO",
    "CHANGES_ACTIVE_SITE",
    "CHANGES_RDS",
    "DEPENDS_ON",
    "CORRELATES_WITH",
    "FAILS_WHEN",
})


def _preserve_broad_semantic_roles(
    graph: nx.MultiDiGraph,
    *,
    chunk_id: str,
):
    """Preserve strict broad-catalysis roles without DAC-HER coercion."""
    del chunk_id
    return graph, []


_SCIENTIFIC_TARGETS = frozenset({
    "Catalyst",
    "CatalystModel",
    "Metal",
    "Support",
    "CoordinationMotif",
    "Reaction",
    "ReactionStep",
    "Intermediate",
    "Material",
    "SynthesisMethod",
    "Precursor",
    "ActiveSite",
    "StructuralState",
    "AdsorbateState",
    "InterfacialEnvironment",
    "MechanisticFactor",
    "Descriptor",
})

_CATALYST_LIKE = frozenset({"Catalyst", "CatalystModel"})
_STATE_OR_FACTOR = frozenset({
    "ActiveSite",
    "StructuralState",
    "AdsorbateState",
    "InterfacialEnvironment",
    "MechanisticFactor",
    "Descriptor",
    "ReactionStep",
    "Intermediate",
})
_CLAIMS = frozenset({"ObservationClaim", "MechanismClaim"})
_CAUSAL_TARGETS = _STATE_OR_FACTOR | frozenset({"Catalyst", "CatalystModel"})
_CAUSAL_SOURCES = _STATE_OR_FACTOR | frozenset({"Catalyst", "CatalystModel"})


CATALYSIS_MECHANISM_RELATION_CONSTRAINTS = (
    RelationConstraint(
        "STUDIES",
        source_types=frozenset({"Paper"}),
        target_types=_SCIENTIFIC_TARGETS,
    ),
    RelationConstraint(
        "HAS_METAL",
        source_types=_CATALYST_LIKE,
        target_types=frozenset({"Metal"}),
    ),
    RelationConstraint(
        "SUPPORTED_ON",
        source_types=_CATALYST_LIKE,
        target_types=frozenset({"Support"}),
    ),
    RelationConstraint(
        "HAS_MOTIF",
        source_types=_CATALYST_LIKE,
        target_types=frozenset({"CoordinationMotif"}),
    ),
    RelationConstraint(
        "CATALYZES",
        source_types=frozenset({"Catalyst"}),
        target_types=frozenset({"Reaction"}),
    ),
    RelationConstraint(
        "MODEL_OF",
        source_types=frozenset({"CatalystModel"}),
        target_types=frozenset({"Catalyst"}),
    ),
    RelationConstraint(
        "HAS_ACTIVE_SITE",
        source_types=_CATALYST_LIKE,
        target_types=frozenset({"ActiveSite"}),
    ),
    RelationConstraint(
        "HAS_STRUCTURAL_STATE",
        source_types=_CATALYST_LIKE | frozenset({"ActiveSite"}),
        target_types=frozenset({"StructuralState"}),
    ),
    RelationConstraint(
        "HAS_ADSORBATE_STATE",
        source_types=_CATALYST_LIKE | frozenset({"ActiveSite", "Reaction"}),
        target_types=frozenset({"AdsorbateState"}),
    ),
    RelationConstraint(
        "HAS_ENVIRONMENT",
        source_types=_CATALYST_LIKE | frozenset({"Reaction"}),
        target_types=frozenset({"InterfacialEnvironment"}),
    ),
    RelationConstraint(
        "HAS_DESCRIPTOR",
        source_types=_CATALYST_LIKE | frozenset({"ActiveSite"}),
        target_types=frozenset({"Descriptor"}),
    ),
    RelationConstraint(
        "RECONSTRUCTS_TO",
        source_types=frozenset({"StructuralState"}),
        target_types=frozenset({"StructuralState"}),
    ),
    RelationConstraint(
        "FACILITATES_STEP",
        source_types=frozenset({"ActiveSite"}),
        target_types=frozenset({"ReactionStep"}),
    ),
    RelationConstraint(
        "INHIBITS_STEP",
        source_types=frozenset({"ActiveSite"}),
        target_types=frozenset({"ReactionStep"}),
    ),
    RelationConstraint(
        "CHANGES_ACTIVE_SITE",
        source_types=_CAUSAL_SOURCES,
        target_types=frozenset({"ActiveSite"}),
    ),
    RelationConstraint(
        "CHANGES_RDS",
        source_types=_CAUSAL_SOURCES,
        target_types=frozenset({"ReactionStep"}),
    ),
    RelationConstraint(
        "FAILS_WHEN",
        source_types=frozenset({"Descriptor", "MechanisticFactor"}),
        target_types=_STATE_OR_FACTOR,
    ),
    *(
        RelationConstraint(
            relation,
            source_types=_CAUSAL_SOURCES,
            target_types=_CAUSAL_TARGETS,
        )
        for relation in (
            "INDUCES",
            "MODULATES",
            "STABILIZES",
            "DESTABILIZES",
            "PROMOTES",
            "SUPPRESSES",
            "DEPENDS_ON",
            "CORRELATES_WITH",
        )
    ),
    RelationConstraint(
        "APPLIES_TO",
        source_types=_CLAIMS,
        target_types=_SCIENTIFIC_TARGETS,
    ),
)


CATALYSIS_MECHANISM_GRAPH_ADAPTER = GraphDomainAdapter(
    adapter_id="catalysis_mechanism",
    domain_profile_id="catalysis_mechanism",
    semantic_role_policy=(
        "Preserve broad-catalysis entity roles exactly. No DAC-HER-specific "
        "Catalyst/Material coercion is applied to abstract-derived graphs."
    ),
    semantic_role_normalizer=_preserve_broad_semantic_roles,
    relation_constraints=CATALYSIS_MECHANISM_RELATION_CONSTRAINTS,
)
