from __future__ import annotations

import networkx as nx
import pytest

from domains.sers.context_compiler import (
    SERSContextCompilationError,
    SERSContextCompiler,
)
from pipeline_core.discovery.discovery_contracts import (
    DiscoveryInspiration,
    DiscoveryScoreBreakdown,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisEvidenceStatement,
)


def _score() -> DiscoveryScoreBreakdown:
    return DiscoveryScoreBreakdown(
        endpoint_relevance=0.5,
        mechanistic_content=0.5,
        cross_paper_span=0.0,
        community_span=0.0,
        relation_rarity=0.0,
        exploratory_mode_bonus=0.0,
        grounding_redundancy_penalty=0.0,
        navigation_burden_penalty=0.0,
        reverse_burden_penalty=0.0,
        total=0.5,
    )


def _inspiration(
    *,
    inspiration_id: str,
    label: str,
    node_ids: list[str],
    entry: str,
    exit: str,
) -> DiscoveryInspiration:
    return DiscoveryInspiration(
        inspiration_id=inspiration_id,
        source_path_id="path:1",
        source_corpus_id="corpus:1",
        source_mode="exploratory",
        paper_ids=["paper:wide-union"],
        node_ids=node_ids,
        edge_ids=[],
        relation_sequence=[],
        rendered_path=label,
        exploration_score=0.3,
        score_breakdown=_score(),
        candidate_unit_id=(
            "candidate:"
            + inspiration_id
        ),
        candidate_unit_label=label,
        candidate_entry_anchor_id=entry,
        candidate_entry_anchor_label=entry,
        candidate_exit_anchor_id=exit,
        candidate_exit_anchor_label=exit,
        candidate_proposed_subject="subject",
        candidate_proposed_relation="VARIES_WITH",
        candidate_proposed_object="object",
        candidate_unit_score=0.31,
    )


def _compiler(
    graph: nx.MultiDiGraph,
) -> SERSContextCompiler:
    return SERSContextCompiler(
        graph=graph,
        domain_profile_id="sers_au_ag",
    )


def _values(
    signature,
    dimension: str,
) -> set[str]:
    return {
        row.value
        for row in signature.facts
        if (
            row.dimension == dimension
            and row.value is not None
        )
    }


def _unknown_subject_tags(
    signature,
) -> set[str]:
    tags: set[str] = set()

    for row in signature.facts:
        if (
            row.dimension
            == "material_state"
            and row.knowledge_state
            == "unknown"
        ):
            tags.update(
                row.tags
            )

    return tags


def test_h1_keeps_pyramid_morphology_and_gap_separate() -> None:
    graph = nx.MultiDiGraph()

    bridge = "paper::p1::bridge"
    ag = "paper::p1::ag"
    auag = "paper::p1::auag"
    si = "paper::p1::si"
    combo = "paper::p1::combo"

    graph.add_node(
        bridge,
        type="BridgeConcept",
        label=(
            "Local electric-field intensity "
            "varies with inserted-pyramid geometry"
        ),
        source_paper_id="p1",
    )

    graph.add_node(
        ag,
        type="Nanostructure",
        label="Ag nanoparticle",
        source_paper_id="p1",
    )

    graph.add_node(
        auag,
        type="Nanostructure",
        label="Au@Ag nanoparticle",
        source_paper_id="p1",
    )

    graph.add_node(
        si,
        type="PlasmonicSubstrate",
        label="3D-Si substrate",
        source_paper_id="p1",
    )

    graph.add_node(
        combo,
        type="PlasmonicSubstrate",
        label="Au@Ag/3D-Si substrate",
        source_paper_id="p1",
    )

    for index, anchor in enumerate(
        [ag, auag, si, combo],
        start=1,
    ):
        graph.add_edge(
            anchor,
            bridge,
            relation=(
                "GROUNDS_SEMANTIC_CANDIDATE"
            ),
            edge_id=f"ground:{index}",
        )

    gap_15 = "paper::p1::gap15"
    gap_10 = "paper::p1::gap10"
    pyramid = "paper::p1::pyramid"

    graph.add_node(
        gap_15,
        type="StructuralMotif",
        label="15 nm nanoparticle gap",
        source_paper_id="p1",
    )

    graph.add_node(
        gap_10,
        type="StructuralMotif",
        label="10 nm nanoparticle gap",
        source_paper_id="p1",
    )

    graph.add_node(
        pyramid,
        type="Morphology",
        label=(
            "3D-Si with inserted small pyramid"
        ),
        source_paper_id="p1",
    )

    graph.add_edge(
        ag,
        gap_15,
        relation="HAS_STRUCTURAL_MOTIF",
        edge_id="edge:gap15",
    )

    graph.add_edge(
        auag,
        gap_10,
        relation="HAS_STRUCTURAL_MOTIF",
        edge_id="edge:gap10",
    )

    graph.add_edge(
        si,
        pyramid,
        relation="HAS_MORPHOLOGY",
        edge_id="edge:pyramid",
    )

    inspiration = _inspiration(
        inspiration_id="insp:h1",
        label=(
            "Local electric-field intensity "
            "varies with inserted-pyramid geometry"
        ),
        node_ids=[
            bridge,
            ag,
            auag,
            si,
            combo,
        ],
        entry=auag,
        exit=ag,
    )

    signature = (
        _compiler(
            graph
        ).compile_axis_inspiration(
            inspiration
        )
    )

    assert _values(
        signature,
        "morphology",
    ) == {
        "3D-Si with inserted small pyramid"
    }

    assert _values(
        signature,
        "gap_regime",
    ) == {
        "10 nm nanoparticle gap",
        "15 nm nanoparticle gap",
    }

    # The compiler must never synthesize a hybrid source fact.
    all_values = {
        row.value
        for row in signature.facts
        if row.value is not None
    }

    assert (
        "inserted-pyramid-like nanogap geometry"
        not in all_values
    )


def test_h2_uses_direct_cu_components_but_not_same_paper_cuo() -> None:
    graph = nx.MultiDiGraph()

    bridge = "paper::p2::bridge"
    cu_ag = "paper::p2::cu_ag"
    cu_au = "paper::p2::cu_au"

    graph.add_node(
        bridge,
        type="BridgeConcept",
        label=(
            "Copper–silver/gold synergy "
            "improves SERS substrate performance"
        ),
        source_paper_id="p2",
    )

    graph.add_node(
        cu_ag,
        type="PlasmonicSubstrate",
        label=(
            "Copper substrate with "
            "nanostructured silver"
        ),
        source_paper_id="p2",
    )

    graph.add_node(
        cu_au,
        type="PlasmonicSubstrate",
        label=(
            "Copper substrate with "
            "nanostructured gold"
        ),
        source_paper_id="p2",
    )

    graph.add_edge(
        cu_ag,
        bridge,
        relation=(
            "GROUNDS_SEMANTIC_CANDIDATE"
        ),
        edge_id="ground:cu_ag",
    )

    graph.add_edge(
        cu_au,
        bridge,
        relation=(
            "GROUNDS_SEMANTIC_CANDIDATE"
        ),
        edge_id="ground:cu_au",
    )

    copper = "paper::p2::cu"
    silver = "paper::p2::ag"
    gold = "paper::p2::au"

    graph.add_node(
        copper,
        type="Metal",
        label="Copper",
        source_paper_id="p2",
    )

    graph.add_node(
        silver,
        type="Metal",
        label="Silver",
        source_paper_id="p2",
    )

    graph.add_node(
        gold,
        type="Metal",
        label="Gold",
        source_paper_id="p2",
    )

    graph.add_edge(
        cu_ag,
        copper,
        relation="HAS_COMPONENT",
        edge_id="edge:cu1",
    )

    graph.add_edge(
        cu_ag,
        silver,
        relation="HAS_COMPONENT",
        edge_id="edge:ag",
    )

    graph.add_edge(
        cu_au,
        copper,
        relation="HAS_COMPONENT",
        edge_id="edge:cu2",
    )

    graph.add_edge(
        cu_au,
        gold,
        relation="HAS_COMPONENT",
        edge_id="edge:au",
    )

    # Same paper, but deliberately disconnected from the candidate-local
    # closure. This must NEVER be inherited.
    cuo = "paper::p2::cuo"

    graph.add_node(
        cuo,
        type="Material",
        label="CuO",
        source_paper_id="p2",
    )

    inspiration = _inspiration(
        inspiration_id="insp:h2",
        label=(
            "Copper–silver/gold synergy "
            "improves SERS substrate performance"
        ),
        node_ids=[
            bridge,
            cu_ag,
            cu_au,
        ],
        entry=cu_au,
        exit=cu_ag,
    )

    signature = (
        _compiler(
            graph
        ).compile_axis_inspiration(
            inspiration
        )
    )

    assert _values(
        signature,
        "substrate",
    ) == {
        (
            "Copper substrate with "
            "nanostructured silver"
        ),
        (
            "Copper substrate with "
            "nanostructured gold"
        ),
    }

    materials = _values(
        signature,
        "material_identity",
    )

    assert materials == {
        "Copper",
        "Silver",
        "Gold",
    }

    assert "CuO" not in materials

    unknown_tags = (
        _unknown_subject_tags(
            signature
        )
    )

    assert (
        "material_subject:copper"
        in unknown_tags
    )

    assert not any(
        row.dimension
        == "material_state"
        and row.knowledge_state
        == "explicit"
        and row.value == "CuO"
        for row in signature.facts
    )


def test_grounded_claim_uses_applies_to_then_direct_context_only() -> None:
    graph = nx.MultiDiGraph()

    claim = "paper::p3::claim"
    substrate = "paper::p3::substrate"
    gold = "paper::p3::gold"
    silver = "paper::p3::silver"
    architecture = "paper::p3::arch"
    nanostar = "paper::p3::morph"

    graph.add_node(
        claim,
        type="MechanismClaim",
        label=(
            "Maximum electromagnetic enhancement "
            "occurs near an LSPR condition"
        ),
        source_paper_id="p3",
    )

    graph.add_node(
        substrate,
        type="PlasmonicSubstrate",
        label=(
            "Silver-coated gold nanostars "
            "(AuNSt@Ag)"
        ),
        source_paper_id="p3",
    )

    graph.add_edge(
        claim,
        substrate,
        relation="APPLIES_TO",
        edge_id="edge:applies",
    )

    graph.add_node(
        gold,
        type="Metal",
        label="Gold",
        source_paper_id="p3",
    )

    graph.add_node(
        silver,
        type="Metal",
        label="Silver",
        source_paper_id="p3",
    )

    graph.add_node(
        architecture,
        type="StructuralMotif",
        label=(
            "Gold-core/silver-shell arrangement"
        ),
        source_paper_id="p3",
    )

    graph.add_node(
        nanostar,
        type="Morphology",
        label="Nanostar morphology",
        source_paper_id="p3",
    )

    graph.add_edge(
        substrate,
        gold,
        relation="HAS_COMPONENT",
        edge_id="edge:gold",
    )

    graph.add_edge(
        substrate,
        silver,
        relation="HAS_COMPONENT",
        edge_id="edge:silver",
    )

    graph.add_edge(
        substrate,
        architecture,
        relation="HAS_ARCHITECTURE",
        edge_id="edge:arch",
    )

    graph.add_edge(
        substrate,
        nanostar,
        relation="HAS_MORPHOLOGY",
        edge_id="edge:morph",
    )

    # Again: same paper, disconnected context must not leak.
    unrelated = "paper::p3::unrelated"

    graph.add_node(
        unrelated,
        type="Morphology",
        label="Unrelated nanocube morphology",
        source_paper_id="p3",
    )

    statement = (
        HypothesisEvidenceStatement(
            statement_id="stmt:p3",
            text=(
                "LSPR placement is relevant "
                "to enhancement."
            ),
            epistemic_role="reported",
            claim_kind="mechanism",
            paper_ids=["p3"],
            scientific_support_node_ids=[
                claim
            ],
            scientific_support_edge_ids=[
                "edge:applies"
            ],
            eligible_as_premise=True,
        )
    )

    signature = (
        _compiler(
            graph
        ).compile_grounded_statement(
            statement
        )
    )

    assert _values(
        signature,
        "substrate",
    ) == {
        (
            "Silver-coated gold nanostars "
            "(AuNSt@Ag)"
        )
    }

    assert _values(
        signature,
        "material_identity",
    ) == {
        "Gold",
        "Silver",
    }

    assert _values(
        signature,
        "morphology",
    ) == {
        "Nanostar morphology"
    }

    assert (
        "Unrelated nanocube morphology"
        not in _values(
            signature,
            "morphology",
        )
    )


def test_physical_support_root_is_context_root_without_applies_to() -> None:
    graph = nx.MultiDiGraph()

    substrate = (
        "paper::p4::substrate"
    )
    gold = "paper::p4::gold"
    silver = "paper::p4::silver"

    graph.add_node(
        substrate,
        type="PlasmonicSubstrate",
        label=(
            "Bimetallic Au–Ag "
            "core–shell substrate"
        ),
        source_paper_id="p4",
    )

    graph.add_node(
        gold,
        type="Metal",
        label="Gold",
        source_paper_id="p4",
    )

    graph.add_node(
        silver,
        type="Metal",
        label="Silver",
        source_paper_id="p4",
    )

    graph.add_edge(
        substrate,
        gold,
        relation="HAS_COMPONENT",
        edge_id="edge:p4:gold",
    )

    graph.add_edge(
        substrate,
        silver,
        relation="HAS_COMPONENT",
        edge_id="edge:p4:silver",
    )

    statement = (
        HypothesisEvidenceStatement(
            statement_id="stmt:p4",
            text=(
                "Au–Ag structures improve "
                "SERS response."
            ),
            epistemic_role="reported",
            claim_kind="observation",
            paper_ids=["p4"],
            scientific_support_node_ids=[
                substrate
            ],
            eligible_as_premise=True,
        )
    )

    signature = (
        _compiler(
            graph
        ).compile_grounded_statement(
            statement
        )
    )

    assert _values(
        signature,
        "substrate",
    ) == {
        (
            "Bimetallic Au–Ag "
            "core–shell substrate"
        )
    }

    assert _values(
        signature,
        "material_identity",
    ) == {
        "Gold",
        "Silver",
    }


def test_compilation_is_deterministic_across_edge_insertion_order() -> None:
    def build(
        reverse: bool,
    ) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()

        bridge = "paper::p5::bridge"
        a = "paper::p5::a"
        b = "paper::p5::b"
        cu = "paper::p5::cu"
        au = "paper::p5::au"

        graph.add_node(
            bridge,
            type="BridgeConcept",
            label="candidate",
            source_paper_id="p5",
        )

        graph.add_node(
            a,
            type="PlasmonicSubstrate",
            label="Copper substrate A",
            source_paper_id="p5",
        )

        graph.add_node(
            b,
            type="PlasmonicSubstrate",
            label="Copper substrate B",
            source_paper_id="p5",
        )

        graph.add_node(
            cu,
            type="Metal",
            label="Copper",
            source_paper_id="p5",
        )

        graph.add_node(
            au,
            type="Metal",
            label="Gold",
            source_paper_id="p5",
        )

        edges = [
            (
                a,
                bridge,
                "GROUNDS_SEMANTIC_CANDIDATE",
                "e1",
            ),
            (
                b,
                bridge,
                "GROUNDS_SEMANTIC_CANDIDATE",
                "e2",
            ),
            (
                a,
                cu,
                "HAS_COMPONENT",
                "e3",
            ),
            (
                b,
                au,
                "HAS_COMPONENT",
                "e4",
            ),
        ]

        if reverse:
            edges.reverse()

        for source, target, relation, edge_id in edges:
            graph.add_edge(
                source,
                target,
                relation=relation,
                edge_id=edge_id,
            )

        return graph

    inspiration = _inspiration(
        inspiration_id="insp:p5",
        label="candidate",
        node_ids=[
            "paper::p5::bridge",
            "paper::p5::a",
            "paper::p5::b",
        ],
        entry="paper::p5::a",
        exit="paper::p5::b",
    )

    first = (
        _compiler(
            build(False)
        ).compile_axis_inspiration(
            inspiration
        )
    )

    second = (
        _compiler(
            build(True)
        ).compile_axis_inspiration(
            inspiration
        )
    )

    assert (
        first.model_dump(
            mode="json"
        )
        == second.model_dump(
            mode="json"
        )
    )


def test_axis_fails_closed_when_lineage_anchor_is_not_bridge_grounded() -> None:
    graph = nx.MultiDiGraph()

    bridge = "paper::p6::bridge"
    a = "paper::p6::a"
    b = "paper::p6::b"

    graph.add_node(
        bridge,
        type="BridgeConcept",
        label="candidate",
    )

    graph.add_node(
        a,
        type="PlasmonicSubstrate",
        label="A",
    )

    graph.add_node(
        b,
        type="PlasmonicSubstrate",
        label="B",
    )

    graph.add_edge(
        a,
        bridge,
        relation=(
            "GROUNDS_SEMANTIC_CANDIDATE"
        ),
        edge_id="ground:a",
    )

    inspiration = _inspiration(
        inspiration_id="insp:p6",
        label="candidate",
        node_ids=[
            bridge,
            a,
            b,
        ],
        entry=a,
        exit=b,
    )

    with pytest.raises(
        SERSContextCompilationError,
        match=(
            "candidate entry/exit lineage"
        ),
    ):
        _compiler(
            graph
        ).compile_axis_inspiration(
            inspiration
        )


def test_grounded_statement_without_claim_local_context_fails_closed() -> None:
    graph = nx.MultiDiGraph()

    claim = "paper::p7::claim"

    graph.add_node(
        claim,
        type="MechanismClaim",
        label="generic mechanism claim",
    )

    statement = (
        HypothesisEvidenceStatement(
            statement_id="stmt:p7",
            text="generic mechanism",
            epistemic_role="reported",
            claim_kind="mechanism",
            scientific_support_node_ids=[
                claim
            ],
            eligible_as_premise=True,
        )
    )

    with pytest.raises(
        SERSContextCompilationError,
        match=(
            "claim-local closure produced "
            "no SERS context facts"
        ),
    ):
        _compiler(
            graph
        ).compile_grounded_statement(
            statement
        )


def test_axis_includes_direct_bridge_anchor_absent_from_path_node_ids() -> None:
    graph = nx.MultiDiGraph()

    bridge = "paper::p8::bridge"
    entry = "paper::p8::entry"
    exit = "paper::p8::exit"

    # This scientific anchor directly grounds the candidate, but is
    # deliberately absent from DiscoveryInspiration.node_ids.
    support = "paper::p8::support"
    morphology = "paper::p8::morphology"

    graph.add_node(
        bridge,
        type="BridgeConcept",
        label="candidate",
        source_paper_id="p8",
    )

    graph.add_node(
        entry,
        type="Nanostructure",
        label="Au@Ag nanoparticle",
        source_paper_id="p8",
    )

    graph.add_node(
        exit,
        type="Nanostructure",
        label="Ag nanoparticle",
        source_paper_id="p8",
    )

    graph.add_node(
        support,
        type="PlasmonicSubstrate",
        label="3D-Si substrate",
        source_paper_id="p8",
    )

    graph.add_node(
        morphology,
        type="Morphology",
        label="3D-Si with inserted small pyramid",
        source_paper_id="p8",
    )

    graph.add_edge(
        entry,
        bridge,
        relation="GROUNDS_SEMANTIC_CANDIDATE",
        edge_id="ground:entry",
    )

    graph.add_edge(
        exit,
        bridge,
        relation="GROUNDS_SEMANTIC_CANDIDATE",
        edge_id="ground:exit",
    )

    graph.add_edge(
        support,
        bridge,
        relation="GROUNDS_SEMANTIC_CANDIDATE",
        edge_id="ground:support",
    )

    graph.add_edge(
        support,
        morphology,
        relation="HAS_MORPHOLOGY",
        edge_id="edge:morphology",
    )

    inspiration = _inspiration(
        inspiration_id="insp:p8",
        label="candidate",

        # support is intentionally NOT listed here.
        node_ids=[
            bridge,
            entry,
            exit,
        ],

        entry=entry,
        exit=exit,
    )

    signature = (
        _compiler(
            graph
        ).compile_axis_inspiration(
            inspiration
        )
    )

    assert _values(
        signature,
        "substrate",
    ) == {
        "3D-Si substrate"
    }

    assert _values(
        signature,
        "morphology",
    ) == {
        "3D-Si with inserted small pyramid"
    }


def test_axis_does_not_inherit_unconnected_same_paper_anchor() -> None:
    graph = nx.MultiDiGraph()

    bridge = "paper::p9::bridge"
    entry = "paper::p9::entry"
    exit = "paper::p9::exit"

    disconnected = "paper::p9::disconnected"
    cuo = "paper::p9::cuo"

    graph.add_node(
        bridge,
        type="BridgeConcept",
        label="candidate",
        source_paper_id="p9",
    )

    graph.add_node(
        entry,
        type="PlasmonicSubstrate",
        label="Copper substrate with gold",
        source_paper_id="p9",
    )

    graph.add_node(
        exit,
        type="PlasmonicSubstrate",
        label="Copper substrate with silver",
        source_paper_id="p9",
    )

    graph.add_edge(
        entry,
        bridge,
        relation="GROUNDS_SEMANTIC_CANDIDATE",
        edge_id="ground:entry",
    )

    graph.add_edge(
        exit,
        bridge,
        relation="GROUNDS_SEMANTIC_CANDIDATE",
        edge_id="ground:exit",
    )

    # Same paper, but not directly grounding the candidate bridge.
    graph.add_node(
        disconnected,
        type="PlasmonicSubstrate",
        label="Unrelated copper oxide substrate",
        source_paper_id="p9",
    )

    graph.add_node(
        cuo,
        type="Material",
        label="CuO",
        source_paper_id="p9",
    )

    graph.add_edge(
        disconnected,
        cuo,
        relation="HAS_COMPONENT",
        edge_id="edge:unrelated-cuo",
    )

    inspiration = _inspiration(
        inspiration_id="insp:p9",
        label="candidate",
        node_ids=[
            bridge,
            entry,
            exit,
        ],
        entry=entry,
        exit=exit,
    )

    signature = (
        _compiler(
            graph
        ).compile_axis_inspiration(
            inspiration
        )
    )

    assert "CuO" not in _values(
        signature,
        "material_identity",
    )

    assert (
        "Unrelated copper oxide substrate"
        not in _values(
            signature,
            "substrate",
        )
    )
