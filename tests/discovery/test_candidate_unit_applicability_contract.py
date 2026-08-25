from __future__ import annotations

import argparse
from types import SimpleNamespace

import networkx as nx

from domains.registry import get_domain_profile
from domains.sers.candidate_unit_applicability import (
    SERS_AU_AG_CANDIDATE_UNIT_APPLICABILITY,
)
from scripts.discovery.run_candidate_unit_traversal import (
    _materialize_route,
)
from scripts.discovery.run_dac_discovery_e2e import (
    _candidate_unit_correction_contract,
)


def _unit(
    label: str,
    *,
    anchors: tuple[str, ...] = (),
):
    return SimpleNamespace(
        unit_id="candidate_unit:test",
        candidate_node_id="candidate:test",
        label=label,
        proposed_subject="",
        proposed_relation="",
        proposed_object="",
        anchors=tuple(
            SimpleNamespace(label=value)
            for value in anchors
        ),
    )


def test_sers_owner_gate_accepts_strict_gap_owner() -> None:
    adapter = SERS_AU_AG_CANDIDATE_UNIT_APPLICABILITY

    row = adapter.classify(
        _unit(
            "SERS enhancement varies with nanogap size"
        ),
        stop="nanogap",
    )

    assert row.owner_class == "UNIT_OWNED_GAP_CONTROL"
    assert row.gap_class == "STRICT_GAP_CONTROL"
    assert row.eligible is True


def test_sers_owner_gate_accepts_only_validated_structural_proxies() -> None:
    adapter = SERS_AU_AG_CANDIDATE_UNIT_APPLICABILITY

    spacing = adapter.classify(
        _unit(
            "Intrinsic field enhancement varies with sidewall spacing"
        ),
        stop="nanogap",
    )

    separation = adapter.classify(
        _unit(
            "Near-field enhancement varies with interparticle separation"
        ),
        stop="nanogap",
    )

    unrelated_distance = adapter.classify(
        _unit(
            "SERS response varies with adsorbate-surface distance"
        ),
        stop="nanogap",
    )

    assert spacing.owner_class == "UNIT_OWNED_GAP_CONTROL"
    assert spacing.gap_class == "GAP_CONTROL_PROXY"
    assert spacing.eligible is True

    assert separation.owner_class == "UNIT_OWNED_GAP_CONTROL"
    assert separation.gap_class == "GAP_CONTROL_PROXY"
    assert separation.eligible is True

    assert unrelated_distance.owner_class == "NO_GAP_ATTACHMENT"
    assert unrelated_distance.eligible is False


def test_sers_owner_gate_blocks_anchor_only_context() -> None:
    adapter = SERS_AU_AG_CANDIDATE_UNIT_APPLICABILITY

    row = adapter.classify(
        _unit(
            "Local electric-field intensity varies with inserted-pyramid geometry",
            anchors=(
                "nanogap geometry",
                "3D Si inserted-pyramid substrate",
            ),
        ),
        stop="nanogap",
    )

    assert row.owner_class == "ANCHOR_CONTEXT_ONLY"
    assert row.eligible is False


def test_sers_owner_gate_blocks_no_gap_attachment() -> None:
    adapter = SERS_AU_AG_CANDIDATE_UNIT_APPLICABILITY

    row = adapter.classify(
        _unit(
            "Copper-silver/gold synergy improves SERS substrate performance"
        ),
        stop="nanogap",
    )

    assert row.owner_class == "NO_GAP_ATTACHMENT"
    assert row.eligible is False


def test_relevance_context_reproduces_rcf_stop_query_contract() -> None:
    adapter = SERS_AU_AG_CANDIDATE_UNIT_APPLICABILITY

    stop_context = adapter.relevance_context(
        "nanogap"
    )

    query = (
        "Au-Ag bimetallic nanostructure"
        f" ; {stop_context}"
        " ; candidate scientific bridge"
        " ; electromagnetic enhancement"
    )

    assert query == (
        "Au-Ag bimetallic nanostructure"
        " ; nanogap control"
        " ; candidate scientific bridge"
        " ; electromagnetic enhancement"
    )


def test_e2e_activates_candidate_depth13_without_changing_grounding_depth() -> None:
    args = argparse.Namespace(
        stop="nanogap",
        max_depth=12,
    )

    profile = get_domain_profile(
        "sers_au_ag"
    )

    argv, meta = (
        _candidate_unit_correction_contract(
            args,
            profile,
        )
    )

    assert meta["active"] is True
    assert meta["grounding_max_depth"] == 12
    assert meta["candidate_max_depth"] == 13
    assert meta["grounding_depth_changed"] is False

    assert argv == [
        "--semantic-stop",
        "nanogap",
        "--corrected-route-contract",
        "--max-depth",
        "13",
    ]


def test_e2e_does_not_activate_locked_chain_for_unsupported_stop() -> None:
    args = argparse.Namespace(
        stop="surface composition",
        max_depth=12,
    )

    profile = get_domain_profile(
        "sers_au_ag"
    )

    argv, meta = (
        _candidate_unit_correction_contract(
            args,
            profile,
        )
    )

    assert meta["active"] is False
    assert argv == [
        "--max-depth",
        "12",
    ]


class _FakeScore:
    def to_dict(self):
        return {
            "score": 1.0,
        }


class _FakeQuality:
    def score(self, row):
        return _FakeScore()


class _FakeRoute:
    def to_dict(self):
        return {
            "route_id": "route:test",
            "candidate_unit": {
                "unit_id": "candidate_unit:test",
                "candidate_node_id": "C",
                "entry_anchor_id": "A",
                "exit_anchor_id": "B",
                "entry_anchor_label": "A",
                "exit_anchor_label": "B",
                "label": "gap candidate",
                "proposed_subject": "",
                "proposed_relation": "",
                "proposed_object": "",
            },
            "source_match": {
                "node_id": "S",
                "semantic_similarity": 1.0,
            },
            "target_match": {
                "node_id": "T",
                "semantic_similarity": 1.0,
            },
            "nodes": [
                "S",
                "H",
                "A",
                "C",
                "B",
                "H",
                "T",
            ],
            "total_cost": 7.0,
            "candidate_unit_selection": {
                "total": 0.4,
            },
            "context_node_labels": [],
            # Corrected selector provenance.
            "visited_paper_ids": [
                "P1",
                "P2",
            ],
        }


def _edge(
    *,
    edge_class="scientific_confirmed",
    candidate=False,
    reverse=False,
):
    return {
        "relation":
            "GROUNDS_SEMANTIC_CANDIDATE"
            if candidate
            else "APPLIES_TO",
        "edge_class":
            "semantic_candidate"
            if candidate
            else edge_class,
        "exploration_cost": 1.0,
        "requires_verification": candidate,
        "reverse_navigation": reverse,
        "source_paper_ids_json": "[]",
    }


def test_corrected_materializer_does_not_repollute_alignment_hub_provenance() -> None:
    g = nx.DiGraph()

    g.add_node("S", type="Material")
    g.add_node(
        "H",
        type="CorpusAlignment",
        graph_layer="corpus_alignment",
        source_paper_ids_json='["HUB_A","HUB_B","HUB_C"]',
    )
    g.add_node(
        "A",
        type="Nanostructure",
        source_paper_id="P1",
    )
    g.add_node(
        "C",
        type="BridgeConcept",
        requires_verification=True,
    )
    g.add_node(
        "B",
        type="Nanostructure",
        source_paper_id="P2",
    )
    g.add_node("T", type="MechanismClaim")

    g.add_edge(
        "S",
        "H",
        **_edge(edge_class="registry_alignment"),
    )
    g.add_edge(
        "H",
        "A",
        **_edge(edge_class="registry_alignment"),
    )
    g.add_edge(
        "A",
        "C",
        **_edge(candidate=True),
    )
    g.add_edge(
        "C",
        "B",
        **_edge(candidate=True, reverse=True),
    )
    g.add_edge(
        "B",
        "H",
        **_edge(edge_class="registry_alignment"),
    )
    g.add_edge(
        "H",
        "T",
        **_edge(edge_class="registry_alignment"),
    )

    corrected = _materialize_route(
        g,
        _FakeRoute(),
        _FakeQuality(),
        corrected_route_contract=True,
        semantic_stop="nanogap",
    )

    legacy = _materialize_route(
        g,
        _FakeRoute(),
        _FakeQuality(),
        corrected_route_contract=False,
    )

    assert set(
        corrected["visited_paper_ids"]
    ) == {
        "P1",
        "P2",
    }

    assert "HUB_A" not in corrected["visited_paper_ids"]
    assert "HUB_B" not in corrected["visited_paper_ids"]
    assert "HUB_C" not in corrected["visited_paper_ids"]

    assert "HUB_A" in legacy["visited_paper_ids"]
