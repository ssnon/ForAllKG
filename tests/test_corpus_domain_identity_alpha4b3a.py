from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from dac_her.corpus_graph import ProjectionBundle, build_corpus_graph
from domains.sers.profile import SERS_AU_AG_PROFILE


def _bundle(
    paper_id: str,
    *,
    domain_profile_id: str | None,
) -> ProjectionBundle:
    graph = nx.MultiDiGraph()
    graph.add_node("ag", type="Metal", label="Ag")
    summary = {}
    if domain_profile_id is not None:
        summary["domain_profile_id"] = domain_profile_id
    return ProjectionBundle(
        paper_id=paper_id,
        mode="exploratory",
        root=Path("."),
        graph_path=Path(f"{paper_id}.graphml"),
        node_text_path=Path(f"{paper_id}.nodes.jsonl"),
        edge_evidence_path=Path(f"{paper_id}.edges.jsonl"),
        summary_path=Path(f"{paper_id}.summary.json"),
        graph=graph,
        node_rows=[],
        evidence_rows=[],
        summary=summary,
        sha256={},
    )


def test_alpha4b3a_explicit_domain_rejects_mixed_projection_bundles() -> None:
    with pytest.raises(ValueError, match="Corpus domain mismatch"):
        build_corpus_graph(
            [
                _bundle("S1", domain_profile_id="sers_au_ag"),
                _bundle("H1", domain_profile_id="dac_her"),
            ],
            corpus_id="mixed",
            mode="exploratory",
            domain_profile=SERS_AU_AG_PROFILE,
        )


def test_alpha4b3a_explicit_domain_rejects_missing_projection_identity() -> None:
    with pytest.raises(ValueError, match="Corpus domain mismatch"):
        build_corpus_graph(
            [_bundle("S1", domain_profile_id=None)],
            corpus_id="missing",
            mode="exploratory",
            domain_profile=SERS_AU_AG_PROFILE,
        )
