from pipeline_core.discovery.adaptive_graph_retraversal import (
    audit_context_delta,
    context_footprint,
    materialize_selected_traversal,
    merge_footprints,
    select_retraversal_paths,
    traversal_footprint,
)


def _path(
    pid,
    edges,
    papers,
    *,
    nodes=None,
    mechanism=0.8,
):
    nodes = nodes or [
        "s",
        *(f"n{i}" for i in range(len(edges) - 1)),
        "t",
    ]
    return {
        "path_id": pid,
        "nodes": list(nodes),
        "steps": [
            {
                "selected_original_edge_id": edge,
                "source_paper_ids": [
                    papers[min(i, len(papers) - 1)]
                ]
                if papers
                else [],
            }
            for i, edge in enumerate(edges)
        ],
        "visited_paper_ids": list(papers),
        "total_cost": 1.0,
        "hop_count": len(edges),
        "path_quality": {
            "path_type": "MECHANISTIC",
            "mechanistic_content_score": mechanism,
        },
    }


def test_retraversal_excludes_exact_old_path():
    old = _path("old", ["e1", "e2"], ["p1"])
    new = _path("new", ["e1", "e3"], ["p1", "p2"])
    prior = traversal_footprint({"paths": [old]})
    selected, report = select_retraversal_paths(
        [old, new],
        prior=prior,
        top_k=4,
        min_new_edge_fraction=0.25,
        min_new_paper_fraction=0.25,
    )
    assert [row["path_id"] for row in selected] == ["new"]
    by_id = {
        row["path_id"]: row
        for row in report["diagnostics"]
    }
    assert by_id["old"]["exact_path_reuse"] is True


def test_context_and_traversal_footprints_merge():
    context = {
        "evidence_statements": [
            {
                "statement_id": "s1",
                "support_path_ids": ["p-old"],
                "alignment_path_ids": [],
                "scientific_support_edge_ids": ["e-context"],
                "scientific_support_node_ids": ["n-context"],
                "paper_ids": ["paper-context"],
            }
        ],
        "mechanism_routes": [],
    }
    traversal = {
        "paths": [
            _path(
                "p-traversal",
                ["e-traversal"],
                ["paper-traversal"],
            )
        ]
    }
    merged = merge_footprints(
        [
            context_footprint(context),
            traversal_footprint(traversal),
        ]
    )
    assert "e-context" in merged["edge_ids"]
    assert "e-traversal" in merged["edge_ids"]
    assert "paper-context" in merged["paper_ids"]
    assert "paper-traversal" in merged["paper_ids"]


def test_materialized_retraversal_is_path_scoped():
    selected = [
        _path("new", ["e3", "e4"], ["p2"])
    ]
    broad = {
        "corpus_id": "c",
        "domain_profile_id": "d",
        "mode": "mechanism",
        "algorithm": "top_n",
        "paths": selected,
        "candidate_paths": selected,
        "direct_concept_hits": [
            {"hit_id": "old-direct"}
        ],
        "direct_concept_hit_count": 1,
    }
    report = {
        "selection_id": "sel",
        "selected_path_ids": ["new"],
    }
    materialized = materialize_selected_traversal(
        broad,
        selected_paths=selected,
        selection_report=report,
        source_context_id="ctx-old",
    )
    assert materialized["path_count"] == 1
    assert materialized["paths"][0]["path_id"] == "new"
    assert materialized["direct_concept_hits"] == []
    assert (
        materialized["adaptive_graph_retraversal"]["source_context_id"]
        == "ctx-old"
    )


def test_retraversal_reserves_new_paper_path_when_available():
    old = _path("old", ["e1"], ["p1"], mechanism=0.9)
    same_paper = _path(
        "same-paper",
        ["e2", "e3"],
        ["p1"],
        mechanism=1.0,
    )
    new_paper = _path(
        "new-paper",
        ["e4", "e5"],
        ["p2"],
        mechanism=0.1,
    )
    prior = traversal_footprint({"paths": [old]})
    selected, report = select_retraversal_paths(
        [same_paper, new_paper],
        prior=prior,
        top_k=1,
        paper_expansion_reserve=1,
    )
    assert [row["path_id"] for row in selected] == ["new-paper"]
    assert report["paper_expansion_candidate_count"] == 1
    assert report["selected_paper_expansion_count"] == 1


def test_context_delta_requires_structural_support_not_new_id_only():
    source = {
        "context_id": "old",
        "evidence_statements": [
            {
                "statement_id": "s-old",
                "text": "A increases B",
                "epistemic_role": "reported",
                "eligible_as_premise": True,
                "requires_verification": False,
                "premise_restrictions": [],
                "scientific_support_edge_ids": ["e1"],
                "scientific_support_node_ids": ["n1"],
                "support_path_ids": ["p1"],
                "alignment_path_ids": [],
                "paper_ids": ["paper1"],
            }
        ],
    }
    output = {
        "context_id": "new",
        "evidence_statements": [
            {
                "statement_id": "s-renamed",
                "text": "A increases B",
                "epistemic_role": "reported",
                "eligible_as_premise": True,
                "requires_verification": False,
                "premise_restrictions": [],
                "scientific_support_edge_ids": ["e1"],
                "scientific_support_node_ids": ["n1"],
                "support_path_ids": ["p1"],
                "alignment_path_ids": [],
                "paper_ids": ["paper1"],
            },
            {
                "statement_id": "s-structural",
                "text": "C moderates B",
                "epistemic_role": "reported",
                "eligible_as_premise": True,
                "requires_verification": False,
                "premise_restrictions": [],
                "scientific_support_edge_ids": ["e2"],
                "scientific_support_node_ids": ["n2"],
                "support_path_ids": ["p2"],
                "alignment_path_ids": [],
                "paper_ids": ["paper1"],
            },
        ],
    }
    report = audit_context_delta(
        source_context=source,
        output_context=output,
    )
    assert report["id_new_eligible_premise_count"] == 2
    assert report["structurally_new_eligible_premise_ids"] == [
        "s-structural"
    ]
    by_id = {
        row["statement_id"]: row
        for row in report["statement_deltas"]
    }
    assert by_id["s-renamed"]["structurally_new_eligible_premise"] is False
    assert by_id["s-structural"]["novelty_class"] == "NEW_GRAPH_SUPPORT"
