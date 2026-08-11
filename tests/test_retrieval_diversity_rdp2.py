from __future__ import annotations

from dac_her.path_bundle import PathBundlePolicy, PathBundleSelector


def _path(path_id, papers, endpoint, edge):
    return {
        "path_id": path_id,
        "visited_paper_ids": papers,
        "endpoint_pair": {
            "source_node_id": endpoint[0],
            "target_node_id": endpoint[1],
        },
        "steps": [
            {
                "navigation_edge_id": edge,
                "source": endpoint[0],
                "target": endpoint[1],
                "relation": "RELATED_TO",
            }
        ],
    }


def _selected_ids(result):
    return [row["path_id"] for row in result.selected_paths]


def _selected_papers(result):
    papers = set()
    for row in result.selected_paths:
        papers.update(row.get("visited_paper_ids", []))
    return papers


def test_coverage_first_skips_high_rank_repeats_to_expose_new_papers():
    rows = [
        _path("p1", ["A", "B"], ("s1", "t1"), "e1"),
        _path("p2", ["A", "B"], ("s2", "t2"), "e2"),
        _path("p3", ["A", "C"], ("s3", "t3"), "e3"),
        _path("p4", ["D", "E"], ("s4", "t4"), "e4"),
    ]
    selector = PathBundleSelector(
        policy=PathBundlePolicy(
            max_per_endpoint_pair=2,
            max_per_paper_signature=2,
            max_edge_jaccard=0.8,
        ),
        coverage_first=True,
    )
    result = selector.select(rows, top_k=3)
    assert _selected_ids(result) == ["p1", "p3", "p4"]
    assert _selected_papers(result) == {"A", "B", "C", "D", "E"}


def test_new_signature_pass_runs_after_paper_coverage_is_exhausted():
    rows = [
        _path("p1", ["A", "B"], ("s1", "t1"), "e1"),
        _path("p2", ["A", "B"], ("s2", "t2"), "e2"),
        _path("p3", ["A", "C"], ("s3", "t3"), "e3"),
        _path("p4", ["B", "C"], ("s4", "t4"), "e4"),
    ]
    result = PathBundleSelector(coverage_first=True).select(rows, top_k=3)
    assert _selected_ids(result) == ["p1", "p3", "p4"]
    assert result.selected_paths[2]["bundle_selection"]["selection_pass"] == "strict_new_signature"


def test_disabling_coverage_first_reproduces_legacy_scan_order():
    rows = [
        _path("p1", ["A", "B"], ("s1", "t1"), "e1"),
        _path("p2", ["A", "B"], ("s2", "t2"), "e2"),
        _path("p3", ["A", "C"], ("s3", "t3"), "e3"),
    ]
    result = PathBundleSelector(coverage_first=False).select(rows, top_k=2)
    assert _selected_ids(result) == ["p1", "p2"]


def test_coverage_first_still_fills_when_candidate_pool_has_no_diversity():
    rows = [
        _path("p1", ["A", "B"], ("s1", "t1"), "e1"),
        _path("p2", ["A", "B"], ("s2", "t2"), "e2"),
    ]
    result = PathBundleSelector(coverage_first=True).select(rows, top_k=2)
    assert _selected_ids(result) == ["p1", "p2"]
