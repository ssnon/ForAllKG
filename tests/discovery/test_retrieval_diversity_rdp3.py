from __future__ import annotations

from scripts.discovery.run_graph_traversal import _path_sort_key


def _row(
    path_id,
    pair_score=0.60,
    cost=10.0,
    hops=7,
    mechanism="low",
    mechanism_score=0.0,
    navigation=0.8,
    reverse=0.2,
):
    return {
        "path_id": path_id,
        "total_cost": cost,
        "hop_count": hops,
        "endpoint_pair": {
            "semantic_tier": 2,
            "pair_score": pair_score,
        },
        "path_quality": {
            "mechanistic_content": mechanism,
            "mechanistic_content_score": mechanism_score,
            "navigation_edge_fraction": navigation,
            "reverse_fraction": reverse,
        },
    }


def test_quality_aware_prefers_mechanistic_path_with_same_endpoint_relevance():
    cheap_bridge = _row(
        "bridge",
        cost=8.0,
        mechanism="low",
        navigation=0.85,
    )
    mechanistic = _row(
        "mechanistic",
        cost=11.0,
        mechanism="high",
        mechanism_score=0.40,
        navigation=0.70,
    )
    ranked = sorted(
        [cheap_bridge, mechanistic],
        key=lambda row: _path_sort_key(
            row,
            quality_aware=True,
        ),
    )
    assert [row["path_id"] for row in ranked] == [
        "mechanistic",
        "bridge",
    ]


def test_legacy_sort_is_preserved_when_quality_aware_is_disabled():
    cheap_bridge = _row(
        "bridge",
        cost=8.0,
        mechanism="low",
    )
    mechanistic = _row(
        "mechanistic",
        cost=11.0,
        mechanism="high",
        mechanism_score=0.50,
    )
    ranked = sorted(
        [cheap_bridge, mechanistic],
        key=lambda row: _path_sort_key(
            row,
            quality_aware=False,
        ),
    )
    assert [row["path_id"] for row in ranked] == [
        "bridge",
        "mechanistic",
    ]


def test_endpoint_pair_score_remains_primary_over_path_quality():
    higher_endpoint_bridge = _row(
        "higher-endpoint",
        pair_score=0.61,
        mechanism="low",
        cost=8.0,
    )
    lower_endpoint_mechanistic = _row(
        "lower-endpoint",
        pair_score=0.60,
        mechanism="high",
        mechanism_score=0.9,
        navigation=0.1,
        cost=20.0,
    )
    ranked = sorted(
        [
            lower_endpoint_mechanistic,
            higher_endpoint_bridge,
        ],
        key=lambda row: _path_sort_key(
            row,
            quality_aware=True,
        ),
    )
    assert ranked[0]["path_id"] == "higher-endpoint"


def test_navigation_burden_breaks_tie_inside_same_mechanistic_band():
    nav_heavy = _row(
        "nav-heavy",
        mechanism="medium",
        mechanism_score=0.30,
        navigation=0.90,
        cost=7.0,
    )
    nav_light = _row(
        "nav-light",
        mechanism="medium",
        mechanism_score=0.30,
        navigation=0.40,
        cost=12.0,
    )
    ranked = sorted(
        [nav_heavy, nav_light],
        key=lambda row: _path_sort_key(
            row,
            quality_aware=True,
        ),
    )
    assert ranked[0]["path_id"] == "nav-light"
