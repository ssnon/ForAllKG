from __future__ import annotations

import os
from pathlib import Path

import networkx as nx

from domains.context_review_registry import (
    get_context_review_adapter,
)
from domains.sers.context_review_adapter import (
    SERSDiscoveryAxisContextReviewer,
)


MAKER = Path(
    "scripts/discovery/"
    "run_discovery_axis_hypothesis_maker.py"
)


def test_domain_adapter_builds_dual_lane_backend_without_network():
    os.environ[
        "SERS_CONTEXT_TEST_KEY_DO_NOT_USE"
    ] = "dummy-key"

    adapter = get_context_review_adapter(
        "sers_au_ag"
    )

    reviewer = (
        adapter.build_openai_compatible(
            grounded_graph=nx.MultiDiGraph(),
            axis_graph=nx.MultiDiGraph(),
            model="test-context-model",
            api_key_env=(
                "SERS_CONTEXT_TEST_KEY_DO_NOT_USE"
            ),
        )
    )

    assert isinstance(
        reviewer,
        SERSDiscoveryAxisContextReviewer,
    )

    assert (
        reviewer.grounded_compiler
        is not reviewer.axis_compiler
    )


def test_shared_graph_override_remains_explicitly_supported():
    os.environ[
        "SERS_CONTEXT_TEST_KEY_DO_NOT_USE"
    ] = "dummy-key"

    adapter = get_context_review_adapter(
        "sers_au_ag"
    )

    graph = nx.MultiDiGraph()

    reviewer = (
        adapter.build_openai_compatible(
            graph=graph,
            model="test-context-model",
            api_key_env=(
                "SERS_CONTEXT_TEST_KEY_DO_NOT_USE"
            ),
        )
    )

    assert (
        reviewer.compiler
        is not None
    )


def test_generic_maker_resolves_external_context_capability():
    source = MAKER.read_text(
        encoding="utf-8"
    )

    assert (
        "available_context_review_profiles"
        in source
    )

    assert (
        "get_context_review_adapter"
        in source
    )

    assert (
        "from domains.sers"
        not in source
    )


def test_maker_injects_context_reviewer_into_runtime():
    source = MAKER.read_text(
        encoding="utf-8"
    )

    assert (
        "context_reviewer=context_reviewer"
        in source
    )


def test_maker_defaults_to_dual_source_graph_lanes():
    source = MAKER.read_text(
        encoding="utf-8"
    )

    # Grounded lane:
    # mechanism/navigation/node_index -> mechanism/graph.graphml
    assert (
        "index_dir.parents[1]"
        in source
    )

    # Axis lane:
    # mechanism/navigation/node_index -> corpus root
    # -> exploratory/graph.graphml
    assert (
        "index_dir.parents[2]"
        in source
    )

    assert (
        '/ "exploratory"'
        in source
    )

    assert (
        "context_grounded_graph_path"
        in source
    )

    assert (
        "context_axis_graph_path"
        in source
    )

    assert (
        "grounded_graph=("
        in source
    )

    assert (
        "axis_graph=("
        in source
    )


def test_context_model_defaults_to_inference_critic():
    source = MAKER.read_text(
        encoding="utf-8"
    )

    assert (
        "args.context_critic_model"
        in source
    )

    assert (
        "or args.inference_critic_model"
        in source
    )


def test_maker_persists_dual_lane_context_artifact_v2():
    source = MAKER.read_text(
        encoding="utf-8"
    )

    for token in (
        "discovery-axis-context-artifact-v2",
        "sers-dual-lane-claim-local-v1",
        '"grounded_source_graph"',
        '"grounded_source_graph_sha256"',
        '"axis_source_graph"',
        '"axis_source_graph_sha256"',
        "outcome.context_reviews",
        "outcome.context_review_history",
    ):
        assert token in source


def test_s1_context_findings_are_not_action_policy():
    source = MAKER.read_text(
        encoding="utf-8"
    )

    assert (
        'context_review.status == "reframe_required"'
        not in source
    )

    assert (
        'context_review.status != "pass"'
        not in source
    )
