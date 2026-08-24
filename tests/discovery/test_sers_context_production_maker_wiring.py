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


def test_domain_adapter_builds_context_backend_without_network():
    os.environ[
        "SERS_CONTEXT_TEST_KEY_DO_NOT_USE"
    ] = "dummy-key"

    adapter = get_context_review_adapter(
        "sers_au_ag"
    )

    reviewer = (
        adapter.build_openai_compatible(
            graph=nx.MultiDiGraph(),
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


def test_maker_uses_mechanism_source_graph():
    source = MAKER.read_text(
        encoding="utf-8"
    )

    assert (
        "index_dir.parents[1]"
        in source
    )

    assert (
        '/ "graph.graphml"'
        in source
    )

    assert (
        "nx.read_graphml("
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


def test_maker_persists_final_and_historical_context_reviews():
    source = MAKER.read_text(
        encoding="utf-8"
    )

    assert (
        "discovery-axis-context-artifact-v1"
        in source
    )

    assert (
        '".context.json"'
        in source
    )

    assert (
        "outcome.context_reviews"
        in source
    )

    assert (
        "outcome.context_review_history"
        in source
    )

    assert (
        '"source_graph_sha256"'
        in source
    )


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
