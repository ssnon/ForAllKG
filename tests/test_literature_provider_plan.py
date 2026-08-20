from __future__ import annotations

import json

import pytest

from pipeline_core.discovery.prior_art_provider_plan import (
    build_literature_providers,
    require_standard_or_full_auto_plan,
    resolve_literature_provider_plan,
)


def test_auto_without_s2_key_is_standard_two_provider():
    env = {
        "OPENALEX_API_KEY": "oa-secret",
        "CROSSREF_MAILTO": "a@example.org",
    }
    plan = resolve_literature_provider_plan(
        env=env
    )
    assert plan.mode == "STANDARD_2_PROVIDER"
    assert plan.active_providers == [
        "openalex",
        "crossref",
    ]
    assert (
        plan.semantic_scholar_api_key_configured
        is False
    )


def test_auto_with_s2_key_is_full_three_provider():
    env = {
        "OPENALEX_API_KEY": "oa-secret",
        "CROSSREF_MAILTO": "a@example.org",
        "SEMANTIC_SCHOLAR_API_KEY": "s2-secret",
    }
    plan = resolve_literature_provider_plan(
        env=env
    )
    assert plan.mode == "FULL_3_PROVIDER"
    assert plan.active_providers == [
        "openalex",
        "crossref",
        "semantic_scholar",
    ]


def test_auto_without_openalex_is_degraded_not_standard():
    env = {
        "CROSSREF_MAILTO": "a@example.org",
    }
    plan = resolve_literature_provider_plan(
        env=env
    )
    assert plan.mode == "DEGRADED_PROVIDER_SET"
    assert plan.active_providers == [
        "crossref",
    ]


def test_plan_never_persists_secret_values():
    env = {
        "OPENALEX_API_KEY": "oa-super-secret",
        "SEMANTIC_SCHOLAR_API_KEY": "s2-super-secret",
        "CROSSREF_MAILTO": "a@example.org",
    }
    plan = resolve_literature_provider_plan(
        env=env
    )
    encoded = json.dumps(
        plan.model_dump(mode="json")
    )
    assert "oa-super-secret" not in encoded
    assert "s2-super-secret" not in encoded


def test_explicit_s2_without_key_fails_closed():
    with pytest.raises(
        ValueError,
        match="SEMANTIC_SCHOLAR_API_KEY",
    ):
        resolve_literature_provider_plan(
            env={
                "OPENALEX_API_KEY":
                    "oa-secret",
            },
            requested=[
                "semantic_scholar",
                "crossref",
            ],
        )


def test_instances_follow_frozen_provider_order():
    env = {
        "OPENALEX_API_KEY": "oa-secret",
        "CROSSREF_MAILTO": "a@example.org",
    }
    plan = resolve_literature_provider_plan(
        env=env
    )
    providers = build_literature_providers(
        plan,
        env=env,
    )
    assert [
        row.provider_name
        for row in providers
    ] == [
        "openalex",
        "crossref",
    ]


def test_environment_drift_after_freeze_is_rejected():
    plan = resolve_literature_provider_plan(
        env={
            "OPENALEX_API_KEY": "oa-secret",
        }
    )
    with pytest.raises(
        RuntimeError,
        match="changed after",
    ):
        build_literature_providers(
            plan,
            env={
                "OPENALEX_API_KEY": "oa-secret",
                "SEMANTIC_SCHOLAR_API_KEY":
                    "later-added",
            },
        )


def test_explicit_provider_order_is_canonicalized():
    env = {
        "OPENALEX_API_KEY": "oa-secret",
        "SEMANTIC_SCHOLAR_API_KEY": "s2-secret",
    }
    plan = resolve_literature_provider_plan(
        env=env,
        requested=[
            "semantic_scholar",
            "crossref",
            "openalex",
        ],
    )
    assert plan.mode == "EXPLICIT_PROVIDER_SET"
    assert plan.active_providers == [
        "openalex",
        "crossref",
        "semantic_scholar",
    ]


def test_auto_crossref_only_is_rejected_for_default_scientific_run():
    plan = resolve_literature_provider_plan(
        env={
            "CROSSREF_MAILTO": "a@example.org",
        }
    )
    with pytest.raises(
        RuntimeError,
        match="OPENALEX_API_KEY",
    ):
        require_standard_or_full_auto_plan(
            plan
        )
