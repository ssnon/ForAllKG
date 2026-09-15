from __future__ import annotations

import inspect

from scripts.discovery import run_dac_discovery_e2e as e2e


def _src() -> str:
    return inspect.getsource(
        e2e._run_scientific_novelty_action_shadow_chain
    )


def test_scientific_shadow_derives_all_three_diagnostic_siblings():
    src = _src()

    assert ".diagnostic_queries.json" in src
    assert ".diagnostic_prior_art.json" in src
    assert ".diagnostic_review.json" in src


def test_scientific_shadow_forwards_complete_diagnostic_bundle_to_10sa():
    src = _src()

    assert '"--external-diagnostic-query-plan"' in src
    assert '"--external-diagnostic-prior-art"' in src
    assert '"--external-diagnostic-review"' in src
    assert "*scientific_diagnostic_args" in src


def test_semantic_shadow_forwards_diagnostic_prior_and_review_to_10sb():
    src = _src()

    assert "*semantic_diagnostic_args" in src


def test_partial_diagnostic_provenance_fails_closed():
    src = _src()

    assert "any(diagnostic_presence) and not all(diagnostic_presence)" in src
    assert "External novelty diagnostic provenance is partial" in src
