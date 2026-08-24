from __future__ import annotations

from pathlib import Path

from domains.registry import (
    get_domain_profile,
)
from scripts.discovery.run_dac_discovery_e2e import (
    _resolve_context_review_capability,
)


E2E = Path(
    "scripts/discovery/"
    "run_dac_discovery_e2e.py"
)

MAKER = Path(
    "scripts/discovery/"
    "run_discovery_axis_hypothesis_maker.py"
)


def test_sers_e2e_resolves_context_capability():
    adapter = (
        _resolve_context_review_capability(
            get_domain_profile(
                "sers_au_ag"
            )
        )
    )

    assert adapter is not None
    assert adapter.adapter_id == "sers_au_ag"


def test_non_sers_domains_do_not_inherit_context_capability():
    for profile_id in (
        "dac_her",
        "catalysis_mechanism",
    ):
        assert (
            _resolve_context_review_capability(
                get_domain_profile(
                    profile_id
                )
            )
            is None
        )


def test_e2e_requires_context_artifact_only_through_capability():
    source = E2E.read_text(
        encoding="utf-8"
    )

    assert (
        'axis_context = run / '
        '"hypothesis_axis_a4.context.json"'
        in source
    )

    assert (
        "if context_review_adapter is not None:"
        in source
    )

    assert (
        "stage8_expected.append("
        in source
    )

    assert (
        "axis_context"
        in source
    )

    assert (
        "expected=stage8_expected"
        in source
    )


def test_e2e_explicitly_forwards_context_critic_model():
    source = E2E.read_text(
        encoding="utf-8"
    )

    assert (
        '"--context-critic-model"'
        in source
    )

    assert (
        "str(args.critic_model)"
        in source
    )


def test_e2e_manifest_preserves_s1_g1_boundary():
    source = E2E.read_text(
        encoding="utf-8"
    )

    assert (
        '"context_review"'
        in source
    )

    assert (
        '"action_policy_applied"'
        in source
    )

    assert (
        '"g1_action_policy_deferred"'
        in source
    )

    assert (
        'context_payload.get('
        in source
    )


def test_maker_context_artifact_has_minimum_provenance():
    source = MAKER.read_text(
        encoding="utf-8"
    )

    for token in (
        '"model": context_model',
        '"final_record_count"',
        '"review_history_count"',
        '"action_policy_applied": False',
        '"g1_action_policy_deferred": True',
        '"grounded_source_graph_sha256"',
        '"axis_source_graph_sha256"',
        '"context_source_policy"',
    ):
        assert token in source


def test_context_artifact_record_count_must_match_alpha4_acceptance():
    source = E2E.read_text(
        encoding="utf-8"
    )

    assert (
        "len(records)"
        in source
    )

    assert (
        "initial_hypotheses"
        in source
    )

    assert (
        "Final context-review record count"
        in source
    )
